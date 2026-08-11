from __future__ import annotations

import argparse
import copy
import gc
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ofgq.models import prepare_llama_for_inference
from ofgq.quantization.fp4 import quantize_tensor as quantize_fp4
from ofgq.utils.data import iter_sequence_batches, load_token_stream


METHODS = {
    "E3M0": 0,
    "E2M1": 1,
    "E1M2": 2,
    "OFGQ": None,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    return cfg


def resolve_dtype(name):
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[key]


def get_llama_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "layers")
    ):
        return model.model.model.layers
    raise RuntimeError("Could not locate LLaMA decoder layers.")


def infer_input_device(model):
    return model.get_input_embeddings().weight.device


def extract_hidden(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("Could not extract decoder hidden states from layer output.")


def parse_layers(layer_values, num_layers):
    if isinstance(layer_values, str):
        if layer_values.strip().lower() == "all":
            return list(range(num_layers))
        values = [int(x.strip()) for x in layer_values.split(",") if x.strip()]
    else:
        values = [int(x) for x in layer_values]
    indices = [x - 1 for x in values]
    invalid = [x for x in indices if x < 0 or x >= num_layers]
    if invalid:
        raise IndexError(
            f"Layer numbers are out of range for a {num_layers}-layer model: "
            f"{[x + 1 for x in invalid]}"
        )
    return indices


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "selected" not in metadata:
        raise KeyError("The metadata file does not contain a 'selected' field.")
    return metadata


def fixed_metadata(metadata, format_index):
    result = copy.deepcopy(metadata)
    for key, layers in result["selected"].items():
        for layer, values in layers.items():
            layers[layer] = [int(format_index)] * len(values)
    return result


def load_model(cfg):
    model_cfg = cfg["model"]
    runtime_cfg = cfg.get("runtime", {})
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        torch_dtype=resolve_dtype(model_cfg.get("dtype", "float16")),
        low_cpu_mem_usage=True,
        device_map=model_cfg.get("device_map", "auto"),
        attn_implementation=runtime_cfg.get("attn_implementation", "eager"),
    )
    model.eval()
    model.config.use_cache = False
    return model


def load_samples(cfg, tokenizer):
    data_cfg = cfg["dataset"]
    analysis_cfg = cfg["numerical_error"]
    input_ids = load_token_stream(
        tokenizer,
        data_cfg["evaluation_path"],
        text_column=data_cfg.get("text_column", "text"),
    )
    samples = []
    for batch in iter_sequence_batches(
        input_ids,
        int(analysis_cfg["sequence_length"]),
        max_samples=int(analysis_cfg["num_samples"]),
    ):
        samples.append(batch.clone().cpu())
    if not samples:
        raise RuntimeError("No complete evaluation sequences were found.")
    return samples


def local_format_data(metadata, layer_idx, method, hidden, group_size):
    if method == "OFGQ":
        try:
            return metadata["selected"]["q_proj"][str(layer_idx)]
        except KeyError as exc:
            raise KeyError(
                f"Missing q_proj metadata for decoder layer {layer_idx + 1}."
            ) from exc
    format_index = METHODS[method]
    padded_cols = ((hidden.shape[-1] + group_size - 1) // group_size) * group_size
    count = padded_cols // group_size
    return [format_index] * count


def evaluate_reference(cfg, samples, metadata, selected_layers):
    group_size = int(cfg["quantization"]["group_size"])
    model = load_model(cfg)
    if str(model.config.model_type).lower() != "llama":
        raise ValueError("numerical_error_eval.py currently supports LLaMA-family models.")
    layers = get_llama_layers(model)
    local_sse = defaultdict(float)
    local_count = defaultdict(int)
    reference_outputs = {layer_idx: [] for layer_idx in selected_layers}
    handles = []

    for layer_idx in selected_layers:
        layer = layers[layer_idx]

        def make_pre_hook(idx):
            def hook(_module, args):
                if not args or not isinstance(args[0], torch.Tensor):
                    return
                hidden = args[0].detach()
                for method in METHODS:
                    format_data = local_format_data(
                        metadata,
                        idx,
                        method,
                        hidden,
                        group_size,
                    )
                    quantized, _ = quantize_fp4(
                        hidden,
                        num_bits=4,
                        block_row=1,
                        block_col=group_size,
                        format_data=format_data,
                    )
                    diff = hidden.float() - quantized.float()
                    local_sse[(idx, method)] += float((diff * diff).sum().item())
                    local_count[(idx, method)] += int(diff.numel())
            return hook

        def make_post_hook(idx):
            def hook(_module, _args, output):
                hidden = extract_hidden(output).detach().to("cpu")
                reference_outputs[idx].append(hidden)
            return hook

        handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx)))
        handles.append(layer.register_forward_hook(make_post_hook(layer_idx)))

    input_device = infer_input_device(model)
    try:
        with torch.inference_mode():
            for sample in samples:
                model(
                    input_ids=sample.to(input_device),
                    use_cache=False,
                    output_attentions=False,
                    return_dict=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    local_rows = []
    for layer_idx in selected_layers:
        row = {"layer": layer_idx + 1}
        for method in METHODS:
            key = (layer_idx, method)
            row[method] = local_sse[key] / max(local_count[key], 1)
        local_rows.append(row)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(local_rows), reference_outputs


def evaluate_cascaded_method(cfg, samples, metadata, selected_layers, reference_outputs, method):
    model = load_model(cfg)
    if method == "OFGQ":
        method_metadata = metadata
    else:
        method_metadata = fixed_metadata(metadata, METHODS[method])

    model = prepare_llama_for_inference(
        model,
        metadata=method_metadata,
        num_bits=int(cfg["quantization"].get("num_bits", 4)),
        group_size=int(cfg["quantization"]["group_size"]),
    )

    layers = get_llama_layers(model)
    sse = defaultdict(float)
    count = defaultdict(int)
    sample_index = {"value": -1}
    handles = []

    for layer_idx in selected_layers:
        layer = layers[layer_idx]

        def make_hook(idx):
            def hook(_module, _args, output):
                hidden = extract_hidden(output).detach()
                ref = reference_outputs[idx][sample_index["value"]].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                    non_blocking=True,
                )
                diff = hidden.float() - ref.float()
                sse[idx] += float((diff * diff).sum().item())
                count[idx] += int(diff.numel())
            return hook

        handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    input_device = infer_input_device(model)
    try:
        with torch.inference_mode():
            for idx, sample in enumerate(samples):
                sample_index["value"] = idx
                model(
                    input_ids=sample.to(input_device),
                    use_cache=False,
                    output_attentions=False,
                    return_dict=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    result = {layer_idx: sse[layer_idx] / max(count[layer_idx], 1) for layer_idx in selected_layers}

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    cfg = load_config(args.config)
    analysis_cfg = cfg["numerical_error"]
    metadata = load_metadata(analysis_cfg["metadata_path"])

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["path"],
        use_fast=bool(cfg.get("runtime", {}).get("use_fast_tokenizer", False)),
    )
    samples = load_samples(cfg, tokenizer)

    probe = load_model(cfg)
    if str(probe.config.model_type).lower() != "llama":
        raise ValueError("numerical_error_eval.py currently supports LLaMA-family models.")
    num_layers = len(get_llama_layers(probe))
    selected_layers = parse_layers(analysis_cfg["layers"], num_layers)
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    local_df, reference_outputs = evaluate_reference(
        cfg,
        samples,
        metadata,
        selected_layers,
    )

    cascaded_rows = {layer_idx: {"layer": layer_idx + 1} for layer_idx in selected_layers}
    for method in METHODS:
        values = evaluate_cascaded_method(
            cfg,
            samples,
            metadata,
            selected_layers,
            reference_outputs,
            method,
        )
        for layer_idx in selected_layers:
            cascaded_rows[layer_idx][method] = values[layer_idx]

    cascaded_df = pd.DataFrame([cascaded_rows[idx] for idx in selected_layers])
    combined_df = pd.concat(
        [
            local_df.assign(error_type="Local"),
            cascaded_df.assign(error_type="Cascaded"),
        ],
        ignore_index=True,
    )[["error_type", "layer", "E3M0", "E2M1", "E1M2", "OFGQ"]]

    output_dir = Path(analysis_cfg.get("output_dir", "outputs/numerical_error"))
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "local_decoder_input_mse.csv"
    cascaded_path = output_dir / "cascaded_decoder_output_mse.csv"
    combined_path = output_dir / "combined_mse.csv"

    local_df.to_csv(local_path, index=False, float_format="%.10e")
    cascaded_df.to_csv(cascaded_path, index=False, float_format="%.10e")
    combined_df.to_csv(combined_path, index=False, float_format="%.10e")

    print("Local MSE")
    print(local_df.to_string(index=False))
    print("Cascaded MSE")
    print(cascaded_df.to_string(index=False))
    print(f"Saved: {local_path}")
    print(f"Saved: {cascaded_path}")
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
