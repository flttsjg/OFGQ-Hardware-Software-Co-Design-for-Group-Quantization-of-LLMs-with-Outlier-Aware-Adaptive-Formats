
























from __future__ import annotations

import argparse
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sample-wise R_cons.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must contain a mapping.")
    return cfg


def get_required(cfg: dict, section: str, key: str):
    if section not in cfg or key not in cfg[section]:
        raise KeyError(f"Missing config entry: {section}.{key}")
    return cfg[section][key]


def resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = str(name).lower()
    if normalized not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[normalized]






def load_texts(dataset_path: str, text_column: str, max_rows: int = 0) -> List[str]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=[text_column])
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=[text_column])
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
        if text_column not in df.columns:
            raise KeyError(
                f'Column "{text_column}" not found. Available columns: {list(df.columns)}'
            )
        df = df[[text_column]]
    elif suffix in {".txt", ".text"}:
        with path.open("r", encoding="utf-8") as f:
            values = [line.rstrip("\n") for line in f]
        if max_rows > 0:
            values = values[:max_rows]
        texts = [x for x in values if isinstance(x, str) and x.strip()]
        if not texts:
            raise ValueError("No non-empty text was found.")
        return texts
    else:
        raise ValueError(
            "Unsupported dataset type. Supported: parquet, csv, json/jsonl, txt."
        )

    if max_rows > 0:
        df = df.iloc[:max_rows]

    texts = [
        str(value)
        for value in df[text_column].tolist()
        if isinstance(value, str) and value.strip()
    ]
    if not texts:
        raise ValueError(f'No non-empty text was found in column "{text_column}".')
    return texts


def tokenize_corpus(
    tokenizer,
    texts: Sequence[str],
    eos_token_id: Optional[int],
    batch_rows: int = 256,
) -> Tensor:
    chunks: List[Tensor] = []
    separator = [] if eos_token_id is None else [int(eos_token_id)]

    for start in range(0, len(texts), batch_rows):
        batch = list(texts[start : start + batch_rows])
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        flattened: List[int] = []
        for token_list in encoded:
            flattened.extend(token_list)
            flattened.extend(separator)
        if flattened:
            chunks.append(torch.tensor(flattened, dtype=torch.long))

    if not chunks:
        raise ValueError("Tokenization produced no tokens.")
    return torch.cat(chunks, dim=0)


def build_samples(
    token_ids: Tensor,
    num_samples: int,
    sequence_length: int,
    sample_mode: str,
    seed: int,
) -> Tensor:
    if num_samples <= 0:
        raise ValueError("analysis.num_samples must be positive.")
    if sequence_length <= 0:
        raise ValueError("analysis.sequence_length must be positive.")
    if token_ids.numel() < sequence_length:
        raise ValueError(
            f"Corpus contains {token_ids.numel()} tokens, fewer than "
            f"sequence_length={sequence_length}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if sample_mode == "random":
        max_start = token_ids.numel() - sequence_length
        if max_start == 0:
            starts = torch.zeros(num_samples, dtype=torch.long)
        else:
            starts = torch.randint(
                0, max_start + 1, (num_samples,), generator=generator
            )
    elif sample_mode == "sequential":
        available = token_ids.numel() // sequence_length
        if available < num_samples:
            raise ValueError(
                f"Sequential mode provides {available} complete samples, "
                f"fewer than requested {num_samples}."
            )
        starts = torch.arange(num_samples, dtype=torch.long) * sequence_length
    else:
        raise ValueError('analysis.sample_mode must be "random" or "sequential".')

    return torch.stack(
        [
            token_ids[int(start) : int(start) + sequence_length]
            for start in starts
        ],
        dim=0,
    )






def parse_layer_spec(spec: str, num_layers: int) -> List[int]:
    normalized = str(spec).strip().lower()
    if normalized == "all":
        return list(range(num_layers))

    selected = set()
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if re.fullmatch(r"\d+", item):
            selected.add(int(item))
            continue
        if re.fullmatch(r"\d+\s*-\s*\d+", item):
            left, right = [int(x.strip()) for x in item.split("-")]
            if left > right:
                left, right = right, left
            selected.update(range(left, right + 1))
            continue
        raise ValueError(
            f'Invalid layer item "{item}". Use "all", "0,8,16", or "0-3,8".'
        )

    result = sorted(selected)
    invalid = [idx for idx in result if idx < 0 or idx >= num_layers]
    if invalid:
        raise IndexError(
            f"Layer indices out of range: {invalid}; valid range is 0-{num_layers - 1}."
        )
    if not result:
        raise ValueError("No layers were selected.")
    return result


def get_decoder_layers(model: nn.Module, model_type: str) -> Sequence[nn.Module]:
    getters: List[Callable[[nn.Module], Sequence[nn.Module]]]

    if model_type == "opt":
        getters = [
            lambda m: m.model.decoder.layers,
            lambda m: m.model.model.decoder.layers,
        ]
    elif model_type == "llama":
        getters = [
            lambda m: m.model.layers,
            lambda m: m.model.model.layers,
        ]
    else:
        raise ValueError(
            f'Unsupported model_type="{model_type}". This analysis supports OPT and LLaMA.'
        )

    for getter in getters:
        try:
            layers = getter(model)
            if len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise AttributeError(f"Could not locate decoder layers for model_type={model_type}.")


def get_activation_input_modules(
    layer: nn.Module,
    model_type: str,
) -> Mapping[str, nn.Module]:

    if model_type == "opt":
        return {
            "X_qkv": layer.self_attn.q_proj,
            "X_ao": layer.self_attn.out_proj,
            "X_fc1": layer.fc1,
            "X_fc2": layer.fc2,
        }
    if model_type == "llama":
        return {
            "X_qkv": layer.self_attn.q_proj,
            "X_ao": layer.self_attn.o_proj,

            "X_fc1": layer.mlp.gate_proj,
            "X_fc2": layer.mlp.down_proj,
        }
    raise ValueError(f"Unsupported model_type={model_type}")


def resolve_num_heads(config, layer: nn.Module) -> int:
    candidates = [
        getattr(layer.self_attn, "num_heads", None),
        getattr(layer.self_attn, "num_attention_heads", None),
        getattr(config, "num_attention_heads", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise AttributeError("Could not determine the number of attention heads.")


def reshape_projected_q(q: Tensor, num_heads: int) -> Tensor:

    if q.ndim != 3:
        raise ValueError(
            f"Expected projected Q with shape [B,S,D], got {tuple(q.shape)}."
        )
    batch, sequence, width = q.shape
    if width % num_heads != 0:
        raise ValueError(f"Q width {width} is not divisible by num_heads={num_heads}.")
    head_dim = width // num_heads
    return q.view(batch, sequence, num_heads, head_dim).transpose(1, 2)






@dataclass
class SampleResult:
    sample: int
    layer: int
    activation: str
    d_intra: float
    d_inter: float
    r_cons: float
    rows_used: int
    num_blocks: int
    feature_dim: int
    group_size: int
    num_bins: int
    pair_samples: int
    q_capture: str


def jensen_shannon_divergence(p: Tensor, q: Tensor) -> Tensor:
    eps = 1e-12
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (p.log() - midpoint.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def activation_to_histograms(
    activation: Tensor,
    group_size: int,
    num_bins: int,
    rows_per_sample: int,
    generator: torch.Generator,
) -> Tuple[Tensor, int, int, int]:





    if not isinstance(activation, Tensor):
        raise TypeError("Activation must be a torch.Tensor.")
    if activation.ndim < 2:
        raise ValueError(
            f"Activation must have at least 2 dimensions, got {activation.ndim}."
        )

    x = activation.detach()
    feature_dim = int(x.shape[-1])
    if feature_dim % group_size != 0:
        raise ValueError(
            f"Feature dimension {feature_dim} is not divisible by group_size={group_size}."
        )

    x = x.reshape(-1, feature_dim)
    total_rows = int(x.shape[0])
    if total_rows < 2:
        raise ValueError(f"At least two rows are required, got {total_rows}.")

    rows_used = min(rows_per_sample, total_rows)
    if rows_used < 2:
        raise ValueError("analysis.rows_per_sample must select at least two rows.")

    row_indices = torch.randperm(
        total_rows, generator=generator, device="cpu"
    )[:rows_used].to(x.device)
    selected = x.index_select(0, row_indices).float()

    num_blocks = feature_dim // group_size
    if num_blocks < 2:
        raise ValueError(
            f"R_cons requires at least two column blocks, but feature_dim={feature_dim} "
            f"and group_size={group_size} produce {num_blocks}."
        )

    groups = selected.reshape(rows_used, num_blocks, group_size)
    max_abs = groups.abs().amax(dim=-1, keepdim=True)
    normalized = groups / max_abs.clamp_min(1e-12)
    normalized = torch.where(
        max_abs > 0, normalized, torch.zeros_like(normalized)
    ).clamp(-1.0, 1.0)

    bin_indices = torch.floor(
        (normalized + 1.0) * (num_bins / 2.0)
    ).to(torch.long)
    bin_indices.clamp_(0, num_bins - 1)

    histograms = torch.zeros(
        (rows_used, num_blocks, num_bins),
        device=selected.device,
        dtype=torch.float32,
    )
    histograms.scatter_add_(
        dim=-1,
        index=bin_indices,
        src=torch.ones_like(normalized, dtype=torch.float32),
    )
    histograms.div_(float(group_size))
    return histograms, rows_used, num_blocks, feature_dim


@torch.no_grad()
def compute_sample_rcons(
    activation: Tensor,
    sample_idx: int,
    layer_idx: int,
    activation_name: str,
    group_size: int,
    num_bins: int,
    rows_per_sample: int,
    pair_samples: int,
    base_seed: int,
    q_capture: str,
) -> SampleResult:
    if pair_samples <= 0:
        raise ValueError("analysis.pair_samples must be positive.")


    name_offset = sum(
        (idx + 1) * ord(char) for idx, char in enumerate(activation_name)
    )
    local_seed = (
        int(base_seed)
        + int(sample_idx) * 1_000_003
        + int(layer_idx) * 10_007
        + int(name_offset)
    ) % (2**31 - 1)

    row_generator = torch.Generator(device="cpu")
    row_generator.manual_seed(local_seed)
    hist, rows_used, num_blocks, feature_dim = activation_to_histograms(
        activation=activation,
        group_size=group_size,
        num_bins=num_bins,
        rows_per_sample=rows_per_sample,
        generator=row_generator,
    )

    pair_generator = torch.Generator(device="cpu")
    pair_generator.manual_seed(local_seed + 97)


    row_a = torch.randint(0, rows_used, (pair_samples,), generator=pair_generator)
    row_b_raw = torch.randint(
        0, rows_used - 1, (pair_samples,), generator=pair_generator
    )
    row_b = row_b_raw + (row_b_raw >= row_a).to(torch.long)

    block_a = torch.randint(
        0, num_blocks, (pair_samples,), generator=pair_generator
    )

    row_a_d = row_a.to(hist.device)
    row_b_d = row_b.to(hist.device)
    block_a_d = block_a.to(hist.device)


    p_intra = hist[row_a_d, block_a_d]
    q_intra = hist[row_b_d, block_a_d]
    d_intra = float(jensen_shannon_divergence(p_intra, q_intra).mean().item())


    block_b_raw = torch.randint(
        0, num_blocks - 1, (pair_samples,), generator=pair_generator
    )
    block_b = block_b_raw + (block_b_raw >= block_a).to(torch.long)
    block_b_d = block_b.to(hist.device)

    p_inter = hist[row_a_d, block_a_d]
    q_inter = hist[row_b_d, block_b_d]
    d_inter = float(jensen_shannon_divergence(p_inter, q_inter).mean().item())

    r_cons = d_inter / max(d_intra, 1e-12)
    return SampleResult(
        sample=sample_idx,
        layer=layer_idx,
        activation=activation_name,
        d_intra=d_intra,
        d_inter=d_inter,
        r_cons=r_cons,
        rows_used=rows_used,
        num_blocks=num_blocks,
        feature_dim=feature_dim,
        group_size=group_size,
        num_bins=num_bins,
        pair_samples=pair_samples,
        q_capture=q_capture,
    )


class SampleWiseRecorder:
    def __init__(
        self,
        selected_layers: Sequence[int],
        group_size: int,
        num_bins: int,
        rows_per_sample: int,
        pair_samples: int,
        seed: int,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.group_size = group_size
        self.num_bins = num_bins
        self.rows_per_sample = rows_per_sample
        self.pair_samples = pair_samples
        self.seed = seed
        self.current_sample: Optional[int] = None
        self.results: List[SampleResult] = []
        self.recorded_keys: set[Tuple[int, int, str]] = set()

    def begin_sample(self, sample_idx: int) -> None:
        self.current_sample = int(sample_idx)

    def end_sample(self) -> None:
        self.current_sample = None

    def capture(
        self,
        layer_idx: int,
        activation_name: str,
        activation: Tensor,
        q_capture: str = "not_applicable",
    ) -> None:
        if self.current_sample is None or layer_idx not in self.selected_layers:
            return

        key = (self.current_sample, int(layer_idx), activation_name)
        if key in self.recorded_keys:
            return

        result = compute_sample_rcons(
            activation=activation,
            sample_idx=self.current_sample,
            layer_idx=int(layer_idx),
            activation_name=activation_name,
            group_size=self.group_size,
            num_bins=self.num_bins,
            rows_per_sample=self.rows_per_sample,
            pair_samples=self.pair_samples,
            base_seed=self.seed,
            q_capture=q_capture,
        )
        self.results.append(result)
        self.recorded_keys.add(key)


class HookManager:
    def __init__(self) -> None:
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def add_input_hook(
        self,
        module: nn.Module,
        layer_idx: int,
        activation_name: str,
        recorder: SampleWiseRecorder,
    ) -> None:
        def hook(_module: nn.Module, args: Tuple[object, ...]) -> None:
            if not args or not isinstance(args[0], Tensor):
                return
            recorder.capture(
                layer_idx=layer_idx,
                activation_name=activation_name,
                activation=args[0],
            )

        self.handles.append(module.register_forward_pre_hook(hook))

    def add_q_output_hook(
        self,
        module: nn.Module,
        layer_idx: int,
        num_heads: int,
        recorder: SampleWiseRecorder,
        q_capture: str,
    ) -> None:
        def hook(
            _module: nn.Module,
            _args: Tuple[object, ...],
            output: object,
        ) -> None:
            if not isinstance(output, Tensor):
                return
            q = reshape_projected_q(output, num_heads=num_heads)
            recorder.capture(
                layer_idx=layer_idx,
                activation_name="Q",
                activation=q,
                q_capture=q_capture,
            )

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()






def summarize_results(
    sample_df: pd.DataFrame,
    expected_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_df.empty:
        raise RuntimeError("No R_cons results were captured.")

    grouped = sample_df.groupby(["layer", "activation"], sort=True, observed=True)
    summary = grouped["R_cons"].agg(
        sample_count="count",
        mean_R_cons="mean",
        std_R_cons="std",
        median_R_cons="median",
        min_R_cons="min",
        max_R_cons="max",
    ).reset_index()

    count_gt1 = grouped["R_cons"].apply(
        lambda values: int((values > 1.0).sum())
    ).rename("count_R_cons_gt_1").reset_index()

    summary = summary.merge(count_gt1, on=["layer", "activation"], how="left")
    summary["percent_R_cons_gt_1"] = (
        100.0 * summary["count_R_cons_gt_1"] / summary["sample_count"]
    )
    summary["expected_samples"] = int(expected_samples)
    summary["missing_samples"] = summary["expected_samples"] - summary["sample_count"]

    percent_wide = summary.pivot(
        index="layer", columns="activation", values="percent_R_cons_gt_1"
    ).sort_index().reset_index()
    percent_wide.columns.name = None

    mean_wide = summary.pivot(
        index="layer", columns="activation", values="mean_R_cons"
    ).sort_index().reset_index()
    mean_wide.columns.name = None


    avg_percent = (
        summary.groupby("activation", sort=True)["percent_R_cons_gt_1"]
        .mean()
        .rename("Avg_percent_R_cons_gt_1")
        .reset_index()
    )

    return summary.sort_values(["layer", "activation"]), percent_wide, mean_wide, avg_percent


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "model"






def load_model_eager(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool,
):
    common_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if device == "auto":
        common_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            **common_kwargs,
        )
    except TypeError:

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        setattr(config, "_attn_implementation", "eager")
        setattr(config, "_attn_implementation_internal", "eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, **common_kwargs
        )

    if device != "auto":
        model.to(torch.device(device))

    model.eval()
    model.config.use_cache = False
    set_impl = getattr(model, "set_attn_implementation", None)
    if callable(set_impl):
        try:
            set_impl("eager")
        except Exception:
            pass
    return model


def infer_input_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device






def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("dataset", {})
    ana_cfg = cfg.get("analysis", {})

    model_path = get_required(cfg, "model", "path")
    dataset_path = get_required(cfg, "dataset", "path")

    text_column = data_cfg.get("text_column", "text")
    dtype_name = model_cfg.get("dtype", "float16")
    device = model_cfg.get("device", "cuda:0")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    layers_spec = str(ana_cfg.get("layers", "all"))
    num_samples = int(ana_cfg.get("num_samples", 100))
    sequence_length = int(ana_cfg.get("sequence_length", 2048))
    group_size = int(ana_cfg.get("group_size", 64))
    num_bins = int(ana_cfg.get("num_bins", 80))
    rows_per_sample = int(ana_cfg.get("rows_per_sample", 32))
    pair_samples = int(ana_cfg.get("pair_samples", 1024))
    sample_mode = str(ana_cfg.get("sample_mode", "random")).lower()
    seed = int(ana_cfg.get("seed", 42))
    max_corpus_rows = int(ana_cfg.get("max_corpus_rows", 0))
    output_dir = Path(ana_cfg.get("output_dir", "outputs/rcons"))
    output_prefix = ana_cfg.get("output_prefix")

    if group_size <= 0:
        raise ValueError("analysis.group_size must be positive.")
    if num_bins < 2:
        raise ValueError("analysis.num_bins must be at least 2.")
    if rows_per_sample < 2:
        raise ValueError("analysis.rows_per_sample must be at least 2.")
    if pair_samples <= 0:
        raise ValueError("analysis.pair_samples must be positive.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("[1/7] Reading model configuration and tokenizer...")
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model_type = str(config.model_type).lower()
    if model_type not in {"opt", "llama"}:
        raise ValueError(
            f'Unsupported model_type="{model_type}". Expected OPT or LLaMA.'
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/7] Loading and tokenizing dataset...")
    texts = load_texts(dataset_path, text_column, max_rows=max_corpus_rows)
    corpus_ids = tokenize_corpus(tokenizer, texts, tokenizer.eos_token_id)
    samples = build_samples(
        corpus_ids,
        num_samples=num_samples,
        sequence_length=sequence_length,
        sample_mode=sample_mode,
        seed=seed,
    )
    del texts, corpus_ids
    gc.collect()

    print("[3/7] Loading model...")
    model = load_model_eager(
        model_path=model_path,
        dtype=resolve_dtype(dtype_name),
        device=device,
        trust_remote_code=trust_remote_code,
    )

    layers = get_decoder_layers(model, model_type)
    selected_layers = parse_layer_spec(layers_spec, len(layers))
    print(
        f"Model type: {model_type}; total layers: {len(layers)}; "
        f"selected layers: {selected_layers}"
    )

    recorder = SampleWiseRecorder(
        selected_layers=selected_layers,
        group_size=group_size,
        num_bins=num_bins,
        rows_per_sample=rows_per_sample,
        pair_samples=pair_samples,
        seed=seed,
    )
    hooks = HookManager()

    print("[4/7] Registering activation hooks...")
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        for activation_name, module in get_activation_input_modules(
            layer, model_type
        ).items():
            hooks.add_input_hook(
                module=module,
                layer_idx=layer_idx,
                activation_name=activation_name,
                recorder=recorder,
            )


        q_capture = "projected_q" if model_type == "opt" else "pre_rope"
        hooks.add_q_output_hook(
            module=layer.self_attn.q_proj,
            layer_idx=layer_idx,
            num_heads=resolve_num_heads(config, layer),
            recorder=recorder,
            q_capture=q_capture,
        )

    input_device = infer_input_device(model)
    print(f"Input device: {input_device}")

    print("[5/7] Running one forward pass per sample...")
    try:
        with torch.inference_mode():
            for sample_idx in range(num_samples):
                recorder.begin_sample(sample_idx)
                input_ids = samples[sample_idx : sample_idx + 1].to(
                    input_device, non_blocking=True
                )
                attention_mask = torch.ones_like(input_ids, device=input_device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=False,
                )
                del output, input_ids, attention_mask
                recorder.end_sample()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"\r  Processed sample {sample_idx + 1}/{num_samples}",
                    end="",
                    flush=True,
                )
        print()
    finally:
        hooks.close()

    print("[6/7] Validating captured tensors...")
    sample_df = pd.DataFrame(
        [
            {
                "sample": result.sample,
                "layer": result.layer,
                "activation": result.activation,
                "D_intra": result.d_intra,
                "D_inter": result.d_inter,
                "R_cons": result.r_cons,
                "R_cons_gt_1": result.r_cons > 1.0,
                "rows_used": result.rows_used,
                "num_blocks": result.num_blocks,
                "feature_dim": result.feature_dim,
                "group_size": result.group_size,
                "num_bins": result.num_bins,
                "pair_samples": result.pair_samples,
                "Q_capture": result.q_capture,
            }
            for result in recorder.results
        ]
    )

    if sample_df.empty:
        raise RuntimeError("No activation statistics were captured.")

    expected_activations = {"X_qkv", "Q", "X_ao", "X_fc1", "X_fc2"}
    captured = set(sample_df["activation"].unique())
    missing = sorted(expected_activations - captured)
    if missing:
        raise RuntimeError(f"The following activation tensors were not captured: {missing}")

    duplicate_mask = sample_df.duplicated(
        subset=["sample", "layer", "activation"], keep=False
    )
    if duplicate_mask.any():
        duplicates = sample_df.loc[
            duplicate_mask, ["sample", "layer", "activation"]
        ]
        raise RuntimeError(
            "Duplicate sample/layer/activation results were captured:\n"
            f"{duplicates.to_string(index=False)}"
        )

    summary_long, percent_wide, mean_wide, avg_percent = summarize_results(
        sample_df, expected_samples=num_samples
    )

    print("[7/7] Writing result tables...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = sanitize_name(Path(model_path.rstrip("/")).name)
    layer_tag = "all" if layers_spec.lower() == "all" else sanitize_name(layers_spec)
    prefix = output_prefix or (
        f"samplewise_rcons_{model_name}_layers-{layer_tag}_n{num_samples}_g{group_size}"
    )

    paths = {
        "per_sample": output_dir / f"{prefix}_per_sample.csv",
        "summary_long": output_dir / f"{prefix}_summary_long.csv",
        "mean": output_dir / f"{prefix}_mean_rcons.csv",
        "percent": output_dir / f"{prefix}_percent_gt1.csv",
        "avg_percent": output_dir / f"{prefix}_avg_percent_gt1.csv",
    }

    sample_df.sort_values(["sample", "layer", "activation"]).to_csv(
        paths["per_sample"], index=False, float_format="%.8f"
    )
    summary_long.to_csv(paths["summary_long"], index=False, float_format="%.8f")
    mean_wide.to_csv(paths["mean"], index=False, float_format="%.8f")
    percent_wide.to_csv(paths["percent"], index=False, float_format="%.4f")
    avg_percent.to_csv(paths["avg_percent"], index=False, float_format="%.4f")

    print("\nPercentage of samples satisfying R_cons > 1:")
    print(
        percent_wide.to_string(
            index=False, float_format=lambda value: f"{value:.2f}%"
        )
    )
    print("\nAverage percentage over selected layers:")
    print(
        avg_percent.to_string(
            index=False, float_format=lambda value: f"{value:.2f}%"
        )
    )
    print("\nSaved files:")
    for label, path in paths.items():
        print(f"  {label:12s}: {path.resolve()}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
