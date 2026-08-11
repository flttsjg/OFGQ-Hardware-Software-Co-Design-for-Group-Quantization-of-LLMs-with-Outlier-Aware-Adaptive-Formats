


from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ofgq.models import (
    prepare_llama_for_inference,
)
from ofgq.utils.config import load_config
from ofgq.utils.data import (
    iter_sequence_batches,
    load_token_stream,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name.lower() not in table:
        raise ValueError(
            f"Unsupported dtype: {name}"
        )
    return table[name.lower()]


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(
        cfg.get("runtime", {}).get(
            "seed", 0
        )
    )
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False)

    model_cfg = cfg["model"]
    runtime_cfg = cfg.get("runtime", {})
    q_cfg = cfg["quantization"]
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["dataset"]

    metadata_path = Path(
        eval_cfg["metadata_path"]
    )
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        torch_dtype=resolve_dtype(
            model_cfg.get(
                "dtype", "float16"
            )
        ),
        low_cpu_mem_usage=True,
        device_map=model_cfg.get(
            "device_map", "auto"
        ),
        attn_implementation=runtime_cfg.get(
            "attn_implementation",
            "eager",
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        use_fast=bool(
            runtime_cfg.get(
                "use_fast_tokenizer",
                False,
            )
        ),
    )

    model = prepare_llama_for_inference(
        model,
        metadata=metadata,
        num_bits=int(
            q_cfg.get("num_bits", 4)
        ),
        group_size=int(
            q_cfg.get(
                "group_size", 64
            )
        ),
    )

    input_ids = load_token_stream(
        tokenizer,
        data_cfg["evaluation_path"],
        text_column=data_cfg.get(
            "text_column", "text"
        ),
    ).to(model.device)

    sequence_length = int(
        eval_cfg["sequence_length"]
    )
    max_samples = eval_cfg.get(
        "max_samples"
    )

    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []
    processed = 0

    for batch in tqdm(
        iter_sequence_batches(
            input_ids,
            sequence_length,
            max_samples=max_samples,
        ),
        desc="PPL evaluation",
    ):
        with torch.no_grad():
            outputs = model(batch)

        if isinstance(outputs, tuple):
            lm_logits = outputs[0]
        else:
            lm_logits = outputs.logits

        shift_logits = (
            lm_logits[:, :-1, :]
            .contiguous()
            .float()
        )
        shift_labels = batch[:, 1:]

        loss = loss_fct(
            shift_logits.view(
                -1,
                shift_logits.size(-1),
            ),
            shift_labels.reshape(-1),
        )

        nlls.append(
            loss.float() * sequence_length
        )
        processed += 1

    if processed == 0:
        raise RuntimeError(
            "No complete evaluation sequence was found."
        )

    ppl = torch.exp(
        torch.stack(nlls).sum()
        / (processed * sequence_length)
    )

    print(
        f"WikiText-2 Perplexity: "
        f"{ppl.item():.6f}"
    )
    print(
        f"Evaluated samples: {processed}"
    )


if __name__ == "__main__":
    main()
