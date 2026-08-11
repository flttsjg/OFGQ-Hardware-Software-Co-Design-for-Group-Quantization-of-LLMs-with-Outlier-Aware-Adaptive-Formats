


from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ofgq.calibration import (
    FormatCollector,
    build_majority_vote_metadata,
    save_metadata,
)
from ofgq.models import (
    prepare_llama_for_calibration,
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
    cal_cfg = cfg["calibration"]
    data_cfg = cfg["dataset"]

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

    collector = FormatCollector()

    model = prepare_llama_for_calibration(
        model,
        collector=collector,
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
        data_cfg["calibration_path"],
        text_column=data_cfg.get(
            "text_column", "text"
        ),
    ).to(model.device)

    sequence_length = int(
        cal_cfg["sequence_length"]
    )
    num_samples = int(
        cal_cfg["num_samples"]
    )

    processed = 0
    for batch in tqdm(
        iter_sequence_batches(
            input_ids,
            sequence_length,
            max_samples=num_samples,
        ),
        total=num_samples,
        desc="OFGQ calibration",
    ):
        with torch.no_grad():
            model(batch)
        processed += 1

    if processed == 0:
        raise RuntimeError(
            "No complete calibration sequence was found. "
            "Check the dataset and sequence length."
        )

    metadata = build_majority_vote_metadata(
        collector,
        num_samples=processed,
    )

    save_metadata(
        metadata,
        json_path=cal_cfg[
            "output_json"
        ],
        txt_path=cal_cfg.get(
            "output_txt"
        ),
    )

    print(
        f"Calibration complete: "
        f"{processed} samples."
    )
    print(
        f"Saved JSON metadata to "
        f"{cal_cfg['output_json']}"
    )
    if cal_cfg.get("output_txt"):
        print(
            f"Saved text summary to "
            f"{cal_cfg['output_txt']}"
        )


if __name__ == "__main__":
    main()
