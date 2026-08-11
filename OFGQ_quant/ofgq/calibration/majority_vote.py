

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from ofgq.quantization.fp4 import FORMAT_NAMES


def build_majority_vote_metadata(
    collector,
    num_samples: int,
) -> dict:









    selected = {}
    counts = {}

    for key in sorted(collector.data.keys()):
        selected[key] = {}
        counts[key] = {}

        for layer_idx in sorted(collector.data[key].keys()):
            tensors = collector.data[key][layer_idx]
            if not tensors:
                continue

            lengths = {int(t.numel()) for t in tensors}
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent format-index length for "
                    f"{key}, layer {layer_idx}: {sorted(lengths)}"
                )

            stacked = torch.stack(
                [t.reshape(-1) for t in tensors],
                dim=0,
            )

            layer_counts = []
            for fmt_idx in range(3):
                layer_counts.append(
                    (stacked == fmt_idx).sum(dim=0)
                )

            count_tensor = torch.stack(
                layer_counts, dim=-1
            )
            winners = torch.argmax(
                count_tensor, dim=-1
            )

            selected[key][str(layer_idx)] = (
                winners.tolist()
            )
            counts[key][str(layer_idx)] = (
                count_tensor.tolist()
            )

    return {
        "format_index_mapping": {
            "0": FORMAT_NAMES[0],
            "1": FORMAT_NAMES[1],
            "2": FORMAT_NAMES[2],
        },
        "majority_vote_num_samples": int(num_samples),
        "selected": selected,
        "counts": counts,
    }


def save_metadata(
    metadata: dict,
    json_path: str,
    txt_path: str | None = None,
) -> None:
    json_file = Path(json_path)
    json_file.parent.mkdir(
        parents=True, exist_ok=True
    )
    json_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if txt_path is None:
        return

    txt_file = Path(txt_path)
    txt_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    total_samples = metadata[
        "majority_vote_num_samples"
    ]

    lines = []
    for key, layers in metadata["counts"].items():
        lines.append(f"key: {key}")

        for layer_idx, positions in layers.items():
            lines.append(f"  layer {layer_idx}:")

            winners = metadata[
                "selected"
            ][key][layer_idx]

            for pos, count_triplet in enumerate(positions):
                percentages = [
                    100.0 * c / total_samples
                    for c in count_triplet
                ]
                lines.append(
                    "    position "
                    f"{pos}: counts={count_triplet}, "
                    f"percentages="
                    f"{[round(x, 2) for x in percentages]}, "
                    f"winner={winners[pos]}"
                )

    txt_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
