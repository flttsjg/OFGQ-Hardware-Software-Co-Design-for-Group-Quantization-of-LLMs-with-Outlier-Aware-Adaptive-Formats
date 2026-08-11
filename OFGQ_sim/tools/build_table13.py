"""Aggregate layer-zero rows into the latency columns used by Table 13."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="llama2-7b")
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    selected: dict[str, list[dict[str, str]]] = {}
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["model"] == args.model
                    and row["phase"] == "prefill"
                    and int(row["sequence_length"]) == args.sequence_length
                    and row["transformer_layer"] == "0"
                ):
                    selected.setdefault(row["method"], []).append(row)
    output: list[dict[str, object]] = []
    for method in ("ANT", "OLIVE", "TENDER", "OFGQ-RQ", "OFGQ"):
        rows = selected.get(method)
        if not rows:
            continue
        frequency = float(rows[0]["frequency_mhz"])
        quantized = any(row["quantized"].lower() == "true" for row in rows)
        output.append(
            {
                "method": method,
                "compute_latency_ms": sum(float(row["compute_cycles"]) for row in rows) / frequency / 1000.0,
                "quantization_latency_ns": (
                    sum(float(row["quantization_penalty_cycles"]) for row in rows) * 1000.0 / frequency
                    if quantized
                    else ""
                ),
                "dram_latency_ms": sum(float(row["dram_cycles"]) for row in rows) / frequency / 1000.0,
                "layer_latency_ms": sum(float(row["layer_cycles"]) for row in rows) / frequency / 1000.0,
                "note": "Compute, quantization, and DRAM overlap; component columns are not additive.",
            }
        )
    if not output:
        raise ValueError("No matching layer-zero rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Table 13 CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
