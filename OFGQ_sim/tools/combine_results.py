"""Normalize independent accelerator summaries to the matching ANT run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict


KEY_FIELDS = ("model", "phase", "sequence_length", "batch_size")


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--add-average", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.inputs)
    ant = {
        tuple(row[field] for field in KEY_FIELDS): row
        for row in rows
        if row["method"] == "ANT"
    }
    output: list[dict[str, object]] = []
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        if key not in ant:
            raise ValueError(f"No matching ANT baseline for {key}")
        baseline = ant[key]
        ant_cycles = float(baseline["total_cycles"])
        ant_energy = float(baseline["total_energy_pj"])
        enriched: dict[str, object] = dict(row)
        enriched.update(
            {
                "speedup_vs_ant": ant_cycles / float(row["total_cycles"]),
                "energy_ratio_vs_ant": float(row["total_energy_pj"]) / ant_energy,
                "dram_energy_ratio_vs_ant_total": float(row["dram_energy_pj"]) / ant_energy,
                "sram_energy_ratio_vs_ant_total": float(row["sram_energy_pj"]) / ant_energy,
                "compute_energy_ratio_vs_ant_total": float(row["compute_energy_pj"]) / ant_energy,
                "attention_energy_ratio_vs_ant_total": float(row["attention_energy_pj"]) / ant_energy,
                "linear_energy_ratio_vs_ant_total": float(row["linear_energy_pj"]) / ant_energy,
            }
        )
        output.append(enriched)
    if args.add_average:
        groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in output:
            groups[(str(row["method"]), str(row["phase"]), str(row["sequence_length"]), str(row["batch_size"]))].append(row)
        for group in groups.values():
            average = dict(group[0])
            average["model"] = "average"
            average["model_label"] = "Average"
            for field in average:
                if field in {"model", "model_label"}:
                    continue
                try:
                    values = [float(row[field]) for row in group]
                except (TypeError, ValueError):
                    continue
                average[field] = sum(values) / len(values)
            output.append(average)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Comparison CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
