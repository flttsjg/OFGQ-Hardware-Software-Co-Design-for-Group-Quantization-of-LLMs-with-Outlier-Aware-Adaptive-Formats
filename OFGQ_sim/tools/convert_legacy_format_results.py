"""Convert private legacy LINEAR/Q/A CSV files to the public single-file schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ofgq_sim.model_profiles import normalize_model_name


FIELDS = ("model", "tensor", "module", "layer", "head", "column_block", "format")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        nargs=4,
        action="append",
        metavar=("MODEL", "LINEAR_CSV", "Q_CSV", "A_CSV"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: list[dict[str, object]] = []
    for raw_model, linear_path, q_path, a_path in args.entry:
        model = normalize_model_name(raw_model)
        with Path(linear_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                output.append({"model": model, "tensor": "LINEAR", "module": row["module"], "layer": "", "head": "", "column_block": row["column_block"], "format": row["format"]})
        with Path(q_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                output.append({"model": model, "tensor": "Q", "module": "", "layer": int(row["Layer"]) - 1, "head": row["Head"], "column_block": row["ColumnBlock"], "format": row["Format"]})
        with Path(a_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                output.append({"model": model, "tensor": "A", "module": "", "layer": int(row["Layer"]) - 1, "head": row["Head"], "column_block": "", "format": row["Format"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    print(f"Converted format file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
