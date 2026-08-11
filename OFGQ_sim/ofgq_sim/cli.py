"""Common command-line interface used by the four accelerator entry points."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .config import ACCELERATOR_DEFAULTS, AcceleratorConfig, QUANT_MODULES, SRAMConfig
from .format_selection import FormatSelections
from .model_profiles import MODEL_SPECS, resolve_models
from .simulator import HardwareSimulator


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser(accelerator: str) -> argparse.ArgumentParser:
    default = ACCELERATOR_DEFAULTS[accelerator]
    parser = argparse.ArgumentParser(
        description=f"Run the {accelerator} analytical hardware simulator and write CSV results."
    )
    parser.add_argument("--model", nargs="+", default=["llama1-7b"], help=f"Model keys: {', '.join(MODEL_SPECS)}")
    parser.add_argument("--mode", "--run-mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--sequence-length", nargs="+", type=_positive_int, default=[2048])
    parser.add_argument("--batch-size", type=_positive_int, default=default.batch_size)
    parser.add_argument("--frequency-mhz", type=_positive_float, default=default.frequency_mhz)

    parser.add_argument("--pe-rows", type=_positive_int, default=default.rows)
    parser.add_argument("--pe-cols", type=_positive_int, default=default.cols)
    parser.add_argument("--pe-dot-product-size", type=_positive_int, default=default.pe_dot_product_size)
    parser.add_argument("--pe-energy-pj", type=_positive_float, default=default.pe_energy_pj)
    parser.add_argument("--weight-bits", type=_positive_int, default=default.weight_bits)
    parser.add_argument("--input-bits", type=_positive_int, default=default.input_bits)
    parser.add_argument("--output-buffer-bits", type=_positive_int, default=default.output_buffer_bits)
    parser.add_argument("--offchip-output-bits", type=_positive_int, default=default.offchip_output_bits)
    parser.add_argument("--q-bits", type=_positive_int, default=default.q_bits)
    parser.add_argument("--k-bits", type=_positive_int, default=default.k_bits)
    parser.add_argument("--a-bits", type=_positive_int, default=default.a_bits)
    parser.add_argument("--v-bits", type=_positive_int, default=default.v_bits)

    parser.add_argument("--weight-sram-kib", type=_positive_int, default=default.sram.weight_kib)
    parser.add_argument("--input-sram-kib", type=_positive_int, default=default.sram.input_kib)
    parser.add_argument("--output-sram-kib", type=_positive_int, default=default.sram.output_kib)
    parser.add_argument("--sram-read-energy-pj-per-bit", type=_positive_float, default=default.sram.read_energy_pj_per_bit)
    parser.add_argument("--sram-write-energy-pj-per-bit", type=_positive_float, default=default.sram.write_energy_pj_per_bit)
    parser.add_argument("--sram-min-write-bits", type=_positive_int, default=default.sram.min_write_bits)
    parser.add_argument("--sram-read-ports-per-bank", type=_positive_int, default=default.sram.read_ports_per_bank)
    parser.add_argument("--sram-write-ports-per-bank", type=_positive_int, default=default.sram.write_ports_per_bank)
    parser.add_argument("--weight-sram-banks", type=_positive_int)
    parser.add_argument("--input-sram-banks", type=_positive_int)
    parser.add_argument("--output-sram-banks", type=_positive_int)

    parser.add_argument("--format-group-size", type=_positive_int, default=default.format_group_size)
    parser.add_argument("--format-index-bits", type=_positive_int, default=default.format_index_bits)
    parser.add_argument("--decode-active-row-group", type=_positive_int, default=default.decode_active_row_group)
    parser.add_argument(
        "--runtime-quantization",
        action=argparse.BooleanOptionalAction,
        default=default.runtime_quantization,
    )
    for module in ("comparator", "divider", "multiplier", "encoder"):
        parser.add_argument(
            f"--quant-{module}-cycles",
            type=_positive_int,
            default=default.quant_cycles[module],
        )

    parser.add_argument("--offchip-bandwidth-gbps", type=_positive_float, required=True)
    parser.add_argument("--offchip-energy-pj-per-bit", type=_positive_float, required=True)
    for module in QUANT_MODULES:
        parser.add_argument(
            f"--quant-{module}-energy-pj-per-cycle",
            type=_positive_float,
            required=True,
        )
    parser.add_argument("--format-selection-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-stem")
    return parser


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows generated for {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _config_from_args(default: AcceleratorConfig, args: argparse.Namespace) -> AcceleratorConfig:
    sram = SRAMConfig(
        weight_kib=args.weight_sram_kib,
        input_kib=args.input_sram_kib,
        output_kib=args.output_sram_kib,
        read_energy_pj_per_bit=args.sram_read_energy_pj_per_bit,
        write_energy_pj_per_bit=args.sram_write_energy_pj_per_bit,
        min_write_bits=args.sram_min_write_bits,
        read_ports_per_bank=args.sram_read_ports_per_bank,
        write_ports_per_bank=args.sram_write_ports_per_bank,
        weight_banks=args.weight_sram_banks,
        input_banks=args.input_sram_banks,
        output_banks=args.output_sram_banks,
    )
    energies = {
        module: getattr(args, f"quant_{module}_energy_pj_per_cycle")
        for module in QUANT_MODULES
    }
    cycles = {
        module: getattr(args, f"quant_{module}_cycles")
        for module in ("comparator", "divider", "multiplier", "encoder")
    }
    return replace(
        default,
        rows=args.pe_rows,
        cols=args.pe_cols,
        pe_dot_product_size=args.pe_dot_product_size,
        pe_energy_pj=args.pe_energy_pj,
        weight_bits=args.weight_bits,
        input_bits=args.input_bits,
        output_buffer_bits=args.output_buffer_bits,
        offchip_output_bits=args.offchip_output_bits,
        q_bits=args.q_bits,
        k_bits=args.k_bits,
        a_bits=args.a_bits,
        v_bits=args.v_bits,
        runtime_quantization=args.runtime_quantization,
        batch_size=args.batch_size,
        frequency_mhz=args.frequency_mhz,
        format_group_size=args.format_group_size,
        format_index_bits=args.format_index_bits,
        decode_active_row_group=args.decode_active_row_group,
        offchip_bandwidth_gbps=args.offchip_bandwidth_gbps,
        offchip_energy_pj_per_bit=args.offchip_energy_pj_per_bit,
        quant_energy_pj_per_cycle=energies,
        quant_cycles=cycles,
        sram=sram,
    )


def run_main(accelerator: str) -> tuple[Path, Path]:
    parser = build_parser(accelerator)
    args = parser.parse_args()
    config = _config_from_args(ACCELERATOR_DEFAULTS[accelerator], args)
    selections = FormatSelections(args.format_selection_file)
    summaries: list[dict[str, object]] = []
    layers: list[dict[str, object]] = []
    for model in resolve_models(args.model):
        for length in args.sequence_length:
            summary, detail = HardwareSimulator(
                config=config,
                model=model,
                sequence_length=length,
                mode=args.mode,
                selections=selections,
            ).evaluate()
            summary["format_selection_file"] = str(args.format_selection_file.resolve())
            summaries.append(summary)
            layers.extend(detail)
    stem = args.output_stem or f"{config.method_label.lower()}_{args.mode}"
    stem = stem.replace(" ", "_")
    summary_path = args.output_dir / f"{stem}_summary.csv"
    layer_path = args.output_dir / f"{stem}_layers.csv"
    _write_csv(summary_path, summaries)
    _write_csv(layer_path, layers)
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Layer CSV:   {layer_path.resolve()}")
    return summary_path, layer_path
