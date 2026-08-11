"""Strict reader for externally supplied OFGQ format selections."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Dict

from .model_profiles import ModelSpec, normalize_model_name


FORMAT_CYCLES = {"E3M0": 1, "E2M1": 2, "E1M2": 3}
REQUIRED_FIELDS = ("model", "tensor", "module", "layer", "head", "column_block", "format")


@dataclass(frozen=True)
class FormatTiming:
    source: str
    num_blocks: int
    cycle_sum: float
    average_cycles: float
    e3m0: float
    e2m1: float
    e1m2: float


class FormatSelections:
    def __init__(self, path: Path):
        self.path = path
        self.linear: Dict[tuple[str, str], Dict[int, str]] = {}
        self.q: Dict[tuple[str, int, int], Dict[int, str]] = {}
        self.a: Dict[tuple[str, int, int], str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Format-selection file not found: {self.path}")
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Format-selection CSV is missing columns: {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                model = normalize_model_name((row["model"] or "").strip())
                tensor = (row["tensor"] or "").strip().upper()
                fmt = (row["format"] or "").strip().upper()
                if fmt not in FORMAT_CYCLES:
                    raise ValueError(f"Line {line}: unsupported format '{fmt}'")
                if tensor == "LINEAR":
                    module = (row["module"] or "").strip()
                    block = int(row["column_block"])
                    self._insert(self.linear.setdefault((model, module), {}), block, fmt, line)
                elif tensor == "Q":
                    key = (model, int(row["layer"]), int(row["head"]))
                    block = int(row["column_block"])
                    self._insert(self.q.setdefault(key, {}), block, fmt, line)
                elif tensor == "A":
                    key = (model, int(row["layer"]), int(row["head"]))
                    if key in self.a:
                        raise ValueError(f"Line {line}: duplicate A selection {key}")
                    self.a[key] = fmt
                else:
                    raise ValueError(f"Line {line}: tensor must be LINEAR, Q, or A")

    @staticmethod
    def _insert(target: Dict[int, str], key: int, value: str, line: int) -> None:
        if key in target:
            raise ValueError(f"Line {line}: duplicate column block {key}")
        target[key] = value

    def validate_model(self, spec: ModelSpec, linear_shapes: Dict[str, tuple[int, int, int]], group_size: int) -> None:
        for module, (_, _, cin) in linear_shapes.items():
            if module == "lm_head" or "attn_qk" in module or "attn_v" in module:
                continue
            expected = math.ceil(cin / group_size)
            blocks = self.linear.get((spec.key, module), {})
            if set(blocks) != set(range(expected)):
                raise ValueError(f"Incomplete LINEAR selection: {spec.key}/{module}; expected {expected} blocks")
        q_blocks = math.ceil((spec.hidden // spec.attention_heads) / group_size)
        for layer in range(spec.layers):
            for head in range(spec.attention_heads):
                q = self.q.get((spec.key, layer, head), {})
                if set(q) != set(range(q_blocks)):
                    raise ValueError(f"Incomplete Q selection: {spec.key}/layer={layer}/head={head}")
                if (spec.key, layer, head) not in self.a:
                    raise ValueError(f"Missing A selection: {spec.key}/layer={layer}/head={head}")

    def global_linear_average(self, model: str) -> float:
        formats = [fmt for (key, _), blocks in self.linear.items() if key == model for fmt in blocks.values()]
        if not formats:
            raise ValueError(f"No LINEAR selections for {model}")
        return sum(FORMAT_CYCLES[fmt] for fmt in formats) / len(formats)

    @staticmethod
    def _layer_index(name: str) -> int:
        match = re.search(r"model\.layers\.(\d+)\..*\.(attn_qk|attn_v)$", name)
        if not match:
            raise ValueError(f"Cannot extract attention layer from {name}")
        return int(match.group(1))

    @staticmethod
    def _timing(source: str, formats: list[str], divisor: int = 1) -> FormatTiming:
        counts = {name: formats.count(name) / divisor for name in FORMAT_CYCLES}
        blocks = len(formats) // divisor
        cycle_sum = sum(counts[name] * FORMAT_CYCLES[name] for name in FORMAT_CYCLES)
        return FormatTiming(source, blocks, cycle_sum, cycle_sum / blocks, counts["E3M0"], counts["E2M1"], counts["E1M2"])

    def timing_for(self, spec: ModelSpec, name: str, reduction_steps: int) -> FormatTiming:
        if name == "lm_head":
            avg = self.global_linear_average(spec.key)
            return FormatTiming("model_average_for_lm_head", reduction_steps, reduction_steps * avg, avg, 0.0, 0.0, 0.0)
        if "attn_qk" in name:
            layer = self._layer_index(name)
            formats = [self.q[(spec.key, layer, head)][block] for head in range(spec.attention_heads) for block in range(reduction_steps)]
            return self._timing("external_Q_per_head_per_block", formats, spec.attention_heads)
        if "attn_v" in name:
            layer = self._layer_index(name)
            formats = [self.a[(spec.key, layer, head)] for head in range(spec.attention_heads) for _ in range(reduction_steps)]
            return self._timing("external_A_per_head_tensor", formats, spec.attention_heads)
        blocks = self.linear[(spec.key, name)]
        formats = [blocks[index] for index in range(reduction_steps)]
        return self._timing("external_LINEAR_per_block", formats)


def fixed_timing(reduction_steps: int, source: str) -> FormatTiming:
    return FormatTiming(source, reduction_steps, float(reduction_steps), 1.0, float(reduction_steps), 0.0, 0.0)
