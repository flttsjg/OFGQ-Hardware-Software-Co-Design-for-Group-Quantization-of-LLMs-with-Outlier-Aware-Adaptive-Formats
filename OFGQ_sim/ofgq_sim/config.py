"""Hardware configuration records and paper defaults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


SRAM_READ_PJ_PER_BIT = 183.1088 / 4096.0
SRAM_WRITE_PJ_PER_BIT = 416.116 / 4096.0


@dataclass(frozen=True)
class SRAMConfig:
    weight_kib: int = 256
    input_kib: int = 256
    output_kib: int = 64
    read_energy_pj_per_bit: float = SRAM_READ_PJ_PER_BIT
    write_energy_pj_per_bit: float = SRAM_WRITE_PJ_PER_BIT
    min_write_bits: int = 64
    read_ports_per_bank: int = 1
    write_ports_per_bank: int = 1
    weight_banks: int | None = None
    input_banks: int | None = None
    output_banks: int | None = None


@dataclass(frozen=True)
class AcceleratorConfig:
    name: str
    rows: int
    cols: int
    pe_dot_product_size: int
    pe_energy_pj: float
    weight_bits: int
    input_bits: int
    output_buffer_bits: int
    offchip_output_bits: int
    q_bits: int
    k_bits: int
    a_bits: int
    v_bits: int
    runtime_quantization: bool
    batch_size: int
    frequency_mhz: float
    format_group_size: int
    format_index_bits: int
    decode_active_row_group: int
    offchip_bandwidth_gbps: float
    offchip_energy_pj_per_bit: float
    quant_energy_pj_per_cycle: Mapping[str, float]
    quant_cycles: Mapping[str, int]
    sram: SRAMConfig

    @property
    def method_label(self) -> str:
        if self.name == "OFGQ" and self.runtime_quantization:
            return "OFGQ-RQ"
        return self.name

    @property
    def pe_count(self) -> int:
        return self.rows * self.cols

    @property
    def offchip_bits_per_cycle(self) -> float:
        return self.offchip_bandwidth_gbps * 1000.0 * 8.0 / self.frequency_mhz

    def with_overrides(self, **kwargs: object) -> "AcceleratorConfig":
        return replace(self, **kwargs)


def _default(name: str, rows: int, cols: int, dp: int, pe_energy: float, attn_bits: int) -> dict[str, object]:
    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "pe_dot_product_size": dp,
        "pe_energy_pj": pe_energy,
        "weight_bits": 4,
        "input_bits": 4,
        "output_buffer_bits": 16,
        "offchip_output_bits": 4,
        "q_bits": attn_bits,
        "k_bits": attn_bits,
        "a_bits": attn_bits,
        "v_bits": attn_bits,
        "runtime_quantization": name == "OFGQ",
        "batch_size": 1,
        "frequency_mhz": 1000.0,
        "format_group_size": 64,
        "format_index_bits": 2,
        "decode_active_row_group": 4,
        "offchip_bandwidth_gbps": 0.0,
        "offchip_energy_pj_per_bit": 0.0,
        "quant_energy_pj_per_cycle": {},
        "quant_cycles": {
            "comparator": 5,
            "divider": 12,
            "multiplier": 4,
            "encoder": 1,
        },
        "sram": SRAMConfig(),
    }


ACCELERATOR_DEFAULTS = {
    "ANT": AcceleratorConfig(**_default("ANT", 42, 64, 1, 0.3730005, 8)),
    "OLIVE": AcceleratorConfig(**_default("OLIVE", 46, 64, 1, 0.3730005, 8)),
    "TENDER": AcceleratorConfig(**_default("TENDER", 52, 64, 1, 0.2661133, 8)),
    "OFGQ": AcceleratorConfig(**_default("OFGQ", 16, 16, 64, 4.7265625, 4)),
}


QUANT_MODULES = ("comparator", "divider", "multiplier", "encoder", "control")
