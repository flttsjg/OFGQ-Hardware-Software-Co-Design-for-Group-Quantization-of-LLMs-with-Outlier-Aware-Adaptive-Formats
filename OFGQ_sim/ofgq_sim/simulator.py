"""Shared cycle, traffic, SRAM, quantizer, and energy model."""

from __future__ import annotations

import math
import re
from typing import Dict

from .config import AcceleratorConfig, QUANT_MODULES
from .format_selection import FormatSelections, fixed_timing
from .model_profiles import ModelSpec, build_shapes


def is_attention(name: str) -> bool:
    return "attn_qk" in name or "attn_v" in name


def operation_class(name: str) -> str:
    return "Attention" if is_attention(name) else "Linear"


class HardwareSimulator:
    def __init__(
        self,
        *,
        config: AcceleratorConfig,
        model: ModelSpec,
        sequence_length: int,
        mode: str,
        selections: FormatSelections,
    ):
        if mode not in {"prefill", "decode"}:
            raise ValueError("mode must be prefill or decode")
        if sequence_length <= 0 or config.batch_size <= 0:
            raise ValueError("sequence length and batch size must be positive")
        if config.rows <= 0 or config.cols <= 0 or config.pe_dot_product_size <= 0:
            raise ValueError("PE dimensions must be positive")
        if config.offchip_bandwidth_gbps <= 0 or config.offchip_energy_pj_per_bit <= 0:
            raise ValueError("Off-chip bandwidth and energy must be positive")
        missing = set(QUANT_MODULES) - set(config.quant_energy_pj_per_cycle)
        if missing:
            raise ValueError(f"Missing quantizer energies: {sorted(missing)}")
        self.config = config
        self.model = model
        self.sequence_length = sequence_length
        self.mode = mode
        self.selections = selections
        self.shapes = build_shapes(
            model,
            batch_size=config.batch_size,
            sequence_length=sequence_length,
            mode=mode,
        )
        if config.name == "OFGQ":
            selections.validate_model(model, self.shapes.weight_dim, config.format_group_size)
        self.memory_bytes = self._calculate_memory_bytes()
        self.refetch = self._calculate_refetch_counts()

    def _precisions(self, name: str) -> tuple[int, int, int]:
        if "attn_qk" in name:
            return self.config.k_bits, self.config.q_bits, self.config.offchip_output_bits
        if "attn_v" in name:
            return self.config.v_bits, self.config.a_bits, self.config.offchip_output_bits
        return self.config.weight_bits, self.config.input_bits, self.config.offchip_output_bits

    def _pe_latency(self, name: str) -> float:
        if "attn_qk" in name:
            return self.config.q_bits * self.config.k_bits / 16.0
        if "attn_v" in name:
            return self.config.a_bits * self.config.v_bits / 16.0
        return 1.0

    def _calculate_memory_bytes(self) -> Dict[str, tuple[int, int, int]]:
        result: Dict[str, tuple[int, int, int]] = {}
        for name in self.shapes.layer_names:
            batch_w, cout, cin = self.shapes.weight_dim[name]
            batch_i, tokens_i, input_channels = self.shapes.input_dim[name]
            batch_o, tokens_o, output_channels = self.shapes.output_dim[name]
            if cin != input_channels or cout != output_channels:
                raise ValueError(f"Shape mismatch in {name}")
            weight_bits, input_bits, output_bits = self._precisions(name)
            result[name] = (
                math.ceil(cin * weight_bits / 8.0) * cout * batch_w,
                math.ceil(input_channels * input_bits / 8.0) * tokens_i * batch_i,
                math.ceil(output_channels * output_bits / 8.0) * tokens_o * batch_o,
            )
        return result

    def _calculate_refetch_counts(self) -> Dict[str, tuple[int, int]]:
        result: Dict[str, tuple[int, int]] = {}
        weight_capacity = self.config.sram.weight_kib * 1024
        input_capacity = self.config.sram.input_kib * 1024
        for name, (weight_bytes, input_bytes, _) in self.memory_bytes.items():
            if weight_bytes > weight_capacity and input_bytes > input_capacity:
                input_refetch = math.ceil(weight_bytes / weight_capacity)
                weight_refetch = math.ceil(input_bytes / input_capacity)
                weight_traffic = weight_refetch * weight_bytes + input_bytes
                input_traffic = input_refetch * input_bytes + weight_bytes
                result[name] = (weight_refetch, 1) if weight_traffic < input_traffic else (1, input_refetch)
            else:
                result[name] = (1, 1)
        return result

    def _active_rows(self) -> int:
        if self.mode == "prefill":
            return self.config.rows
        return math.ceil(self.config.rows / self.config.decode_active_row_group)

    def _organization(self, name: str) -> dict[str, int]:
        active_rows = self._active_rows()
        active_pes = active_rows * self.config.cols
        weight_bits, input_bits, _ = self._precisions(name)
        default_weight_banks = self.config.rows if self.mode == "prefill" else active_pes
        default_output_banks = self.config.rows if self.mode == "prefill" else active_pes
        weight_banks = self.config.sram.weight_banks or default_weight_banks
        input_banks = self.config.sram.input_banks or self.config.cols
        output_banks = self.config.sram.output_banks or default_output_banks
        weight_bank_width = self.config.pe_dot_product_size * weight_bits
        input_bank_width = self.config.pe_dot_product_size * input_bits
        read_ports = self.config.sram.read_ports_per_bank
        write_ports = self.config.sram.write_ports_per_bank
        return {
            "active_pe_rows": active_rows,
            "active_pe_count": active_pes,
            "weight_sram_banks": weight_banks,
            "weight_sram_bits_per_bank_access": weight_bank_width,
            "weight_sram_bandwidth_bits_per_cycle": weight_banks * weight_bank_width * read_ports,
            "input_sram_banks": input_banks,
            "input_sram_bits_per_bank_access": input_bank_width,
            "input_sram_bandwidth_bits_per_cycle": input_banks * input_bank_width * read_ports,
            "output_sram_banks": output_banks,
            "output_sram_bits_per_bank_access": self.config.output_buffer_bits,
            "output_sram_bandwidth_bits_per_cycle": output_banks * self.config.output_buffer_bits * write_ports,
            "sram_read_ports_per_bank": read_ports,
            "sram_write_ports_per_bank": write_ports,
        }

    def _format_timing(self, name: str, reduction_steps: int):
        if self.config.name == "OFGQ":
            return self.selections.timing_for(self.model, name, reduction_steps)
        source = "fixed_attention_precision" if is_attention(name) else "fixed_linear_precision"
        return fixed_timing(reduction_steps, source)

    def _output_tiles(self, name: str) -> int:
        _, cout, _ = self.shapes.weight_dim[name]
        batch, tokens, _ = self.shapes.output_dim[name]
        if self.mode == "prefill":
            return batch * math.ceil(cout / self.config.rows) * math.ceil(tokens / self.config.cols)
        active_pes = self._active_rows() * self.config.cols
        return batch * tokens * math.ceil(cout / active_pes)

    def _compute_timing(self, name: str) -> dict[str, object]:
        _, _, cin = self.shapes.weight_dim[name]
        reduction_steps = math.ceil(cin / self.config.pe_dot_product_size)
        selected = self._format_timing(name, reduction_steps)
        output_tiles = self._output_tiles(name)
        reduction_cycles = selected.cycle_sum * self._pe_latency(name)
        drain_cycles = self._active_rows() + self.config.cols - 2
        tile_cycles = max(reduction_cycles, float(drain_cycles))
        return {
            "output_tiles": output_tiles,
            "reduction_steps": reduction_steps,
            "format_selection_source": selected.source,
            "format_average_cycles": selected.average_cycles,
            "format_cycle_sum_per_output_tile": selected.cycle_sum,
            "format_E3M0_blocks": selected.e3m0,
            "format_E2M1_blocks": selected.e2m1,
            "format_E1M2_blocks": selected.e1m2,
            "pe_operand_latency": self._pe_latency(name),
            "reduction_cycles_per_output_tile": reduction_cycles,
            "array_drain_cycles_per_output_tile": drain_cycles,
            "compute_cycles_per_output_tile": tile_cycles,
            "drain_limited": drain_cycles > reduction_cycles,
            "compute_cycles": output_tiles * tile_cycles,
            "physical_tiles": output_tiles * reduction_steps,
        }

    def _traffic(self, name: str) -> dict[str, float]:
        weight_bytes, input_bytes, output_bytes = self.memory_bytes[name]
        weight_fetches, input_fetches = self.refetch[name]
        read_bits = 8.0 * (weight_bytes * weight_fetches + input_bytes * input_fetches)
        write_bits = 8.0 * output_bytes
        return {
            "offchip_read_bits": read_bits,
            "offchip_write_bits": write_bits,
            "dram_read_cycles": read_bits / self.config.offchip_bits_per_cycle,
            "dram_write_cycles": write_bits / self.config.offchip_bits_per_cycle,
            "dram_energy_pj": (read_bits + write_bits) * self.config.offchip_energy_pj_per_bit,
        }

    def _sram_energy(self, name: str, timing: dict[str, object], organization: dict[str, int]) -> dict[str, float]:
        weight_bits, input_bits, _ = self._precisions(name)
        batch_w, cout, cin = self.shapes.weight_dim[name]
        batch_i, tokens_i, input_channels = self.shapes.input_dim[name]
        batch_o, tokens_o, output_channels = self.shapes.output_dim[name]
        weight_fetches, input_fetches = self.refetch[name]
        minimum = self.config.sram.min_write_bits
        read_bits_per_tile = (
            organization["weight_sram_bandwidth_bits_per_cycle"]
            + organization["input_sram_bandwidth_bits_per_cycle"]
        )
        read_energy = (
            int(timing["physical_tiles"])
            * float(timing["format_average_cycles"])
            * read_bits_per_tile
            * self.config.sram.read_energy_pj_per_bit
        )
        writes = (
            math.ceil(cin * weight_bits / minimum) * cout * batch_w * weight_fetches
            + math.ceil(input_channels * input_bits / minimum) * tokens_i * batch_i * input_fetches
            + math.ceil(output_channels * self.config.output_buffer_bits / minimum) * tokens_o * batch_o
        )
        write_energy = writes * minimum * self.config.sram.write_energy_pj_per_bit
        return {"base_sram_read_energy_pj": read_energy, "base_sram_write_energy_pj": write_energy}

    def _quantizer(self, name: str, compute_cycles: float) -> dict[str, float | int | bool]:
        if not self.config.runtime_quantization or name == "lm_head":
            return {
                "quantized": False,
                "quant_intervals": 0,
                "quant_interval_compute_cycles": 0.0,
                "quant_latency_cycles": 0,
                "quant_ii_cycles": 0,
                "quant_stall_cycles": 0.0,
                "quant_tail_cycles": 0.0,
                "quantization_penalty_cycles": 0.0,
                "quantization_energy_pj": 0.0,
            }
        batch, tokens, channels = self.shapes.output_dim[name]
        groups_per_row = math.ceil(channels / self.config.format_group_size)
        if self.mode == "prefill":
            intervals = batch * math.ceil(tokens / self.config.rows) * groups_per_row
            segments_per_interval = math.ceil(self.config.format_group_size / self.config.cols) * self.config.rows
            groups_per_interval = self.config.rows
        else:
            intervals = batch * tokens * groups_per_row
            segments_per_interval = math.ceil(self.config.format_group_size / self.config.cols)
            groups_per_interval = 1
        pipeline = sum(self.config.quant_cycles.values())
        initiation_interval = segments_per_interval
        # The first segment enters the comparator in its arrival cycle.
        latency = initiation_interval + pipeline - 1
        interval_compute = compute_cycles / intervals
        stall = max(0, intervals - 1) * max(0.0, initiation_interval - interval_compute)
        segments = intervals * segments_per_interval
        groups = intervals * groups_per_interval
        active = {
            "comparator": segments + self.config.quant_cycles["comparator"] - 1,
            "divider": groups + self.config.quant_cycles["divider"] - 1,
            "multiplier": segments + groups + self.config.quant_cycles["multiplier"] - 1,
            "encoder": segments + self.config.quant_cycles["encoder"] - 1,
            "control": latency + max(0, intervals - 1) * initiation_interval,
        }
        energy = sum(self.config.quant_energy_pj_per_cycle[key] * cycles for key, cycles in active.items())
        return {
            "quantized": True,
            "quant_intervals": intervals,
            "quant_interval_compute_cycles": interval_compute,
            "quant_latency_cycles": latency,
            "quant_ii_cycles": initiation_interval,
            "quant_stall_cycles": stall,
            "quant_tail_cycles": float(latency),
            "quantization_penalty_cycles": stall + latency,
            "quantization_energy_pj": energy,
            **{f"quant_{key}_active_cycles": value for key, value in active.items()},
        }

    def _extra_buffer_energy(self, name: str) -> dict[str, float | int]:
        batch, tokens, channels = self.shapes.output_dim[name]
        elements = batch * tokens * channels
        pe_output_write = elements * self.config.output_buffer_bits * self.config.sram.write_energy_pj_per_bit
        empty = {
            "output_elements": elements,
            "format_index_count": 0,
            "pe_output_write_energy_pj": pe_output_write,
            "runtime_output_read_energy_pj": 0.0,
            "runtime_output_writeback_energy_pj": 0.0,
            "format_index_read_energy_pj": 0.0,
            "format_index_write_energy_pj": 0.0,
        }
        if not self.config.runtime_quantization or name == "lm_head":
            return empty
        if "attn_qk" in name:
            index_count = self.config.batch_size * self.model.attention_heads
        else:
            index_count = batch * tokens * math.ceil(channels / self.config.format_group_size)
        index_bits = index_count * self.config.format_index_bits
        return {
            "output_elements": elements,
            "format_index_count": index_count,
            "pe_output_write_energy_pj": pe_output_write,
            "runtime_output_read_energy_pj": elements * self.config.output_buffer_bits * self.config.sram.read_energy_pj_per_bit,
            "runtime_output_writeback_energy_pj": elements * self.config.offchip_output_bits * self.config.sram.write_energy_pj_per_bit,
            "format_index_read_energy_pj": index_bits * self.config.sram.read_energy_pj_per_bit,
            "format_index_write_energy_pj": index_bits * self.config.sram.write_energy_pj_per_bit,
        }

    @staticmethod
    def _periodic_write_finish(first_ready: float, spacing: float, jobs: int, write_cycles: float, read_cycles: float) -> float:
        if jobs <= 0:
            return read_cycles
        service = write_cycles / jobs
        return max(
            read_cycles + jobs * service,
            first_ready + jobs * service,
            first_ready + (jobs - 1) * spacing + service,
        )

    @staticmethod
    def _transformer_layer(name: str) -> int | str:
        match = re.search(r"layers\.(\d+)\.", name)
        return int(match.group(1)) if match else ""

    def _simulate_layer(self, name: str) -> dict[str, object]:
        timing = self._compute_timing(name)
        organization = self._organization(name)
        traffic = self._traffic(name)
        sram = self._sram_energy(name, timing, organization)
        quant = self._quantizer(name, float(timing["compute_cycles"]))
        buffers = self._extra_buffer_energy(name)
        dram_cycles = traffic["dram_read_cycles"] + traffic["dram_write_cycles"]
        compute_cycles = float(timing["compute_cycles"])
        if bool(quant["quantized"]):
            intervals = int(quant["quant_intervals"])
            interval_compute = float(quant["quant_interval_compute_cycles"])
            baseline_cycles = self._periodic_write_finish(interval_compute, interval_compute, intervals, traffic["dram_write_cycles"], traffic["dram_read_cycles"])
            layer_cycles = self._periodic_write_finish(
                interval_compute + float(quant["quant_latency_cycles"]),
                max(interval_compute, float(quant["quant_ii_cycles"])),
                intervals,
                traffic["dram_write_cycles"],
                traffic["dram_read_cycles"],
            )
        else:
            baseline_cycles = max(compute_cycles, dram_cycles)
            layer_cycles = baseline_cycles
        extra_read = buffers["runtime_output_read_energy_pj"] + buffers["format_index_read_energy_pj"]
        extra_write = buffers["runtime_output_writeback_energy_pj"] + buffers["format_index_write_energy_pj"]
        sram_read_energy = sram["base_sram_read_energy_pj"] + extra_read
        sram_write_energy = sram["base_sram_write_energy_pj"] + extra_write
        sram_energy = sram_read_energy + sram_write_energy
        active_pe_count = organization["active_pe_count"] if self.mode == "decode" else self.config.pe_count
        pe_energy = self.config.pe_energy_pj * active_pe_count * compute_cycles
        compute_energy = pe_energy + float(quant["quantization_energy_pj"])
        total_energy = traffic["dram_energy_pj"] + sram_energy + compute_energy
        weight_bits, input_bits, _ = self._precisions(name)
        return {
            "model": self.model.key,
            "model_label": self.model.label,
            "phase": self.mode,
            "sequence_length": self.sequence_length,
            "batch_size": self.config.batch_size,
            "method": self.config.method_label,
            "layer": name,
            "transformer_layer": self._transformer_layer(name),
            "operation_class": operation_class(name),
            "weight_operand_bits": weight_bits,
            "input_operand_bits": input_bits,
            "output_buffer_bits": self.config.output_buffer_bits,
            "offchip_output_bits": self.config.offchip_output_bits,
            "pe_rows": self.config.rows,
            "pe_cols": self.config.cols,
            "pe_dot_product_size": self.config.pe_dot_product_size,
            "pe_energy_pj": self.config.pe_energy_pj,
            "frequency_mhz": self.config.frequency_mhz,
            "offchip_bandwidth_gbps": self.config.offchip_bandwidth_gbps,
            "offchip_energy_pj_per_bit": self.config.offchip_energy_pj_per_bit,
            **organization,
            **timing,
            **traffic,
            **quant,
            **buffers,
            "dram_cycles": dram_cycles,
            "baseline_layer_cycles": baseline_cycles,
            "visible_quantization_cycles": layer_cycles - baseline_cycles,
            "layer_cycles": layer_cycles,
            "sram_read_energy_pj": sram_read_energy,
            "sram_write_energy_pj": sram_write_energy,
            "sram_energy_pj": sram_energy,
            "pe_compute_energy_pj": pe_energy,
            "compute_energy_pj": compute_energy,
            "total_energy_pj": total_energy,
            "weight_refetches": self.refetch[name][0],
            "input_refetches": self.refetch[name][1],
        }

    def evaluate(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        layers = [self._simulate_layer(name) for name in self.shapes.layer_names]
        attention = [row for row in layers if row["operation_class"] == "Attention"]
        linear = [row for row in layers if row["operation_class"] == "Linear"]
        total_cycles = sum(float(row["layer_cycles"]) for row in layers)
        dram_energy = sum(float(row["dram_energy_pj"]) for row in layers)
        sram_energy = sum(float(row["sram_energy_pj"]) for row in layers)
        pe_energy = sum(float(row["pe_compute_energy_pj"]) for row in layers)
        quant_energy = sum(float(row["quantization_energy_pj"]) for row in layers)
        compute_energy = pe_energy + quant_energy
        total_energy = dram_energy + sram_energy + compute_energy
        summary = {
            "model": self.model.key,
            "model_label": self.model.label,
            "phase": self.mode,
            "sequence_length": self.sequence_length,
            "batch_size": self.config.batch_size,
            "method": self.config.method_label,
            "runtime_quantization": self.config.runtime_quantization,
            "pe_rows": self.config.rows,
            "pe_cols": self.config.cols,
            "pe_count": self.config.pe_count,
            "pe_dot_product_size": self.config.pe_dot_product_size,
            "pe_energy_pj": self.config.pe_energy_pj,
            "frequency_mhz": self.config.frequency_mhz,
            "weight_bits": self.config.weight_bits,
            "input_bits": self.config.input_bits,
            "output_buffer_bits": self.config.output_buffer_bits,
            "offchip_output_bits": self.config.offchip_output_bits,
            "q_bits": self.config.q_bits,
            "k_bits": self.config.k_bits,
            "a_bits": self.config.a_bits,
            "v_bits": self.config.v_bits,
            "weight_sram_kib": self.config.sram.weight_kib,
            "input_sram_kib": self.config.sram.input_kib,
            "output_sram_kib": self.config.sram.output_kib,
            "sram_read_energy_pj_per_bit": self.config.sram.read_energy_pj_per_bit,
            "sram_write_energy_pj_per_bit": self.config.sram.write_energy_pj_per_bit,
            "sram_min_write_bits": self.config.sram.min_write_bits,
            "sram_read_ports_per_bank": self.config.sram.read_ports_per_bank,
            "sram_write_ports_per_bank": self.config.sram.write_ports_per_bank,
            "weight_sram_banks_override": self.config.sram.weight_banks or "",
            "input_sram_banks_override": self.config.sram.input_banks or "",
            "output_sram_banks_override": self.config.sram.output_banks or "",
            "offchip_bandwidth_gbps": self.config.offchip_bandwidth_gbps,
            "offchip_energy_pj_per_bit": self.config.offchip_energy_pj_per_bit,
            **{
                f"quant_{module}_energy_pj_per_cycle": self.config.quant_energy_pj_per_cycle[module]
                for module in QUANT_MODULES
            },
            **{
                f"quant_{module}_cycles": cycles
                for module, cycles in self.config.quant_cycles.items()
            },
            "total_cycles": total_cycles,
            "latency_ms": total_cycles / self.config.frequency_mhz / 1000.0,
            "attention_cycles": sum(float(row["layer_cycles"]) for row in attention),
            "linear_cycles": sum(float(row["layer_cycles"]) for row in linear),
            "dram_energy_pj": dram_energy,
            "sram_energy_pj": sram_energy,
            "pe_compute_energy_pj": pe_energy,
            "quantization_energy_pj": quant_energy,
            "compute_energy_pj": compute_energy,
            "attention_energy_pj": sum(float(row["total_energy_pj"]) for row in attention),
            "linear_energy_pj": sum(float(row["total_energy_pj"]) for row in linear),
            "total_energy_pj": total_energy,
            "total_energy_uj": total_energy / 1e6,
        }
        return summary, layers
