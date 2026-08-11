"""Built-in public model dimensions used to construct operation shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


Shape = Tuple[int, int, int]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    architecture: str
    layers: int
    hidden: int
    attention_heads: int
    key_value_heads: int
    intermediate: int
    vocab: int


@dataclass
class ModelShapes:
    layer_names: List[str]
    weight_dim: Dict[str, Shape]
    input_dim: Dict[str, Shape]
    output_dim: Dict[str, Shape]


MODEL_SPECS = {
    "opt-6.7b": ModelSpec("opt-6.7b", "OPT-6.7B", "opt", 32, 4096, 32, 32, 16384, 50272),
    "opt-13b": ModelSpec("opt-13b", "OPT-13B", "opt", 40, 5120, 40, 40, 20480, 50272),
    "llama1-7b": ModelSpec("llama1-7b", "LLaMA-1-7B", "llama", 32, 4096, 32, 32, 11008, 32000),
    "llama1-13b": ModelSpec("llama1-13b", "LLaMA-1-13B", "llama", 40, 5120, 40, 40, 13824, 32000),
    "llama1-30b": ModelSpec("llama1-30b", "LLaMA-1-30B", "llama", 60, 6656, 52, 52, 17920, 32000),
    "llama1-65b": ModelSpec("llama1-65b", "LLaMA-1-65B", "llama", 80, 8192, 64, 64, 22016, 32000),
    "llama2-7b": ModelSpec("llama2-7b", "LLaMA-2-7B", "llama", 32, 4096, 32, 32, 11008, 32000),
    "llama2-13b": ModelSpec("llama2-13b", "LLaMA-2-13B", "llama", 40, 5120, 40, 40, 13824, 32000),
}


ALIASES = {
    "facebook/opt-6.7b": "opt-6.7b",
    "facebook/opt-13b": "opt-13b",
    "huggyllama/llama-7b": "llama1-7b",
    "huggyllama/llama-13b": "llama1-13b",
    "huggyllama/llama-30b": "llama1-30b",
    "huggyllama/llama-65b": "llama1-65b",
    "NousResearch/Llama-2-7b-hf": "llama2-7b",
    "NousResearch/Llama-2-13b-hf": "llama2-13b",
    **{key: key for key in MODEL_SPECS},
}


def normalize_model_name(name: str) -> str:
    try:
        return ALIASES[name]
    except KeyError as exc:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model '{name}'. Choose from: {supported}") from exc


def resolve_models(names: Iterable[str]) -> list[ModelSpec]:
    return [MODEL_SPECS[normalize_model_name(name)] for name in names]


def _linear_layers(spec: ModelSpec) -> list[tuple[str, tuple[int, int]]]:
    rows: list[tuple[str, tuple[int, int]]] = []
    for layer in range(spec.layers):
        if spec.architecture == "llama":
            prefix = f"model.layers.{layer}"
            rows.extend(
                [
                    (f"{prefix}.self_attn.q_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.k_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.v_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.o_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.mlp.gate_proj", (spec.intermediate, spec.hidden)),
                    (f"{prefix}.mlp.up_proj", (spec.intermediate, spec.hidden)),
                    (f"{prefix}.mlp.down_proj", (spec.hidden, spec.intermediate)),
                ]
            )
        else:
            prefix = f"model.decoder.layers.{layer}"
            rows.extend(
                [
                    (f"{prefix}.self_attn.k_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.v_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.q_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.self_attn.out_proj", (spec.hidden, spec.hidden)),
                    (f"{prefix}.fc1", (spec.intermediate, spec.hidden)),
                    (f"{prefix}.fc2", (spec.hidden, spec.intermediate)),
                ]
            )
    rows.append(("lm_head", (spec.vocab, spec.hidden)))
    return rows


def build_shapes(spec: ModelSpec, *, batch_size: int, sequence_length: int, mode: str) -> ModelShapes:
    weight: Dict[str, Shape] = {}
    inputs: Dict[str, Shape] = {}
    outputs: Dict[str, Shape] = {}
    for name, (cout, cin) in _linear_layers(spec):
        weight[name] = (1, cout, cin)
        tokens = sequence_length if mode == "prefill" else 1
        inputs[name] = (batch_size, tokens, cin)
        outputs[name] = (batch_size, tokens, cout)

    head_size = spec.hidden // spec.attention_heads
    query_share = spec.attention_heads // spec.key_value_heads
    attention_batch = batch_size * spec.key_value_heads
    query_tokens = query_share * (sequence_length if mode == "prefill" else 1)
    for layer in range(spec.layers):
        qk = f"model.layers.{layer}.self_attn.attn_qk"
        weight[qk] = (attention_batch, sequence_length, head_size)
        inputs[qk] = (attention_batch, query_tokens, head_size)
        outputs[qk] = (attention_batch, query_tokens, sequence_length)

        av = f"model.layers.{layer}.self_attn.attn_v"
        weight[av] = (attention_batch, head_size, sequence_length)
        inputs[av] = (attention_batch, query_tokens, sequence_length)
        outputs[av] = (attention_batch, query_tokens, head_size)

    return ModelShapes(list(weight), weight, inputs, outputs)
