





from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)

from ofgq.quantization.fp4 import quantize_tensor as qfp
from ofgq.quantization.int4 import quantize_tensor as qint


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _extract_layer_idx(full_name: str) -> Optional[int]:
    match = _LAYER_RE.search(full_name)
    if match is None:
        return None
    return int(match.group(1))


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = (
        hidden_states.shape
    )
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[
        :, :, None, :, :
    ].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )

    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )


class QuantizedLinearWrapper(nn.Module):


    def __init__(
        self,
        original_module: nn.Linear,
        local_name: str,
        full_name: str,
        mode: str,
        collector=None,
        metadata=None,
        num_bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__()
        self.original_module = original_module
        self.local_name = local_name
        self.full_name = full_name
        self.layer_idx = _extract_layer_idx(
            full_name
        )
        self.mode = mode
        self.collector = collector
        self.metadata = metadata
        self.num_bits = num_bits
        self.group_size = group_size

        if local_name != "lm_head":
            with torch.no_grad():
                self.original_module.weight.data = qint(
                    self.original_module.weight.data,
                    num_bits=self.num_bits,
                    block_row=1,
                    block_col=self.group_size,
                )

    def _metadata_key(self):
        if self.local_name in {
            "k_proj",
            "v_proj",
        }:
            return "q_proj"
        return self.local_name

    def forward(self, x):
        if self.local_name == "lm_head":
            return self.original_module(x)

        if self.mode == "calibration":
            x_q, best_indices = qfp(
                x,
                num_bits=self.num_bits,
                block_row=1,
                block_col=self.group_size,
            )

            if (
                self.collector is not None
                and self.layer_idx is not None
                and self.local_name
                not in {"k_proj", "v_proj"}
            ):
                self.collector.add(
                    self.local_name,
                    self.layer_idx,
                    best_indices,
                )

        elif self.mode == "inference":
            if self.layer_idx is None:
                raise RuntimeError(
                    f"Cannot determine layer index for "
                    f"{self.full_name}"
                )

            key = self._metadata_key()
            format_data = self.metadata[
                "selected"
            ][key][str(self.layer_idx)]

            x_q, _ = qfp(
                x,
                num_bits=self.num_bits,
                block_row=1,
                block_col=self.group_size,
                format_data=format_data,
            )

        else:
            raise ValueError(
                f"Unknown mode: {self.mode}"
            )

        return self.original_module(x_q)


def _ofgq_attention_forward(
    module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):

    mode = module._ofgq_mode
    collector = module._ofgq_collector
    metadata = module._ofgq_metadata
    layer_idx = module._ofgq_layer_idx
    num_bits = module._ofgq_num_bits
    group_size = module._ofgq_group_size

    bsz, q_len, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    def shape(x):
        return x.view(
            bsz,
            -1,
            module.config.num_attention_heads,
            module.head_dim,
        ).transpose(1, 2)

    query_states = shape(query_states)
    key_states = shape(key_states)
    value_states = shape(value_states)

    cos, sin = module.rotary_emb(
        hidden_states,
        position_ids=position_ids,
    )
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    n_rep = (
        module.config.num_attention_heads
        // module.config.num_key_value_heads
    )
    key_states = repeat_kv(
        key_states, n_rep
    )
    value_states = repeat_kv(
        value_states, n_rep
    )

    if mode == "calibration":
        query_states_q, q_indices = qfp(
            query_states,
            num_bits=num_bits,
            block_row=1,
            block_col=group_size,
        )
        collector.add(
            "Q",
            layer_idx,
            q_indices,
        )

    elif mode == "inference":
        q_formats = metadata[
            "selected"
        ]["Q"][str(layer_idx)]
        query_states_q, _ = qfp(
            query_states,
            num_bits=num_bits,
            block_row=1,
            block_col=group_size,
            format_data=q_formats,
        )

    else:
        raise ValueError(
            f"Unknown OFGQ attention mode: {mode}"
        )

    key_states_q = qint(
        key_states,
        num_bits=num_bits,
        block_row=1,
        block_col=group_size,
    )

    attn_weights = torch.matmul(
        query_states_q,
        key_states_q.transpose(2, 3),
    ) * (module.head_dim ** -0.5)

    if attention_mask is not None:
        causal_mask = attention_mask[
            :, :, :, :key_states.shape[-2]
        ]
        attn_weights = (
            attn_weights + causal_mask
        )

    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)

    if mode == "calibration":
        attn_weights_q, a_indices = qfp(
            attn_weights,
            num_bits=num_bits,
            block_row=1,
            block_col=group_size,
        )
        collector.add(
            "A",
            layer_idx,
            a_indices,
        )
    else:
        a_formats = metadata[
            "selected"
        ]["A"][str(layer_idx)]
        attn_weights_q, _ = qfp(
            attn_weights,
            num_bits=num_bits,
            block_row=1,
            block_col=group_size,
            format_data=a_formats,
        )

    value_states_q = qint(
        value_states,
        num_bits=num_bits,
        block_row=group_size,
        block_col=1,
    )

    attn_output = torch.matmul(
        attn_weights_q,
        value_states_q,
    )

    attn_output = (
        attn_output
        .transpose(1, 2)
        .contiguous()
        .reshape(bsz, q_len, -1)
        .contiguous()
    )

    attn_output = module.o_proj(attn_output)

    if output_attentions:
        return attn_output, attn_weights_q
    return attn_output, None


def _prepare_llama(
    model,
    mode: str,
    collector=None,
    metadata=None,
    num_bits: int = 4,
    group_size: int = 64,
):

    model.eval()



    LlamaAttention.forward = _ofgq_attention_forward

    def recursive_apply(
        module,
        prefix="",
    ):
        for name, child in list(
            module.named_children()
        ):
            full_name = (
                f"{prefix}.{name}"
                if prefix
                else name
            )

            if isinstance(
                child, LlamaAttention
            ):
                layer_idx = _extract_layer_idx(
                    full_name
                )
                if layer_idx is None:
                    raise RuntimeError(
                        "Could not determine layer index "
                        f"for attention module {full_name}"
                    )

                child._ofgq_mode = mode
                child._ofgq_collector = collector
                child._ofgq_metadata = metadata
                child._ofgq_layer_idx = layer_idx
                child._ofgq_num_bits = num_bits
                child._ofgq_group_size = group_size

                if not hasattr(
                    child, "rotary_emb"
                ):
                    from transformers.models.llama.modeling_llama import (
                        LlamaRotaryEmbedding,
                    )
                    child.rotary_emb = (
                        LlamaRotaryEmbedding(
                            child.config
                        )
                    )



            if len(
                list(child.children())
            ) > 0:
                recursive_apply(
                    child,
                    full_name,
                )

            if isinstance(child, nn.Linear):
                wrapper = QuantizedLinearWrapper(
                    original_module=child,
                    local_name=name,
                    full_name=full_name,
                    mode=mode,
                    collector=collector,
                    metadata=metadata,
                    num_bits=num_bits,
                    group_size=group_size,
                )
                setattr(
                    module,
                    name,
                    wrapper,
                )

    recursive_apply(model)
    return model


def prepare_llama_for_calibration(
    model,
    collector,
    num_bits: int = 4,
    group_size: int = 64,
):
    return _prepare_llama(
        model=model,
        mode="calibration",
        collector=collector,
        metadata=None,
        num_bits=num_bits,
        group_size=group_size,
    )


def prepare_llama_for_inference(
    model,
    metadata,
    num_bits: int = 4,
    group_size: int = 64,
):
    return _prepare_llama(
        model=model,
        mode="inference",
        collector=None,
        metadata=metadata,
        num_bits=num_bits,
        group_size=group_size,
    )
