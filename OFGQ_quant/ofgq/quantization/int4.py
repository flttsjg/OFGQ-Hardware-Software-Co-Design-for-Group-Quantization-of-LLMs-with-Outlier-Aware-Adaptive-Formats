












from __future__ import annotations

import torch
import torch.nn.functional as F


E1M2_VALUES = [
    -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
]


def get_midpoints(allow_values):
    return [
        (allow_values[i] + allow_values[i + 1]) / 2
        for i in range(len(allow_values) - 1)
    ]


def quantize_with_format(blocks, fp_format):
    allow_tensor = torch.tensor(
        fp_format, dtype=blocks.dtype, device=blocks.device
    )
    mid_tensor = torch.tensor(
        get_midpoints(fp_format),
        dtype=blocks.dtype,
        device=blocks.device,
    )

    qmax = max(abs(x) for x in fp_format)
    rmax = (
        blocks.abs()
        .max(dim=-1, keepdim=True).values
        .max(dim=-2, keepdim=True).values
    )
    scale = rmax / qmax
    safe_scale = torch.where(
        scale == 0, torch.ones_like(scale), scale
    )

    x = blocks / safe_scale
    bin_idx = torch.bucketize(x, mid_tensor)
    q_tensor = allow_tensor[bin_idx]
    return q_tensor * scale


def quantize_block_mse_analysis(blocks):
    return quantize_with_format(blocks, E1M2_VALUES)


def quantize_tensor(
    tensor,
    num_bits=4,
    block_row=1,
    block_col=64,
):
    if num_bits != 4:
        raise ValueError(
            "The supplied experimental quantizer is 4-bit only."
        )

    original_shape = tensor.shape
    dims = len(original_shape)

    if dims == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif dims == 3:
        tensor = tensor.unsqueeze(1)
    elif dims == 4:
        pass
    else:
        raise ValueError(
            f"Unsupported tensor dimension: {dims}"
        )

    b, s, rows, cols = tensor.shape

    pad_row = (block_row - rows % block_row) % block_row
    pad_col = (block_col - cols % block_col) % block_col
    if pad_row != 0 or pad_col != 0:
        tensor = F.pad(
            tensor,
            (0, pad_col, 0, pad_row),
            mode="constant",
            value=0,
        )

    b, s, rows, cols = tensor.shape

    tensor = tensor.view(
        b,
        s,
        rows // block_row,
        block_row,
        cols // block_col,
        block_col,
    )
    tensor = (
        tensor
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
    )
    tensor = tensor.view(-1, block_row, block_col)

    quantized_blocks = quantize_block_mse_analysis(tensor)

    quantized_blocks = quantized_blocks.view(
        b,
        rows // block_row,
        cols // block_col,
        s,
        block_row,
        block_col,
    )
    quantized_blocks = (
        quantized_blocks
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
    )
    quantized = quantized_blocks.view(
        b, s, rows, cols
    )

    quantized = quantized[
        :, :, :original_shape[-2], :original_shape[-1]
    ]

    if dims == 2:
        quantized = quantized.squeeze(0).squeeze(0)
    elif dims == 3:
        quantized = quantized.squeeze(1)

    return quantized
