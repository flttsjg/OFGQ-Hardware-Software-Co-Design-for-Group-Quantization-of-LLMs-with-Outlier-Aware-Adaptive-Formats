









from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


FP4_E3M0 = [
    -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25,
    0.0,
    0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
]

FP4_E2M1 = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
]

FP4_E1M2 = [
    -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
]

FP4_FORMATS = [FP4_E3M0, FP4_E2M1, FP4_E1M2]
FORMAT_NAMES = ["E3M0", "E2M1", "E1M2"]


def get_midpoints(allow_values: Sequence[float]) -> list[float]:
    return [
        (allow_values[i] + allow_values[i + 1]) / 2
        for i in range(len(allow_values) - 1)
    ]


def quantize_with_format(
    blocks: torch.Tensor,
    fp_format: Sequence[float],
) -> torch.Tensor:





    allow_tensor = torch.tensor(
        fp_format, dtype=blocks.dtype, device=blocks.device
    )
    mid_tensor = torch.tensor(
        get_midpoints(fp_format), dtype=blocks.dtype, device=blocks.device
    )

    qmax = max(abs(x) for x in fp_format)
    rmax = (
        blocks.abs()
        .max(dim=-1, keepdim=True).values
        .max(dim=-2, keepdim=True).values
    )
    scale = rmax / qmax




    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x = blocks / safe_scale

    bin_idx = torch.bucketize(x, mid_tensor)
    q_tensor = allow_tensor[bin_idx]
    quantized = q_tensor * scale
    return quantized


def _select_formats_for_columns(
    columns_of_blocks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:









    (
        num_block_cols,
        num_blocks_in_col,
        block_row,
        block_col,
    ) = columns_of_blocks.shape

    original_flat = columns_of_blocks.reshape(
        -1, block_row, block_col
    )

    mse_by_format = []
    quantized_options = []

    for fp_format in FP4_FORMATS:
        q_flat = quantize_with_format(original_flat, fp_format)
        quantized_options.append(q_flat)

        mse_per_block = F.mse_loss(
            q_flat, original_flat, reduction="none"
        ).mean(dim=[-1, -2])

        mse_grid = mse_per_block.view(
            num_block_cols, num_blocks_in_col
        )
        total_mse_per_column = mse_grid.sum(dim=1)
        mse_by_format.append(total_mse_per_column)

    all_mses = torch.stack(mse_by_format, dim=0)
    best_format_indices = torch.argmin(all_mses, dim=0)

    stacked = torch.stack(quantized_options, dim=0)

    index_per_block = best_format_indices.repeat_interleave(
        num_blocks_in_col
    )
    gather_index = (
        index_per_block
        .view(1, -1, 1, 1)
        .expand(1, -1, block_row, block_col)
    )

    selected_flat = torch.gather(
        stacked, dim=0, index=gather_index
    ).squeeze(0)

    selected = selected_flat.view(
        num_block_cols,
        num_blocks_in_col,
        block_row,
        block_col,
    )
    return selected, best_format_indices


def _apply_preselected_formats(
    columns_of_blocks: torch.Tensor,
    format_result: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    (
        num_block_cols,
        num_blocks_in_col,
        block_row,
        block_col,
    ) = columns_of_blocks.shape

    original_flat = columns_of_blocks.reshape(
        -1, block_row, block_col
    )

    quantized_options = [
        quantize_with_format(original_flat, fmt)
        for fmt in FP4_FORMATS
    ]
    stacked = torch.stack(quantized_options, dim=0)

    best_format_indices = torch.as_tensor(
        format_result,
        dtype=torch.long,
        device=stacked.device,
    )

    if best_format_indices.numel() != num_block_cols:
        raise ValueError(
            "Number of preselected format indices does not match "
            f"the current tensor layout: got "
            f"{best_format_indices.numel()}, expected {num_block_cols}."
        )

    index_per_block = best_format_indices.repeat_interleave(
        num_blocks_in_col
    )
    gather_index = (
        index_per_block
        .view(1, -1, 1, 1)
        .expand(1, -1, block_row, block_col)
    )

    selected_flat = torch.gather(
        stacked, dim=0, index=gather_index
    ).squeeze(0)

    selected = selected_flat.view(
        num_block_cols,
        num_blocks_in_col,
        block_row,
        block_col,
    )
    return selected, best_format_indices


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int = 4,
    block_row: int = 1,
    block_col: int = 64,
    format_data: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:








    if num_bits != 4:
        raise ValueError(
            "The supplied OFGQ implementation supports FP4 only."
        )

    original_shape = tensor.shape
    if tensor.ndim < 2:
        raise ValueError(
            f"Expected tensor with at least 2 dimensions, got {tensor.ndim}."
        )

    rows, cols = original_shape[-2], original_shape[-1]
    batch_size = tensor.numel() // (rows * cols)

    reshaped = tensor.reshape(batch_size, rows, cols)

    pad_row = (block_row - rows % block_row) % block_row
    pad_col = (block_col - cols % block_col) % block_col
    if pad_row != 0 or pad_col != 0:
        reshaped = F.pad(
            reshaped, (0, pad_col, 0, pad_row)
        )

    padded_rows, padded_cols = (
        reshaped.shape[1],
        reshaped.shape[2],
    )
    num_block_rows = padded_rows // block_row
    num_block_cols = padded_cols // block_col

    blocks = reshaped.view(
        batch_size,
        num_block_rows,
        block_row,
        num_block_cols,
        block_col,
    )

    blocks_by_col = (
        blocks
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )

    merged = blocks_by_col.view(
        -1,
        num_block_rows,
        block_row,
        block_col,
    )

    if format_data is None:
        quantized_blocks, best_indices = _select_formats_for_columns(
            merged
        )
    else:
        quantized_blocks, best_indices = _apply_preselected_formats(
            merged, format_data
        )

    quantized_blocks = quantized_blocks.view(
        batch_size,
        num_block_cols,
        num_block_rows,
        block_row,
        block_col,
    )

    quantized_blocks = (
        quantized_blocks
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )

    quantized_padded = quantized_blocks.view(
        batch_size,
        padded_rows,
        padded_cols,
    )

    quantized_no_pad = quantized_padded[:, :rows, :cols]
    quantized_tensor = quantized_no_pad.reshape(original_shape)

    return quantized_tensor, best_indices
