

from __future__ import annotations

from datasets import load_dataset


def load_token_stream(
    tokenizer,
    parquet_path: str,
    text_column: str = "text",
):
    dataset = load_dataset(
        "parquet",
        data_files={"data": parquet_path},
        split="data",
    )

    if text_column not in dataset.column_names:
        raise KeyError(
            f"Column '{text_column}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    text = "\n\n".join(dataset[text_column])
    encoded = tokenizer(
        text,
        return_tensors="pt",
    )
    return encoded.input_ids


def iter_sequence_batches(
    input_ids,
    sequence_length: int,
    max_samples=None,
):
    total_complete = (
        input_ids.numel() // sequence_length
    )
    if max_samples is not None:
        total_complete = min(
            total_complete, int(max_samples)
        )

    for i in range(total_complete):
        yield input_ids[
            :,
            i * sequence_length:
            (i + 1) * sequence_length,
        ]
