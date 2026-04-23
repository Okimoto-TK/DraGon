"""Collate functions for network-facing training batches."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import (
    FLOAT_BATCH_KEYS,
    INT_BATCH_KEYS,
    NETWORK_SAMPLE_SHAPES,
    _validate_network_sample,
)


def estimate_network_sample_bytes() -> int:
    """Estimate the bytes of one fully adapted sample after collate dtypes."""

    total = 0
    for key in FLOAT_BATCH_KEYS:
        shape = NETWORK_SAMPLE_SHAPES[key]
        total += int(np.prod(shape)) * torch.tensor([], dtype=torch.bfloat16).element_size()
    for key in INT_BATCH_KEYS:
        shape = NETWORK_SAMPLE_SHAPES[key]
        total += int(np.prod(shape)) * np.dtype(np.int64).itemsize
    return total


def resolve_collate_chunk_size(
    *,
    batch_size: int,
    target_chunk_mb: float = 16.0,
) -> int:
    """Bound worker-side temporary batch assembly by splitting stack ops into chunks."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    target_bytes = max(int(target_chunk_mb * 1024 * 1024), 1)
    per_sample_bytes = max(estimate_network_sample_bytes(), 1)
    chunk_size = max(1, target_bytes // per_sample_bytes)
    return min(int(batch_size), int(chunk_size))


def _stack_samples_chunked(
    samples: list[dict[str, np.ndarray]],
    *,
    key: str,
    dtype: np.dtype,
    chunk_size: int,
) -> np.ndarray:
    first = np.asarray(samples[0][key], dtype=dtype)
    out = np.empty((len(samples), *first.shape), dtype=dtype)
    for start in range(0, len(samples), chunk_size):
        end = min(start + chunk_size, len(samples))
        out[start:end] = np.stack(
            [np.asarray(sample[key], dtype=dtype) for sample in samples[start:end]],
            axis=0,
        )
    return out


def collate_network_batch(
    samples: list[dict[str, np.ndarray]],
    *,
    chunk_size: int | None = None,
    validate_shapes: bool = True,
) -> dict[str, torch.Tensor]:
    """Stack adapted network samples into a torch batch."""

    if not samples:
        raise ValueError("collate_network_batch requires at least one sample.")

    if validate_shapes:
        for sample in samples:
            _validate_network_sample(sample)

    effective_chunk_size = (
        int(chunk_size)
        if chunk_size is not None
        else resolve_collate_chunk_size(batch_size=len(samples))
    )
    batch: dict[str, torch.Tensor] = {}
    for key in FLOAT_BATCH_KEYS:
        stacked = _stack_samples_chunked(
            samples,
            key=key,
            dtype=np.float32,
            chunk_size=effective_chunk_size,
        )
        batch[key] = torch.from_numpy(stacked).to(dtype=torch.bfloat16)

    for key in INT_BATCH_KEYS:
        stacked = _stack_samples_chunked(
            samples,
            key=key,
            dtype=np.int64,
            chunk_size=effective_chunk_size,
        )
        batch[key] = torch.from_numpy(stacked).to(dtype=torch.int64)

    if validate_shapes:
        batch_size = len(samples)
        for key, expected in NETWORK_SAMPLE_SHAPES.items():
            expected_batch_shape = (batch_size, *expected)
            if tuple(batch[key].shape) != expected_batch_shape:
                raise ValueError(
                    f"{key} batch shape mismatch: expected {expected_batch_shape}, "
                    f"got {tuple(batch[key].shape)}."
                )

    return batch
