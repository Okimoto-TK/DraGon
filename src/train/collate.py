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


def collate_network_batch(
    samples: list[dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Stack adapted network samples into a torch batch."""

    if not samples:
        raise ValueError("collate_network_batch requires at least one sample.")

    for sample in samples:
        _validate_network_sample(sample)

    batch: dict[str, torch.Tensor] = {}
    for key in FLOAT_BATCH_KEYS:
        stacked = np.stack(
            [np.asarray(sample[key], dtype=np.float32) for sample in samples],
            axis=0,
        )
        batch[key] = torch.from_numpy(stacked).to(dtype=torch.float32)

    for key in INT_BATCH_KEYS:
        stacked = np.stack(
            [np.asarray(sample[key], dtype=np.int64) for sample in samples],
            axis=0,
        )
        batch[key] = torch.from_numpy(stacked).to(dtype=torch.int64)

    batch_size = len(samples)
    for key, expected in NETWORK_SAMPLE_SHAPES.items():
        expected_batch_shape = (batch_size, *expected)
        if tuple(batch[key].shape) != expected_batch_shape:
            raise ValueError(
                f"{key} batch shape mismatch: expected {expected_batch_shape}, "
                f"got {tuple(batch[key].shape)}."
            )

    return batch
