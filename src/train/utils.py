"""Shared utilities for training loops."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import torch
from torch import Tensor

_TASKS = ("S", "M", "MDD", "RV")


def move_batch_to_device(batch: Mapping[str, Tensor], device: str) -> dict[str, Tensor]:
    """Move tensor values in ``batch`` onto ``device``."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def grad_norm(parameters) -> float:
    """Compute the global L2 gradient norm over parameters that have gradients."""
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total) if total > 0.0 else 0.0


class MetricTracker:
    """Accumulate weighted scalar metrics."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0

    def update(
        self,
        metrics: Mapping[str, Tensor | float],
        weight: int,
    ) -> None:
        self._count += int(weight)
        for key, value in metrics.items():
            scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
            self._sums[key] += scalar * weight

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {key: 0.0 for key in self._sums}
        return {key: total / self._count for key, total in self._sums.items()}


def batch_prediction_metrics(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Compute MAE and scale statistics for one batch."""
    mean_metrics: dict[str, Tensor] = {}

    for task in _TASKS:
        pred = outputs[f"pred_{task}"]
        target = batch[f"label_{task}"]
        scale = outputs[f"scale_{task}"]

        mean_metrics[f"mae_{task}"] = torch.mean(torch.abs(pred - target)).detach()
        mean_metrics[f"scale_{task}_mean"] = scale.mean().detach()

    return mean_metrics


__all__ = [
    "MetricTracker",
    "batch_prediction_metrics",
    "grad_norm",
    "move_batch_to_device",
]
