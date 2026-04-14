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
        self._sums: dict[str, Tensor] = {}
        self._count = 0
        self._device: torch.device | None = None

    def _as_scalar_tensor(self, value: Tensor | float) -> Tensor:
        if isinstance(value, Tensor):
            detached = value.detach()
            if detached.ndim != 0:
                detached = detached.mean()
            if self._device is None:
                self._device = detached.device
            return detached

        device = self._device or torch.device("cpu")
        return torch.tensor(float(value), device=device)

    def update(
        self,
        metrics: Mapping[str, Tensor | float],
        weight: int,
    ) -> None:
        self._count += int(weight)
        for key, value in metrics.items():
            scalar = self._as_scalar_tensor(value)
            weighted = scalar * float(weight)
            if key in self._sums:
                self._sums[key] = self._sums[key] + weighted
            else:
                self._sums[key] = weighted

    def compute(self, *, as_python: bool = True) -> dict[str, Tensor | float]:
        if self._count == 0:
            if as_python:
                return {key: 0.0 for key in self._sums}
            return {
                key: torch.zeros((), device=value.device, dtype=value.dtype)
                for key, value in self._sums.items()
            }

        averaged = {key: total / float(self._count) for key, total in self._sums.items()}
        if not as_python:
            return {key: value.detach() for key, value in averaged.items()}
        return {key: float(value.detach().cpu().item()) for key, value in averaged.items()}


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
