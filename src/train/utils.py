"""Shared utilities for training loops."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from src.task_labels import detect_task_from_outputs


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
    """GPU-resident weighted scalar metric accumulator."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._device: torch.device | None = None if device is None else torch.device(device)
        self._dtype = dtype
        self._sums: dict[str, Tensor] = {}
        self._count: Tensor | None = None
        if self._device is not None:
            self._count = torch.zeros((), device=self._device, dtype=torch.int64)

    def _ensure_device(self, value: Tensor | float | int) -> torch.device:
        if self._device is not None:
            return self._device
        if isinstance(value, Tensor):
            self._device = value.device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._count = torch.zeros((), device=self._device, dtype=torch.int64)
        return self._device

    def _as_scalar_tensor(self, value: Tensor | float | int, device: torch.device) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.detach()
            if tensor.ndim != 0:
                tensor = tensor.mean()
            return tensor.to(device=device, dtype=self._dtype)
        return torch.tensor(float(value), device=device, dtype=self._dtype)

    def update(
        self,
        metrics: Mapping[str, Tensor | float],
        weight: int,
    ) -> None:
        if not metrics:
            return
        first_value = next(iter(metrics.values()))
        device = self._ensure_device(first_value)
        if self._count is None:
            self._count = torch.zeros((), device=device, dtype=torch.int64)
        weight_i64 = torch.tensor(int(weight), device=device, dtype=torch.int64)
        weight_fp = weight_i64.to(dtype=self._dtype)
        self._count.add_(weight_i64)
        for key, value in metrics.items():
            scalar = self._as_scalar_tensor(value, device)
            if key not in self._sums:
                self._sums[key] = torch.zeros((), device=device, dtype=self._dtype)
            self._sums[key].add_(scalar * weight_fp)

    def reset(self) -> None:
        for total in self._sums.values():
            total.zero_()
        if self._count is not None:
            self._count.zero_()

    def _mean_tensor_by_key(self, key: str) -> Tensor:
        assert self._count is not None
        denom = self._count.clamp_min(1).to(dtype=self._dtype)
        return self._sums[key] / denom

    def compute(self) -> dict[str, float]:
        return self.compute_and_reset(reset=False)

    def compute_and_reset(self, *, reset: bool = True) -> dict[str, float]:
        if self._count is None or not self._sums:
            return {}
        keys = sorted(self._sums.keys())
        packed = torch.stack([self._mean_tensor_by_key(key) for key in keys], dim=0)
        values = packed.detach().cpu().tolist()
        result = {key: float(value) for key, value in zip(keys, values)}
        if reset:
            self.reset()
        return result


def batch_prediction_metrics(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Compute single-task prediction diagnostics for one batch."""
    task = detect_task_from_outputs(outputs)
    target = batch[f"label_{task}"]
    metrics: dict[str, Tensor] = {}

    if task == "Persist":
        pred = outputs["pred_Persist"]
        metrics["mae_Persist"] = torch.mean(torch.abs(pred - target)).detach()
        metrics["brier_Persist"] = torch.mean((pred - target) * (pred - target)).detach()
        metrics["prob_Persist_mean"] = pred.mean().detach()
        metrics["unc_Persist_mean"] = outputs["Persist_unc"].mean().detach()
        return metrics

    pred = outputs[f"pred_{task}"]
    metrics[f"mae_{task}"] = torch.mean(torch.abs(pred - target)).detach()
    metrics[f"unc_{task}_mean"] = outputs[f"unc_{task}"].mean().detach()
    return metrics


__all__ = [
    "MetricTracker",
    "batch_prediction_metrics",
    "grad_norm",
    "move_batch_to_device",
]
