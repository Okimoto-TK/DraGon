from __future__ import annotations

import torch
import torch.nn as nn

from ._runtime_checks import tensor_value_checks_enabled


class GammaNLLLoss(nn.Module):
    """Gamma negative log-likelihood under mean-shape parameterization."""

    def __init__(
        self,
        _eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if _eps <= 0:
            raise ValueError(
                f"_eps must be > 0, got {_eps}. Valid range: (0, +inf)."
            )

        self._eps = float(_eps)

    def forward(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        shape: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_column_tensor("target", target)
        _validate_column_tensor("mean", mean)
        _validate_column_tensor("shape", shape)
        if sample_weight is not None:
            _validate_column_tensor("sample_weight", sample_weight)

        if target.shape != mean.shape or target.shape != shape.shape:
            raise ValueError(
                "target, mean, and shape must share the same shape, "
                f"got target={tuple(target.shape)}, mean={tuple(mean.shape)}, shape={tuple(shape.shape)}. "
                "Valid shape: [B, 1] with a shared batch size."
            )
        if sample_weight is not None and sample_weight.shape != target.shape:
            raise ValueError(
                "sample_weight must share the same shape as target, mean, and shape, "
                f"got sample_weight={tuple(sample_weight.shape)}, target={tuple(target.shape)}."
            )
        if tensor_value_checks_enabled() and torch.any(mean <= 0):
            raise ValueError(
                f"mean must be strictly positive, got min={mean.min().item()}. Valid range: (0, +inf)."
            )
        if tensor_value_checks_enabled() and torch.any(shape <= 0):
            raise ValueError(
                f"shape must be strictly positive, got min={shape.min().item()}. Valid range: (0, +inf)."
            )

        target_safe = target.clamp_min(self._eps)
        nll = (
            shape * (target_safe / mean)
            + shape * torch.log(mean)
            - shape * torch.log(shape)
            + torch.lgamma(shape)
            - (shape - 1.0) * torch.log(target_safe)
        )
        if sample_weight is None:
            return nll.mean()
        if tensor_value_checks_enabled() and torch.any(sample_weight < 0):
            raise ValueError(
                f"sample_weight must be >= 0, got min={sample_weight.min().item()}."
            )
        weight_sum = sample_weight.sum().clamp_min(self._eps)
        return (nll * sample_weight).sum() / weight_sum


def _validate_column_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B, 1], got shape={tuple(value.shape)}. Valid shape: [B, 1]."
        )
