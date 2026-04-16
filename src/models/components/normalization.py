"""Normalization helpers."""
from __future__ import annotations

from torch import Tensor, nn


def ada_layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Adaptive layer norm with per-sample affine modulation."""
    y = norm(x)
    return y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


__all__ = ["ada_layer_norm"]

