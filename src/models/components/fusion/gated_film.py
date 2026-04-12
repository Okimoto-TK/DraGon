"""Gated FiLM fusion blocks."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn


class GatedFiLM(nn.Module):
    """Apply residual FiLM modulation with a learned gate."""

    def __init__(self, dim: int = DEFAULT_LMF_DIM) -> None:
        super().__init__()

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        self.dim = dim
        self.modulation = nn.Linear(dim, dim * 3)

    def forward(self, x: Tensor, signal: Tensor) -> Tensor:
        if x.ndim != 3 or signal.ndim != 3:
            msg = f"Expected x and signal to be [B, K, D], got {tuple(x.shape)} and {tuple(signal.shape)}"
            raise ValueError(msg)

        if x.shape != signal.shape:
            msg = f"Expected x and signal to have the same shape, got {tuple(x.shape)} and {tuple(signal.shape)}"
            raise ValueError(msg)

        if x.shape[-1] != self.dim:
            msg = f"Expected feature dim {self.dim}, got {x.shape[-1]}"
            raise ValueError(msg)

        gamma, beta, gate = self.modulation(signal).chunk(3, dim=-1)
        gate = gate.sigmoid()
        modulated = x * (1.0 + gamma) + beta
        return x + gate * (modulated - x)


__all__ = ["GatedFiLM"]
