"""Attentive pooling over 1D token sequences."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn


class AttentivePool1d(nn.Module):
    """Pool [B, K, D] into [B, D] with learned token weights."""

    def __init__(self, dim: int = DEFAULT_LMF_DIM) -> None:
        super().__init__()

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        self.dim = dim
        self.input_norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, K, D], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[-1] != self.dim:
            msg = f"Expected feature dim {self.dim}, got {x.shape[-1]}"
            raise ValueError(msg)

        x = self.input_norm(x)
        scores = self.score(x)
        weights = scores.softmax(dim=1)
        return self.output_norm((weights * x).sum(dim=1))


__all__ = ["AttentivePool1d"]
