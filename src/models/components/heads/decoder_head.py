"""Decoder head for direct multi-target prediction."""
from __future__ import annotations

from torch import Tensor, nn


class DecoderHead(nn.Module):
    """Map fused TFN features into 8 direct outputs."""

    def __init__(self, in_dim: int = 289, hidden_dim1: int = 128, hidden_dim2: int = 64, out_dim: int = 8) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim1 <= 0 or hidden_dim2 <= 0 or out_dim <= 0:
            raise ValueError(
                f"All dimensions must be positive, got in_dim={in_dim}, hidden_dim1={hidden_dim1}, "
                f"hidden_dim2={hidden_dim2}, out_dim={out_dim}"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [B, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {x.shape[-1]}")
        return self.net(x)


__all__ = ["DecoderHead"]
