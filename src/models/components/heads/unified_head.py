"""Unified prediction head."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn

from src.models.components.pooling import AttentivePool1d


class UnifiedHead(nn.Module):
    """Pool token features and map them to the requested output dimension."""

    def __init__(
        self,
        out_dim: int,
        dim: int = DEFAULT_LMF_DIM,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if out_dim <= 0:
            msg = f"out_dim must be positive, got {out_dim}"
            raise ValueError(msg)

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        if hidden_dim is not None and hidden_dim <= 0:
            msg = f"hidden_dim must be positive when provided, got {hidden_dim}"
            raise ValueError(msg)

        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0, 1), got {dropout}"
            raise ValueError(msg)

        resolved_hidden_dim = hidden_dim if hidden_dim is not None else dim * 2

        self.dim = dim
        self.out_dim = out_dim
        self.hidden_dim = resolved_hidden_dim
        self.pool = AttentivePool1d(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, resolved_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(resolved_hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, K, D], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[-1] != self.dim:
            msg = f"Expected feature dim {self.dim}, got {x.shape[-1]}"
            raise ValueError(msg)

        pooled = self.pool(x)
        return self.mlp(pooled)


__all__ = ["UnifiedHead"]
