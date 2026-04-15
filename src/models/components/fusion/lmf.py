"""Low-rank multimodal fusion modules."""
from __future__ import annotations

import torch
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import lmf_rank as DEFAULT_LMF_RANK
from torch import Tensor, nn

from src.models.components.normalization import LayerNorm2d


class LowRankFusion(nn.Module):
    """Fuse two vectors with low-rank multiplicative interactions."""

    def __init__(
        self,
        dx: int = DEFAULT_LMF_DIM,
        dy: int = DEFAULT_LMF_DIM,
        d_out: int = DEFAULT_LMF_DIM,
        rank: int = DEFAULT_LMF_RANK,
    ) -> None:
        super().__init__()

        if dx <= 0:
            msg = f"dx must be positive, got {dx}"
            raise ValueError(msg)

        if dy <= 0:
            msg = f"dy must be positive, got {dy}"
            raise ValueError(msg)

        if d_out <= 0:
            msg = f"d_out must be positive, got {d_out}"
            raise ValueError(msg)

        if rank <= 0:
            msg = f"rank must be positive, got {rank}"
            raise ValueError(msg)

        self.dx = dx
        self.dy = dy
        self.d_out = d_out
        self.rank = rank
        self.x_norm = nn.LayerNorm(dx)
        self.y_norm = nn.LayerNorm(dy)
        self.x_proj = nn.Linear(dx, rank * d_out)
        self.y_proj = nn.Linear(dy, rank * d_out)
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.shape[:-1] != y.shape[:-1]:
            msg = f"Leading dimensions must match, got {tuple(x.shape)} and {tuple(y.shape)}"
            raise ValueError(msg)

        if x.shape[-1] != self.dx:
            msg = f"Expected x last dim {self.dx}, got {x.shape[-1]}"
            raise ValueError(msg)

        if y.shape[-1] != self.dy:
            msg = f"Expected y last dim {self.dy}, got {y.shape[-1]}"
            raise ValueError(msg)

        x = self.x_norm(x)
        y = self.y_norm(y)
        a = self.x_proj(x).view(*x.shape[:-1], self.rank, self.d_out)
        b = self.y_proj(y).view(*y.shape[:-1], self.rank, self.d_out)
        return self.out_norm((a * b).sum(dim=-2))


class PairwiseLMFMap(nn.Module):
    """Apply low-rank fusion to every pair of sequence positions."""

    def __init__(
        self,
        dx: int = DEFAULT_LMF_DIM,
        dy: int = DEFAULT_LMF_DIM,
        d_out: int = DEFAULT_LMF_DIM,
        rank: int = DEFAULT_LMF_RANK,
    ) -> None:
        super().__init__()

        if dx <= 0 or dy <= 0 or d_out <= 0 or rank <= 0:
            msg = f"Expected positive dims/rank, got dx={dx}, dy={dy}, d_out={d_out}, rank={rank}"
            raise ValueError(msg)

        self.dx = dx
        self.dy = dy
        self.d_out = d_out
        self.rank = rank
        self.x_norm = nn.LayerNorm(dx)
        self.y_norm = nn.LayerNorm(dy)
        self.x_proj = nn.Linear(dx, rank * d_out)
        self.y_proj = nn.Linear(dy, rank * d_out)
        self.out_norm = LayerNorm2d(d_out)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 3 or y.ndim != 3:
            msg = f"Expected x and y to be [B, L, D], got {tuple(x.shape)} and {tuple(y.shape)}"
            raise ValueError(msg)

        if x.shape[0] != y.shape[0]:
            msg = f"Batch sizes must match, got {x.shape[0]} and {y.shape[0]}"
            raise ValueError(msg)

        if x.shape[-1] != self.dx:
            msg = f"Expected x last dim {self.dx}, got {x.shape[-1]}"
            raise ValueError(msg)

        if y.shape[-1] != self.dy:
            msg = f"Expected y last dim {self.dy}, got {y.shape[-1]}"
            raise ValueError(msg)

        x = self.x_norm(x)
        y = self.y_norm(y)
        a = self.x_proj(x).view(x.shape[0], x.shape[1], self.rank, self.d_out)
        b = self.y_proj(y).view(y.shape[0], y.shape[1], self.rank, self.d_out)
        fused = torch.einsum("biro,bjro->bijo", a, b).permute(0, 3, 1, 2)
        return self.out_norm(fused)


class TokenLMF(nn.Module):
    """Apply low-rank fusion token by token along aligned sequences."""

    def __init__(
        self,
        dx: int = DEFAULT_LMF_DIM,
        dy: int = DEFAULT_LMF_DIM,
        d_out: int = DEFAULT_LMF_DIM,
        rank: int = DEFAULT_LMF_RANK,
    ) -> None:
        super().__init__()

        if dx <= 0 or dy <= 0 or d_out <= 0 or rank <= 0:
            msg = f"Expected positive dims/rank, got dx={dx}, dy={dy}, d_out={d_out}, rank={rank}"
            raise ValueError(msg)

        self.dx = dx
        self.dy = dy
        self.d_out = d_out
        self.rank = rank
        self.x_norm = nn.LayerNorm(dx)
        self.y_norm = nn.LayerNorm(dy)
        self.x_proj = nn.Linear(dx, rank * d_out)
        self.y_proj = nn.Linear(dy, rank * d_out)
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 3 or y.ndim != 3:
            msg = f"Expected x and y to be [B, K, D], got {tuple(x.shape)} and {tuple(y.shape)}"
            raise ValueError(msg)

        if x.shape[:2] != y.shape[:2]:
            msg = f"Batch/token dimensions must match, got {tuple(x.shape[:2])} and {tuple(y.shape[:2])}"
            raise ValueError(msg)

        if x.shape[-1] != self.dx:
            msg = f"Expected x last dim {self.dx}, got {x.shape[-1]}"
            raise ValueError(msg)

        if y.shape[-1] != self.dy:
            msg = f"Expected y last dim {self.dy}, got {y.shape[-1]}"
            raise ValueError(msg)

        x = self.x_norm(x)
        y = self.y_norm(y)
        a = self.x_proj(x).view(x.shape[0], x.shape[1], self.rank, self.d_out)
        b = self.y_proj(y).view(y.shape[0], y.shape[1], self.rank, self.d_out)
        return self.out_norm((a * b).sum(dim=2))


__all__ = ["LowRankFusion", "PairwiseLMFMap", "TokenLMF"]
