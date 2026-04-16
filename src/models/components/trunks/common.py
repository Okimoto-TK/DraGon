"""Common tensor helpers and lightweight blocks."""
from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F


def to_time_major(x: Tensor) -> Tensor:
    return x.transpose(1, 2).contiguous()


def to_channel_major(x: Tensor) -> Tensor:
    return x.transpose(1, 2).contiguous()


def pool_tokens(x: Tensor) -> Tensor:
    if x.ndim == 4:
        return x.mean(dim=(1, 2))
    if x.ndim == 3:
        return x.mean(dim=1)
    raise ValueError(f"Unsupported tensor rank for pooling: {tuple(x.shape)}")


def expand_to_bt(query: Tensor, batch: int, steps: int) -> Tensor:
    return query.unsqueeze(0).unsqueeze(0).expand(batch, steps, -1, -1).reshape(batch * steps, query.shape[0], query.shape[1])


def flatten_tokens(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,T,K,D], got {tuple(x.shape)}")
    bsz, steps, tokens, dim = x.shape
    return x.reshape(bsz, steps * tokens, dim)


def reshape_tokens(x: Tensor, *, steps: int, tokens: int) -> Tensor:
    bsz, _, dim = x.shape
    return x.reshape(bsz, steps, tokens, dim)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or dim * 2)
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        value = self.value(x)
        gate = F.silu(self.gate(x))
        return self.out(value * gate)


class ConvRefineBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = to_channel_major(x)
        y = self.dw(y)
        y = self.pw(y)
        y = F.silu(y)
        y = to_time_major(y)
        return self.norm(residual + y)


class SmallTokenRefine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))


__all__ = [
    "ConvRefineBlock",
    "SmallTokenRefine",
    "SwiGLUFFN",
    "expand_to_bt",
    "flatten_tokens",
    "pool_tokens",
    "reshape_tokens",
    "to_channel_major",
    "to_time_major",
]

