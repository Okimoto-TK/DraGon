"""Shared layer-normalization helpers for channel-first tensors."""
from __future__ import annotations

from torch import Tensor, nn


class LayerNorm1d(nn.Module):
    """LayerNorm over channels for [B, C, L] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, C, L], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for [B, C, H, W] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


__all__ = ["LayerNorm1d", "LayerNorm2d"]
