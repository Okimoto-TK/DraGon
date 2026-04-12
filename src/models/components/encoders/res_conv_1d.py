"""Residual 1D convolution block."""
from __future__ import annotations

from torch import Tensor, nn


class ResConv1dBlock(nn.Module):
    """A length-preserving residual Conv1d block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            msg = f"channels must be positive, got {channels}"
            raise ValueError(msg)

        if kernel_size <= 0 or kernel_size % 2 == 0:
            msg = f"kernel_size must be a positive odd integer, got {kernel_size}"
            raise ValueError(msg)

        if expansion <= 0:
            msg = f"expansion must be positive, got {expansion}"
            raise ValueError(msg)

        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0, 1), got {dropout}"
            raise ValueError(msg)

        hidden_channels = channels * expansion
        padding = kernel_size // 2

        self.channels = channels
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, C, L], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.channels:
            msg = f"Expected {self.channels} channels, got {x.shape[1]}"
            raise ValueError(msg)

        return x + self.block(x)


__all__ = ["ResConv1dBlock"]
