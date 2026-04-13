"""2D joint feature refinement blocks with UniRepLK-style large-kernel trunks."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
import torch
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN2d(nn.Module):
    """Global response normalization for NCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        gx = x.norm(p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ResConv2dBlock(nn.Module):
    """UniRepLK-style residual block with large-kernel depthwise token mixing."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        large_kernel_size: int = 13,
        small_kernel_size: int = 5,
        expansion: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if large_kernel_size <= 0 or large_kernel_size % 2 == 0:
            raise ValueError(f"large_kernel_size must be a positive odd integer, got {large_kernel_size}")
        if small_kernel_size <= 0 or small_kernel_size % 2 == 0:
            raise ValueError(f"small_kernel_size must be a positive odd integer, got {small_kernel_size}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if layer_scale_init < 0.0:
            raise ValueError(f"layer_scale_init must be non-negative, got {layer_scale_init}")

        hidden_channels = channels * expansion
        large_padding = large_kernel_size // 2
        small_padding = small_kernel_size // 2

        self.channels = channels
        self.token_mixer_norm = nn.BatchNorm2d(channels)
        self.dw_large = nn.Conv2d(
            channels,
            channels,
            kernel_size=large_kernel_size,
            stride=1,
            padding=large_padding,
            groups=channels,
            bias=bias,
        )
        self.dw_small = nn.Conv2d(
            channels,
            channels,
            kernel_size=small_kernel_size,
            stride=1,
            padding=small_padding,
            groups=channels,
            bias=bias,
        )
        self.channel_norm = LayerNorm2d(channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=bias),
            nn.GELU(),
            GRN2d(hidden_channels),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias),
        )
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")

        residual = x
        x = self.token_mixer_norm(x)
        x = self.dw_large(x) + self.dw_small(x)
        x = self.channel_mlp(self.channel_norm(x))
        return residual + self.layer_scale * x


class JointNet2D(nn.Module):
    """Large-kernel 2D trunk for refining pairwise interaction maps."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        num_blocks: int = 3,
        large_kernel_size: int = 13,
        small_kernel_size: int = 5,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.channels = channels
        blocks = [
            ResConv2dBlock(
                channels=channels,
                large_kernel_size=large_kernel_size,
                small_kernel_size=small_kernel_size,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
            nn.GELU(),
            *blocks,
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return x + self.net(x)


__all__ = ["JointNet2D", "ResConv2dBlock"]
