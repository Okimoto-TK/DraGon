"""Sidechain encoder."""
from __future__ import annotations

from config.config import hidden_dim as DEFAULT_HIDDEN_DIM
from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn

from src.models.components.encoders.res_conv_1d import ResConv1dBlock


class SidechainEncoder(nn.Module):
    """Encode sidechain features from [B, 8, L] to [B, lmf_dim, L]."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            msg = f"in_channels must be positive, got {in_channels}"
            raise ValueError(msg)

        if hidden_dim <= 0:
            msg = f"hidden_dim must be positive, got {hidden_dim}"
            raise ValueError(msg)

        if lmf_dim <= 0:
            msg = f"lmf_dim must be positive, got {lmf_dim}"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lmf_dim = lmf_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.GELU(),
            ResConv1dBlock(hidden_dim, kernel_size=3),
            ResConv1dBlock(hidden_dim, kernel_size=3),
            nn.Conv1d(hidden_dim, lmf_dim, kernel_size=1, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, C, L], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.in_channels:
            msg = f"Expected {self.in_channels} channels, got {x.shape[1]}"
            raise ValueError(msg)

        return self.encoder(x)


__all__ = ["SidechainEncoder"]
