"""Map 2D interaction maps into a fixed number of latent tokens."""
from __future__ import annotations

from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn


class InteractionMapToTokens(nn.Module):
    """Compress [B, C, Lx, Ly] into [B, latent_token, C]."""

    def __init__(self, latent_token: int = DEFAULT_LATENT_TOKEN, dim: int = DEFAULT_LMF_DIM) -> None:
        super().__init__()

        if latent_token <= 0:
            msg = f"latent_token must be positive, got {latent_token}"
            raise ValueError(msg)

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        self.latent_token = latent_token
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool1d(latent_token)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            msg = f"Expected input shape [B, C, Lx, Ly], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.dim:
            msg = f"Expected channel dim {self.dim}, got {x.shape[1]}"
            raise ValueError(msg)

        bsz, channels, lx, ly = x.shape
        tokens = x.view(bsz, channels, lx * ly)
        tokens = self.pool(tokens)
        return tokens.transpose(1, 2)


__all__ = ["InteractionMapToTokens"]
