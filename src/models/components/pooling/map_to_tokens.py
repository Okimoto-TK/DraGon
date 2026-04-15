"""Map 2D interaction maps into a fixed number of latent tokens."""
from __future__ import annotations

import torch
from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn


class InteractionMapToTokens(nn.Module):
    """Compress [B, C, Lx, Ly] into [B, latent_token, C] with latent cross-attention."""

    def __init__(
        self,
        latent_token: int = DEFAULT_LATENT_TOKEN,
        dim: int = DEFAULT_LMF_DIM,
        num_heads: int = 4,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()

        if latent_token <= 0:
            msg = f"latent_token must be positive, got {latent_token}"
            raise ValueError(msg)

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        if num_heads <= 0 or dim % num_heads != 0:
            msg = f"num_heads must be positive and divide dim, got num_heads={num_heads}, dim={dim}"
            raise ValueError(msg)

        if ff_mult <= 0:
            msg = f"ff_mult must be positive, got {ff_mult}"
            raise ValueError(msg)

        self.latent_token = latent_token
        self.dim = dim
        self.num_heads = num_heads

        self.latent_queries = nn.Parameter(torch.randn(latent_token, dim) * 0.02)
        self.coord_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def _coord_tokens(self, *, lx: int, ly: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        x_coord = torch.linspace(-1.0, 1.0, steps=lx, device=device, dtype=dtype)
        y_coord = torch.linspace(-1.0, 1.0, steps=ly, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(x_coord, y_coord, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(lx * ly, 2)
        return self.coord_proj(coords)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            msg = f"Expected input shape [B, C, Lx, Ly], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.dim:
            msg = f"Expected channel dim {self.dim}, got {x.shape[1]}"
            raise ValueError(msg)

        bsz, channels, lx, ly = x.shape
        fmap_tokens = x.reshape(bsz, channels, lx * ly).transpose(1, 2)
        coord_tokens = self._coord_tokens(lx=lx, ly=ly, device=x.device, dtype=x.dtype)
        fmap_tokens = self.input_norm(fmap_tokens + coord_tokens.unsqueeze(0))

        queries = self.query_norm(self.latent_queries.unsqueeze(0).expand(bsz, -1, -1))
        attended, _ = self.cross_attn(queries, fmap_tokens, fmap_tokens, need_weights=False)
        tokens = queries + attended
        tokens = tokens + self.ffn(self.post_attn_norm(tokens))
        return self.output_norm(tokens)


__all__ = ["InteractionMapToTokens"]
