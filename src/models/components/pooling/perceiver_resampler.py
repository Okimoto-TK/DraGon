"""Lightweight Perceiver-style resampler for short token sequences."""
from __future__ import annotations

import torch
from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_INPUT_DIM
from config.config import token_dim as DEFAULT_TOKEN_DIM
from torch import Tensor, nn


class PerceiverResampler(nn.Module):
    """Resample [B, L, D] into fixed latent tokens [B, K, D]."""

    def __init__(
        self,
        latent_token: int = DEFAULT_LATENT_TOKEN,
        dim: int = DEFAULT_TOKEN_DIM,
        *,
        input_dim: int = DEFAULT_INPUT_DIM,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_token <= 0:
            raise ValueError(f"latent_token must be positive, got {latent_token}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide dim, got num_heads={num_heads}, dim={dim}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.latent_token = latent_token
        self.dim = dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(torch.randn(latent_token, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.kv_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.debug_enabled = False
        self.last_cross_attn: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, L, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.shape[-1]}")

        bsz = x.shape[0]
        latents = self.query_norm(self.latents.unsqueeze(0).expand(bsz, -1, -1))
        kv = self.kv_norm(self.kv_proj(x))
        attended, attn = self.cross_attn(
            latents,
            kv,
            kv,
            need_weights=self.debug_enabled,
            average_attn_weights=False,
        )
        self.last_cross_attn = attn.detach() if self.debug_enabled and attn is not None else None
        tokens = latents + attended
        return tokens + self.ffn(self.post_attn_norm(tokens))

    def get_last_debug(self) -> dict[str, Tensor | None]:
        return {"cross_attn": self.last_cross_attn}

    def set_debug_capture(self, enabled: bool) -> None:
        self.debug_enabled = bool(enabled)


__all__ = ["PerceiverResampler"]
