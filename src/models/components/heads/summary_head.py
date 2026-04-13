"""Summary heads based on learnable tokens and pre-norm self-attention."""
from __future__ import annotations

import torch
from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn
import torch.nn.functional as F


class _PreNormSelfAttentionBlock(nn.Module):
    """A small pre-norm transformer encoder block."""

    def __init__(self, dim: int, *, num_heads: int = 4, ff_mult: int = 4) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide dim, got num_heads={num_heads}, dim={dim}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SummaryHead(nn.Module):
    """Summarize token sequences into one vector via learnable summary tokens."""

    def __init__(
        self,
        dim: int = DEFAULT_LMF_DIM,
        *,
        num_summary_tokens: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_summary_tokens <= 0:
            raise ValueError(f"num_summary_tokens must be positive, got {num_summary_tokens}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        resolved_summary_dim = dim if summary_dim is None else int(summary_dim)
        if resolved_summary_dim <= 0:
            raise ValueError(f"summary_dim must be positive, got {resolved_summary_dim}")

        self.dim = dim
        self.num_summary_tokens = num_summary_tokens
        self.summary_dim = resolved_summary_dim
        summary_tokens = nn.Parameter(Tensor(num_summary_tokens, dim))
        nn.init.trunc_normal_(summary_tokens, std=0.02)
        self.summary_tokens = summary_tokens
        self.layers = nn.ModuleList(
            [_PreNormSelfAttentionBlock(dim, num_heads=num_heads, ff_mult=ff_mult) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(num_summary_tokens * dim, resolved_summary_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, K, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {x.shape[-1]}")

        bsz = x.shape[0]
        summary = self.summary_tokens.unsqueeze(0).expand(bsz, -1, -1)
        tokens = F.layer_norm(torch.cat((summary, x), dim=1), (self.dim,))
        for layer in self.layers:
            tokens = layer(tokens)
        summary_tokens = tokens[:, : self.num_summary_tokens, :]
        return self.out_proj(summary_tokens.reshape(bsz, self.num_summary_tokens * self.dim))


__all__ = ["SummaryHead"]
