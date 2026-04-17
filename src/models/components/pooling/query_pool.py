"""Learned token pooling blocks."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.components.trunks.common import expand_to_bt


class QueryTokenPooling(nn.Module):
    def __init__(self, dim: int, *, out_tokens: int, num_heads: int) -> None:
        super().__init__()
        self.learned_query = nn.Parameter(torch.randn(out_tokens, dim) * 0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)

    def forward(self, x: Tensor) -> Tensor:
        bsz, steps, tokens, dim = x.shape
        kv = x.reshape(bsz * steps, tokens, dim)
        q = expand_to_bt(self.learned_query, bsz, steps)
        out, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        return out.reshape(bsz, steps, self.learned_query.shape[0], dim)


__all__ = ["QueryTokenPooling"]
