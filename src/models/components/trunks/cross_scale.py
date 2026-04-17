"""Cross-scale conditioning and fusion blocks."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.components.normalization import ada_layer_norm
from src.models.components.trunks.common import SwiGLUFFN, flatten_tokens, pool_tokens, reshape_tokens


class MultiContextCondition(nn.Module):
    def __init__(self, dim: int, *, num_contexts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * num_contexts, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2 + 1),
        )

    def forward(self, *contexts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = torch.cat([pool_tokens(ctx) for ctx in contexts], dim=-1)
        out = self.net(pooled)
        gamma, beta, gate = torch.split(out, [contexts[0].shape[-1], contexts[0].shape[-1], 1], dim=-1)
        return gamma, beta, torch.sigmoid(gate)


class CrossScaleCrossAttention(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, num_contexts: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cond = MultiContextCondition(dim, num_contexts=num_contexts)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query: Tensor, source: Tensor, *contexts: Tensor) -> Tensor:
        bsz, q_steps, q_tokens, dim = query.shape
        gamma, beta, gate = self.cond(*contexts)
        q_flat = flatten_tokens(query)
        src_flat = flatten_tokens(source)
        q_mod = ada_layer_norm(q_flat, gamma, beta, self.q_norm)
        src_mod = self.kv_norm(src_flat)
        delta, _ = self.attn(q_mod, src_mod, src_mod, need_weights=False)
        out = self.out_norm(q_flat + gate.unsqueeze(1) * delta)
        return reshape_tokens(out, steps=q_steps, tokens=q_tokens)


class CrossScaleFFN(nn.Module):
    def __init__(self, dim: int, *, num_contexts: int) -> None:
        super().__init__()
        self.cond = MultiContextCondition(dim, num_contexts=num_contexts)
        self.norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *contexts: Tensor) -> Tensor:
        bsz, steps, tokens, dim = x.shape
        gamma, beta, gate = self.cond(*contexts)
        flat = x.reshape(bsz * steps, tokens, dim)
        gamma_expanded = gamma.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, dim)
        beta_expanded = beta.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, dim)
        mod = ada_layer_norm(flat, gamma_expanded, beta_expanded, self.norm)
        delta = self.ffn(mod)
        flat = self.out_norm(flat + gate.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, 1, 1) * delta)
        return flat.reshape(bsz, steps, tokens, dim)


__all__ = ["CrossScaleCrossAttention", "CrossScaleFFN", "MultiContextCondition"]
