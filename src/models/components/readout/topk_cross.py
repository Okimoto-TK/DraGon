"""Local dense cross readout with soft gate on logits."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.models.components.trunks.common import flatten_tokens, reshape_tokens
from src.models.components.trunks.cross_scale import MultiContextCondition


def build_local_mask(*, q_steps: int, q_tokens: int, kv_steps: int, kv_tokens: int, window: int) -> Tensor:
    centers = torch.round(torch.linspace(0, kv_steps - 1, q_steps))
    mask = torch.full((q_steps * q_tokens, kv_steps * kv_tokens), float("-inf"))
    for t in range(q_steps):
        center = int(centers[t].item())
        left = max(0, center - window)
        right = min(kv_steps - 1, center + window)
        valid_times = torch.arange(left, right + 1, dtype=torch.long)
        valid_idx = []
        for time_idx in valid_times.tolist():
            start = time_idx * kv_tokens
            valid_idx.extend(range(start, start + kv_tokens))
        row_slice = slice(t * q_tokens, (t + 1) * q_tokens)
        mask[row_slice, valid_idx] = 0.0
    return mask


class LocalDenseCrossReadout(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        q_steps: int,
        q_tokens: int,
        kv_steps: int,
        kv_tokens: int,
        window: int,
        num_contexts: int,
    ) -> None:
        super().__init__()
        self.q_tokens = q_tokens
        self.q_steps = q_steps
        self.scale = dim ** -0.5
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.cond = MultiContextCondition(dim, num_contexts=num_contexts)
        self.register_buffer("mask", build_local_mask(q_steps=q_steps, q_tokens=q_tokens, kv_steps=kv_steps, kv_tokens=kv_tokens, window=window))

        # Learned soft gate projection (低秩)
        gate_rank = min(32, dim)
        self.gate_q = nn.Linear(dim, gate_rank, bias=False)
        self.gate_k = nn.Linear(dim, gate_rank, bias=False)

    def forward(self, query: Tensor, source: Tensor, *contexts: Tensor) -> Tensor:
        bsz, _, _, dim = query.shape
        gamma, beta, _ = self.cond(*contexts)
        q_flat = flatten_tokens(query)
        src_flat = flatten_tokens(source)
        q_base = self.q_norm(q_flat)
        src_base = self.kv_norm(src_flat)
        q_mod = (q_base * (1.0 + gamma.unsqueeze(1))) + beta.unsqueeze(1)
        q_proj = self.q_proj(q_mod)
        k_proj = self.k_proj(src_base)
        v_proj = self.v_proj(src_base)

        # 1) 全量 attention logits
        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2)) * self.scale

        # 2) Local window mask: window 外为 -inf
        local_bias = self.mask.unsqueeze(0)

        # 3) Learned soft gate bias
        gate_q = self.gate_q(q_proj)  # [B, Lq, R]
        gate_k = self.gate_k(k_proj)  # [B, Lk, R]
        gate_logits = torch.matmul(gate_q, gate_k.transpose(-1, -2)) / math.sqrt(gate_q.shape[-1])
        gate = torch.sigmoid(gate_logits)
        gate_bias = torch.log(gate + 1e-6)

        # 合并所有 bias 后 softmax
        scores = scores + local_bias + gate_bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_proj)
        out = self.out_proj(out)
        return reshape_tokens(out, steps=self.q_steps, tokens=self.q_tokens)


__all__ = ["LocalDenseCrossReadout", "build_local_mask"]
