"""Top-k local cross readout."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.models.components.trunks.common import flatten_tokens, reshape_tokens
from src.models.components.trunks.cross_scale import CrossScaleCondition


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


class LocalTopKCrossReadout(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        q_steps: int,
        q_tokens: int,
        kv_steps: int,
        kv_tokens: int,
        window: int,
        topk: int,
        num_contexts: int,
    ) -> None:
        super().__init__()
        self.q_tokens = q_tokens
        self.q_steps = q_steps
        self.topk = int(topk)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.cond = CrossScaleCondition(dim, num_contexts=num_contexts)
        self.register_buffer("mask", build_local_mask(q_steps=q_steps, q_tokens=q_tokens, kv_steps=kv_steps, kv_tokens=kv_tokens, window=window))

    def forward(self, query: Tensor, source: Tensor, *contexts: Tensor) -> Tensor:
        bsz, _, _, dim = query.shape
        gamma, beta, _ = self.cond(*contexts)
        q_flat = flatten_tokens(query)
        src_flat = flatten_tokens(source)
        q_mod = (q_flat * (1.0 + gamma.unsqueeze(1))) + beta.unsqueeze(1)
        q_proj = self.q_proj(q_mod)
        k_proj = self.k_proj(src_flat)
        v_proj = self.v_proj(src_flat)

        scores = torch.matmul(q_proj, k_proj.transpose(1, 2)) / math.sqrt(dim)
        scores = scores + self.mask.unsqueeze(0)
        topk_scores, topk_idx = torch.topk(scores, k=self.topk, dim=-1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        selected_v = v_proj.unsqueeze(1).expand(-1, q_proj.shape[1], -1, -1).gather(2, gather_idx)
        attn = torch.softmax(topk_scores, dim=-1)
        out = torch.sum(attn.unsqueeze(-1) * selected_v, dim=2)
        out = self.out_proj(out)
        return reshape_tokens(out, steps=self.q_steps, tokens=self.q_tokens)


__all__ = ["LocalTopKCrossReadout", "build_local_mask"]

