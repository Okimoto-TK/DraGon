"""Sidechain context encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.components.fusion.token_attention import SequenceCrossAttention
from src.models.components.trunks.common import to_time_major


class SeqGroupEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, dim, kernel_size=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.proj(x)
        y = self.dw(y)
        y = self.pw(y)
        y = F.silu(y)
        return self.norm(to_time_major(y))


class CausalFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        l = self.left(left)
        r = self.right(right)
        prod = l * r
        absdiff = torch.abs(l - r)
        return self.out_norm(self.out(torch.cat((prod, absdiff), dim=-1)))


class SideContextEncoder(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.gap_encoder = SeqGroupEncoder(2, dim)
        self.moneyflow_encoder = SeqGroupEncoder(3, dim)
        self.liquidity_encoder = SeqGroupEncoder(3, dim)
        self.mf_reads_liq = SequenceCrossAttention(dim, num_heads=num_heads)
        self.liq_reads_mf = SequenceCrossAttention(dim, num_heads=num_heads)
        self.cause_fuse = CausalFusion(dim)
        self.gap_reads_cause = SequenceCrossAttention(dim, num_heads=num_heads)

    def forward(self, sidechain: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        gap = self.gap_encoder(sidechain[:, 0:2, :])
        moneyflow = self.moneyflow_encoder(sidechain[:, 2:5, :])
        liquidity = self.liquidity_encoder(sidechain[:, 5:8, :])
        z_moneyflow = self.mf_reads_liq(moneyflow, liquidity)
        z_liquidity_reg = self.liq_reads_mf(liquidity, moneyflow)
        z_causal = self.cause_fuse(z_moneyflow, z_liquidity_reg)
        z_gap_context = self.gap_reads_cause(gap, z_causal)
        e_d = torch.stack((z_moneyflow, z_liquidity_reg, z_causal, z_gap_context), dim=2)
        return e_d, {
            "z_moneyflow": z_moneyflow,
            "z_liquidity_reg": z_liquidity_reg,
            "z_causal": z_causal,
            "z_gap_context": z_gap_context,
        }


__all__ = ["CausalFusion", "SeqGroupEncoder", "SideContextEncoder"]

