"""Sidechain context encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.components.fusion.token_attention import SequenceCrossBlock
from src.models.components.trunks.common import to_time_major


class SequenceGroupEncoder(nn.Module):
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


class CausalFuse(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        l = self.left(left)
        r = self.right(right)
        prod = l * r
        absdiff = torch.abs(l - r)
        return self.out(torch.cat((prod, absdiff), dim=-1))


class SidechainContextEncoder(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.gap_encoder = SequenceGroupEncoder(2, dim)
        self.moneyflow_encoder = SequenceGroupEncoder(3, dim)
        self.liquidity_encoder = SequenceGroupEncoder(3, dim)
        self.mf_reads_liq = SequenceCrossBlock(dim, num_heads=num_heads)
        self.liq_reads_mf = SequenceCrossBlock(dim, num_heads=num_heads)
        self.cause_fuse = CausalFuse(dim)
        self.gap_reads_cause = SequenceCrossBlock(dim, num_heads=num_heads)

    def forward(self, sidechain: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        gap = self.gap_encoder(sidechain[:, 0:2, :])
        moneyflow = self.moneyflow_encoder(sidechain[:, 2:5, :])
        liquidity = self.liquidity_encoder(sidechain[:, 5:8, :])
        z_mf1 = self.mf_reads_liq(moneyflow, liquidity)
        z_liqreg1 = self.liq_reads_mf(liquidity, moneyflow)
        z_cause = self.cause_fuse(z_mf1, z_liqreg1)
        z_gap_ctx = self.gap_reads_cause(gap, z_cause)
        e_d = torch.stack((z_mf1, z_liqreg1, z_cause, z_gap_ctx), dim=2)
        return e_d, {
            "z_mf1": z_mf1,
            "z_liqreg1": z_liqreg1,
            "z_cause": z_cause,
            "z_gap_ctx": z_gap_ctx,
        }


__all__ = ["CausalFuse", "SequenceGroupEncoder", "SidechainContextEncoder"]

