"""Pairwise and dual-stream fusion modules."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.components.fusion.token_attention import PerTimeCrossAttention, TokenSelfAttentionBlock
from src.models.components.pooling.query_pool import QueryTokenPooling


class PairwiseCross(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)
        self.diff = nn.Linear(dim, dim)
        self.absdiff = nn.Linear(dim, dim)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prod = self.left(left) * self.right(right)
        diff = self.diff(left - right)
        absdiff = self.absdiff(torch.abs(left - right))
        return prod, diff, absdiff


class PairInteractionGrid(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj_p = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, price: Tensor, liquid: Tensor) -> Tensor:
        p = self.proj_p(price).unsqueeze(3)
        v = self.proj_v(liquid).unsqueeze(2)
        prod = p * v
        absdiff = torch.abs(p - v)
        p_b = p.expand(-1, -1, -1, liquid.shape[2], -1)
        v_b = v.expand(-1, -1, price.shape[2], -1, -1)
        pair = torch.cat((p_b, v_b, prod, absdiff), dim=-1)
        grid = self.fuse(pair)
        return grid.reshape(grid.shape[0], grid.shape[1], grid.shape[2] * grid.shape[3], grid.shape[4])


class PriceLiquidityDualFusion(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, joint_tokens: int) -> None:
        super().__init__()
        self.price_self = TokenSelfAttentionBlock(dim, num_heads=num_heads)
        self.liquid_self = TokenSelfAttentionBlock(dim, num_heads=num_heads)
        self.price_to_liquid = PerTimeCrossAttention(dim, num_heads=num_heads)
        self.liquid_to_price = PerTimeCrossAttention(dim, num_heads=num_heads)
        self.pair_grid = PairInteractionGrid(dim)
        self.joint_readout = QueryTokenPooling(dim, out_tokens=joint_tokens, num_heads=num_heads)

    def forward(self, price: Tensor, liquid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        price_self = self.price_self(price)
        liquid_self = self.liquid_self(liquid)
        price_dual = price_self + self.price_to_liquid(price_self, liquid_self)
        liquid_dual = liquid_self + self.liquid_to_price(liquid_self, price_self)
        pair_grid = self.pair_grid(price_dual, liquid_dual)
        joint = self.joint_readout(pair_grid)
        return price_dual, liquid_dual, pair_grid, joint


__all__ = ["PairInteractionGrid", "PairwiseCross", "PriceLiquidityDualFusion"]

