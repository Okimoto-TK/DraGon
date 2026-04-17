"""Pairwise and dual-stream fusion modules."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.fusion.token_attention import PerTimeCrossAttention
from src.models.components.pooling.query_pool import QueryTokenPooling


class PairCross(nn.Module):
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


class GatedDualFusion(nn.Module):
    """Gated bilinear fusion for price-liquid interaction.
    要求 Kp == Kv,即 price 和 liquid 的 token 数相同。
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.price_norm = nn.LayerNorm(dim)
        self.liquid_norm = nn.LayerNorm(dim)
        self.price_proj1 = nn.Linear(dim, dim)
        self.liquid_proj1 = nn.Linear(dim, dim)
        self.price_proj2 = nn.Linear(dim, dim)
        self.liquid_proj2 = nn.Linear(dim, dim)
        self.down = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, price: Tensor, liquid: Tensor) -> Tensor:
        assert price.shape[:3] == liquid.shape[:3], "price 和 liquid 的 [B, T, K] 必须一致"

        p = self.price_norm(price)
        v = self.liquid_norm(liquid)

        fused = F.silu(self.price_proj1(p)) * self.liquid_proj1(v) + F.silu(self.liquid_proj2(v)) * self.price_proj2(p)
        fused = self.down(fused)

        interaction = self.out_norm(price + liquid + fused)
        return interaction


class PriceLiquidityDualFusion(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, joint_tokens: int) -> None:
        super().__init__()
        # 保留双向 cross-attn,删除 self-attn
        self.price_to_liquid = PerTimeCrossAttention(dim, num_heads=num_heads)
        self.liquid_to_price = PerTimeCrossAttention(dim, num_heads=num_heads)
        # Gated bilinear fusion 替代 pair_grid
        self.interaction_fusion = GatedDualFusion(dim)
        self.joint_readout = QueryTokenPooling(dim, out_tokens=joint_tokens, num_heads=num_heads)
        self.price_res_norm = nn.LayerNorm(dim)
        self.liquid_res_norm = nn.LayerNorm(dim)

    def forward(self, price: Tensor, liquid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # 双向 cross-attn
        price_ctx = self.price_to_liquid(price, liquid)
        liquid_ctx = self.liquid_to_price(liquid, price)

        price_dual = self.price_res_norm(price + price_ctx)
        liquid_dual = self.liquid_res_norm(liquid + liquid_ctx)

        # Gated bilinear fusion
        interaction_tokens = self.interaction_fusion(price_dual, liquid_dual)
        joint = self.joint_readout(interaction_tokens)

        return price_dual, liquid_dual, interaction_tokens, joint


__all__ = ["GatedDualFusion", "PairCross", "PriceLiquidityDualFusion"]
