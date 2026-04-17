"""Path branch encoders."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.fusion.token_attention import TokenSelfAttention
from src.models.components.pooling.query_pool import QueryTokenPooling
from src.models.components.trunks.common import ConvRefineBlock, SmallTokenRefine, to_channel_major, to_time_major


_HAAR_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class WNOBranch(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x)
        y = self.pw(y)
        return F.silu(y)


class WNOBlock(nn.Module):
    def __init__(self, dim: int, *, levels: int) -> None:
        super().__init__()
        self.levels = int(levels)
        self.pre_norm = nn.LayerNorm(dim)
        self.approx_branch = WNOBranch(dim)
        self.detail_branches = nn.ModuleList([WNOBranch(dim) for _ in range(self.levels)])
        self.out_norm = nn.LayerNorm(dim)

    def _decompose(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        current = x
        details: list[Tensor] = []
        for _ in range(self.levels):
            even = current[..., 0::2]
            odd = current[..., 1::2]
            approx = (even + odd) * _HAAR_INV_SQRT2
            detail = (even - odd) * _HAAR_INV_SQRT2
            details.append(detail)
            current = approx
        return current, details

    def _reconstruct(self, approx: Tensor, details: list[Tensor]) -> Tensor:
        current = approx
        for detail in reversed(details):
            even = (current + detail) * _HAAR_INV_SQRT2
            odd = (current - detail) * _HAAR_INV_SQRT2
            current = torch.stack((even, odd), dim=-1).reshape(current.shape[0], current.shape[1], -1)
        return current

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = to_channel_major(self.pre_norm(x))
        approx, details = self._decompose(y)
        approx = self.approx_branch(approx)
        details = [branch(detail) for branch, detail in zip(self.detail_branches, details)]
        y = self._reconstruct(approx, details)
        y = to_time_major(y)
        return self.out_norm(residual + y)


class ParallelWNOEncoder(nn.Module):
    """共享的单特征编码器,输入 [B*4, T],输出 [B*4, T, D]"""
    def __init__(self, dim: int, *, levels: int, num_blocks: int, local_kernel: int) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, dim, kernel_size=1)
        self.blocks = nn.ModuleList([WNOBlock(dim, levels=levels) for _ in range(num_blocks)])
        self.local_refine = ConvRefineBlock(dim, kernel_size=local_kernel)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B*4, T] -> [B*4, D, T]
        y = self.lift(x.unsqueeze(1))
        y = to_time_major(y)
        for block in self.blocks:
            y = block(y)
        y = self.local_refine(y)
        return y


class PathXYMixer(nn.Module):
    """共享的 XY 条件融合模块,输入 [B*4, T, D] + [B*4, T, 2],输出 [B*4, T, D]"""
    def __init__(self, dim: int, *, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim + 2, dim, kernel_size=kernel_size, padding=padding)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, xy: Tensor) -> Tensor:
        # h: [B*4, T, D], xy: [B*4, T, 2]
        residual = h
        y = torch.cat((h, xy), dim=-1)
        y = to_channel_major(y)
        y = self.conv(y)
        y = F.silu(y)
        y = self.proj(y)
        y = to_time_major(y)
        return self.norm(residual + y)


class VectorizedPairCross(nn.Module):
    """向量化 pairwise 交叉特征计算
    固定 pair 定义: [(0,1), (0,2), (1,2), (0,3), (1,3)]
    输入: [B, 4, T, D]
    输出: (prod, diff, absdiff), 每个 shape [B, 5, T, D]
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("pair_index", torch.tensor([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3]]))
        self.fuse_prod = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.fuse_diff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.fuse_absdiff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # feat: [B, 4, T, D]
        B, _, T, D = feat.shape

        left = feat[:, self.pair_index[:, 0]]
        right = feat[:, self.pair_index[:, 1]]

        prod = left * right
        diff = left - right
        absdiff = diff.abs()

        flat_shape = (B * 5, T, D)
        prod = self.fuse_prod(prod.reshape(flat_shape)).reshape(B, 5, T, D)
        diff = self.fuse_diff(diff.reshape(flat_shape)).reshape(B, 5, T, D)
        absdiff = self.fuse_absdiff(absdiff.reshape(flat_shape)).reshape(B, 5, T, D)

        return prod, diff, absdiff


class PathEncoder(nn.Module):
    """向量化 PathEncoder

    Token schema (总计 19 tokens):
    - [0:4]   : 4 个 feature tokens
    - [4:9]   : 5 个 pair-prod tokens
    - [9:14]  : 5 个 pair-diff tokens
    - [14:19] : 5 个 pair-absdiff tokens

    Pair 顺序 (5 pairs):
    - (0,1), (0,2), (1,2), (0,3), (1,3)
    """
    def __init__(self, dim: int, *, levels: int, num_blocks: int, local_kernel: int, out_tokens: int, num_heads: int) -> None:
        super().__init__()
        self.feature_encoder = ParallelWNOEncoder(dim, levels=levels, num_blocks=num_blocks, local_kernel=local_kernel)
        self.xy_mixer = PathXYMixer(dim, kernel_size=local_kernel)
        self.pairwise_cross = VectorizedPairCross(dim)
        self.token_block = TokenSelfAttention(dim, num_heads=num_heads)
        self.pool = QueryTokenPooling(dim, out_tokens=out_tokens, num_heads=num_heads)
        self.refine = SmallTokenRefine(dim)

        self.feature_type_embed = nn.Parameter(torch.empty(4, dim))
        self.xy_feature_bias = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.feature_type_embed, std=0.02)
        nn.init.normal_(self.xy_feature_bias, std=0.02)

    def _encode_features(self, x: Tensor) -> Tensor:
        """Feature encoding + feature-type embed
        x: [B, 4, T] -> feat: [B, 4, T, D]
        """
        B = x.shape[0]
        x_flat = x.reshape(B * 4, -1)
        feat_flat = self.feature_encoder(x_flat)
        feat = feat_flat.reshape(B, 4, -1, feat_flat.shape[-1])
        D = feat.shape[-1]
        feat = feat + self.feature_type_embed.view(1, 4, 1, D)
        return feat

    def _mix_xy(self, feat: Tensor, xy: Tensor) -> Tensor:
        """XY 条件融合 + feature bias
        feat: [B, 4, T, D], xy: [B, T, 2] -> [B, 4, T, D]
        """
        B, _, _, D = feat.shape
        feat_for_xy = feat + self.xy_feature_bias.view(1, 4, 1, D)
        xy_rep = xy.unsqueeze(1).expand(B, 4, *xy.shape[1:])
        xy_flat = xy_rep.reshape(B * 4, -1, xy.shape[-1])
        feat_flat = feat_for_xy.reshape(B * 4, -1, D)
        feat_mixed = self.xy_mixer(feat_flat, xy_flat)
        return feat_mixed.reshape(B, 4, -1, D)

    def _build_pair_tokens(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """向量化 pair 计算
        feat: [B, 4, T, D] -> (prod, diff, absdiff), 每个 [B, 5, T, D]
        """
        return self.pairwise_cross(feat)

    def forward(self, x: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        # x: [B, 4, T], xy: [B, T, 2]
        feat = self._encode_features(x)          # [B, 4, T, D]
        feat = self._mix_xy(feat, xy)            # [B, 4, T, D]
        prod, diff, absdiff = self._build_pair_tokens(feat)  # each [B, 5, T, D]

        # [B, K, T, D] -> [B, T, K, D]
        tokens_19 = torch.cat([feat, prod, diff, absdiff], dim=1).transpose(1, 2).contiguous()

        tokens_19 = self.token_block(tokens_19)
        z_price = self.refine(self.pool(tokens_19))
        return z_price, tokens_19


__all__ = [
    "ParallelWNOEncoder",
    "PathEncoder",
    "PathXYMixer",
    "VectorizedPairCross",
    "WNOBlock",
    "WNOBranch",
]