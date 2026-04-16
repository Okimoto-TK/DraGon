"""Path branch encoders."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.fusion.pairwise import PairwiseCross
from src.models.components.fusion.token_attention import TokenSelfAttentionBlock
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


class FeatureWNOBlock(nn.Module):
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


class SingleFeaturePathEncoder(nn.Module):
    def __init__(self, dim: int, *, levels: int, num_blocks: int, local_kernel: int) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, dim, kernel_size=1)
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([FeatureWNOBlock(dim, levels=levels) for _ in range(num_blocks)])
        self.local_refine = ConvRefineBlock(dim, kernel_size=local_kernel)

    def forward(self, x: Tensor) -> Tensor:
        y = self.lift(x.unsqueeze(1))
        y = to_time_major(y)
        y = self.in_norm(y)
        for block in self.blocks:
            y = block(y)
        y = self.local_refine(y)
        return y


class XYConditionedMixer(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim + 2, dim, kernel_size=kernel_size, padding=padding)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, xy: Tensor) -> Tensor:
        residual = h
        y = torch.cat((h, xy), dim=-1)
        y = to_channel_major(y)
        y = self.conv(y)
        y = F.silu(y)
        y = self.proj(y)
        y = to_time_major(y)
        return self.norm(residual + y)


class PathEncoder(nn.Module):
    _pairs: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3))

    def __init__(self, dim: int, *, levels: int, num_blocks: int, local_kernel: int, out_tokens: int, num_heads: int) -> None:
        super().__init__()
        self.feature_encoders = nn.ModuleList(
            [SingleFeaturePathEncoder(dim, levels=levels, num_blocks=num_blocks, local_kernel=local_kernel) for _ in range(4)]
        )
        self.xy_mixers = nn.ModuleList([XYConditionedMixer(dim, kernel_size=local_kernel) for _ in range(4)])
        self.crosses = nn.ModuleList([PairwiseCross(dim) for _ in self._pairs])
        self.token_block = TokenSelfAttentionBlock(dim, num_heads=num_heads)
        self.pool = QueryTokenPooling(dim, out_tokens=out_tokens, num_heads=num_heads)
        self.refine = SmallTokenRefine(dim)

    def forward(self, x: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        features: list[Tensor] = []
        for idx, (encoder, mixer) in enumerate(zip(self.feature_encoders, self.xy_mixers)):
            h = encoder(x[:, idx, :])
            features.append(mixer(h, xy))

        token_list = [*features]
        for (i, j), cross in zip(self._pairs, self.crosses):
            prod, diff, absdiff = cross(features[i], features[j])
            token_list.extend((prod, diff, absdiff))
        tokens_19 = torch.stack(token_list, dim=2)
        tokens_19 = self.token_block(tokens_19)
        z_price = self.refine(self.pool(tokens_19))
        return z_price, tokens_19


__all__ = [
    "FeatureWNOBlock",
    "PathEncoder",
    "SingleFeaturePathEncoder",
    "WNOBranch",
    "XYConditionedMixer",
]

