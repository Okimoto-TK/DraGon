"""Liquidity and hard-state encoders."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.trunks.common import to_channel_major, to_time_major


class RBFEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        centers = torch.log(torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float32))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", torch.full_like(centers, 0.5))

    def forward(self, x: Tensor) -> Tensor:
        diff = x - self.centers.view(1, 1, -1)
        return torch.exp(-0.5 * diff.square() / self.widths.view(1, 1, -1).square())


class FourierEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.tensor([1.0, 2.0], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        angles = x * self.freqs.view(1, 1, -1)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class LiquidityXYMixer(nn.Module):
    """与 PathEncoder 风格一致的共享 XY mixer"""
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim + 2, dim, kernel_size=kernel_size, padding=padding)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, xy: Tensor) -> Tensor:
        residual = h
        y = torch.cat((h, xy), dim=-1)  # [B, T, D+2]
        y = to_channel_major(y)
        y = self.conv(y)
        y = F.silu(y)
        y = self.proj(y)
        y = to_time_major(y)
        return self.norm(residual + y)


class LiquidityBranch(nn.Module):
    def __init__(self, dim: int, *, out_tokens: int, local_kernel: int) -> None:
        super().__init__()
        self.rbf = RBFEncoding()
        self.fourier = FourierEncoding()
        self.film = nn.Sequential(
            nn.Linear(5, dim),
            nn.SiLU(),
            nn.Linear(dim, 8),
        )
        # 不含 xy 的 liquidity feature 投影
        self.pre_xy_proj = nn.Sequential(
            nn.Linear(1 + 4, dim),  # r(1) + direction(4) = 5
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Path 风格的 XY mixer (内部已含 norm)
        self.xy_mixer = LiquidityXYMixer(dim, kernel_size=local_kernel)

        # Gated Slot Projection 替代原来的 token_proj
        self.token_content = nn.Linear(dim, out_tokens * dim)
        self.token_gate = nn.Linear(dim, out_tokens * dim)
        self.slot_embed = nn.Parameter(torch.empty(out_tokens, dim))
        self.token_norm = nn.LayerNorm(dim)

        nn.init.normal_(self.slot_embed, std=0.02)
        self.out_tokens = out_tokens
        self.eps = 1e-6

    def forward(self, f4: Tensor, f5: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        r = f4.clamp(0.0, 10.0).unsqueeze(-1)
        log_r = torch.log(r + self.eps)
        e4 = self.rbf(log_r)
        gamma_beta = self.film(e4)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        u5 = torch.sign(f5).unsqueeze(-1) * torch.log1p(f5.abs().unsqueeze(-1))
        q5 = self.fourier(u5)
        direction = F.layer_norm((1.0 + gamma) * q5 + beta, (q5.shape[-1],))

        # 1) 先得到不含 xy 的 liquidity feature
        liq_feat = self.pre_xy_proj(torch.cat([r, direction], dim=-1))

        # 2) 用 XY mixer 融合 xy (内部已有 norm)
        base_seq = self.xy_mixer(liq_feat, xy)

        # 3) Gated Slot Projection
        B, T, D = base_seq.shape
        Kv = self.out_tokens
        content = self.token_content(base_seq).view(B, T, Kv, D)
        gate = torch.sigmoid(self.token_gate(base_seq).view(B, T, Kv, D))
        tokens = content * gate + self.slot_embed.view(1, 1, Kv, D)
        z_liquid = self.token_norm(tokens)

        return z_liquid, base_seq


class StateQueryEncoder(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.embedding = nn.Embedding(16, dim)
        self.time_proj = nn.Linear(2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, f6: Tensor, xy: Tensor) -> Tensor:
        idx = f6.round().clamp(0.0, 15.0).long()
        state = self.embedding(idx)
        y = self.norm(state + self.time_proj(xy))
        residual = y
        y = y.transpose(1, 2).contiguous()
        y = self.dw(y)
        y = self.pw(y)
        y = F.silu(y)
        y = y.transpose(1, 2).contiguous()
        return self.out_norm(residual + y).unsqueeze(2)


__all__ = ["FourierEncoding", "LiquidityBranch", "RBFEncoding", "StateQueryEncoder"]
