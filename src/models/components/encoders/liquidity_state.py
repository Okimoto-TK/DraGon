"""Liquidity and hard-state encoders."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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


class LiquidityBranch(nn.Module):
    def __init__(self, dim: int, *, out_tokens: int) -> None:
        super().__init__()
        self.rbf = RBFEncoding()
        self.fourier = FourierEncoding()
        self.film = nn.Sequential(
            nn.Linear(5, dim),
            nn.SiLU(),
            nn.Linear(dim, 8),
        )
        self.base_proj = nn.Sequential(
            nn.Linear(7, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.base_norm = nn.LayerNorm(dim)
        self.token_proj = nn.Linear(dim, out_tokens * dim)
        self.token_norm = nn.LayerNorm(dim)
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

        base = torch.cat((r, direction, xy), dim=-1)
        base_seq = self.base_norm(self.base_proj(base))
        token_raw = self.token_proj(base_seq).reshape(base_seq.shape[0], base_seq.shape[1], self.out_tokens, base_seq.shape[-1])
        z_liquid = self.token_norm(token_raw)
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
