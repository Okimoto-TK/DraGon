"""LayerNorm helpers for 1D sequence layouts."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm1dCF(nn.Module):
    """Apply LayerNorm over channel dim for channel-first sequences."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, C, T], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.num_channels}, got {x.shape[1]}."
            )
        y = torch.nn.functional.layer_norm(
            x.transpose(1, 2),
            (self.num_channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        return y.transpose(1, 2)


class LayerNorm1dLast(nn.Module):
    """Apply standard last-dim LayerNorm for [B, T, C] tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, T, C], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.num_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.num_channels}, got {x.shape[-1]}."
            )
        return torch.nn.functional.layer_norm(
            x,
            (self.num_channels,),
            self.weight,
            self.bias,
            self.eps,
        )


class AdaLayerNorm1DLast(nn.Module):
    """Last-dim LayerNorm followed by conditional affine modulation."""

    def __init__(self, num_channels: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {cond_dim}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.num_channels = int(num_channels)
        self.cond_dim = int(cond_dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))
        self.to_gamma_beta = nn.Linear(self.cond_dim, 2 * self.num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, T, C], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if cond.ndim != 3:
            raise ValueError(
                "cond must have shape [B, T, cond_dim], "
                f"got ndim={cond.ndim}, shape={tuple(cond.shape)}."
            )
        if x.shape[0] != cond.shape[0] or x.shape[1] != cond.shape[1]:
            raise ValueError(
                "x/cond batch or patch mismatch: "
                f"x shape={tuple(x.shape)}, cond shape={tuple(cond.shape)}."
            )
        if x.shape[-1] != self.num_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.num_channels}, got {x.shape[-1]}."
            )
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond cond_dim mismatch: expected {self.cond_dim}, got {cond.shape[-1]}."
            )

        x = torch.nn.functional.layer_norm(
            x,
            (self.num_channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return x * (1.0 + gamma) + beta
