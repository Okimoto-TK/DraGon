"""Feature-wise linear modulation for 1D hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM1D(nn.Module):
    """Apply FiLM modulation from condition sequences."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if cond_dim <= 0:
            raise ValueError(
                f"cond_dim must be > 0, got {cond_dim}. Expected positive integer."
            )

        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.to_gamma_beta = nn.Conv1d(self.cond_dim, 2 * self.hidden_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [Bf, hidden_dim, N], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if cond.ndim != 3:
            raise ValueError(
                "cond must have shape [Bf, cond_dim, N], "
                f"got ndim={cond.ndim}, shape={tuple(cond.shape)}."
            )
        if x.shape[0] != cond.shape[0] or x.shape[2] != cond.shape[2]:
            raise ValueError(
                "x/cond batch or patch mismatch: "
                f"x shape={tuple(x.shape)}, cond shape={tuple(cond.shape)}."
            )
        if x.shape[1] != self.hidden_dim:
            raise ValueError(
                f"x hidden_dim mismatch: expected {self.hidden_dim}, got {x.shape[1]}."
            )
        if cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"cond cond_dim mismatch: expected {self.cond_dim}, got {cond.shape[1]}."
            )

        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        out = x * (1.0 + gamma) + beta
        return out

