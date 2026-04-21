"""Channel-wise feed-forward network for 1D hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelFFN1D(nn.Module):
    """Lightweight FFN along channel dimension with pointwise convolutions."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if ffn_ratio <= 0:
            raise ValueError(
                f"ffn_ratio must be > 0, got {ffn_ratio}. Expected positive value."
            )
        if dropout < 0:
            raise ValueError(
                f"dropout must be >= 0, got {dropout}. Expected non-negative value."
            )

        self.hidden_dim = int(hidden_dim)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)

        self._ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        if self._ffn_dim <= 0:
            raise ValueError(
                f"_ffn_dim must be > 0, got {self._ffn_dim}. "
                f"Check hidden_dim={hidden_dim}, ffn_ratio={ffn_ratio}."
            )

        self.net = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self._ffn_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self._ffn_dim, self.hidden_dim, kernel_size=1),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [Bf, hidden_dim, N], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != self.hidden_dim:
            raise ValueError(
                f"x hidden_dim mismatch: expected {self.hidden_dim}, got {x.shape[1]}."
            )

        return self.net(x)


class ChannelFFN1DLast(nn.Module):
    """Lightweight FFN over last dim for [B, T, C] hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if ffn_ratio <= 0:
            raise ValueError(
                f"ffn_ratio must be > 0, got {ffn_ratio}. Expected positive value."
            )
        if dropout < 0:
            raise ValueError(
                f"dropout must be >= 0, got {dropout}. Expected non-negative value."
            )

        self.hidden_dim = int(hidden_dim)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)

        self._ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        if self._ffn_dim <= 0:
            raise ValueError(
                f"_ffn_dim must be > 0, got {self._ffn_dim}. "
                f"Check hidden_dim={hidden_dim}, ffn_ratio={ffn_ratio}."
            )

        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, self._ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self._ffn_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, T, hidden_dim], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x hidden_dim mismatch: expected {self.hidden_dim}, got {x.shape[-1]}."
            )

        return self.net(x)
