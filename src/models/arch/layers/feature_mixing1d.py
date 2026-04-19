"""Feature mixing layer for lightweight conditioning encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMixing1D(nn.Module):
    """Mix along feature axis for tensors shaped [B, T, D]."""

    def __init__(
        self,
        d_cond: int,
        _feature_mlp_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_cond <= 0:
            raise ValueError(
                f"d_cond must be > 0, got {d_cond}. Expected positive integer."
            )
        if _feature_mlp_mult <= 0:
            raise ValueError(
                "_feature_mlp_mult must be > 0, "
                f"got {_feature_mlp_mult}. Expected positive integer."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Expected range [0, 1)."
            )

        self.d_cond = int(d_cond)
        self._feature_mlp_mult = int(_feature_mlp_mult)
        self.dropout = float(dropout)
        feature_hidden = self._feature_mlp_mult * self.d_cond

        self.fc1 = nn.Linear(self.d_cond, feature_hidden)
        self.fc2 = nn.Linear(feature_hidden, self.d_cond)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, T, {self.d_cond}], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.d_cond:
            raise ValueError(
                "feature dimension mismatch: "
                f"expected {self.d_cond}, got {x.shape[-1]}. "
                f"Expected x.shape[-1] == {self.d_cond}."
            )

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
