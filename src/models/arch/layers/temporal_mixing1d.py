"""Temporal mixing layer for lightweight conditioning encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixing1D(nn.Module):
    """Mix along temporal axis for tensors shaped [B, 64, D]."""

    def __init__(
        self,
        seq_len: int = 64,
        _temporal_mlp_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(
                f"seq_len must be > 0, got {seq_len}. Expected positive integer."
            )
        if _temporal_mlp_mult <= 0:
            raise ValueError(
                "_temporal_mlp_mult must be > 0, "
                f"got {_temporal_mlp_mult}. Expected positive integer."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Expected range [0, 1)."
            )

        self.seq_len = int(seq_len)
        self._temporal_mlp_mult = int(_temporal_mlp_mult)
        self.dropout = float(dropout)
        temporal_hidden = self._temporal_mlp_mult * self.seq_len

        self.fc1 = nn.Linear(self.seq_len, temporal_hidden)
        self.fc2 = nn.Linear(temporal_hidden, self.seq_len)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, {self.seq_len}, D], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != self.seq_len:
            raise ValueError(
                "temporal dimension mismatch: "
                f"expected {self.seq_len}, got {x.shape[1]}. "
                f"Expected x.shape[1] == {self.seq_len}."
            )

        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)
        return x
