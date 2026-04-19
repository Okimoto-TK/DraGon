"""Lightweight TSMixer-style conditioning encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.arch.layers import FeatureMixing1D, TemporalMixing1D


class ConditioningMixerBlock(nn.Module):
    """One lightweight mixer block with temporal then feature mixing."""

    def __init__(
        self,
        d_cond: int,
        dropout: float = 0.0,
        _temporal_mlp_mult: int = 2,
        _feature_mlp_mult: int = 2,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_cond <= 0:
            raise ValueError(
                f"d_cond must be > 0, got {d_cond}. Expected positive integer."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Expected range [0, 1)."
            )
        if _temporal_mlp_mult <= 0:
            raise ValueError(
                "_temporal_mlp_mult must be > 0, "
                f"got {_temporal_mlp_mult}. Expected positive integer."
            )
        if _feature_mlp_mult <= 0:
            raise ValueError(
                "_feature_mlp_mult must be > 0, "
                f"got {_feature_mlp_mult}. Expected positive integer."
            )
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Expected positive value."
            )

        self.d_cond = int(d_cond)
        self.dropout = float(dropout)
        self._temporal_mlp_mult = int(_temporal_mlp_mult)
        self._feature_mlp_mult = int(_feature_mlp_mult)
        self._norm_eps = float(_norm_eps)

        self.temporal_norm = nn.LayerNorm(self.d_cond, eps=self._norm_eps)
        self.temporal_mixing = TemporalMixing1D(
            seq_len=64,
            _temporal_mlp_mult=self._temporal_mlp_mult,
            dropout=self.dropout,
        )
        self.feature_norm = nn.LayerNorm(self.d_cond, eps=self._norm_eps)
        self.feature_mixing = FeatureMixing1D(
            d_cond=self.d_cond,
            _feature_mlp_mult=self._feature_mlp_mult,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, 64, {self.d_cond}], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != 64:
            raise ValueError(
                f"x temporal length mismatch: expected 64, got {x.shape[1]}. Expected x.shape[1] == 64."
            )
        if x.shape[2] != self.d_cond:
            raise ValueError(
                "x d_cond mismatch: "
                f"expected {self.d_cond}, got {x.shape[2]}. "
                f"Expected x.shape[2] == {self.d_cond}."
            )

        u = self.temporal_norm(x)
        x = x + self.temporal_mixing(u)

        v = self.feature_norm(x)
        x = x + self.feature_mixing(v)
        return x


class ConditioningEncoder(nn.Module):
    """Encode conditioning sequence [B, F_cond, 64] into seq/global condition states."""

    def __init__(
        self,
        d_cond: int = 32,
        input_features: int = 13,
        num_blocks: int = 1,
        dropout: float = 0.0,
        _temporal_mlp_mult: int = 2,
        _feature_mlp_mult: int = 2,
        _norm_eps: float = 1e-6,
        _pool_type: str = "mean",
    ) -> None:
        super().__init__()
        if d_cond <= 0:
            raise ValueError(
                f"d_cond must be > 0, got {d_cond}. Expected positive integer."
            )
        if input_features <= 0:
            raise ValueError(
                "input_features must be > 0, "
                f"got {input_features}. Expected positive integer."
            )
        if num_blocks <= 0:
            raise ValueError(
                f"num_blocks must be > 0, got {num_blocks}. Expected positive integer."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Expected range [0, 1)."
            )
        if _temporal_mlp_mult <= 0:
            raise ValueError(
                "_temporal_mlp_mult must be > 0, "
                f"got {_temporal_mlp_mult}. Expected positive integer."
            )
        if _feature_mlp_mult <= 0:
            raise ValueError(
                "_feature_mlp_mult must be > 0, "
                f"got {_feature_mlp_mult}. Expected positive integer."
            )
        if _pool_type != "mean":
            raise ValueError(
                f"_pool_type must be 'mean', got {_pool_type!r}. Expected one of ['mean']."
            )
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Expected positive value."
            )

        self.d_cond = int(d_cond)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self._temporal_mlp_mult = int(_temporal_mlp_mult)
        self._feature_mlp_mult = int(_feature_mlp_mult)
        self._norm_eps = float(_norm_eps)
        self._pool_type = _pool_type
        self._input_features = int(input_features)
        self._seq_len = 64

        self.input_proj = nn.Linear(self._input_features, self.d_cond)
        self.blocks = nn.ModuleList(
            [
                ConditioningMixerBlock(
                    d_cond=self.d_cond,
                    dropout=self.dropout,
                    _temporal_mlp_mult=self._temporal_mlp_mult,
                    _feature_mlp_mult=self._feature_mlp_mult,
                    _norm_eps=self._norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, x_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_cond.ndim != 3:
            raise ValueError(
                f"x_cond must have shape [B, {self._input_features}, 64], "
                f"got ndim={x_cond.ndim}, shape={tuple(x_cond.shape)}. Expected ndim == 3."
            )
        if x_cond.shape[1] != self._input_features:
            raise ValueError(
                f"x_cond feature dimension mismatch: got {x_cond.shape[1]}, expected {self._input_features}."
            )
        if x_cond.shape[2] != self._seq_len:
            raise ValueError(
                f"x_cond temporal length mismatch: got {x_cond.shape[2]}, expected {self._seq_len}."
            )
        ref_param = next(self.parameters())
        if x_cond.device != ref_param.device:
            raise ValueError(
                "x_cond device mismatch: "
                f"input device={x_cond.device}, module device={ref_param.device}."
            )
        if x_cond.dtype != ref_param.dtype:
            raise ValueError(
                "x_cond dtype mismatch: "
                f"input dtype={x_cond.dtype}, module dtype={ref_param.dtype}. "
                "Move the module with `.to(...)` before calling forward."
            )

        x = x_cond.transpose(1, 2)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        cond_seq = x.transpose(1, 2)
        cond_global = x.mean(dim=1)
        return cond_seq, cond_global
