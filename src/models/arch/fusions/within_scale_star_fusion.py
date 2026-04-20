"""Within-scale STAR-style feature fusion for encoded single-scale features."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.arch.layers import StochasticPooling1D


class STARAggregateRedistributeBlock(nn.Module):
    """STAR-style aggregate-redistribute block over feature axis."""

    def __init__(
        self,
        hidden_dim: int,
        core_dim: int,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _pool_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
            )
        if core_dim <= 0:
            raise ValueError(
                f"core_dim must be > 0, got {core_dim}. Valid range: positive integers."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Valid range: [0, 1)."
            )
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Valid range: (0, +inf)."
            )
        if _pool_temperature <= 0:
            raise ValueError(
                "_pool_temperature must be > 0, "
                f"got {_pool_temperature}. Valid range: (0, +inf)."
            )

        self.hidden_dim = int(hidden_dim)
        self.core_dim = int(core_dim)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)
        self._pool_temperature = float(_pool_temperature)

        self.norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.core_proj = nn.Linear(self.hidden_dim, self.core_dim)
        self.score_proj = nn.Linear(self.hidden_dim, 1)
        self.pool = StochasticPooling1D(_pool_temperature=self._pool_temperature)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.core_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [Bq, F, hidden_dim], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[2] != self.hidden_dim:
            raise ValueError(
                "x hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {x.shape[2]}. Valid value: hidden_dim."
            )

        u = self.norm(x)
        core_values = self.core_proj(u)
        core_scores = self.score_proj(u)
        core = self.pool(core_values, core_scores)
        core_rep = core.expand(-1, x.shape[1], -1)
        fuse_in = torch.cat([u, core_rep], dim=-1)
        delta = self.fuse_mlp(fuse_in)
        out = x + delta
        return out


class WithinScaleSTARFusion(nn.Module):
    """Within-scale STAR fusion for z_scale with shape [B, F, D, N]."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_features: int = 9,
        core_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _pool_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
            )
        if num_features <= 0:
            raise ValueError(
                f"num_features must be > 0, got {num_features}. Valid range: positive integers."
            )
        if core_dim <= 0:
            raise ValueError(
                f"core_dim must be > 0, got {core_dim}. Valid range: positive integers."
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {num_layers}. Valid range: positive integers."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Valid range: [0, 1)."
            )
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Valid range: (0, +inf)."
            )
        if _pool_temperature <= 0:
            raise ValueError(
                "_pool_temperature must be > 0, "
                f"got {_pool_temperature}. Valid range: (0, +inf)."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_features = int(num_features)
        self.core_dim = int(core_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)
        self._pool_temperature = float(_pool_temperature)

        self.blocks = nn.ModuleList(
            [
                STARAggregateRedistributeBlock(
                    hidden_dim=self.hidden_dim,
                    core_dim=self.core_dim,
                    dropout=self.dropout,
                    _norm_eps=self._norm_eps,
                    _pool_temperature=self._pool_temperature,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, z_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_scale.ndim != 4:
            raise ValueError(
                f"z_scale must have ndim == 4, got ndim={z_scale.ndim}, shape={tuple(z_scale.shape)}."
            )
        bsz, num_features, hidden_dim, num_patches = z_scale.shape
        if num_features != self.num_features:
            raise ValueError(
                "z_scale feature dimension mismatch: "
                f"expected {self.num_features}, got {num_features}. Valid value: num_features."
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "z_scale hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {hidden_dim}. Valid value: hidden_dim."
            )
        ref_param = next(self.parameters())
        if z_scale.device != ref_param.device:
            raise ValueError(
                "z_scale device mismatch: "
                f"input device={z_scale.device}, module device={ref_param.device}."
            )
        allowed_amp_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if z_scale.dtype != ref_param.dtype and not (
            ref_param.dtype == torch.float32 and z_scale.dtype in allowed_amp_dtypes
        ):
            raise ValueError(
                "z_scale dtype mismatch: "
                f"input dtype={z_scale.dtype}, module dtype={ref_param.dtype}. "
                "Expected matching dtypes, or an AMP/autocast input in "
                "{torch.float16, torch.bfloat16, torch.float32} for float32 modules."
            )

        z = z_scale.permute(0, 3, 1, 2)
        z = z.reshape(bsz * num_patches, num_features, hidden_dim)
        for block in self.blocks:
            z = block(z)
        z = z.view(bsz, num_patches, num_features, hidden_dim)
        z_fused = z.permute(0, 2, 3, 1)
        scale_seq = z_fused.mean(dim=1)
        return z_fused, scale_seq
