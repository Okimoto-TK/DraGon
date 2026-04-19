from __future__ import annotations

import torch
import torch.nn as nn

from .task_query_tower import TaskQueryTower


class MultiTaskHeads(nn.Module):
    """Task-specific towers and heads for ret, rv, and q predictions."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_latents: int = 8,
        tower_num_heads: int = 4,
        tower_ffn_ratio: float = 2.0,
        tower_dropout: float = 0.0,
        _tower_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
            )
        if num_latents <= 0:
            raise ValueError(
                f"num_latents must be > 0, got {num_latents}. Valid range: positive integers."
            )
        if tower_num_heads <= 0:
            raise ValueError(
                f"tower_num_heads must be > 0, got {tower_num_heads}. Valid range: positive integers."
            )
        if hidden_dim % tower_num_heads != 0:
            raise ValueError(
                f"hidden_dim % tower_num_heads must be 0, got hidden_dim={hidden_dim}, "
                f"tower_num_heads={tower_num_heads}. Valid range: hidden_dim divisible by tower_num_heads."
            )
        if tower_ffn_ratio <= 0:
            raise ValueError(
                f"tower_ffn_ratio must be > 0, got {tower_ffn_ratio}. Valid range: (0, +inf)."
            )
        if tower_dropout < 0 or tower_dropout >= 1:
            raise ValueError(
                f"tower_dropout must satisfy 0 <= tower_dropout < 1, got {tower_dropout}. "
                "Valid range: [0, 1)."
            )
        if _tower_norm_eps <= 0:
            raise ValueError(
                f"_tower_norm_eps must be > 0, got {_tower_norm_eps}. Valid range: (0, +inf)."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_latents = int(num_latents)
        self.tower_num_heads = int(tower_num_heads)
        self.tower_ffn_ratio = float(tower_ffn_ratio)
        self.tower_dropout = float(tower_dropout)
        self._tower_norm_eps = float(_tower_norm_eps)

        tower_kwargs = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.tower_num_heads,
            "ffn_ratio": self.tower_ffn_ratio,
            "dropout": self.tower_dropout,
            "_norm_eps": self._tower_norm_eps,
        }

        self.ret_tower = TaskQueryTower(**tower_kwargs)
        self.rv_tower = TaskQueryTower(**tower_kwargs)
        self.q_tower = TaskQueryTower(**tower_kwargs)

        self.ret_value_head = self._make_head()
        self.ret_uncertainty_head = self._make_head()
        self.rv_value_head = self._make_head()
        self.rv_uncertainty_head = self._make_head()
        self.q_value_head = self._make_head()
        self.q_uncertainty_head = self._make_head()

    def _make_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=self._tower_norm_eps),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.tower_dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        fused_latents: torch.Tensor,
        fused_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if fused_latents.ndim != 3:
            raise ValueError(
                "fused_latents must have ndim == 3, "
                f"got ndim={fused_latents.ndim}, shape={tuple(fused_latents.shape)}. "
                "Valid shape: [B, hidden_dim, num_latents]."
            )
        if fused_global.ndim != 2:
            raise ValueError(
                "fused_global must have ndim == 2, "
                f"got ndim={fused_global.ndim}, shape={tuple(fused_global.shape)}. "
                "Valid shape: [B, hidden_dim]."
            )

        batch_latents, latent_hidden_dim, latent_count = fused_latents.shape
        batch_global, global_hidden_dim = fused_global.shape

        if latent_hidden_dim != self.hidden_dim:
            raise ValueError(
                "fused_latents hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {latent_hidden_dim}. Valid value: hidden_dim."
            )
        if latent_count != self.num_latents:
            raise ValueError(
                "fused_latents num_latents mismatch: "
                f"expected {self.num_latents}, got {latent_count}. Valid value: num_latents."
            )
        if global_hidden_dim != self.hidden_dim:
            raise ValueError(
                "fused_global hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {global_hidden_dim}. Valid value: hidden_dim."
            )
        if batch_latents != batch_global:
            raise ValueError(
                "Batch size mismatch between fused_latents and fused_global: "
                f"got B_latents={batch_latents}, B_global={batch_global}. "
                "Valid range: both batch sizes must be equal."
            )

        h_ret = self.ret_tower(fused_latents, fused_global)
        h_rv = self.rv_tower(fused_latents, fused_global)
        h_q = self.q_tower(fused_latents, fused_global)

        pred_mu_ret = self.ret_value_head(h_ret)
        pred_scale_ret_raw = self.ret_uncertainty_head(h_ret)

        pred_mean_rv_raw = self.rv_value_head(h_rv)
        pred_shape_rv_raw = self.rv_uncertainty_head(h_rv)

        pred_mu_q = self.q_value_head(h_q)
        pred_scale_q_raw = self.q_uncertainty_head(h_q)

        return {
            "pred_mu_ret": pred_mu_ret,
            "pred_scale_ret_raw": pred_scale_ret_raw,
            "pred_mean_rv_raw": pred_mean_rv_raw,
            "pred_shape_rv_raw": pred_shape_rv_raw,
            "pred_mu_q": pred_mu_q,
            "pred_scale_q_raw": pred_scale_q_raw,
        }
