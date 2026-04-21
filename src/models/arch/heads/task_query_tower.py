from __future__ import annotations

import torch
import torch.nn as nn


class TaskQueryTower(nn.Module):
    """Task-specific query tower over fused latent tokens."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be > 0, got {num_heads}. Valid range: positive integers."
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim % num_heads must be 0, got hidden_dim={hidden_dim}, num_heads={num_heads}. "
                "Valid range: hidden_dim divisible by num_heads."
            )
        if ffn_ratio <= 0:
            raise ValueError(
                f"ffn_ratio must be > 0, got {ffn_ratio}. Valid range: (0, +inf)."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Valid range: [0, 1)."
            )
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Valid range: (0, +inf)."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.task_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.kv_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        _ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, _ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(_ffn_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        fused_latents: torch.Tensor,
        fused_global: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        batch_latents, latent_hidden_dim, _ = fused_latents.shape
        batch_global, global_hidden_dim = fused_global.shape

        if latent_hidden_dim != self.hidden_dim:
            raise ValueError(
                "fused_latents hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {latent_hidden_dim}. Valid value: hidden_dim."
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

        latents = fused_latents.transpose(1, 2)
        q0 = self.task_token.expand(batch_latents, -1, -1) + self.global_proj(
            fused_global
        ).unsqueeze(1)

        q = self.q_norm(q0)
        kv = self.kv_norm(latents)
        delta_attn, attn_weights = self.cross_attn(
            q,
            kv,
            kv,
            need_weights=return_debug,
            average_attn_weights=False,
        )
        q1 = q0 + delta_attn
        q2 = q1 + self.ffn(self.ffn_norm(q1))
        task_repr = q2.squeeze(1)
        if not return_debug:
            return task_repr

        debug = {
            "head_task_repr_l2_mean": _l2_mean(task_repr),
            "head_task_delta_attn_l2_mean": _l2_mean(delta_attn.squeeze(1)),
        }
        if attn_weights is not None:
            probs = attn_weights.squeeze(2).detach().float().clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            max_weight = probs.max(dim=-1).values.mean()
            debug["head_task_attn_entropy_mean"] = float(entropy.cpu())
            debug["head_task_attn_max_weight_mean"] = float(max_weight.cpu())
        return task_repr, debug


def _l2_mean(value: torch.Tensor) -> float:
    flat = value.detach().float().reshape(value.shape[0], -1)
    return float(flat.norm(dim=1).mean().cpu())
