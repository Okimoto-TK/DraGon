"""Prediction head over concatenated mezzo-level time and wavelet tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.arch.heads.single_task_head import VALID_SINGLE_TASKS
from src.models.arch.layers import LayerNorm1dCF

from .task_query_tower import TaskQueryTower


class DualDomainConcatHead(nn.Module):
    """Fuse mezzo time/wavelet tokens by concatenation and read out one task."""

    def __init__(
        self,
        task: str,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if task not in VALID_SINGLE_TASKS:
            raise ValueError(f"task must be one of {VALID_SINGLE_TASKS}, got {task!r}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")

        self.task = task
        self.hidden_dim = int(hidden_dim)
        self.concat_norm = LayerNorm1dCF(2 * self.hidden_dim, eps=float(_norm_eps))
        self.concat_proj = nn.Conv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.global_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim, eps=float(_norm_eps)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.readout = TaskQueryTower(
            hidden_dim=self.hidden_dim,
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            _norm_eps=float(_norm_eps),
        )
        self.value_head = self._make_head(float(dropout), float(_norm_eps))
        self.aux_head = self._make_head(float(dropout), float(_norm_eps))

    def _make_head(self, dropout: float, norm_eps: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=norm_eps),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        mezzo_time_tokens: torch.Tensor,
        mezzo_wavelet_tokens: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        for name, value in (
            ("mezzo_time_tokens", mezzo_time_tokens),
            ("mezzo_wavelet_tokens", mezzo_wavelet_tokens),
        ):
            if value.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [B, hidden_dim, N], got shape={tuple(value.shape)}."
                )
            if value.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{name} hidden_dim mismatch: expected {self.hidden_dim}, got {value.shape[1]}."
                )
        if mezzo_time_tokens.shape != mezzo_wavelet_tokens.shape:
            raise ValueError(
                "mezzo_time_tokens and mezzo_wavelet_tokens must have identical shapes, "
                f"got {tuple(mezzo_time_tokens.shape)} and {tuple(mezzo_wavelet_tokens.shape)}."
            )

        head_tokens = torch.cat([mezzo_time_tokens, mezzo_wavelet_tokens], dim=1)
        head_tokens = self.concat_proj(self.concat_norm(head_tokens))
        head_context = self.global_fuse(
            torch.cat(
                [
                    mezzo_time_tokens.mean(dim=-1),
                    mezzo_wavelet_tokens.mean(dim=-1),
                ],
                dim=-1,
            )
        )

        if return_debug:
            task_repr, readout_debug = self.readout(
                head_tokens,
                head_context,
                return_debug=True,
            )
        else:
            task_repr = self.readout(head_tokens, head_context)

        pred_primary = self.value_head(task_repr)
        pred_aux_raw = self.aux_head(task_repr)
        out: dict[str, torch.Tensor] = {
            "pred_primary": pred_primary,
            "pred_aux_raw": pred_aux_raw,
            "head_context": head_context,
            "task_repr": task_repr,
            "head_tokens": head_tokens,
        }
        if self.task == "ret":
            out["pred_mu_ret"] = pred_primary
            out["pred_scale_ret_raw"] = pred_aux_raw
        elif self.task == "rv":
            out["pred_mean_rv_raw"] = pred_primary
            out["pred_shape_rv_raw"] = pred_aux_raw
        else:
            out["pred_mu_q"] = pred_primary
            out["pred_scale_q_raw"] = pred_aux_raw
        if return_debug:
            out["_debug"] = readout_debug
        return out
