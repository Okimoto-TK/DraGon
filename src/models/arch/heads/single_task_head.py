from __future__ import annotations

import torch
import torch.nn as nn

from .task_query_tower import TaskQueryTower

VALID_SINGLE_TASKS = ("ret", "rv", "q")


class SingleTaskHead(nn.Module):
    """Read out one task from micro tokens plus macro/mezzo summaries."""

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
            raise ValueError(
                f"task must be one of {VALID_SINGLE_TASKS}, got {task!r}."
            )
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

        self.task = task
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.context_fuse = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim, eps=self._norm_eps),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.readout = TaskQueryTower(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
            _norm_eps=self._norm_eps,
        )
        self.value_head = self._make_head()
        self.aux_head = self._make_head()

    def _make_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=self._norm_eps),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        micro_td: torch.Tensor,
        mezzo_ctx: torch.Tensor,
        macro_ctx: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        if micro_td.ndim != 3:
            raise ValueError(
                "micro_td must have ndim == 3, "
                f"got ndim={micro_td.ndim}, shape={tuple(micro_td.shape)}. "
                "Valid shape: [B, hidden_dim, N_micro]."
            )
        if mezzo_ctx.ndim != 2:
            raise ValueError(
                "mezzo_ctx must have ndim == 2, "
                f"got ndim={mezzo_ctx.ndim}, shape={tuple(mezzo_ctx.shape)}. "
                "Valid shape: [B, hidden_dim]."
            )
        if macro_ctx.ndim != 2:
            raise ValueError(
                "macro_ctx must have ndim == 2, "
                f"got ndim={macro_ctx.ndim}, shape={tuple(macro_ctx.shape)}. "
                "Valid shape: [B, hidden_dim]."
            )

        batch_size, hidden_dim, _ = micro_td.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"micro_td hidden_dim mismatch: expected {self.hidden_dim}, got {hidden_dim}. "
                "Valid value: hidden_dim."
            )
        for name, ctx in (("mezzo_ctx", mezzo_ctx), ("macro_ctx", macro_ctx)):
            if ctx.shape != (batch_size, self.hidden_dim):
                raise ValueError(
                    f"{name} shape mismatch: expected ({batch_size}, {self.hidden_dim}), got {tuple(ctx.shape)}."
                )

        micro_ctx = micro_td.mean(dim=-1)
        ctx_input = torch.cat([micro_ctx, mezzo_ctx, macro_ctx], dim=-1)
        head_context = micro_ctx + self.context_fuse(ctx_input)

        if return_debug:
            task_repr, readout_debug = self.readout(
                micro_td,
                head_context,
                return_debug=True,
            )
        else:
            task_repr = self.readout(micro_td, head_context)

        pred_primary = self.value_head(task_repr)
        pred_aux_raw = self.aux_head(task_repr)
        out: dict[str, torch.Tensor] = {
            "pred_primary": pred_primary,
            "pred_aux_raw": pred_aux_raw,
            "head_context": head_context,
            "task_repr": task_repr,
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
            out["_debug"] = {
                "head_head_context_l2_mean": _l2_mean(head_context),
                **readout_debug,
            }
        return out


def _l2_mean(value: torch.Tensor) -> float:
    flat = value.detach().float().reshape(value.shape[0], -1)
    return float(flat.norm(dim=1).mean().cpu())
