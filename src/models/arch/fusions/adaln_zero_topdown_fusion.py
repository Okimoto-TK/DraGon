"""AdaLN-Zero top-down modulation over macro -> mezzo -> micro sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNZeroTopDownBlock(nn.Module):
    """Modulate a child sequence from a parent context vector with an identity init."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: float,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
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
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.parent_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.child_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.modulation = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        self.in_proj = nn.Linear(self.hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.out_proj = nn.Linear(ffn_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout)

        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    def forward(
        self,
        child_seq: torch.Tensor,
        parent_ctx: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if child_seq.ndim != 3:
            raise ValueError(
                f"child_seq must have shape [B, hidden_dim, N], got shape={tuple(child_seq.shape)}."
            )
        if parent_ctx.ndim != 2:
            raise ValueError(
                f"parent_ctx must have shape [B, hidden_dim], got shape={tuple(parent_ctx.shape)}."
            )
        if child_seq.shape[1] != self.hidden_dim:
            raise ValueError(
                f"child_seq hidden_dim mismatch: expected {self.hidden_dim}, got {child_seq.shape[1]}."
            )
        if parent_ctx.shape != (child_seq.shape[0], self.hidden_dim):
            raise ValueError(
                "parent_ctx shape mismatch: "
                f"expected ({child_seq.shape[0]}, {self.hidden_dim}), got {tuple(parent_ctx.shape)}."
            )

        child_tokens = child_seq.transpose(1, 2)
        shift, scale, gate = self.modulation(self.parent_norm(parent_ctx)).chunk(3, dim=-1)
        modulated = self.child_norm(child_tokens)
        modulated = modulated * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        delta = self.in_proj(modulated)
        delta = self.act(delta)
        delta = self.dropout1(delta)
        delta = self.out_proj(delta)
        delta = self.dropout2(delta)
        out_tokens = child_tokens + gate.unsqueeze(1) * delta
        out_seq = out_tokens.transpose(1, 2)

        if not return_debug:
            return out_seq

        debug = {
            "shift_l2_mean": _l2_mean(shift),
            "scale_l2_mean": _l2_mean(scale),
            "gate_mean": _mean(gate),
            "gate_std": _std(gate),
            "gate_abs_mean": _abs_mean(gate),
            "delta_l2_mean": _l2_mean(delta),
        }
        return out_seq, debug


class AdaLNZeroTopDownFusion(nn.Module):
    """Top-down context chain: macro summary modulates mezzo, then mezzo modulates micro."""

    def __init__(
        self,
        hidden_dim: int = 128,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Valid range: positive integers."
            )
        self.hidden_dim = int(hidden_dim)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.macro_to_mezzo = AdaLNZeroTopDownBlock(
            hidden_dim=self.hidden_dim,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
            _norm_eps=self._norm_eps,
        )
        self.mezzo_to_micro = AdaLNZeroTopDownBlock(
            hidden_dim=self.hidden_dim,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
            _norm_eps=self._norm_eps,
        )

    def _validate_scale_seq(
        self,
        *,
        name: str,
        value: torch.Tensor,
        expected_len: int,
    ) -> None:
        if value.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, hidden_dim, {expected_len}], got shape={tuple(value.shape)}."
            )
        if value.shape[1] != self.hidden_dim:
            raise ValueError(
                f"{name} hidden_dim mismatch: expected {self.hidden_dim}, got {value.shape[1]}."
            )
        if value.shape[2] != expected_len:
            raise ValueError(
                f"{name} length mismatch: expected {expected_len}, got {value.shape[2]}."
            )

    def forward(
        self,
        macro_seq: torch.Tensor,
        mezzo_seq: torch.Tensor,
        micro_seq: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ]:
        self._validate_scale_seq(name="macro_seq", value=macro_seq, expected_len=16)
        self._validate_scale_seq(name="mezzo_seq", value=mezzo_seq, expected_len=24)
        self._validate_scale_seq(name="micro_seq", value=micro_seq, expected_len=36)
        if not (macro_seq.shape[0] == mezzo_seq.shape[0] == micro_seq.shape[0]):
            raise ValueError(
                "Batch size mismatch among macro_seq/mezzo_seq/micro_seq: "
                f"got {macro_seq.shape[0]}, {mezzo_seq.shape[0]}, {micro_seq.shape[0]}."
            )

        macro_ctx = macro_seq.mean(dim=-1)
        if return_debug:
            mezzo_td, macro_debug = self.macro_to_mezzo(
                mezzo_seq,
                macro_ctx,
                return_debug=True,
            )
        else:
            mezzo_td = self.macro_to_mezzo(mezzo_seq, macro_ctx)

        mezzo_ctx = mezzo_td.mean(dim=-1)
        if return_debug:
            micro_td, mezzo_debug = self.mezzo_to_micro(
                micro_seq,
                mezzo_ctx,
                return_debug=True,
            )
        else:
            micro_td = self.mezzo_to_micro(micro_seq, mezzo_ctx)

        micro_ctx = micro_td.mean(dim=-1)
        if not return_debug:
            return micro_td, macro_ctx, mezzo_ctx, micro_ctx

        debug: dict[str, float] = {
            "cross_scale_macro_ctx_l2_mean": _l2_mean(macro_ctx),
            "cross_scale_mezzo_ctx_l2_mean": _l2_mean(mezzo_ctx),
            "cross_scale_micro_ctx_l2_mean": _l2_mean(micro_ctx),
            "cross_scale_macro_to_mezzo_shift_l2_mean": macro_debug["shift_l2_mean"],
            "cross_scale_macro_to_mezzo_scale_l2_mean": macro_debug["scale_l2_mean"],
            "cross_scale_macro_to_mezzo_gate_mean": macro_debug["gate_mean"],
            "cross_scale_macro_to_mezzo_gate_std": macro_debug["gate_std"],
            "cross_scale_macro_to_mezzo_gate_abs_mean": macro_debug["gate_abs_mean"],
            "cross_scale_macro_to_mezzo_delta_l2_mean": macro_debug["delta_l2_mean"],
            "cross_scale_mezzo_to_micro_shift_l2_mean": mezzo_debug["shift_l2_mean"],
            "cross_scale_mezzo_to_micro_scale_l2_mean": mezzo_debug["scale_l2_mean"],
            "cross_scale_mezzo_to_micro_gate_mean": mezzo_debug["gate_mean"],
            "cross_scale_mezzo_to_micro_gate_std": mezzo_debug["gate_std"],
            "cross_scale_mezzo_to_micro_gate_abs_mean": mezzo_debug["gate_abs_mean"],
            "cross_scale_mezzo_to_micro_delta_l2_mean": mezzo_debug["delta_l2_mean"],
        }
        return micro_td, macro_ctx, mezzo_ctx, micro_ctx, debug


def _l2_mean(value: torch.Tensor) -> float:
    flat = value.detach().float().reshape(value.shape[0], -1)
    return float(flat.norm(dim=1).mean().cpu())


def _mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())


def _std(value: torch.Tensor) -> float:
    return float(value.detach().float().std(unbiased=False).cpu())


def _abs_mean(value: torch.Tensor) -> float:
    return float(value.detach().float().abs().mean().cpu())
