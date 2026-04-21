"""Top-down scale-context bridge over macro -> mezzo -> micro sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from .exogenous_bridge_fusion import ExogenousBridgeFusion


class ScaleContextBridgeFusion(nn.Module):
    """Inject higher-scale context into lower-scale sequences with bridge blocks."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {num_layers}. Valid range: positive integers."
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}. Valid range: [0, 1)."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        bridge_kwargs = {
            "hidden_dim": self.hidden_dim,
            "exogenous_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        self.macro_to_mezzo = ExogenousBridgeFusion(**bridge_kwargs)
        self.mezzo_to_micro = ExogenousBridgeFusion(**bridge_kwargs)

    def _validate_scale_seq(
        self,
        *,
        name: str,
        value: torch.Tensor,
        expected_len: int,
    ) -> None:
        if value.ndim != 3:
            raise ValueError(
                f"{name} must have ndim == 3, got ndim={value.ndim}, shape={tuple(value.shape)}. "
                f"Valid shape: [B, hidden_dim, {expected_len}]."
            )
        if value.shape[1] != self.hidden_dim:
            raise ValueError(
                f"{name} hidden_dim mismatch: expected {self.hidden_dim}, got {value.shape[1]}. "
                "Valid value: hidden_dim."
            )
        if value.shape[2] != expected_len:
            raise ValueError(
                f"{name} length mismatch: expected {expected_len}, got {value.shape[2]}. "
                f"Valid value: {expected_len}."
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
        if macro_seq.shape[0] != mezzo_seq.shape[0] or macro_seq.shape[0] != micro_seq.shape[0]:
            raise ValueError(
                "Batch size mismatch among macro_seq/mezzo_seq/micro_seq: "
                f"got B_macro={macro_seq.shape[0]}, B_mezzo={mezzo_seq.shape[0]}, B_micro={micro_seq.shape[0]}. "
                "Valid range: all three batch sizes must be equal."
            )

        macro_ctx = macro_seq.mean(dim=-1)
        if return_debug:
            mezzo_td, _, macro_debug = self.macro_to_mezzo(
                endogenous_seq=mezzo_seq,
                exogenous_seq=macro_seq,
                exogenous_global=macro_ctx,
                return_debug=True,
            )
        else:
            mezzo_td, _ = self.macro_to_mezzo(
                endogenous_seq=mezzo_seq,
                exogenous_seq=macro_seq,
                exogenous_global=macro_ctx,
            )

        mezzo_ctx = mezzo_td.mean(dim=-1)
        if return_debug:
            micro_td, _, mezzo_debug = self.mezzo_to_micro(
                endogenous_seq=micro_seq,
                exogenous_seq=mezzo_td,
                exogenous_global=mezzo_ctx,
                return_debug=True,
            )
        else:
            micro_td, _ = self.mezzo_to_micro(
                endogenous_seq=micro_seq,
                exogenous_seq=mezzo_td,
                exogenous_global=mezzo_ctx,
            )

        micro_ctx = micro_td.mean(dim=-1)
        if not return_debug:
            return micro_td, macro_ctx, mezzo_ctx, micro_ctx

        debug: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "macro_ctx": macro_ctx,
            "mezzo_ctx": mezzo_ctx,
            "micro_ctx": micro_ctx,
            "macro_to_mezzo": macro_debug,
            "mezzo_to_micro": mezzo_debug,
        }
        return micro_td, macro_ctx, mezzo_ctx, micro_ctx, debug
