"""Semantic gated channel fusion for diffusion-side token streams."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from config.config import token_dim as DEFAULT_TOKEN_DIM
from torch import Tensor, nn

from src.models.components.pooling.attentive_pool_1d import AttentivePool1d


def _should_record_debug() -> bool:
    return not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _token_l2_mean(x: Tensor) -> Tensor:
    return x.detach().float().norm(dim=-1).mean()


class _GEGLU(nn.Module):
    """GEGLU activation for token MLPs."""

    def forward(self, x: Tensor) -> Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * nn.functional.gelu(gate)


class _SemanticFusionBlock(nn.Module):
    """One explicit summary-level interaction fusion block."""

    def __init__(self, dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * 8
        self.norm = nn.LayerNorm(dim * 4)
        self.fc_in = nn.Linear(dim * 4, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.last_term_norms: dict[str, Tensor] = {}
        self.last_gate_activation: Tensor | None = None

    def forward(self, x: Tensor, summary: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        product = x * summary
        difference = x - summary
        interaction = torch.cat((x, summary, product, difference), dim=-1)
        record_debug = _should_record_debug()
        if record_debug:
            self.last_term_norms = {
                "x": x.detach().norm(dim=-1).mean(),
                "summary": summary.detach().norm(dim=-1).mean(),
                "product": product.detach().norm(dim=-1).mean(),
                "difference": difference.detach().norm(dim=-1).mean(),
            }
        hidden = self.fc_in(self.norm(interaction))
        value, gate = hidden.chunk(2, dim=-1)
        gate_act = F.gelu(gate)
        if record_debug:
            self.last_gate_activation = gate_act.detach()
        fused = value * gate_act
        residual_out = x + self.fc_out(self.dropout(fused))
        next_x = self.output_norm(residual_out)
        if not return_debug:
            return next_x
        return next_x, {
            "x_norm": _token_l2_mean(x),
            "summary_norm": _token_l2_mean(summary),
            "product_norm": _token_l2_mean(product),
            "difference_norm": _token_l2_mean(difference),
            "gate_norm": _token_l2_mean(gate_act),
            "residual_out_norm": _token_l2_mean(residual_out),
            "out_norm": _token_l2_mean(next_x),
        }


class SemanticGatedChannelFusion(nn.Module):
    """Stacked explicit summary-level interaction fusion for diffusion tokens."""

    def __init__(self, dim: int = DEFAULT_TOKEN_DIM, *, num_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = dim
        self.side_pool = AttentivePool1d(dim=dim)
        self.blocks = nn.ModuleList([_SemanticFusionBlock(dim, dropout=dropout) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.last_side_global: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        side_tokens: Tensor,
        *,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or side_tokens.ndim != 3:
            raise ValueError(
                f"Expected x and side_tokens to be [B, K, D], got {tuple(x.shape)} and {tuple(side_tokens.shape)}"
            )
        if x.shape != side_tokens.shape:
            raise ValueError(f"Expected matching shapes, got {tuple(x.shape)} and {tuple(side_tokens.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {x.shape[-1]}")

        side_global = self.side_pool(side_tokens)
        if _should_record_debug():
            self.last_side_global = side_global.detach()
        side_broadcast = side_global.unsqueeze(1).expand(-1, x.shape[1], -1)
        tokens = x
        if not return_debug:
            for block in self.blocks:
                tokens = block(tokens, side_broadcast)
            return self.output_norm(tokens)

        debug: dict[str, Tensor] = {
            "fusion/side_global_norm": _token_l2_mean(side_global),
        }
        for block_idx, block in enumerate(self.blocks, start=1):
            tokens, block_debug = block(tokens, side_broadcast, return_debug=True)
            for name, value in block_debug.items():
                debug[f"fusion/diffusion_block{block_idx}_{name}"] = value
        return self.output_norm(tokens), debug

    def get_last_debug(self) -> dict[str, object]:
        return {
            "side_global": self.last_side_global,
            "term_norms": [block.last_term_norms for block in self.blocks],
            "gate_activations": [block.last_gate_activation for block in self.blocks],
        }


__all__ = ["SemanticGatedChannelFusion"]
