"""Explicit, stacked dual cross-attention fusion for short token sequences."""
from __future__ import annotations

import torch
from config.config import token_dim as DEFAULT_TOKEN_DIM
from torch import Tensor, nn


def _should_record_debug() -> bool:
    return not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _token_l2_mean(x: Tensor) -> Tensor:
    return x.detach().float().norm(dim=-1).mean()


class _FeedForward(nn.Module):
    """Residual FFN with large expansion."""

    def __init__(self, dim: int, *, ff_mult: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        residual_out = x + self.net(self.norm(x))
        next_x = self.output_norm(residual_out)
        if not return_debug:
            return next_x
        return next_x, {
            "residual_out_norm": _token_l2_mean(residual_out),
            "out_norm": _token_l2_mean(next_x),
        }


class _SelfAttentionRefine(nn.Module):
    """Light self-attention refinement after cross-attention."""

    def __init__(self, dim: int, *, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        refined, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        residual_out = x + refined
        next_x = self.output_norm(residual_out)
        if not return_debug:
            return next_x
        return next_x, {
            "residual_out_norm": _token_l2_mean(residual_out),
            "out_norm": _token_l2_mean(next_x),
        }


class _BidirectionalFusionBlock(nn.Module):
    """One strong dual fusion block with cross-attn, self-attn and FFN."""

    def __init__(self, dim: int, *, num_heads: int = 4, ff_mult: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        self.x_q_norm = nn.LayerNorm(dim)
        self.x_kv_norm = nn.LayerNorm(dim)
        self.y_q_norm = nn.LayerNorm(dim)
        self.y_kv_norm = nn.LayerNorm(dim)
        self.x_cross_res_norm = nn.LayerNorm(dim)
        self.y_cross_res_norm = nn.LayerNorm(dim)
        self.x_cross = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.y_cross = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.x_self = _SelfAttentionRefine(dim, num_heads=num_heads, dropout=dropout)
        self.y_self = _SelfAttentionRefine(dim, num_heads=num_heads, dropout=dropout)
        self.x_ffn = _FeedForward(dim, ff_mult=ff_mult, dropout=dropout)
        self.y_ffn = _FeedForward(dim, ff_mult=ff_mult, dropout=dropout)
        self.last_x_to_y_attn: Tensor | None = None
        self.last_y_to_x_attn: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        record_debug = _should_record_debug()
        next_x, x_to_y_attn = self.x_cross(
            self.x_q_norm(x),
            self.x_kv_norm(y),
            self.x_kv_norm(y),
            need_weights=record_debug,
            average_attn_weights=False,
        )
        next_y, y_to_x_attn = self.y_cross(
            self.y_q_norm(y),
            self.y_kv_norm(x),
            self.y_kv_norm(x),
            need_weights=record_debug,
            average_attn_weights=False,
        )
        if record_debug:
            self.last_x_to_y_attn = x_to_y_attn.detach()
            self.last_y_to_x_attn = y_to_x_attn.detach()
        x_cross_residual = x + next_x
        y_cross_residual = y + next_y
        x = self.x_cross_res_norm(x_cross_residual)
        y = self.y_cross_res_norm(y_cross_residual)
        x_cross_out = x
        y_cross_out = y
        if not return_debug:
            x = self.x_self(x)
            y = self.y_self(y)
            x = self.x_ffn(x)
            y = self.y_ffn(y)
            return x, y
        x, x_self_debug = self.x_self(x, return_debug=True)
        y, y_self_debug = self.y_self(y, return_debug=True)
        x, x_ffn_debug = self.x_ffn(x, return_debug=True)
        y, y_ffn_debug = self.y_ffn(y, return_debug=True)
        if not return_debug:
            return x, y
        debug = {
            "x_cross_residual_out_norm": _token_l2_mean(x_cross_residual),
            "x_cross_out_norm": _token_l2_mean(x_cross_out),
            "y_cross_residual_out_norm": _token_l2_mean(y_cross_residual),
            "y_cross_out_norm": _token_l2_mean(y_cross_out),
            "x_norm": _token_l2_mean(x),
            "y_norm": _token_l2_mean(y),
        }
        for name, value in x_self_debug.items():
            debug[f"x_self_{name}"] = value
        for name, value in y_self_debug.items():
            debug[f"y_self_{name}"] = value
        for name, value in x_ffn_debug.items():
            debug[f"x_ffn_{name}"] = value
        for name, value in y_ffn_debug.items():
            debug[f"y_ffn_{name}"] = value
        return x, y, debug


class DualCrossAttentionFusion(nn.Module):
    """Four-layer stacked, symmetric dual cross-attention fusion."""

    def __init__(
        self,
        dim: int = DEFAULT_TOKEN_DIM,
        *,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_mult: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide dim, got num_heads={num_heads}, dim={dim}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = dim
        self.layers = nn.ModuleList(
            [
                _BidirectionalFusionBlock(dim, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fuse_norm = nn.LayerNorm(dim * 2)
        self.fuse_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    @staticmethod
    def _match_length(tokens: Tensor, target_len: int) -> Tensor:
        if tokens.shape[1] == target_len:
            return tokens
        pooled = nn.functional.adaptive_avg_pool1d(tokens.transpose(1, 2), target_len)
        return pooled.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Expected x and y to be [B, L, D], got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch sizes must match, got {x.shape[0]} and {y.shape[0]}")
        if x.shape[2] != self.dim or y.shape[2] != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {x.shape[2]} and {y.shape[2]}")

        x_tokens = x
        y_tokens = y
        if not return_debug:
            for layer in self.layers:
                x_tokens, y_tokens = layer(x_tokens, y_tokens)
        else:
            debug: dict[str, Tensor] = {}
            for layer_idx, layer in enumerate(self.layers, start=1):
                x_tokens, y_tokens, layer_debug = layer(x_tokens, y_tokens, return_debug=True)
                for name, value in layer_debug.items():
                    debug[f"fusion/drift_layer{layer_idx}_{name}"] = value

        fused_len = max(x_tokens.shape[1], y_tokens.shape[1])
        x_aligned = self._match_length(x_tokens, fused_len)
        y_aligned = self._match_length(y_tokens, fused_len)
        fused = torch.cat((x_aligned, y_aligned), dim=-1)
        fused_out = self.fuse_proj(self.fuse_norm(fused))
        if not return_debug:
            return fused_out
        debug["fusion/drift_fused_tokens_norm"] = _token_l2_mean(fused_out)
        return fused_out, debug

    def get_last_debug(self) -> dict[str, list[Tensor | None]]:
        return {
            "x_to_y_attn": [layer.last_x_to_y_attn for layer in self.layers],
            "y_to_x_attn": [layer.last_y_to_x_attn for layer in self.layers],
        }


__all__ = ["DualCrossAttentionFusion"]
