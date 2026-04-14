"""Explicit, stacked dual cross-attention fusion for short token sequences."""
from __future__ import annotations

import torch
from config.config import token_dim as DEFAULT_TOKEN_DIM
from torch import Tensor, nn


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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


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

    def forward(self, x: Tensor) -> Tensor:
        refined, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        return x + refined


class _BidirectionalFusionBlock(nn.Module):
    """One strong dual fusion block with cross-attn, self-attn and FFN."""

    def __init__(self, dim: int, *, num_heads: int = 4, ff_mult: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        self.x_q_norm = nn.LayerNorm(dim)
        self.x_kv_norm = nn.LayerNorm(dim)
        self.y_q_norm = nn.LayerNorm(dim)
        self.y_kv_norm = nn.LayerNorm(dim)
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
        self.debug_enabled = False
        self.last_x_to_y_attn: Tensor | None = None
        self.last_y_to_x_attn: Tensor | None = None

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        next_x, x_to_y_attn = self.x_cross(
            self.x_q_norm(x),
            self.x_kv_norm(y),
            self.x_kv_norm(y),
            need_weights=self.debug_enabled,
            average_attn_weights=False,
        )
        next_y, y_to_x_attn = self.y_cross(
            self.y_q_norm(y),
            self.y_kv_norm(x),
            self.y_kv_norm(x),
            need_weights=self.debug_enabled,
            average_attn_weights=False,
        )
        if self.debug_enabled and x_to_y_attn is not None and y_to_x_attn is not None:
            self.last_x_to_y_attn = x_to_y_attn.detach()
            self.last_y_to_x_attn = y_to_x_attn.detach()
        else:
            self.last_x_to_y_attn = None
            self.last_y_to_x_attn = None
        x = x + next_x
        y = y + next_y
        x = self.x_self(x)
        y = self.y_self(y)
        x = self.x_ffn(x)
        y = self.y_ffn(y)
        return x, y


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
        self.debug_enabled = False

    @staticmethod
    def _match_length(tokens: Tensor, target_len: int) -> Tensor:
        if tokens.shape[1] == target_len:
            return tokens
        pooled = nn.functional.adaptive_avg_pool1d(tokens.transpose(1, 2), target_len)
        return pooled.transpose(1, 2)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Expected x and y to be [B, L, D], got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch sizes must match, got {x.shape[0]} and {y.shape[0]}")
        if x.shape[2] != self.dim or y.shape[2] != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {x.shape[2]} and {y.shape[2]}")

        x_tokens = x
        y_tokens = y
        for layer in self.layers:
            layer.debug_enabled = self.debug_enabled
            x_tokens, y_tokens = layer(x_tokens, y_tokens)

        fused_len = max(x_tokens.shape[1], y_tokens.shape[1])
        x_aligned = self._match_length(x_tokens, fused_len)
        y_aligned = self._match_length(y_tokens, fused_len)
        fused = torch.cat((x_aligned, y_aligned), dim=-1)
        return self.fuse_proj(self.fuse_norm(fused))

    def get_last_debug(self) -> dict[str, list[Tensor | None]]:
        return {
            "x_to_y_attn": [layer.last_x_to_y_attn for layer in self.layers],
            "y_to_x_attn": [layer.last_y_to_x_attn for layer in self.layers],
        }

    def set_debug_capture(self, enabled: bool) -> None:
        self.debug_enabled = bool(enabled)
        for layer in self.layers:
            layer.debug_enabled = self.debug_enabled


__all__ = ["DualCrossAttentionFusion"]
