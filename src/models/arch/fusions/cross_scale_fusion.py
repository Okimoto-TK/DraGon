"""Cross-scale bottleneck-latent fusion over three side-aware scale sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossScaleFusionBlock(nn.Module):
    """One latent-space fusion block with cross-attn, self-attn, and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
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

        self.cross_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        _ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=self._norm_eps),
            nn.Linear(self.hidden_dim, _ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(_ffn_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        scale_tokens: torch.Tensor,
    ) -> torch.Tensor:
        q_cross = self.cross_norm(latents)
        cross_delta, _ = self.cross_attn(
            q_cross, scale_tokens, scale_tokens, need_weights=False
        )
        latents = latents + cross_delta

        q_self = self.self_norm(latents)
        self_delta, _ = self.self_attn(q_self, q_self, q_self, need_weights=False)
        latents = latents + self_delta

        latents = latents + self.ffn(latents)
        return latents


class CrossScaleFusion(nn.Module):
    """Cross-scale late fusion using learned bottleneck latents."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_latents: int = 8,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
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
        if _norm_eps <= 0:
            raise ValueError(
                f"_norm_eps must be > 0, got {_norm_eps}. Valid range: (0, +inf)."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_latents = int(num_latents)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.macro_scale_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.mezzo_scale_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.micro_scale_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.latents = nn.Parameter(torch.zeros(1, self.num_latents, self.hidden_dim))

        self.blocks = nn.ModuleList(
            [
                CrossScaleFusionBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    dropout=self.dropout,
                    _norm_eps=self._norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        macro_seq: torch.Tensor,
        mezzo_seq: torch.Tensor,
        micro_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if macro_seq.ndim != 3:
            raise ValueError(
                "macro_seq must have ndim == 3, "
                f"got ndim={macro_seq.ndim}, shape={tuple(macro_seq.shape)}. "
                "Valid shape: [B, hidden_dim, 16]."
            )
        if mezzo_seq.ndim != 3:
            raise ValueError(
                "mezzo_seq must have ndim == 3, "
                f"got ndim={mezzo_seq.ndim}, shape={tuple(mezzo_seq.shape)}. "
                "Valid shape: [B, hidden_dim, 24]."
            )
        if micro_seq.ndim != 3:
            raise ValueError(
                "micro_seq must have ndim == 3, "
                f"got ndim={micro_seq.ndim}, shape={tuple(micro_seq.shape)}. "
                "Valid shape: [B, hidden_dim, 36]."
            )

        b_macro, d_macro, n_macro = macro_seq.shape
        b_mezzo, d_mezzo, n_mezzo = mezzo_seq.shape
        b_micro, d_micro, n_micro = micro_seq.shape

        if d_macro != self.hidden_dim:
            raise ValueError(
                "macro_seq hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {d_macro}. Valid value: hidden_dim."
            )
        if d_mezzo != self.hidden_dim:
            raise ValueError(
                "mezzo_seq hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {d_mezzo}. Valid value: hidden_dim."
            )
        if d_micro != self.hidden_dim:
            raise ValueError(
                "micro_seq hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {d_micro}. Valid value: hidden_dim."
            )
        if n_macro != 16:
            raise ValueError(
                "macro_seq length mismatch: "
                f"expected 16, got {n_macro}. Valid value: 16."
            )
        if n_mezzo != 24:
            raise ValueError(
                "mezzo_seq length mismatch: "
                f"expected 24, got {n_mezzo}. Valid value: 24."
            )
        if n_micro != 36:
            raise ValueError(
                "micro_seq length mismatch: "
                f"expected 36, got {n_micro}. Valid value: 36."
            )
        if b_macro != b_mezzo or b_macro != b_micro:
            raise ValueError(
                "Batch size mismatch among macro_seq/mezzo_seq/micro_seq: "
                f"got B_macro={b_macro}, B_mezzo={b_mezzo}, B_micro={b_micro}. "
                "Valid range: all three batch sizes must be equal."
            )

        macro = macro_seq.transpose(1, 2)
        mezzo = mezzo_seq.transpose(1, 2)
        micro = micro_seq.transpose(1, 2)

        macro = macro + self.macro_scale_embedding
        mezzo = mezzo + self.mezzo_scale_embedding
        micro = micro + self.micro_scale_embedding

        scale_tokens = torch.cat([macro, mezzo, micro], dim=1)

        latents = self.latents.expand(scale_tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, scale_tokens)

        fused_latents = latents.transpose(1, 2)
        fused_global = latents.mean(dim=1)
        return fused_latents, fused_global
