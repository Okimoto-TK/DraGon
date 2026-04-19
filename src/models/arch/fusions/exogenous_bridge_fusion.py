"""TimeXer-style exogenous bridge fusion for single-scale sequence features."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExogenousBridgeBlock(nn.Module):
    """Single bridge block that injects exogenous memory into endogenous tokens."""

    def __init__(
        self,
        hidden_dim: int,
        exogenous_dim: int,
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
        if exogenous_dim <= 0:
            raise ValueError(
                "exogenous_dim must be > 0, "
                f"got {exogenous_dim}. Valid range: positive integers."
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
        self.exogenous_dim = int(exogenous_dim)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.exo_proj = nn.Linear(self.exogenous_dim, self.hidden_dim)
        self.exo_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)

        self.bridge_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.bridge_norm = nn.LayerNorm(self.hidden_dim, eps=self._norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.global_gate = nn.Linear(self.exogenous_dim, self.hidden_dim)

        _ffn_dim = int(self.hidden_dim * self.ffn_ratio)
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim, eps=self._norm_eps),
            nn.Linear(2 * self.hidden_dim, _ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(_ffn_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        endogenous_tokens: torch.Tensor,
        exogenous_tokens: torch.Tensor,
        exogenous_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exo = self.exo_proj(exogenous_tokens)
        exo = self.exo_norm(exo)

        endo_mean = endogenous_tokens.mean(dim=1, keepdim=True)
        bridge0 = endo_mean + self.bridge_token

        q = self.bridge_norm(bridge0)
        bridge_delta, _ = self.cross_attn(q, exo, exo, need_weights=False)
        bridge1 = bridge0 + bridge_delta

        gate = torch.sigmoid(self.global_gate(exogenous_global)).unsqueeze(1)

        bridge_rep = bridge1.expand(-1, endogenous_tokens.shape[1], -1)
        fuse_in = torch.cat([endogenous_tokens, bridge_rep], dim=-1)
        delta = self.fuse_mlp(fuse_in)
        delta = delta * gate
        endogenous_out = endogenous_tokens + delta

        bridge_global = bridge1.squeeze(1)
        return endogenous_out, bridge_global


class ExogenousBridgeFusion(nn.Module):
    """Single-scale endogenous/exogenous bridge fusion module."""

    def __init__(
        self,
        hidden_dim: int = 128,
        exogenous_dim: int = 32,
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
        if exogenous_dim <= 0:
            raise ValueError(
                "exogenous_dim must be > 0, "
                f"got {exogenous_dim}. Valid range: positive integers."
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
        self.exogenous_dim = int(exogenous_dim)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self._norm_eps = float(_norm_eps)

        self.blocks = nn.ModuleList(
            [
                ExogenousBridgeBlock(
                    hidden_dim=self.hidden_dim,
                    exogenous_dim=self.exogenous_dim,
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
        endogenous_seq: torch.Tensor,
        exogenous_seq: torch.Tensor,
        exogenous_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if endogenous_seq.ndim != 3:
            raise ValueError(
                "endogenous_seq must have ndim == 3, "
                f"got ndim={endogenous_seq.ndim}, shape={tuple(endogenous_seq.shape)}. "
                "Valid shape: [B, hidden_dim, N_endo]."
            )
        if exogenous_seq.ndim != 3:
            raise ValueError(
                "exogenous_seq must have ndim == 3, "
                f"got ndim={exogenous_seq.ndim}, shape={tuple(exogenous_seq.shape)}. "
                "Valid shape: [B, exogenous_dim, N_exo]."
            )
        if exogenous_global.ndim != 2:
            raise ValueError(
                "exogenous_global must have ndim == 2, "
                f"got ndim={exogenous_global.ndim}, shape={tuple(exogenous_global.shape)}. "
                "Valid shape: [B, exogenous_dim]."
            )

        b_endo, d_endo, n_endo = endogenous_seq.shape
        b_exo, d_exo, n_exo = exogenous_seq.shape
        b_exo_global, d_exo_global = exogenous_global.shape

        if d_endo != self.hidden_dim:
            raise ValueError(
                "endogenous_seq hidden_dim mismatch: "
                f"expected {self.hidden_dim}, got {d_endo}. Valid value: hidden_dim."
            )
        if d_exo != self.exogenous_dim:
            raise ValueError(
                "exogenous_seq exogenous_dim mismatch: "
                f"expected {self.exogenous_dim}, got {d_exo}. Valid value: exogenous_dim."
            )
        if d_exo_global != self.exogenous_dim:
            raise ValueError(
                "exogenous_global exogenous_dim mismatch: "
                f"expected {self.exogenous_dim}, got {d_exo_global}. Valid value: exogenous_dim."
            )
        if b_endo != b_exo or b_endo != b_exo_global:
            raise ValueError(
                "Batch size mismatch among endogenous_seq/exogenous_seq/exogenous_global: "
                f"got B_endo={b_endo}, B_exo={b_exo}, B_exo_global={b_exo_global}. "
                "Valid range: all three batch sizes must be equal."
            )
        if n_endo <= 0:
            raise ValueError(
                f"N_endo must be > 0, got {n_endo}. Valid range: positive integers."
            )
        if n_exo <= 0:
            raise ValueError(
                f"N_exo must be > 0, got {n_exo}. Valid range: positive integers."
            )

        endo = endogenous_seq.transpose(1, 2)
        exo = exogenous_seq.transpose(1, 2)

        for block in self.blocks:
            endo, bridge_global = block(
                endogenous_tokens=endo,
                exogenous_tokens=exo,
                exogenous_global=exogenous_global,
            )

        endogenous_fused = endo.transpose(1, 2)
        return endogenous_fused, bridge_global
