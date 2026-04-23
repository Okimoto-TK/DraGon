"""Time-domain top-down hierarchical fusion with causal cross-scale attention."""

from __future__ import annotations

import torch
import torch.nn as nn


def _historical_attn_mask(
    query_len: int,
    key_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    q_pos = torch.linspace(0.0, 1.0, steps=query_len, device=device)
    k_pos = torch.linspace(0.0, 1.0, steps=key_len, device=device)
    return k_pos.unsqueeze(0) > q_pos.unsqueeze(1)


class _TimeCrossScaleAttentionBlock(nn.Module):
    """Write parent-scale time context into child tokens with causal visibility."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        *,
        _norm_eps: float,
        _gate_floor: float,
    ) -> None:
        super().__init__()
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
        if _gate_floor < 0 or _gate_floor >= 1:
            raise ValueError(
                f"_gate_floor must satisfy 0 <= _gate_floor < 1, got {_gate_floor}."
            )

        self.hidden_dim = int(hidden_dim)
        self._gate_floor = float(_gate_floor)
        self.query_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.parent_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.write_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        ffn_dim = int(self.hidden_dim * float(ffn_ratio))
        self.ffn_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        child_tokens: torch.Tensor,
        parent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if child_tokens.ndim != 3 or parent_tokens.ndim != 3:
            raise ValueError(
                "child_tokens and parent_tokens must both have shape [B, D, N]."
            )
        if child_tokens.shape[0] != parent_tokens.shape[0]:
            raise ValueError(
                "Batch mismatch between child_tokens and parent_tokens: "
                f"{child_tokens.shape[0]} vs {parent_tokens.shape[0]}."
            )
        if child_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                f"child_tokens hidden_dim mismatch: expected {self.hidden_dim}, got {child_tokens.shape[1]}."
            )
        if parent_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                f"parent_tokens hidden_dim mismatch: expected {self.hidden_dim}, got {parent_tokens.shape[1]}."
            )

        child_seq = child_tokens.transpose(1, 2)
        parent_seq = parent_tokens.transpose(1, 2)
        attn_mask = _historical_attn_mask(
            child_seq.shape[1],
            parent_seq.shape[1],
            device=child_seq.device,
        )

        q = self.query_norm(child_seq)
        kv = self.parent_norm(parent_seq)
        delta, _ = self.cross_attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)

        gate = torch.sigmoid(self.write_gate(kv.mean(dim=1)))
        gate = self._gate_floor + (1.0 - self._gate_floor) * gate
        child_seq = child_seq + gate.unsqueeze(1) * delta
        child_seq = child_seq + self.ffn(self.ffn_norm(child_seq))
        return child_seq.transpose(1, 2)


class TimeTopDownHierarchicalFusion(nn.Module):
    """Time-domain macro -> mezzo -> micro -> mezzo fusion chain."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _gate_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.macro_to_mezzo = nn.ModuleList(
            [
                _TimeCrossScaleAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.mezzo_to_micro = nn.ModuleList(
            [
                _TimeCrossScaleAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.micro_to_mezzo = nn.ModuleList(
            [
                _TimeCrossScaleAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        macro_tokens: torch.Tensor,
        mezzo_tokens: torch.Tensor,
        micro_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.macro_to_mezzo:
            mezzo_tokens = block(mezzo_tokens, macro_tokens)
        for block in self.mezzo_to_micro:
            micro_tokens = block(micro_tokens, mezzo_tokens)
        for block in self.micro_to_mezzo:
            mezzo_tokens = block(mezzo_tokens, micro_tokens)
        return macro_tokens, mezzo_tokens, micro_tokens
