"""Side-memory hierarchy from 64-day conditioning memory to 12/3 day memories."""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionBlock1D(nn.Module):
    """One residual self-attention + FFN block over [B, T, D] tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model % num_heads must be 0, got d_model={d_model}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")

        self.d_model = int(d_model)
        self.norm1 = nn.LayerNorm(self.d_model, eps=float(_norm_eps))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.d_model, eps=float(_norm_eps))
        ffn_dim = int(self.d_model * float(ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, D], got shape={tuple(x.shape)}.")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dim mismatch: expected {self.d_model}, got {x.shape[-1]}."
            )
        u = self.norm1(x)
        delta, _ = self.attn(u, u, u, need_weights=False)
        x = x + delta
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock1D(nn.Module):
    """One residual cross-attention + FFN block over [B, Tq, D] query tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model % num_heads must be 0, got d_model={d_model}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")

        self.d_model = int(d_model)
        self.norm_q = nn.LayerNorm(self.d_model, eps=float(_norm_eps))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.d_model, eps=float(_norm_eps))
        ffn_dim = int(self.d_model * float(ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f"q must have shape [B, Tq, D], got shape={tuple(q.shape)}.")
        if kv.ndim != 3:
            raise ValueError(
                f"kv must have shape [B, Tkv, D], got shape={tuple(kv.shape)}."
            )
        if q.shape[0] != kv.shape[0]:
            raise ValueError(
                f"q/kv batch mismatch: q batch={q.shape[0]}, kv batch={kv.shape[0]}."
            )
        if q.shape[-1] != self.d_model or kv.shape[-1] != self.d_model:
            raise ValueError(
                f"q/kv last dim must be {self.d_model}, got {q.shape[-1]} and {kv.shape[-1]}."
            )
        qn = self.norm_q(q)
        delta, _ = self.attn(qn, kv, kv, need_weights=False)
        out = q + delta
        out = out + self.ffn(self.norm2(out))
        return out


class SideMemoryHierarchy(nn.Module):
    """Build side memories s1/s2/s3 and globals g1/g2/g3 from conditioning memory."""

    def __init__(
        self,
        d_cond: int = 32,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_cond <= 0:
            raise ValueError(f"d_cond must be > 0, got {d_cond}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if d_cond % num_heads != 0:
            raise ValueError(
                f"d_cond % num_heads must be 0, got d_cond={d_cond}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")

        self.d_cond = int(d_cond)
        self._full_days = 64
        self._mezzo_days = 12
        self._micro_days = 3

        self.s1_block = SelfAttentionBlock1D(
            d_model=self.d_cond,
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            _norm_eps=float(_norm_eps),
        )
        self.s2_block = CrossAttentionBlock1D(
            d_model=self.d_cond,
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            _norm_eps=float(_norm_eps),
        )
        self.s3_block = CrossAttentionBlock1D(
            d_model=self.d_cond,
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            _norm_eps=float(_norm_eps),
        )

    def forward(
        self,
        cond_seq: torch.Tensor,
        cond_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cond_seq.ndim != 3:
            raise ValueError(
                f"cond_seq must have shape [B, d_cond, 64], got shape={tuple(cond_seq.shape)}."
            )
        if cond_global.ndim != 2:
            raise ValueError(
                f"cond_global must have shape [B, d_cond], got shape={tuple(cond_global.shape)}."
            )
        batch_size, d_cond, seq_len = cond_seq.shape
        if d_cond != self.d_cond or seq_len != self._full_days:
            raise ValueError(
                "cond_seq shape mismatch: "
                f"expected [{batch_size}, {self.d_cond}, {self._full_days}], got {tuple(cond_seq.shape)}."
            )
        if cond_global.shape != (batch_size, self.d_cond):
            raise ValueError(
                "cond_global shape mismatch: "
                f"expected ({batch_size}, {self.d_cond}), got {tuple(cond_global.shape)}."
            )

        s1_tokens = self.s1_block(cond_seq.transpose(1, 2))
        s1 = s1_tokens.transpose(1, 2)
        g1 = s1.mean(dim=-1)

        q2 = s1_tokens[:, -self._mezzo_days :, :]
        kv2 = s1_tokens[:, : -self._mezzo_days, :]
        s2_tokens = self.s2_block(q=q2, kv=kv2)
        s2 = s2_tokens.transpose(1, 2)
        g2 = s2.mean(dim=-1)

        q3 = s2_tokens[:, -self._micro_days :, :]
        kv3 = s2_tokens[:, : -self._micro_days, :]
        s3_tokens = self.s3_block(q=q3, kv=kv3)
        s3 = s3_tokens.transpose(1, 2)
        g3 = s3.mean(dim=-1)

        return s1, g1, s2, g2, s3, g3
