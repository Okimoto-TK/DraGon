"""Attention-based fusion primitives."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.components.normalization import ada_layer_norm
from src.models.components.trunks.common import SmallTokenRefine, SwiGLUFFN, pool_tokens


class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, steps, tokens, dim = x.shape
        y = x.reshape(bsz * steps, tokens, dim)
        attn_in = self.norm1(y)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        y = y + attn_out
        y = y + self.ffn(self.norm2(y))
        return y.reshape(bsz, steps, tokens, dim)


class PerTimeCrossAttention(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        bsz, steps, q_tokens, dim = q.shape
        _, _, kv_tokens, _ = kv.shape
        q_flat = q.reshape(bsz * steps, q_tokens, dim)
        kv_flat = kv.reshape(bsz * steps, kv_tokens, dim)
        out, _ = self.attn(self.q_norm(q_flat), self.kv_norm(kv_flat), self.kv_norm(kv_flat), need_weights=False)
        return out.reshape(bsz, steps, q_tokens, dim)


class SequenceCrossBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        delta, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        gate = self.gate(torch.cat((pool_tokens(q), pool_tokens(kv)), dim=-1)).unsqueeze(1)
        return self.out_norm(q + gate * delta)


class SideWriteIntoJoint(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.cross_q_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.write_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.cond = nn.Sequential(nn.Linear(dim * 2, dim * 3 + 2), nn.SiLU(), nn.Linear(dim * 3 + 2, dim * 2 + 2))
        self.token_self = TokenSelfAttentionBlock(dim, num_heads=num_heads)
        self.ffn = SmallTokenRefine(dim)
        self.write_out_norm = nn.LayerNorm(dim)
        self.adaln_norm = nn.LayerNorm(dim)

    def forward(self, joint: Tensor, e_d: Tensor, s6_ctx: Tensor) -> Tensor:
        bsz, steps, tokens, dim = joint.shape
        q = joint.reshape(bsz, steps * tokens, dim)
        kv = e_d.reshape(bsz, e_d.shape[1] * e_d.shape[2], dim)
        delta, _ = self.cross(
            self.cross_q_norm(q),
            self.cross_kv_norm(kv),
            self.cross_kv_norm(kv),
            need_weights=False,
        )
        pooled_joint = pool_tokens(joint)
        pooled_side = pool_tokens(e_d)
        pooled_state = pool_tokens(s6_ctx)
        write_gate = self.write_gate(torch.cat((pooled_state, pooled_side, pooled_joint), dim=-1)).unsqueeze(1)
        joint_out = self.write_out_norm(q + write_gate * delta).reshape(bsz, steps, tokens, dim)

        cond = self.cond(torch.cat((pooled_side, pooled_state), dim=-1))
        gamma, beta, g_attn, g_ffn = torch.split(cond, [dim, dim, 1, 1], dim=-1)

        flat = joint_out.reshape(bsz * steps, tokens, dim)
        gamma_bt = gamma.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, dim)
        beta_bt = beta.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, dim)
        attn_gate = g_attn.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, 1, 1)
        ffn_gate = g_ffn.unsqueeze(1).expand(-1, steps, -1).reshape(bsz * steps, 1, 1)

        mod = ada_layer_norm(flat, gamma_bt, beta_bt, self.adaln_norm)
        attn_delta = self.token_self(mod.reshape(bsz, steps, tokens, dim)).reshape(bsz * steps, tokens, dim) - mod
        flat = flat + attn_gate * attn_delta
        ffn_in = ada_layer_norm(flat, gamma_bt, beta_bt, self.adaln_norm)
        ffn_delta = self.ffn(ffn_in.reshape(bsz, steps, tokens, dim)).reshape(bsz * steps, tokens, dim) - ffn_in
        flat = flat + ffn_gate * ffn_delta
        return flat.reshape(bsz, steps, tokens, dim)


class StateQueryJointReader(nn.Module):
    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.cross = PerTimeCrossAttention(dim, num_heads=num_heads)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, s6_ctx: Tensor, joint: Tensor) -> Tensor:
        return self.out_norm(s6_ctx + self.cross(s6_ctx, joint))


__all__ = [
    "PerTimeCrossAttention",
    "SequenceCrossBlock",
    "SideWriteIntoJoint",
    "StateQueryJointReader",
    "TokenSelfAttentionBlock",
]
