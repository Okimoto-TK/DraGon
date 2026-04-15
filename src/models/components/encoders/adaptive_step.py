"""STEP-like adaptive patch encoder for sidechain features."""
from __future__ import annotations

import torch
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import side_hidden_dim as DEFAULT_SIDE_HIDDEN_DIM
from torch import Tensor, nn

from src.models.components.normalization import LayerNorm1d


def _token_l2_mean(x: Tensor) -> Tensor:
    return x.detach().float().norm(dim=-1).mean()


class _PreNormTransformerBlock(nn.Module):
    """A compact pre-norm Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide dim, got num_heads={num_heads}, dim={dim}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_res_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        attn_residual_out = x + attn_out
        attn_out_norm = self.attn_res_norm(attn_residual_out)
        ffn_residual_out = attn_out_norm + self.ffn(self.norm2(attn_out_norm))
        x = self.output_norm(ffn_residual_out)
        if not return_debug:
            return x
        return x, {
            "attn_residual_out_norm": _token_l2_mean(attn_residual_out),
            "attn_out_norm": _token_l2_mean(attn_out_norm),
            "ffn_residual_out_norm": _token_l2_mean(ffn_residual_out),
            "out_norm": _token_l2_mean(x),
        }


class _PatchBranch(nn.Module):
    """One adaptive patch branch with a fixed patch size."""

    def __init__(self, d_model: int, patch_size: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.d_model = d_model
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(d_model * patch_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_tokens: Tensor) -> Tensor:
        if x_tokens.ndim != 3:
            raise ValueError(f"Expected x_tokens shape [B, L, D], got {tuple(x_tokens.shape)}")
        if x_tokens.shape[-1] != self.d_model:
            raise ValueError(f"Expected token dim {self.d_model}, got {x_tokens.shape[-1]}")
        if x_tokens.shape[1] % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {x_tokens.shape[1]} must be divisible by patch_size {self.patch_size}"
            )

        bsz, seq_len, dim = x_tokens.shape
        num_patches = seq_len // self.patch_size
        patches = x_tokens.reshape(bsz, num_patches, self.patch_size * dim)
        patch_tokens = self.norm(self.proj(patches))
        return patch_tokens.repeat_interleave(self.patch_size, dim=1)


class SidechainEncoder(nn.Module):
    """Encode sidechain features with adaptive multi-scale patching.

    Input:
        x: [B, 8, 64]

    Output:
        y: [B, lmf_dim, 64]
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = DEFAULT_SIDE_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
        *,
        d_model: int | None = None,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if lmf_dim <= 0:
            raise ValueError(f"lmf_dim must be positive, got {lmf_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        resolved_d_model = int(hidden_dim if d_model is None else d_model)
        if resolved_d_model < 32:
            raise ValueError(f"d_model must be at least 32, got {resolved_d_model}")
        if resolved_d_model % num_heads != 0:
            raise ValueError(
                f"resolved d_model must be divisible by num_heads, got d_model={resolved_d_model}, num_heads={num_heads}"
            )

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lmf_dim = lmf_dim
        self.d_model = resolved_d_model
        self.patch_sizes = (2, 4, 8)

        self.channel_lift = nn.Sequential(
            nn.Conv1d(in_channels, resolved_d_model, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(resolved_d_model, resolved_d_model, kernel_size=1, stride=1),
        )
        self.channel_lift_norm = LayerNorm1d(resolved_d_model)
        self.branch_2 = _PatchBranch(resolved_d_model, patch_size=2)
        self.branch_4 = _PatchBranch(resolved_d_model, patch_size=4)
        self.branch_8 = _PatchBranch(resolved_d_model, patch_size=8)

        self.global_gate = nn.Sequential(
            nn.Linear(resolved_d_model, resolved_d_model),
            nn.GELU(),
            nn.Linear(resolved_d_model, len(self.patch_sizes)),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(resolved_d_model * len(self.patch_sizes), resolved_d_model),
            nn.GELU(),
            nn.Linear(resolved_d_model, len(self.patch_sizes)),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 64, resolved_d_model) * 0.02)
        self.branch_fuse_norm = nn.LayerNorm(resolved_d_model)
        self.encoder = nn.ModuleList(
            [
                _PreNormTransformerBlock(
                    resolved_d_model,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(resolved_d_model),
            nn.Linear(resolved_d_model, lmf_dim),
        )

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, C, L], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")
        if x.shape[2] != 64:
            raise ValueError(f"Expected sequence length 64, got {x.shape[2]}")

        lifted = self.channel_lift_norm(self.channel_lift(x))  # [B, d_model, 64]
        base_tokens = lifted.transpose(1, 2)  # [B, 64, d_model]

        branch2 = self.branch_2(base_tokens)
        branch4 = self.branch_4(base_tokens)
        branch8 = self.branch_8(base_tokens)

        branch_stack = torch.stack((branch2, branch4, branch8), dim=2)  # [B, 64, 3, d_model]
        global_context = base_tokens.mean(dim=1)
        global_logits = self.global_gate(global_context).unsqueeze(1)  # [B, 1, 3]
        local_logits = self.local_gate(torch.cat((branch2, branch4, branch8), dim=-1))  # [B, 64, 3]
        gates = torch.softmax(local_logits + global_logits, dim=-1)

        fused = torch.sum(branch_stack * gates.unsqueeze(-1), dim=2)
        tokens = self.branch_fuse_norm(fused + base_tokens + self.pos_embed)

        if not return_debug:
            for layer in self.encoder:
                tokens = layer(tokens)
        else:
            debug: dict[str, Tensor] = {
                "encoder/side_channel_lift_norm": _token_l2_mean(base_tokens),
                "encoder/side_branch_fuse_norm": _token_l2_mean(tokens),
            }
            for layer_idx, layer in enumerate(self.encoder, start=1):
                tokens, layer_debug = layer(tokens, return_debug=True)
                for name, value in layer_debug.items():
                    debug[f"encoder/side_block{layer_idx}_{name}"] = value

        out_tokens = self.out_proj(tokens)  # [B, 64, lmf_dim]
        out = out_tokens.transpose(1, 2).contiguous()
        if not return_debug:
            return out
        debug["encoder/side_pre_out_proj_norm"] = _token_l2_mean(tokens)
        return out, debug


__all__ = ["SidechainEncoder"]
