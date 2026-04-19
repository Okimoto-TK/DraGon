"""Patch-aligned embeddings for discrete condition sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEmbedding1D(nn.Module):
    """Embed state/position conditions and align them to patch timeline."""

    def __init__(
        self,
        state_vocab_size: int,
        pos_vocab_size: int,
        cond_dim: int,
        patch_len: int,
        patch_stride: int,
        num_features: int,
    ) -> None:
        super().__init__()
        if state_vocab_size <= 0:
            raise ValueError(
                "state_vocab_size must be > 0, "
                f"got {state_vocab_size}. Expected positive integer."
            )
        if pos_vocab_size <= 0:
            raise ValueError(
                f"pos_vocab_size must be > 0, got {pos_vocab_size}. Expected positive integer."
            )
        if cond_dim <= 0:
            raise ValueError(
                f"cond_dim must be > 0, got {cond_dim}. Expected positive integer."
            )
        if patch_len <= 0:
            raise ValueError(
                f"patch_len must be > 0, got {patch_len}. Expected positive integer."
            )
        if patch_stride <= 0:
            raise ValueError(
                f"patch_stride must be > 0, got {patch_stride}. Expected positive integer."
            )
        if patch_len < patch_stride:
            raise ValueError(
                "patch_len must be >= patch_stride, "
                f"got patch_len={patch_len}, patch_stride={patch_stride}."
            )
        if num_features <= 0:
            raise ValueError(
                f"num_features must be > 0, got {num_features}. Expected positive integer."
            )

        self.state_vocab_size = int(state_vocab_size)
        self.pos_vocab_size = int(pos_vocab_size)
        self.cond_dim = int(cond_dim)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.num_features = int(num_features)

        self._pad_right = self.patch_len - self.patch_stride

        self.state_embedding = nn.Embedding(self.state_vocab_size, self.cond_dim)
        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.cond_dim)
        self.pool = nn.AvgPool1d(
            kernel_size=self.patch_len,
            stride=self.patch_stride,
            ceil_mode=False,
        )

    def forward(
        self,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        if x_state.ndim != 2:
            raise ValueError(
                f"x_state must have shape [B, T], got ndim={x_state.ndim}, shape={tuple(x_state.shape)}."
            )
        if x_pos.ndim != 2:
            raise ValueError(
                f"x_pos must have shape [B, T], got ndim={x_pos.ndim}, shape={tuple(x_pos.shape)}."
            )
        if x_state.shape != x_pos.shape:
            raise ValueError(
                "x_state/x_pos shape mismatch: "
                f"x_state shape={tuple(x_state.shape)}, x_pos shape={tuple(x_pos.shape)}."
            )

        e_state = self.state_embedding(x_state.long())
        e_pos = self.pos_embedding(x_pos.long())
        e = e_state + e_pos  # [B, T, cond_dim]

        e = e.transpose(1, 2)  # [B, cond_dim, T]
        e = F.pad(e, (0, self._pad_right), mode="replicate")
        c = self.pool(e)  # [B, cond_dim, N]

        c = c.unsqueeze(1).expand(-1, self.num_features, -1, -1)
        c = c.reshape(c.shape[0] * self.num_features, self.cond_dim, c.shape[-1])
        return c

