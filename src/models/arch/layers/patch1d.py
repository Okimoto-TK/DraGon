"""Patch projection for 1D sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Patch1D(nn.Module):
    """Project a 1D sequence into patch tokens."""

    def __init__(
        self,
        patch_len: int,
        patch_stride: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError(
                f"patch_len must be > 0, got {patch_len}. Expected positive integer."
            )
        if patch_stride <= 0:
            raise ValueError(
                f"patch_stride must be > 0, got {patch_stride}. Expected positive integer."
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if patch_len < patch_stride:
            raise ValueError(
                "patch_len must be >= patch_stride, "
                f"got patch_len={patch_len}, patch_stride={patch_stride}."
            )

        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.hidden_dim = int(hidden_dim)

        self._pad_right = self.patch_len - self.patch_stride
        self._in_channels = 1
        self._out_channels = self.hidden_dim

        self.proj = nn.Conv1d(
            self._in_channels,
            self._out_channels,
            kernel_size=self.patch_len,
            stride=self.patch_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [Bf, 1, T], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"x channel size must be 1, got {x.shape[1]}. Expected shape [Bf, 1, T]."
            )

        x = F.pad(x, (0, self._pad_right), mode="replicate")
        z = self.proj(x)
        return z

