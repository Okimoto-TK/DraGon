"""Full tensor fusion network block."""
from __future__ import annotations

import torch
from config.config import lmf_dim as DEFAULT_LMF_DIM
from torch import Tensor, nn


class TensorFusion(nn.Module):
    """Fuse two vectors with full outer-product tensor fusion."""

    def __init__(self, dim_x: int = DEFAULT_LMF_DIM, dim_y: int = DEFAULT_LMF_DIM) -> None:
        super().__init__()
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError(f"dim_x and dim_y must be positive, got dim_x={dim_x}, dim_y={dim_y}")
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.out_dim = (dim_x + 1) * (dim_y + 1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Expected x and y to be [B, D], got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch sizes must match, got {x.shape[0]} and {y.shape[0]}")
        if x.shape[1] != self.dim_x or y.shape[1] != self.dim_y:
            raise ValueError(
                f"Expected dims ({self.dim_x}, {self.dim_y}), got ({x.shape[1]}, {y.shape[1]})"
            )

        ones_x = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        ones_y = torch.ones((y.shape[0], 1), device=y.device, dtype=y.dtype)
        x_aug = torch.cat((x, ones_x), dim=1)
        y_aug = torch.cat((y, ones_y), dim=1)
        outer = torch.einsum("bi,bj->bij", x_aug, y_aug)
        return outer.reshape(x.shape[0], self.out_dim)


__all__ = ["TensorFusion"]
