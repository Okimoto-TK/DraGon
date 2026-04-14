"""Decoder head for direct multi-target prediction."""
from __future__ import annotations

from torch import Tensor, nn


class DecoderHead(nn.Module):
    """Map fused TFN features into 8 direct outputs."""

    def __init__(self, in_dim: int = 289, hidden_dim1: int = 128, hidden_dim2: int = 64, out_dim: int = 8) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim1 <= 0 or hidden_dim2 <= 0 or out_dim <= 0:
            raise ValueError(
                f"All dimensions must be positive, got in_dim={in_dim}, hidden_dim1={hidden_dim1}, "
                f"hidden_dim2={hidden_dim2}, out_dim={out_dim}"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim2, out_dim)
        self.debug_enabled = False
        self.last_hidden1: Tensor | None = None
        self.last_hidden2: Tensor | None = None
        self.last_out: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [B, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {x.shape[-1]}")
        hidden1 = self.act1(self.fc1(x))
        hidden2 = self.act2(self.fc2(hidden1))
        out = self.fc3(hidden2)
        if self.debug_enabled:
            self.last_hidden1 = hidden1.detach()
            self.last_hidden2 = hidden2.detach()
            self.last_out = out.detach()
        else:
            self.last_hidden1 = None
            self.last_hidden2 = None
            self.last_out = None
        return out

    def get_last_debug(self) -> dict[str, Tensor | None]:
        return {
            "hidden1": self.last_hidden1,
            "hidden2": self.last_hidden2,
            "out": self.last_out,
        }

    def set_debug_capture(self, enabled: bool) -> None:
        self.debug_enabled = bool(enabled)


__all__ = ["DecoderHead"]
