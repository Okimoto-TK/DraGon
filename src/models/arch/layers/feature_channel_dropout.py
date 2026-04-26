"""Feature-channel dropout over [B, C, T] inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureChannelDropout1D(nn.Module):
    """Drop full feature channels across the whole temporal axis."""

    def __init__(
        self,
        num_channels: int,
        p: float = 0.0,
        special_channel_ps: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(
                f"num_channels must be > 0, got {num_channels}. Valid range: positive integers."
            )
        if p < 0 or p >= 1:
            raise ValueError(
                f"p must satisfy 0 <= p < 1, got {p}. Valid range: [0, 1)."
            )
        if special_channel_ps is None:
            special_channel_ps = {}

        probs = torch.full((int(num_channels),), float(p), dtype=torch.float32)
        for index, value in special_channel_ps.items():
            if index < 0 or index >= num_channels:
                raise ValueError(
                    f"special channel index out of range: got {index}, expected in [0, {num_channels - 1}]."
                )
            if value < 0 or value >= 1:
                raise ValueError(
                    f"special channel p must satisfy 0 <= p < 1, got {value}. Valid range: [0, 1)."
                )
            probs[int(index)] = float(value)

        self.num_channels = int(num_channels)
        self.p = float(p)
        self._has_active_dropout = bool(torch.any(probs > 0).item())
        self.register_buffer("channel_ps", probs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, C, T], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.num_channels}, got {x.shape[1]}."
            )
        if not self.training or not self._has_active_dropout:
            return x

        keep_prob = (1.0 - self.channel_ps).to(device=x.device, dtype=torch.float32)
        keep_prob = keep_prob.unsqueeze(0).unsqueeze(-1)
        rand = torch.rand((x.shape[0], self.num_channels, 1), device=x.device, dtype=torch.float32)
        mask = (rand < keep_prob).to(dtype=x.dtype)
        scaled = mask / keep_prob.clamp_min(1e-6).to(dtype=x.dtype)
        return x * scaled
