"""Temperature-weighted pooling over feature axis for STAR-style aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticPooling1D(nn.Module):
    """Pool feature representations with temperature-scaled attention weights."""

    def __init__(
        self,
        _pool_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if _pool_temperature <= 0:
            raise ValueError(
                "_pool_temperature must be > 0, "
                f"got {_pool_temperature}. Valid range: (0, +inf)."
            )
        self._pool_temperature = float(_pool_temperature)

    def forward(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                f"values must have shape [Bq, F, C], got ndim={values.ndim}, shape={tuple(values.shape)}."
            )
        if scores.ndim != 3:
            raise ValueError(
                f"scores must have shape [Bq, F, 1], got ndim={scores.ndim}, shape={tuple(scores.shape)}."
            )
        if values.shape[0] != scores.shape[0] or values.shape[1] != scores.shape[1]:
            raise ValueError(
                "values and scores must share [Bq, F] dimensions, "
                f"got values shape={tuple(values.shape)}, scores shape={tuple(scores.shape)}."
            )
        if scores.shape[2] != 1:
            raise ValueError(
                f"scores last dimension must be 1, got {scores.shape[2]}. Valid value: 1."
            )

        probs = torch.softmax(scores / self._pool_temperature, dim=1)
        pooled = (probs * values).sum(dim=1, keepdim=True)
        return pooled
