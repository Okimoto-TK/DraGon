"""Shared utilities for processor modules."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm


def normal_rank(s: pl.Series) -> pl.Series:
    """Apply normal rank transformation while preserving NaN slots."""
    arr = s.to_numpy()
    valid_mask = ~np.isnan(arr)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return s

    result = np.full(len(arr), np.nan)
    valid_values = arr[valid_mask]
    ranked = pl.Series(valid_values).rank(method="average").to_numpy()
    result[valid_mask] = norm.ppf((ranked - 0.5) / n_valid)

    return pl.Series(s.name, result)


def fracdiff_weights(d: float, window: int) -> np.ndarray:
    """Build fractional-differencing weights using recursive definition.

    omega_0 = 1
    omega_k = -omega_{k-1} * (d - k + 1) / k
    """
    if not (0.0 < d < 1.0):
        raise ValueError(f"fracdiff order d must be in (0,1), got {d}")
    if window <= 0:
        raise ValueError(f"fracdiff window must be positive, got {window}")

    weights = np.empty(window, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, window):
        weights[k] = -weights[k - 1] * (d - k + 1.0) / k
    return weights


def fracdiff_expr(
    col: str,
    d: float,
    window: int,
    over: str | list[str],
) -> pl.Expr:
    """Build a truncated fractional-differencing expression.

    Uses a sliding window of length ``window`` and sums:
        sum_{k=0}^{window-1} omega_k * X_{t-k}
    Missing history / NaN values are treated as 0 so early timestamps use
    the available prefix:
        sum_{k=0}^{t} omega_k * X_{t-k}
    """
    weights = fracdiff_weights(d=d, window=window)
    over_key: str | list[str]
    if isinstance(over, str):
        over_key = over
    else:
        over_key = list(over)

    return (
        pl.col(col)
        .fill_null(0.0)
        .fill_nan(0.0)
        .rolling_sum(
            window_size=window,
            weights=weights[::-1].tolist(),
            min_samples=1,
        )
        .over(over_key)
    )
