from __future__ import annotations

from datetime import date
import math

import polars as pl
import pytest

from src.data.processor.label import process_label
from src.data.registry.processor import LABEL_WINDOW


def test_process_label_uses_return_minus_one_and_parkinson_window() -> None:
    days = [date(2026, 1, d) for d in range(5, 11)]
    index_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * len(days),
            "trade_date": days,
            "logic_index": list(range(1, len(days) + 1)),
        }
    )
    daily_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * len(days),
            "trade_date": days,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [10.2, 11.55, 12.60, 13.78, 14.98, 15.30],
            "low": [9.8, 10.50, 11.20, 12.20, 13.20, 14.70],
            "close": [10.1, 11.2, 12.3, 13.4, 14.5, 15.1],
        }
    )
    adj_factor_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * len(days),
            "trade_date": days,
            "adj_factor": [1.0] * len(days),
        }
    )

    out = process_label(
        index_df=index_df,
        daily_df=daily_df,
        adj_factor_df=adj_factor_df,
    )
    row = out.filter(pl.col("trade_date") == days[0]).row(0, named=True)

    future_opens = [daily_df["open"][k] for k in range(2, LABEL_WINDOW + 1)]
    entry_open = daily_df["open"][1]
    expected_ret = sum(future_opens) / len(future_opens) / entry_open - 1.0

    future_ranges = []
    for k in range(1, LABEL_WINDOW + 1):
        high_k = daily_df["high"][k]
        low_k = daily_df["low"][k]
        future_ranges.append(math.log(high_k / low_k) ** 2)
    expected_rv = math.sqrt(
        sum(future_ranges) / (LABEL_WINDOW * 4.0 * math.log(2.0))
    )

    assert row["label_ret"] == pytest.approx(expected_ret, rel=1e-6, abs=1e-6)
    assert row["label_rv"] == pytest.approx(expected_rv, rel=1e-6, abs=1e-6)
