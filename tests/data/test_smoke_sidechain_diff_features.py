from __future__ import annotations

from datetime import date
import math

import polars as pl
import pytest
from scipy.stats import norm

from src.data.processor.sidechain import process_sidechain
from src.data.processor.utils import fracdiff_weights


def test_sidechain_mf_concentration_diff_and_rank_semantics() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    index_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "trade_date": days + days,
            "logic_index": [1, 2, 3, 1, 2, 3],
        }
    )
    daily_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "trade_date": days + days,
            "open": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "high": [10.5, 11.5, 12.5, 20.5, 19.5, 18.5],
            "low": [9.5, 10.5, 11.5, 19.5, 18.5, 17.5],
            "close": [10.0, 11.0, 12.0, 20.0, 19.0, 18.0],
            "amount": [100.0] * 6,
        }
    )
    adj_factor_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "trade_date": days + days,
            "adj_factor": [1.0] * 6,
        }
    )
    moneyflow_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "trade_date": days + days,
            "buy_lg_amount": [20.0, 30.0, 10.0, 10.0, 5.0, 25.0],
            "buy_elg_amount": [10.0, 0.0, 5.0, 0.0, 5.0, 0.0],
            "sell_lg_amount": [10.0, 20.0, 10.0, 10.0, 10.0, 20.0],
            "sell_elg_amount": [0.0, 10.0, 5.0, 0.0, 0.0, 5.0],
        }
    )

    out = process_sidechain(
        index_df=index_df,
        daily_df=daily_df,
        adj_factor_df=adj_factor_df,
        moneyflow_df=moneyflow_df,
        diff_d=0.5,
    )
    row = out.filter(
        (pl.col("code") == "000001.SZ") & (pl.col("trade_date") == date(2026, 1, 7))
    ).row(0, named=True)

    omega = fracdiff_weights(0.5, 48)
    expected_diff = omega[0] * 0.3 + omega[1] * 0.6 + omega[2] * 0.4
    score_hi = float(norm.ppf(0.75))
    score_lo = float(norm.ppf(0.25))

    assert "mf_concentration_diff" in out.columns
    assert "mf_concentration_rank" in out.columns
    assert "mf_main_amount_log" in out.columns
    assert "mf_main_amount_log_diff" in out.columns
    assert "mf_main_amount_log_rank" in out.columns
    assert "amihud_rank" in out.columns
    assert "amihud_impact" not in out.columns
    assert row["mf_concentration_diff"] == pytest.approx(
        expected_diff,
        rel=1e-6,
        abs=1e-6,
    )
    assert row["mf_concentration_rank"] == pytest.approx(score_lo, rel=1e-6, abs=1e-6)
    assert score_hi > row["mf_concentration_rank"]

    expected_main_log = math.log(30.0)
    expected_main_log_diff = (
        omega[0] * math.log(30.0)
        + omega[1] * math.log(60.0)
        + omega[2] * math.log(40.0)
    )

    assert row["mf_main_amount_log"] == pytest.approx(expected_main_log, rel=1e-6, abs=1e-6)
    assert row["mf_main_amount_log_diff"] == pytest.approx(
        expected_main_log_diff,
        rel=1e-6,
        abs=1e-6,
    )
    assert row["mf_main_amount_log_rank"] == pytest.approx(score_lo, rel=1e-6, abs=1e-6)
