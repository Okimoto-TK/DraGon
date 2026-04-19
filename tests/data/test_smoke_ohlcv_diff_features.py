from __future__ import annotations

from datetime import date, time
import math

import polars as pl
import pytest

from src.data.processor.ohlcv import _aggregate_30min, _process_ohlcv, process_macro
from src.data.processor.utils import fracdiff_weights


def _index_df(days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for code in ["000001.SZ", "000002.SZ"]:
        for idx, d in enumerate(days, start=1):
            rows.append({"code": code, "trade_date": d, "logic_index": idx})
    return pl.DataFrame(rows)


def _limit_df(days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for code in ["000001.SZ", "000002.SZ"]:
        for d in days:
            rows.append(
                {
                    "code": code,
                    "trade_date": d,
                    "up_limit": 9999.0,
                    "down_limit": 0.01,
                }
            )
    return pl.DataFrame(rows)


def test_macro_f8_f10_use_fracdiff_and_renamed_tail() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    index_df = _index_df(days)
    limit_df = _limit_df(days)
    daily_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 4 + ["000002.SZ"] * 4,
            "trade_date": days + days,
            "open": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 18.0, 17.0],
            "high": [10.5, 11.5, 12.5, 13.5, 20.5, 19.5, 18.5, 17.5],
            "low": [9.5, 10.5, 11.5, 12.5, 19.5, 18.5, 17.5, 16.5],
            "close": [10.0, 11.0, 12.0, 13.0, 20.0, 19.0, 18.0, 17.0],
            "amount": [100.0, 110.0, 220.0, 440.0, 100.0, 95.0, 47.5, 23.75],
        }
    )
    adj_factor_df = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 4 + ["000002.SZ"] * 4,
            "trade_date": days + days,
            "adj_factor": [1.0] * 8,
        }
    )

    out = process_macro(
        index_df=index_df,
        daily_df=daily_df,
        adj_factor_df=adj_factor_df,
        limit_df=limit_df,
        diff_d=0.5,
    )
    row = out.filter(
        (pl.col("code") == "000001.SZ") & (pl.col("trade_date") == date(2026, 1, 8))
    ).row(0, named=True)

    omega = fracdiff_weights(0.5, 48)
    ret_d2 = math.log(11.0) - math.log(10.0)
    ret_d3 = math.log(12.0) - math.log(11.0)
    ret_d4 = math.log(13.0) - math.log(12.0)
    expected_f8 = omega[0] * ret_d4 + omega[1] * ret_d3 + omega[2] * ret_d2

    vol_d2 = math.log(110.0) - math.log(100.0)
    vol_d3 = math.log(220.0) - math.log(110.0)
    vol_d4 = math.log(440.0) - math.log(220.0)
    expected_f9 = omega[0] * vol_d4 + omega[1] * vol_d3 + omega[2] * vol_d2

    assert row["mcr_f8"] == pytest.approx(expected_f8, rel=1e-6, abs=1e-6)
    assert row["mcr_f9"] == pytest.approx(expected_f9, rel=1e-6, abs=1e-6)
    assert row["mcr_f10"] == pytest.approx(expected_f8 - expected_f9, abs=1e-6)
    assert "mcr_f11" not in out.columns
    assert "mcr_f12" not in out.columns


def test_intraday_volume_ratio_uses_prev_day_same_slot_with_fracdiff() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    index_df = _index_df(days)
    limit_df = _limit_df(days)

    rows: list[dict[str, object]] = []
    for code in ["000001.SZ", "000002.SZ"]:
        for d in days:
            for t in [1, 2]:
                rows.append(
                    {
                        "code": code,
                        "trade_date": d,
                        "time_index": t,
                        "open": 10.0 if code == "000001.SZ" else 20.0,
                        "high": 11.0 if code == "000001.SZ" else 21.0,
                        "low": 9.0 if code == "000001.SZ" else 19.0,
                        "close": 10.5 if code == "000001.SZ" else 19.5,
                        "amount": 100.0,
                    }
                )
    ohlcv_df = pl.DataFrame(rows)

    amount_map = {
        ("000001.SZ", date(2026, 1, 5), 1): 100.0,
        ("000001.SZ", date(2026, 1, 5), 2): 100.0,
        ("000001.SZ", date(2026, 1, 6), 1): 300.0,
        ("000001.SZ", date(2026, 1, 6), 2): 50.0,
        ("000001.SZ", date(2026, 1, 7), 1): 600.0,
        ("000001.SZ", date(2026, 1, 7), 2): 200.0,
        ("000001.SZ", date(2026, 1, 8), 1): 1200.0,
        ("000001.SZ", date(2026, 1, 8), 2): 800.0,
        ("000002.SZ", date(2026, 1, 5), 1): 200.0,
        ("000002.SZ", date(2026, 1, 5), 2): 100.0,
        ("000002.SZ", date(2026, 1, 6), 1): 100.0,
        ("000002.SZ", date(2026, 1, 6), 2): 200.0,
        ("000002.SZ", date(2026, 1, 7), 1): 50.0,
        ("000002.SZ", date(2026, 1, 7), 2): 50.0,
        ("000002.SZ", date(2026, 1, 8), 1): 25.0,
        ("000002.SZ", date(2026, 1, 8), 2): 25.0,
    }
    ohlcv_df = ohlcv_df.with_columns(
        pl.struct(["code", "trade_date", "time_index"])
        .map_elements(
            lambda s: amount_map[(s["code"], s["trade_date"], s["time_index"])],
            return_dtype=pl.Float64,
        )
        .alias("amount")
    )

    out = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=ohlcv_df,
        limit_df=limit_df,
        freq="micro",
        diff_d=0.5,
    )
    row = out.filter(
        (pl.col("code") == "000001.SZ")
        & (pl.col("trade_date") == date(2026, 1, 8))
        & (pl.col("time_index") == 2)
    ).row(0, named=True)

    omega = fracdiff_weights(0.5, 48)
    vol_d2 = math.log(50.0) - math.log(100.0)
    vol_d3 = math.log(200.0) - math.log(50.0)
    vol_d4 = math.log(800.0) - math.log(200.0)
    expected = omega[0] * vol_d4 + omega[1] * vol_d3 + omega[2] * vol_d2

    assert row["f9"] == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_process_ohlcv_assume_sorted_matches_default_on_sorted_intraday_input() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    index_df = _index_df(days)
    limit_df = _limit_df(days)

    rows: list[dict[str, object]] = []
    for code, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
        for day_idx, d in enumerate(days):
            for slot in range(1, 4):
                rows.append(
                    {
                        "code": code,
                        "trade_date": d,
                        "time_index": slot,
                        "open": base + day_idx + slot * 0.1,
                        "high": base + day_idx + slot * 0.2,
                        "low": base + day_idx + slot * 0.05,
                        "close": base + day_idx + slot * 0.15,
                        "amount": 100.0 + day_idx * 10.0 + slot,
                    }
                )
    ohlcv_df = pl.DataFrame(rows).sort(["code", "trade_date", "time_index"])

    out_default = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=ohlcv_df,
        limit_df=limit_df,
        freq="micro",
        diff_d=0.5,
    )
    out_fast = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=ohlcv_df,
        limit_df=limit_df,
        freq="micro",
        diff_d=0.5,
        assume_sorted=True,
    )

    assert out_fast.equals(out_default)


def test_aggregate_30min_keeps_bar_order_and_amount_sum() -> None:
    bars = pl.DataFrame(
        {
            "code": ["000001.SZ"] * 12,
            "trade_date": [date(2026, 1, 5)] * 12,
            "time": [
                time(9, 35),
                time(9, 40),
                time(9, 45),
                time(9, 50),
                time(9, 55),
                time(10, 0),
                time(10, 5),
                time(10, 10),
                time(10, 15),
                time(10, 20),
                time(10, 25),
                time(10, 30),
            ],
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    out = _aggregate_30min(bars)

    assert out["time_index"].to_list() == [1, 2]
    assert out["open"].to_list() == [10.0, 20.0]
    assert out["close"].to_list() == [15.5, 25.5]
    assert out["high"].to_list() == [16.0, 26.0]
    assert out["low"].to_list() == [9.0, 19.0]
    assert out["amount"].to_list() == [21.0, 210.0]
