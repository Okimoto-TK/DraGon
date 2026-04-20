"""OHLCV-based processed-data builders for macro, mezzo, and micro features."""
from __future__ import annotations

import polars as pl

from src.data.processor.utils import fracdiff_expr
from src.data.registry.dataset import WARMUP_BARS
from src.data.schemas.processed import (
    PROCESSED_MACRO_SCHEMA,
    PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA,
)
from src.data.validators import validate_table

_EPS = 1e-8
_LIMIT_TOL = 1e-4
_FEATURE_COUNT = 11
_AMIHUD_CLAMP = 1e-3
_VOL_FRACDIFF_CLAMP = 10.0


def process_macro(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    diff_d: float = 0.5,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily OHLCV data into macro backbone features."""
    daily_adj = _adjust_ohlcv_prices(daily_df, adj_factor_df)
    limit_adj = _adjust_limits(limit_df, adj_factor_df)

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=daily_adj,
        limit_df=limit_adj,
        freq="macro",
        diff_d=diff_d,
    )

    result = result.rename({f"f{i}": f"mcr_f{i}" for i in range(_FEATURE_COUNT)})

    validate_table(result, PROCESSED_MACRO_SCHEMA)
    return result


def process_micro(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    diff_d: float = 0.5,
    **_kwargs,
) -> pl.DataFrame:
    """Process 5-minute OHLCV data into micro backbone features."""
    min5_indexed = min5_df.sort(["code", "trade_date", "time"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1)
        .over(["code", "trade_date"])
        .cast(pl.Int32)
    ).drop("time")

    min5_adj = _adjust_ohlcv_prices(min5_indexed, adj_factor_df)
    limit_adj = _adjust_limits(limit_df, adj_factor_df)

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min5_adj,
        limit_df=limit_adj,
        freq="micro",
        diff_d=diff_d,
        assume_sorted=True,
    )

    result = result.rename({f"f{i}": f"mic_f{i}" for i in range(_FEATURE_COUNT)})

    validate_table(result, PROCESSED_MICRO_SCHEMA)
    return result


def process_mezzo(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    diff_d: float = 0.5,
    **_kwargs,
) -> pl.DataFrame:
    """Process 30-minute OHLCV data into mezzo backbone features."""
    min30 = _aggregate_30min(min5_df)

    min30_adj = _adjust_ohlcv_prices(min30, adj_factor_df)
    limit_adj = _adjust_limits(limit_df, adj_factor_df)

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min30_adj,
        limit_df=limit_adj,
        freq="mezzo",
        diff_d=diff_d,
        assume_sorted=True,
    )

    result = result.rename({f"f{i}": f"mzo_f{i}" for i in range(_FEATURE_COUNT)})

    validate_table(result, PROCESSED_MEZZO_SCHEMA)
    return result


def _process_ohlcv(
    index_df: pl.DataFrame,
    ohlcv_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    freq: str = "macro",
    diff_d: float = 0.5,
    assume_sorted: bool = False,
) -> pl.DataFrame:
    """Process OHLCV data into backbone features."""
    if not (0.0 < diff_d < 1.0):
        raise ValueError(f"diff_d must be in (0,1), got {diff_d}")

    df = ohlcv_df.join(
        index_df.select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="left",
    ).filter(pl.col("logic_index").is_not_null())

    df = df.join(
        limit_df.select(["code", "trade_date", "up_limit", "down_limit"]),
        on=["code", "trade_date"],
        how="left",
    )

    has_time_index = "time_index" in df.columns
    sort_cols = ["code", "trade_date"] + (["time_index"] if has_time_index else [])
    lag_group_key = ["code", "time_index"] if has_time_index else ["code"]

    if not assume_sorted:
        df = df.sort(sort_cols)

    if freq == "macro":
        f0_expr = pl.col("close").log() - pl.col("close").shift(1).over("code").log()
    else:
        f0_expr = pl.col("close").log() - pl.col("open").log()

    if has_time_index:
        # For intraday branches, compare against the previous 5 sessions'
        # average amount at the same slot, excluding the current bar.
        amount_ma5 = (
            pl.col("amount")
            .rolling_mean(5)
            .shift(1)
            .over(["code", "time_index"])
        )
    else:
        # For daily macro features, compare against the previous 5 trading days'
        # average amount, excluding the current day.
        amount_ma5 = pl.col("amount").rolling_mean(5).shift(1).over("code")
    amount_prev = (
        pl.col("amount").shift(1).over(lag_group_key)
        if has_time_index
        else pl.col("amount").shift(1).over("code")
    )
    up_limit_tol = pl.col("up_limit").abs() * _LIMIT_TOL + 1e-6
    down_limit_tol = pl.col("down_limit").abs() * _LIMIT_TOL + 1e-6
    amihud_expr = f0_expr / (pl.col("amount") + _EPS)

    df = df.with_columns([
        f0_expr.alias("f0_raw"),
        (pl.col("high") / (pl.col("low") + _EPS)).log().alias("f1_raw"),
        pl.when(pl.col("high") == pl.col("low"))
        .then(pl.lit(0.5))
        .otherwise(
            (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
        )
        .clip(0.0, 1.0)
        .alias("f2_raw"),
        (
            (pl.col("close") - pl.col("down_limit"))
            / (pl.col("up_limit") - pl.col("down_limit") + _EPS)
        )
        .clip(0.0, 1.0)
        .alias("f3_raw"),
        (pl.col("amount") / (amount_ma5 + _EPS)).clip(0.0, 10.0).alias("f4_raw"),
        amihud_expr.clip(-_AMIHUD_CLAMP, _AMIHUD_CLAMP).alias("f5_raw"),
        ((pl.col("amount") + _EPS).log() - (amount_prev + _EPS).log()).alias(
            "_vol_ratio_log_raw"
        ),
        (pl.col("high") >= pl.col("up_limit") - up_limit_tol)
        .cast(pl.Int8)
        .alias("_hit_up_raw"),
        (pl.col("low") <= pl.col("down_limit") + down_limit_tol)
        .cast(pl.Int8)
        .alias("_hit_down_raw"),
        ((pl.col("close") - pl.col("up_limit")).abs() <= up_limit_tol)
        .cast(pl.Int8)
        .alias("_close_up_raw"),
        ((pl.col("close") - pl.col("down_limit")).abs() <= down_limit_tol)
        .cast(pl.Int8)
        .alias("_close_down_raw"),
    ])

    if freq == "micro":
        period = 48
        df = df.with_columns(
            step_idx=pl.when(pl.col("time_index") % period == 0)
            .then(pl.lit(period))
            .otherwise(pl.col("time_index") % period)
        )
    elif freq == "mezzo":
        period = 8
        df = df.with_columns(
            step_idx=pl.when(pl.col("time_index") % period == 0)
            .then(pl.lit(period))
            .otherwise(pl.col("time_index") % period)
        )
    else:
        period = 5
        df = df.with_columns(weekday=pl.col("trade_date").dt.weekday())
        df = df.with_columns(
            step_idx=pl.when(pl.col("weekday") <= 5)
            .then(pl.col("weekday"))
            .otherwise(pl.lit(5))
        )

    df = df.with_columns([
        (
            pl.col("_hit_up_raw") * pl.lit(8, dtype=pl.Int8)
            + pl.col("_hit_down_raw") * pl.lit(4, dtype=pl.Int8)
            + pl.col("_close_up_raw") * pl.lit(2, dtype=pl.Int8)
            + pl.col("_close_down_raw")
        ).cast(pl.Int8).alias("f6_raw"),
        pl.col("step_idx").cast(pl.Int8).alias("f7_raw"),
    ])

    ret_fracdiff = fracdiff_expr(
        col="f0_raw",
        d=diff_d,
        window=WARMUP_BARS,
        over=lag_group_key,
    )
    vol_fracdiff = fracdiff_expr(
        col="_vol_ratio_log_raw",
        d=diff_d,
        window=WARMUP_BARS,
        over=lag_group_key,
    )

    df = df.with_columns([
        ret_fracdiff.alias("f8_raw"),
        vol_fracdiff.clip(-_VOL_FRACDIFF_CLAMP, _VOL_FRACDIFF_CLAMP).alias("f9_raw"),
        (ret_fracdiff - vol_fracdiff).clip(-_VOL_FRACDIFF_CLAMP, _VOL_FRACDIFF_CLAMP).alias("f10_raw"),
    ])

    cols_to_drop = [
        "logic_index",
        "up_limit",
        "down_limit",
        "step_idx",
        "weekday",
        "_hit_up_raw",
        "_hit_down_raw",
        "_close_up_raw",
        "_close_down_raw",
        "_vol_ratio_log_raw",
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])

    df = df.rename({f"f{i}_raw": f"f{i}" for i in range(_FEATURE_COUNT)})

    select_cols = (
        ["code", "trade_date"]
        + (["time_index"] if has_time_index else [])
        + [f"f{i}" for i in range(_FEATURE_COUNT)]
    )
    result = df.select([c for c in select_cols if c in df.columns])

    if "time" in result.columns:
        result = result.drop("time")

    return result


def _aggregate_30min(min5_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 5-minute OHLCV bars into 30-minute bars."""
    sorted_df = min5_df.sort(["code", "trade_date", "time"])
    with_bar_idx = sorted_df.with_columns(
        bar_idx=pl.int_range(0, pl.len()).over(["code", "trade_date"]),
    ).with_columns(
        group_idx=(pl.col("bar_idx") // 6).cast(pl.Int32),
    )

    aggregated = with_bar_idx.group_by(
        ["code", "trade_date", "group_idx"],
        maintain_order=True,
    ).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("amount").sum().alias("amount"),
    )

    return aggregated.with_columns(
        time_index=pl.int_range(1, pl.len() + 1)
        .over(["code", "trade_date"])
        .cast(pl.Int32)
    ).drop("group_idx")


def _adjust_ohlcv_prices(
    ohlcv_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply adj_factor once for all OHLC price columns."""
    return ohlcv_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")


def _adjust_limits(
    limit_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply adj_factor to daily limit prices."""
    return limit_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("up_limit") * pl.col("adj_factor")).alias("up_limit"),
        (pl.col("down_limit") * pl.col("adj_factor")).alias("down_limit"),
    ).drop("adj_factor")
