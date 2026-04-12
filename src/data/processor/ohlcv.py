"""OHLCV-based processed-data builders for macro, mezzo, and micro features."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from src.data.schemas.processed import (
    PROCESSED_MACRO_SCHEMA,
    PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA,
)
from src.data.validators import validate_table

_EPS = 1e-8


def _normal_rank(s: pl.Series) -> pl.Series:
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


def process_macro(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily OHLCV data into macro backbone features."""
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    limit_adj = limit_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("up_limit") * pl.col("adj_factor")).alias("up_limit"),
        (pl.col("down_limit") * pl.col("adj_factor")).alias("down_limit"),
    ).drop("adj_factor")

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=daily_adj,
        limit_df=limit_adj,
        freq="macro",
    )

    for i in range(9):
        result = result.rename({f"f{i}": f"mcr_f{i}"})

    validate_table(result, PROCESSED_MACRO_SCHEMA)
    return result


def process_micro(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process 5-minute OHLCV data into micro backbone features."""
    min5_indexed = min5_df.sort(["code", "trade_date", "time"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1)
        .over(["code", "trade_date"])
        .cast(pl.Int32)
    ).drop("time")

    min5_adj = min5_indexed.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    limit_adj = limit_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("up_limit") * pl.col("adj_factor")).alias("up_limit"),
        (pl.col("down_limit") * pl.col("adj_factor")).alias("down_limit"),
    ).drop("adj_factor")

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min5_adj,
        limit_df=limit_adj,
        freq="micro",
    )

    for i in range(9):
        result = result.rename({f"f{i}": f"mic_f{i}"})

    validate_table(result, PROCESSED_MICRO_SCHEMA)
    return result


def process_mezzo(
    index_df: pl.DataFrame,
    min5_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process 30-minute OHLCV data into mezzo backbone features."""
    min30 = _aggregate_30min(min5_df)

    min30_adj = min30.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("open"),
        (pl.col("high") * pl.col("adj_factor")).alias("high"),
        (pl.col("low") * pl.col("adj_factor")).alias("low"),
        (pl.col("close") * pl.col("adj_factor")).alias("close"),
    ).drop("adj_factor")

    limit_adj = limit_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        (pl.col("up_limit") * pl.col("adj_factor")).alias("up_limit"),
        (pl.col("down_limit") * pl.col("adj_factor")).alias("down_limit"),
    ).drop("adj_factor")

    result = _process_ohlcv(
        index_df=index_df,
        ohlcv_df=min30_adj,
        limit_df=limit_adj,
        freq="mezzo",
    )

    for i in range(9):
        result = result.rename({f"f{i}": f"mzo_f{i}"})

    validate_table(result, PROCESSED_MEZZO_SCHEMA)
    return result


def _process_ohlcv(
    index_df: pl.DataFrame,
    ohlcv_df: pl.DataFrame,
    limit_df: pl.DataFrame,
    freq: str = "macro",
) -> pl.DataFrame:
    """Process OHLCV data into nine backbone features."""
    df = ohlcv_df.join(
        index_df.select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="inner",
    )

    df = df.join(
        limit_df.select(["code", "trade_date", "up_limit", "down_limit"]),
        on=["code", "trade_date"],
        how="left",
    )

    has_time_index = "time_index" in df.columns
    sort_cols = ["code", "trade_date"] + (["time_index"] if has_time_index else [])
    group_key = ["trade_date"] + (["time_index"] if has_time_index else [])

    df = df.sort(sort_cols)

    if freq == "macro":
        f0_expr = pl.col("close").log() - pl.col("close").shift(1).over("code").log()
        f1_expr = (pl.col("amount") + 1).log() - (
            pl.col("amount").shift(1).over("code") + 1
        ).log()
    else:
        f0_expr = pl.col("close").log() - pl.col("open").log()
        f1_expr = (pl.col("amount") + 1).log() - (
            pl.col("amount").shift(1).over(["code", "trade_date"]) + 1
        ).log()

    df = df.with_columns([
        f0_expr.alias("f0_raw"),
        f1_expr.alias("f1_raw"),
        (pl.col("high") / (pl.col("low") + _EPS)).log().alias("f2_raw"),
        pl.when(pl.col("high") == pl.col("low"))
        .then(pl.lit(0.5))
        .otherwise(
            (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
        )
        .clip(0.0, 1.0)
        .alias("f3_raw"),
        (
            (pl.col("close") - pl.col("down_limit"))
            / (pl.col("up_limit") - pl.col("down_limit") + _EPS)
        )
        .clip(0.0, 1.0)
        .alias("f4_raw"),
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
        (pl.col("step_idx") * 2 * np.pi / period).sin().alias("f5_raw"),
        (pl.col("step_idx") * 2 * np.pi / period).cos().alias("f6_raw"),
        pl.col("f0_raw").map_batches(_normal_rank).over(group_key).alias("f7_raw"),
        pl.col("f1_raw").map_batches(_normal_rank).over(group_key).alias("f8_raw"),
    ])

    cols_to_drop = ["logic_index", "up_limit", "down_limit", "step_idx", "weekday"]
    df = df.drop([c for c in cols_to_drop if c in df.columns])

    for i in range(9):
        df = df.rename({f"f{i}_raw": f"f{i}"})

    select_cols = (
        ["code", "trade_date"]
        + (["time_index"] if has_time_index else [])
        + [f"f{i}" for i in range(9)]
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

    aggregated = with_bar_idx.group_by(["code", "trade_date", "group_idx"]).agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("amount").sum().alias("amount"),
    )

    return aggregated.sort(["code", "trade_date", "group_idx"]).with_columns(
        time_index=pl.int_range(1, pl.len() + 1)
        .over(["code", "trade_date"])
        .cast(pl.Int32)
    ).drop("group_idx")
