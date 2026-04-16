"""Sidechain feature builders derived from daily prices and money flow."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from src.data.schemas.processed import PROCESSED_SIDECHAIN_SCHEMA
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


def process_sidechain(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    moneyflow_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process moneyflow and daily data into sidechain features."""
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

    df = daily_adj.join(
        moneyflow_df,
        on=["code", "trade_date"],
        how="inner",
    ).join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    ).sort(["code", "trade_date"])

    df = df.with_columns([
        (
            pl.col("open").log() - pl.col("close").shift(1).over("code").log()
        ).alias("gap"),
        (
            pl.col("close").log() - pl.col("close").shift(1).over("code").log()
        ).alias("velocity_raw"),
        (
            pl.col("amount") / (pl.col("amount").rolling_mean(5).over("code") + _EPS)
        ).alias("amt_surge_raw"),
    ]).with_columns(
        (pl.col("velocity_raw") / (pl.col("amount") + _EPS)).alias("amihud_raw")
    )

    group_key = ["trade_date"]
    df = df.with_columns([
        pl.col("gap").map_batches(_normal_rank).over(group_key).alias("gap_rank"),
        pl.col("velocity_raw")
        .map_batches(_normal_rank)
        .over(group_key)
        .alias("velocity_rank"),
        pl.col("amount")
        .map_batches(_normal_rank)
        .over(group_key)
        .alias("amount_rank"),
        pl.col("amihud_raw")
        .map_batches(_normal_rank)
        .over(group_key)
        .alias("amihud_impact"),
    ])

    buy_main = pl.col("buy_lg_amount") + pl.col("buy_elg_amount")
    sell_main = pl.col("sell_lg_amount") + pl.col("sell_elg_amount")

    df = df.with_columns([
        ((buy_main - sell_main) / (pl.col("amount") + _EPS)).alias("mf_net_ratio"),
        ((buy_main - sell_main) / (pl.col("amount") + _EPS))
        .map_batches(_normal_rank)
        .over(group_key)
        .alias("mf_net_rank"),
        ((buy_main + sell_main) / (pl.col("amount") + _EPS)).alias(
            "mf_concentration"
        ),
    ])

    result = df.select([
        "code",
        "trade_date",
        "gap",
        "gap_rank",
        "mf_net_ratio",
        "mf_net_rank",
        "mf_concentration",
        "amount_rank",
        "velocity_rank",
        "amihud_impact",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_SIDECHAIN_SCHEMA)
    return result
