"""Sidechain feature builders derived from daily prices and money flow."""
from __future__ import annotations

import polars as pl

from src.data.processor.utils import fracdiff_expr, normal_rank
from src.data.registry.dataset import WARMUP_BARS
from src.data.schemas.processed import PROCESSED_SIDECHAIN_SCHEMA
from src.data.validators import validate_table

_EPS = 1e-8


def process_sidechain(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    moneyflow_df: pl.DataFrame,
    diff_d: float = 0.5,
    **_kwargs,
) -> pl.DataFrame:
    """Process moneyflow and daily data into sidechain features."""
    if not (0.0 < diff_d < 1.0):
        raise ValueError(f"diff_d must be in (0,1), got {diff_d}")

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
    ]).with_columns(
        (pl.col("velocity_raw") / (pl.col("amount") + _EPS)).alias("amihud_raw")
    )

    group_key = ["trade_date"]
    df = df.with_columns([
        pl.col("gap").map_batches(normal_rank).over(group_key).alias("gap_rank"),
        pl.col("velocity_raw")
        .map_batches(normal_rank)
        .over(group_key)
        .alias("velocity_rank"),
        pl.col("amount")
        .map_batches(normal_rank)
        .over(group_key)
        .alias("amount_rank"),
        pl.col("amihud_raw")
        .map_batches(normal_rank)
        .over(group_key)
        .alias("amihud_rank"),
    ])

    buy_main = pl.col("buy_lg_amount") + pl.col("buy_elg_amount")
    sell_main = pl.col("sell_lg_amount") + pl.col("sell_elg_amount")

    df = df.with_columns([
        ((buy_main - sell_main) / (pl.col("amount") + _EPS)).alias("mf_net_ratio"),
        ((buy_main - sell_main) / (pl.col("amount") + _EPS))
        .map_batches(normal_rank)
        .over(group_key)
        .alias("mf_net_rank"),
        ((buy_main + sell_main) / (pl.col("amount") + _EPS)).alias(
            "mf_concentration"
        ),
        (buy_main + sell_main).alias("mf_main_amount_raw"),
        (buy_main + sell_main + _EPS).log().alias("mf_main_amount_log"),
    ]).with_columns(
        pl.col("mf_concentration")
        .map_batches(normal_rank)
        .over(group_key)
        .alias("mf_concentration_rank"),
        pl.col("mf_main_amount_log")
        .map_batches(normal_rank)
        .over(group_key)
        .alias("mf_main_amount_log_rank"),
    ).with_columns(
        fracdiff_expr(
            col="mf_concentration",
            d=diff_d,
            window=WARMUP_BARS,
            over="code",
        ).alias("mf_concentration_diff"),
        fracdiff_expr(
            col="mf_main_amount_log",
            d=diff_d,
            window=WARMUP_BARS,
            over="code",
        ).alias("mf_main_amount_log_diff"),
    )

    result = df.select([
        "code",
        "trade_date",
        "gap",
        "gap_rank",
        "mf_net_ratio",
        "mf_net_rank",
        "mf_concentration",
        "mf_concentration_diff",
        "mf_concentration_rank",
        "mf_main_amount_log",
        "mf_main_amount_log_diff",
        "mf_main_amount_log_rank",
        "amount_rank",
        "velocity_rank",
        "amihud_rank",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_SIDECHAIN_SCHEMA)
    return result
