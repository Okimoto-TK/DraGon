"""Label builders for future return and volatility targets."""
from __future__ import annotations

import polars as pl
from tqdm import tqdm

from src.data.registry.processor import LABEL_WINDOW
from src.data.schemas.processed import PROCESSED_LABEL_SCHEMA
from src.data.validators import validate_table

_NUMERIC_EPS = 1e-6


def process_label(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily adjusted prices into ret / rv labels."""
    required_cols = {"high", "low", "open"}
    missing = required_cols.difference(daily_df.columns)
    if missing:
        raise KeyError(f"daily_df must include {sorted(missing)} to build ret/rv labels.")

    tqdm.write("Adjusting prices...")
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        adj_open=pl.col("open") * pl.col("adj_factor"),
        adj_close=pl.col("close") * pl.col("adj_factor"),
        adj_high=pl.col("high") * pl.col("adj_factor"),
        adj_low=pl.col("low") * pl.col("adj_factor"),
    ).sort(["code", "trade_date"])

    df = daily_adj.join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    )

    tqdm.write("Computing ret / rv labels...")

    open_exprs: list[pl.Expr] = []
    rv_terms: list[pl.Expr] = []
    for k in range(1, LABEL_WINDOW + 1):
        open_exprs.append(pl.col("adj_open").shift(-k).over("code").alias(f"_open_{k}"))
        high_k = pl.col("adj_high").shift(-k).over("code")
        low_k = pl.col("adj_low").shift(-k).over("code")
        rv_terms.append((high_k / low_k.clip(lower_bound=_NUMERIC_EPS)).log().pow(2))

    df = df.with_columns(open_exprs)

    future_open_mean = (
        pl.col("_open_2")
        + pl.col("_open_3")
        + pl.col("_open_4")
    ) / 3.0
    entry_open = pl.col("_open_1")
    rv_expr = (sum(rv_terms) / float(LABEL_WINDOW)).sqrt()

    df = df.with_columns([
        (future_open_mean / entry_open.clip(lower_bound=_NUMERIC_EPS)).alias("label_ret"),
        rv_expr.alias("label_rv"),
    ])

    result = df.select([
        "code",
        "trade_date",
        "label_ret",
        "label_rv",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_LABEL_SCHEMA)
    return result
