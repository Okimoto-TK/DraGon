"""Label builders for future path-dependent targets."""
from __future__ import annotations

import polars as pl
from tqdm import tqdm

from src.data.registry.processor import LABEL_WEIGHTS, LABEL_WINDOW
from src.data.schemas.processed import PROCESSED_LABEL_SCHEMA
from src.data.validators import validate_table


def process_label(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily adjusted prices into dense, orthogonal path labels."""
    tqdm.write("Adjusting prices...")
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        adj_open=pl.col("open") * pl.col("adj_factor"),
        adj_close=pl.col("close") * pl.col("adj_factor"),
    ).sort(["code", "trade_date"])

    df = daily_adj.join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    )

    tqdm.write("Computing Dense Orthogonal Labels (S, M, MDD, RV)...")
    next_open_log = pl.col("adj_open").log().shift(-1).over("code")

    weighted_sum = pl.lit(0.0)
    max_ret = pl.lit(-float("inf"))
    max_log_price_so_far = next_open_log
    max_drawdown = pl.lit(0.0)
    sum_sq_daily_ret = pl.lit(0.0)
    prev_log_price = next_open_log

    for k in range(1, LABEL_WINDOW + 1):
        current_log_price = pl.col("adj_close").log().shift(-k).over("code")
        future_cum_ret = current_log_price - next_open_log
        daily_ret = current_log_price - prev_log_price

        weighted_sum = weighted_sum + LABEL_WEIGHTS[k - 1] * future_cum_ret
        max_ret = pl.max_horizontal(max_ret, future_cum_ret)
        max_log_price_so_far = pl.max_horizontal(max_log_price_so_far, current_log_price)
        current_drawdown = max_log_price_so_far - current_log_price
        max_drawdown = pl.max_horizontal(max_drawdown, current_drawdown)
        sum_sq_daily_ret = sum_sq_daily_ret + (daily_ret**2)
        prev_log_price = current_log_price

    df = df.with_columns([
        weighted_sum.alias("label_S"),
        max_ret.alias("label_M"),
        max_drawdown.alias("label_MDD"),
        sum_sq_daily_ret.sqrt().alias("label_RV"),
    ])

    result = df.select([
        "code",
        "trade_date",
        "label_S",
        "label_M",
        "label_MDD",
        "label_RV",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_LABEL_SCHEMA)
    return result
