"""Label builders for future path-dependent targets."""
from __future__ import annotations

import polars as pl
from tqdm import tqdm

from src.data.registry.processor import LABEL_WEIGHTS, LABEL_WINDOW, PERSIST_TAU
from src.data.schemas.processed import PROCESSED_LABEL_SCHEMA
from src.data.validators import validate_table

_ROBUST_Z_EPS = 1e-6


def process_label(
    index_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    adj_factor_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process daily adjusted prices into Edge / Persist / DownRisk labels."""
    if "low" not in daily_df.columns:
        raise KeyError("daily_df must include 'low' to build label_DownRisk from adjusted low path.")

    tqdm.write("Adjusting prices...")
    daily_adj = daily_df.join(
        adj_factor_df.select(["code", "trade_date", "adj_factor"]),
        on=["code", "trade_date"],
        how="left",
    ).with_columns(
        adj_open=pl.col("open") * pl.col("adj_factor"),
        adj_close=pl.col("close") * pl.col("adj_factor"),
        adj_low=pl.col("low") * pl.col("adj_factor"),
    ).sort(["code", "trade_date"])

    df = daily_adj.join(
        index_df.select(["code", "trade_date"]),
        on=["code", "trade_date"],
        how="inner",
    )

    tqdm.write("Computing Edge / Persist / DownRisk labels...")
    df = df.with_columns(
        entry_open=pl.col("adj_open").shift(-1).over("code"),
        entry_log=pl.col("adj_open").log().shift(-1).over("code"),
    )

    prefix_log_sum = pl.lit(0.0)
    q_tilde_exprs: list[pl.Expr] = []
    close_exprs: list[pl.Expr] = []
    low_exprs: list[pl.Expr] = []

    for k in range(1, LABEL_WINDOW + 1):
        close_log_k = pl.col("adj_close").log().shift(-k).over("code")
        prefix_log_sum = prefix_log_sum + (close_log_k - pl.col("entry_log"))
        q_tilde_exprs.append((prefix_log_sum / float(k)).alias(f"_q_tilde_{k}"))
        close_exprs.append(pl.col("adj_close").shift(-k).over("code").alias(f"_close_{k}"))
        low_exprs.append(pl.col("adj_low").shift(-k).over("code").alias(f"_low_{k}"))

    df = df.with_columns(q_tilde_exprs + close_exprs + low_exprs)

    median_exprs: list[pl.Expr] = []
    for k in range(1, LABEL_WINDOW + 1):
        median_exprs.append(pl.col(f"_q_tilde_{k}").median().over("trade_date").alias(f"_q_med_{k}"))
    df = df.with_columns(median_exprs)

    mad_exprs: list[pl.Expr] = []
    for k in range(1, LABEL_WINDOW + 1):
        mad_exprs.append(
            (pl.col(f"_q_tilde_{k}") - pl.col(f"_q_med_{k}")).abs().median().over("trade_date").alias(f"_q_mad_{k}")
        )
    df = df.with_columns(mad_exprs)

    robust_q_exprs: list[pl.Expr] = []
    edge_expr = pl.lit(0.0)
    persist_expr = pl.lit(0.0)
    for k, weight in enumerate(LABEL_WEIGHTS, start=1):
        q_col_name = f"_q_{k}"
        q_expr = (
            (pl.col(f"_q_tilde_{k}") - pl.col(f"_q_med_{k}"))
            / (1.4826 * pl.col(f"_q_mad_{k}") + _ROBUST_Z_EPS)
        ).alias(q_col_name)
        robust_q_exprs.append(q_expr)
        q_col = pl.col(q_col_name)
        edge_expr = edge_expr + float(weight) * q_col
        persist_expr = persist_expr + (1.0 / (1.0 + (-(q_col / PERSIST_TAU)).exp()))

    peak_expr = pl.col("entry_open")
    downrisk_expr = pl.lit(0.0)
    for k in range(1, LABEL_WINDOW + 1):
        low_k = pl.col(f"_low_{k}")
        drawdown_k = ((peak_expr - low_k) / peak_expr.clip(lower_bound=_ROBUST_Z_EPS)).clip(lower_bound=0.0)
        downrisk_expr = pl.max_horizontal(downrisk_expr, drawdown_k)
        peak_expr = pl.max_horizontal(peak_expr, pl.col(f"_close_{k}"))

    df = df.with_columns(robust_q_exprs)
    df = df.with_columns(
        edge_expr.alias("label_Edge"),
        (persist_expr / float(LABEL_WINDOW)).alias("label_Persist"),
        downrisk_expr.alias("label_DownRisk"),
    )

    result = df.select([
        "code",
        "trade_date",
        "label_Edge",
        "label_Persist",
        "label_DownRisk",
    ]).sort(["code", "trade_date"])

    validate_table(result, PROCESSED_LABEL_SCHEMA)
    return result
