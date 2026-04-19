"""Basic processed-data builders for logical index and filter mask."""
from __future__ import annotations

import config.data as config
import polars as pl
from tqdm import tqdm

from src.data.registry.processor import FEAT_WINDOW, LABEL_WINDOW
from src.data.schemas.processed import PROCESSED_INDEX_SCHEMA, PROCESSED_MASK_SCHEMA
from src.data.validators import validate_table


def process_index(suspend_df: pl.DataFrame, **_kwargs) -> pl.DataFrame:
    """Process suspend data into logical index table."""
    result = (
        suspend_df.filter(pl.col("is_suspend") == False)
        .sort(["code", "trade_date"])
        .with_columns(
            logic_index=pl.int_range(1, pl.len() + 1).over("code").cast(pl.Int32)
        )
        .select(["code", "trade_date", "logic_index"])
    )
    validate_table(result, PROCESSED_INDEX_SCHEMA)
    return result


def process_mask(
    suspend_df: pl.DataFrame,
    namechange_df: pl.DataFrame,
    index_df: pl.DataFrame,
    **_kwargs,
) -> pl.DataFrame:
    """Process suspend and namechange data into filter mask table."""
    nc = namechange_df.with_columns(
        is_st=pl.col("name").str.starts_with("ST")
        | pl.col("name").str.starts_with("*ST")
    )

    df = suspend_df.join(
        nc.select(["code", "trade_date", "is_st"]),
        on=["code", "trade_date"],
        how="full",
        coalesce=True,
    ).with_columns(
        pl.col("is_suspend").fill_null(False),
        pl.col("is_st").fill_null(False),
    )

    df = df.join(
        index_df.select(["code", "trade_date", "logic_index"]),
        on=["code", "trade_date"],
        how="left",
    ).sort(["code", "trade_date"])

    codes = df["code"].unique().to_list()
    results = []
    for code in tqdm(codes, desc="Processing mask", disable=config.debug):
        group = df.filter(pl.col("code") == code)
        n = len(group)
        suspend = group["is_suspend"].to_numpy()
        st = group["is_st"].to_numpy()
        logic_idx = group["logic_index"].to_numpy()

        mask = []
        for i in range(n):
            current_idx = logic_idx[i]
            future_end = min(i + LABEL_WINDOW + 1, n)
            future_suspend = suspend[i:future_end].any()
            future_st = st[i:future_end].any()

            lookback_start = current_idx - FEAT_WINDOW
            lookback_mask = (logic_idx >= lookback_start) & (logic_idx <= current_idx)
            window_st = st[lookback_mask].any()

            mask.append(not future_suspend and not future_st and not window_st)

        results.append(group.with_columns(filter_mask=pl.Series(mask, dtype=pl.Boolean)))

    result = (
        pl.concat(results)
        .filter(pl.col("logic_index").is_not_null())
        .sort(["code", "trade_date"])
        .select(["code", "trade_date", "filter_mask"])
    )
    validate_table(result, PROCESSED_MASK_SCHEMA)
    return result
