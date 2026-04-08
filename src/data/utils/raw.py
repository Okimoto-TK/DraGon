"""Raw data utility functions for calendar parsing and DataFrame alignment."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from src.data.schemas.raw import CALENDAR_SCHEMA


def _filter_calendar(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Filter calendar DataFrame by date range."""
    date_fmt = CALENDAR_SCHEMA.get_column("trade_date").fmt
    if start_date is not None:
        df = df.filter(pl.col("trade_date") >= datetime.strptime(start_date, date_fmt))
    if end_date is not None:
        df = df.filter(pl.col("trade_date") <= datetime.strptime(end_date, date_fmt))
    return df


def parse_calendar(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Sequence[str]:
    """Extract sorted trade dates from calendar as formatted strings."""
    date_fmt = CALENDAR_SCHEMA.get_column("trade_date").fmt
    calendar = (
        _filter_calendar(df, start_date=start_date, end_date=end_date)
        .sort("trade_date")
        .with_columns(pl.col("trade_date").dt.to_string(date_fmt))
    )["trade_date"].to_list()
    return calendar


def partition_by(
    df: pl.DataFrame,
    by: str | tuple[str, ...] = "trade_date",
) -> dict[tuple, pl.DataFrame]:
    """Split DataFrame into a dict of DataFrames grouped by columns."""
    return df.partition_by(by=by, as_dict=True)


def get_grid(
    codes: pl.DataFrame,
    calendar: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Generate a full cross-join grid of codes and calendar dates."""
    grid = codes.select("code").join(
        _filter_calendar(calendar, start_date=start_date, end_date=end_date).select(
            "trade_date"
        ),
        how="cross",
    )
    return grid


def align_df(
    df: pl.DataFrame,
    codes: pl.DataFrame,
    calendar: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Align DataFrame to full grid of codes x calendar, filling missing rows."""
    grid = get_grid(
        codes=codes,
        calendar=calendar,
        start_date=start_date,
        end_date=end_date,
    )
    df = grid.join(df, on=["code", "trade_date"], how="left").sort(
        ["code", "trade_date"]
    )
    return df
