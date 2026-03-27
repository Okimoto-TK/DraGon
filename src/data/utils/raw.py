import polars as pl
from datetime import datetime
from typing import Dict, Tuple, Sequence

from src.data.schemas.raw import CALENDAR_SCHEMA


def _filter_calendar(df: pl.DataFrame, start_date:str | None = None, end_date: str | None = None) -> pl.DataFrame:
    if start_date is not None:
        df = df.filter(
            pl.col("trade_date") >= datetime.strptime(start_date, CALENDAR_SCHEMA.get_column("trade_date").fmt)
        )
    if end_date is not None:
        df = df.filter(
            pl.col("trade_date") <= datetime.strptime(end_date, CALENDAR_SCHEMA.get_column("trade_date").fmt)
        )
    return df


def parse_calendar(df: pl.DataFrame, start_date: str | None = None, end_date: str | None = None) -> Sequence[str]:
    calendar = (
        _filter_calendar(df, start_date=start_date, end_date=end_date).sort("trade_date").with_columns(
            pl.col("trade_date").dt.to_string(CALENDAR_SCHEMA.get_column("trade_date").fmt)
        )
    )["trade_date"].to_list()
    return calendar


def partition_by(df: pl.DataFrame, by: str | Tuple[str, ...] = "trade_date") -> Dict[Tuple, pl.DataFrame]:
    groups = df.partition_by(by=by, as_dict=True)
    return groups


def get_grid(
        codes: pl.DataFrame,
        calendar: pl.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
):
    grid = (codes.select("code")
            .join(_filter_calendar(calendar, start_date=start_date, end_date=end_date)
            .select("trade_date"), how="cross"))
    return grid


def align_df(
        _df: pl.DataFrame,
        codes: pl.DataFrame,
        calendar: pl.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
):
    grid = get_grid(
        codes=codes,
        calendar=calendar,
        start_date=start_date,
        end_date=end_date,
    )
    df = grid.join(
        _df,
        on=["code", "trade_date"],
        how="left",
    ).sort(["code", "trade_date"])

    return df
