import polars as pl
from typing import List, Dict, Tuple


def parse_calendar(df: pl.DataFrame, start_date: str | None = None, end_date: str | None = None) -> List[str]:
    if start_date is not None:
        df = df.filter(
            pl.col("trade_date") >= start_date
        )
    if end_date is not None:
        df = df.filter(
            pl.col("trade_date") <= end_date
        )

    calendar = (
        df.filter(pl.col("is_open") == True)
        .select("trade_date")
        .sort("trade_date")
    )["trade_date"].to_list()
    return calendar


def partition_by(df: pl.DataFrame, by: str = "date") -> Dict[Tuple, pl.DataFrame]:
    groups = df.partition_by(by=by, as_dict=True)
    return groups
