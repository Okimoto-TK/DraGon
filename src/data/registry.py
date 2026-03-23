from __future__ import annotations

from typing import Callable
from pathlib import Path
from operator import attrgetter
from pydantic import BaseModel

from config.conf import (
    calendar_path,
    raw
)
from src.data.schemas.raw import (
    TableSchema,
    CALENDAR_SCHEMA,
    RAW_DAILY_SCHEMA,
    RAW_5MIN_SCHEMA,
    RAW_MONEYFLOW_SCHEMA,
    RAW_LIMIT_SCHEMA,
    RAW_ST_SCHEMA,
    RAW_SUSPEND_SCHEMA
)
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
import src.data.storage.raw as raw_storage


class Params(BaseModel):
    api: Callable
    provider: Callable
    reader: Callable
    writer: Callable
    path: Path
    schema: TableSchema
    desc: str = ""


PARAM_MAP = {
    "calendar": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_calendar,
        reader=raw_storage.read_parquet,
        writer=raw_storage.write_parquet,
        path=calendar_path,
        schema=CALENDAR_SCHEMA,
        desc="calendar"
    ),
    "daily": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_daily,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.daily_dir,
        schema=RAW_DAILY_SCHEMA,
        desc="daily"
    ),
    "5min": Params(
        api=attrgetter("mairui"),
        provider=MairuiApi.get_5min,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.r5min_dir,
        schema=RAW_5MIN_SCHEMA,
        desc="5min"
    ),
    "moneyflow": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_moneyflow,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.moneyflow_dir,
        schema=RAW_MONEYFLOW_SCHEMA,
        desc="moneyflow"
    ),
    "limit": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_limit,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.limit_dir,
        schema=RAW_LIMIT_SCHEMA,
        desc="limit"
    ),
    "st": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_st,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.st_dir,
        schema=RAW_ST_SCHEMA,
        desc="st"
    ),
    "suspend": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_suspend,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_by_date,
        path=raw.suspend_dir,
        schema=RAW_SUSPEND_SCHEMA,
        desc="suspend"
    ),
}
