"""Raw pipeline parameter registry and API field mappings."""
from __future__ import annotations

from operator import attrgetter

from config.config import raw_path

from src.data.models import RawParams
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.schemas.raw import (
    CALENDAR_SCHEMA,
    RAW_5MIN_SCHEMA,
    RAW_ADJ_FACTOR_SCHEMA,
    RAW_DAILY_SCHEMA,
    RAW_LIMIT_SCHEMA,
    RAW_MONEYFLOW_SCHEMA,
    RAW_NAMECHANGE_SCHEMA,
    RAW_SUSPEND_SCHEMA,
    UNIVERSE_SCHEMA,
)
from src.data.storage.parquet_io import (
    read_parquet,
    read_parquet_schema,
    read_parquets,
    read_parquets_schema,
    write_parquet,
    write_parquets,
)

# === Pipeline Parameter Registry ===

PARAM_MAP: dict[str, RawParams] = {
    "universe": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_universe,
        reader=read_parquet,
        writer=write_parquet,
        sreader=read_parquet_schema,
        path=raw_path.universe_path,
        schema=UNIVERSE_SCHEMA,
        desc="universe",
    ),
    "calendar": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_calendar,
        reader=read_parquet,
        writer=write_parquet,
        sreader=read_parquet_schema,
        path=raw_path.calendar_path,
        schema=CALENDAR_SCHEMA,
        desc="calendar",
    ),
    "daily": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_daily,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.daily_dir,
        schema=RAW_DAILY_SCHEMA,
        desc="daily",
    ),
    "adj_factor": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_adj_factor,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.adj_factor_dir,
        schema=RAW_ADJ_FACTOR_SCHEMA,
        desc="adj_factor",
    ),
    "5min": RawParams(
        api=attrgetter("mairui"),
        provider=MairuiApi.get_5min,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.r5min_dir,
        schema=RAW_5MIN_SCHEMA,
        desc="5min",
    ),
    "moneyflow": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_moneyflow,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.moneyflow_dir,
        schema=RAW_MONEYFLOW_SCHEMA,
        desc="moneyflow",
    ),
    "limit": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_limit,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.limit_dir,
        schema=RAW_LIMIT_SCHEMA,
        desc="limit",
    ),
    "namechange": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_namechange,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.namechange_dir,
        schema=RAW_NAMECHANGE_SCHEMA,
        desc="namechange",
    ),
    "suspend": RawParams(
        api=attrgetter("tushare"),
        provider=TushareApi.get_suspend,
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=raw_path.suspend_dir,
        schema=RAW_SUSPEND_SCHEMA,
        desc="suspend",
    ),
}
