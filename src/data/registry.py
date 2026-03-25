from __future__ import annotations

from operator import attrgetter

from config.config import raw_path
from src.data.schemas.raw import (
    UNIVERSE_SCHEMA,
    CALENDAR_SCHEMA,
    RAW_DAILY_SCHEMA,
    RAW_ADJ_FACTOR_SCHEMA,
    RAW_5MIN_SCHEMA,
    RAW_MONEYFLOW_SCHEMA,
    RAW_LIMIT_SCHEMA,
    RAW_ST_SCHEMA,
    RAW_SUSPEND_SCHEMA
)
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
import src.data.storage.parquet_io as raw_storage
from src.data.models import Params


PARAM_MAP = {
    "universe": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_universe,
        reader=raw_storage.read_parquet,
        writer=raw_storage.write_parquet,
        path=raw_path.universe_path,
        schema=UNIVERSE_SCHEMA,
        desc="universe",
    ),
    "calendar": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_calendar,
        reader=raw_storage.read_parquet,
        writer=raw_storage.write_parquet,
        path=raw_path.calendar_path,
        schema=CALENDAR_SCHEMA,
        desc="calendar"
    ),
    "daily": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_daily,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.daily_dir,
        schema=RAW_DAILY_SCHEMA,
        desc="daily"
    ),
    "adj_factor": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_adj_factor,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.adj_factor_dir,
        schema=RAW_ADJ_FACTOR_SCHEMA,
        desc="adj_factor"
    ),
    "5min": Params(
        api=attrgetter("mairui"),
        provider=MairuiApi.get_5min,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.r5min_dir,
        schema=RAW_5MIN_SCHEMA,
        desc="5min"
    ),
    "moneyflow": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_moneyflow,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.moneyflow_dir,
        schema=RAW_MONEYFLOW_SCHEMA,
        desc="moneyflow"
    ),
    "limit": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_limit,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.limit_dir,
        schema=RAW_LIMIT_SCHEMA,
        desc="limit"
    ),
    "st": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_st,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.st_dir,
        schema=RAW_ST_SCHEMA,
        desc="st"
    ),
    "suspend": Params(
        api=attrgetter("tushare"),
        provider=TushareApi.get_suspend,
        reader=raw_storage.read_parquets,
        writer=raw_storage.write_parquets,
        path=raw_path.suspend_dir,
        schema=RAW_SUSPEND_SCHEMA,
        desc="suspend"
    ),
}
