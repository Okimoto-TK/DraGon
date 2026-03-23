from __future__ import annotations

import polars as pl
import tushare as ts
from tqdm import tqdm
from typing import List, Callable, Sequence
import inspect

import config.conf as conf
from config.api import TushareConfig
from src.data.providers.base import RawProvider
from src.data.schemas.raw import (
    TableSchema,
    CALENDAR_SCHEMA,
    RAW_DAILY_SCHEMA,
    RAW_MONEYFLOW_SCHEMA,
    RAW_ST_SCHEMA,
    RAW_LIMIT_SCHEMA,
    RAW_SUSPEND_SCHEMA,
)
from src.data.validators.raw import validate_table
from src.data.providers.api.registry import *
from src.data.utils.raw import parse_calendar
from src.data.types import Map
from src.utils.log import vlog
from src.data.types import Query

class TushareApi(RawProvider):
    def __init__(self, config: TushareConfig):
        super().__init__("Tushare")

        vlog(self.api, "Creating Tushare Instance...")

        self.token = config.token
        self.timeout = config.timeout
        self.mode = config.mode
        self.http_url = config.http_url
        self.pro = self._get_pro()

    def _get_pro(self):
        ts.set_token(self.token)
        pro = ts.pro_api(self.token)

        if self.mode == "private":
            self.vlog(f"Using private mode.")

            pro._DataApi__token = self.token

            if self.http_url is None:
                self.vlog("No private http_url provided.")
                raise Exception("No private http_url provided.")

            pro._DataApi__http_url = self.http_url

        else:
            self.vlog(f"Using official mode.")

        return pro

    # noinspection PyMethodMayBeStatic
    def _get_data(
            self,
            interface: Callable,
            fields: Sequence[str],
            schema: TableSchema,
            mappings: Map,
            query: Query,
            calendar: pl.DataFrame | None = None,
    ):
        self.vlog(f"Fetching {query.desc}...")

        results = []
        if query.trade_date is not None:
            self.vlog(f"trade_date exists, asof-date=trade_date.")
            _df = pl.from_pandas(interface(**{
                "trade_date": query.trade_date,
            }, fields=fields)).rename(mappings)

            results.append(_df)

        else:
            if query.desc is not "calendar" and calendar is None:
                self.vlog("No calendar provided while daily fetching.", level="ERROR")
                raise ValueError("No calendar provided while daily fetching.")
            if query.desc is "calendar":
                self.vlog(f"Using start/end_date as default.")
                self.vlog("Requesting through Tushare...")

                _df = pl.from_pandas(interface(**{
                    "start_date": query.start_date,
                    "end_date": query.end_date,
                }, fields=fields)).rename(mappings)

                results.append(_df)
            else:
                self.vlog("Fetching data by date...")

                dates = parse_calendar(calendar, start_date=query.start_date, end_date=query.end_date)
                for date in tqdm(dates, desc=f"Fetching {query.desc}", disable=conf.debug):
                    self.vlog(f"Requesting data for {date}...")

                    _df = pl.from_pandas(interface(**{
                        "trade_date": date,
                    }, fields=fields)).rename(mappings)

                    self.vlog(f"{date} received.")

                    results.append(_df)

        if len(results) == 0:
            self.vlog(f"No data received, building empty dataframe...", level="WARNING")

            df = pl.DataFrame(schema=schema.column_names_and_types)
        else:
            df = pl.concat(results)

        df = df.with_columns(
            pl.col("trade_date").str.to_date(schema.get_column("date").fmt, strict=conf.debug)
        )

        if query.codes is not None:
            df = df.with_columns(
                pl.col("code").is_in(query.codes)
            )

        return df

    def get_calendar(
            self,
            query: Query
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.trade_cal,
            fields=FETCH_FIELD_CAL,
            schema=CALENDAR_SCHEMA,
            mappings=FIELD_MAP_CAL,
            query=query,
        )
        df = df.rename(FIELD_MAP_CAL).with_columns(
            pl.col("is_open").cast(pl.Boolean)
        )
        validate_table(df, CALENDAR_SCHEMA)

        self.vlog(f"{query.query.desc} built.")
        return df

    def get_daily(
            self,
            query: Query,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.daily,
            fields=FETCH_FIELD_DAILY,
            schema=RAW_DAILY_SCHEMA,
            mappings=FIELD_MAP_DAILY,
            query=query,
            calendar=calendar,
        )

        validate_table(df, RAW_DAILY_SCHEMA)

        self.vlog(f"{query.desc} fetched.")
        return df

    def get_5min(
            self,
            query: Query,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_moneyflow(
            self,
            query: Query,
            calendar: List[str] | None = None
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.moneyflow,
            fields=[

            ],
            schema=RAW_MONEYFLOW_SCHEMA,
            mappings=FIELD_MAP_MONEYFLOW,
            query=query,
            calendar=calendar,
        )

        validate_table(df, RAW_MONEYFLOW_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_limit(
            self,
            query: Query,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.stk_limit,
            fields=FETCH_FIELD_LIMIT,
            schema=RAW_LIMIT_SCHEMA,
            mappings=FIELD_MAP_LIMIT,
            query=query,
            calendar=calendar,
        )

        validate_table(df, RAW_LIMIT_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_st(
            self,
            query: Query,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.stock_st,
            fields=FETCH_FIELD_ST,
            schema=RAW_ST_SCHEMA,
            mappings=FIELD_MAP_ST,
            query=query,
            calendar=calendar,
        )

        df = df.with_columns(
           is_st=pl.lit(True)
        )

        validate_table(df, RAW_ST_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_suspend(
            self,
            query: Query,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.suspend_d,
            fields=FETCH_FIELD_SUSPEND,
            schema=RAW_SUSPEND_SCHEMA,
            mappings=FIELD_MAP_SUSPEND,
            query=query,
            calendar=calendar,
        )

        df = df.filter(pl.col("suspend_type") == "S").drop("suspend_type").with_columns(
            is_suspend=pl.lit(True)
        )

        validate_table(df, RAW_SUSPEND_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df
