from __future__ import annotations

import polars as pl
import tushare as ts
from tqdm import tqdm
from typing import List, Callable, Sequence
import inspect

import config.conf as conf
from config.api import TushareConfig
from src.data.providers.base import RawProvider, validate_query_args
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
from src.data.providers.api.mapping import (
    FIELD_MAP_CAL,
    FIELD_MAP_DAILY,
    FIELD_MAP_MONEYFLOW,
    FIELD_MAP_ST,
    FIELD_MAP_LIMIT,
    FIELD_MAP_SUSPEND
)
from src.data.utils.raw import parse_calendar
from src.data.types import Map
from src.utils.log import vlog


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
        desc: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
        calendar: pl.DataFrame | None = None,
    ):
        validate_query_args(trade_date=trade_date, start_date=start_date, end_date=end_date, codes=codes)

        self.vlog(f"Fetching {desc}...")

        results = []
        if trade_date is not None:
            self.vlog(f"trade_date exists, asof-date=trade_date.")
            _df = pl.from_pandas(interface(**{
                "trade_date": trade_date,
            }, fields=fields)).rename(mappings)

            results.append(_df)

        else:
            if desc is not "calendar" and calendar is None:
                self.vlog("No calendar provided while daily fetching.", level="ERROR")
                raise ValueError("No calendar provided while daily fetching.")
            if desc is "calendar":
                self.vlog(f"Using start/end_date as default.")
                self.vlog("Requesting through Tushare...")

                _df = pl.from_pandas(interface(**{
                    "start_date": start_date,
                    "end_date": end_date,
                }, fields=fields)).rename(mappings)

                results.append(_df)
            else:
                self.vlog("Fetching data by date...")

                dates = parse_calendar(calendar, start_date=start_date, end_date=end_date)
                for date in tqdm(dates, desc=f"Fetching {desc}", disable=conf.debug):
                    self.vlog(f"Requesting data for {date}...")

                    _df = pl.from_pandas(interface(**{
                        "trade_date": date,
                    }, fields=fields)).rename(mappings)

                    self.vlog(f"{date} received.")

                    results.append(_df)

        if len(results) == 0:
            self.vlog(f"No data received, building empty dataframe...", level="Warning")

            df = pl.DataFrame(schema=schema.column_names_and_types)
        else:
            df = pl.concat(results)

        df = df.with_columns(
            pl.col("trade_date").str.to_date(schema.get_column("date").fmt, strict=conf.debug)
        )

        return df

    def get_calendar(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.daily,
            fields=[
                "cal_date",
                "is_open",
            ],
            schema=CALENDAR_SCHEMA,
            mappings=FIELD_MAP_CAL,
            desc="calendar",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
        )
        df = df.rename(FIELD_MAP_CAL).with_columns(
            pl.col("is_open").cast(pl.Boolean)
        )
        validate_table(df, CALENDAR_SCHEMA)

        self.vlog(f"Calendar built.")
        return df

    def get_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.daily,
            fields=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol"
            ],
            schema=RAW_DAILY_SCHEMA,
            mappings=FIELD_MAP_DAILY,
            desc="Daily Candles",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        validate_table(df, RAW_DAILY_SCHEMA)

        self.vlog("Daily Fetched.")
        return df

    def get_5min(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
    ) -> None:
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_moneyflow(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: List[str] | None = None) -> pl.DataFrame:
        df = self._get_data(
            self.pro.moneyflow,
            fields=[
                "ts_code",
                "trade_date",
                "buy_sm_vol",
                "buy_sm_amount",
                "sell_sm_vol",
                "sell_sm_amount",
                "buy_md_vol",
                "buy_md_amount",
                "sell_md_vol",
                "sell_md_amount",
                "buy_lg_vol",
                "buy_lg_amount",
                "sell_lg_vol",
                "sell_lg_amount",
                "buy_elg_vol",
                "buy_elg_amount",
                "sell_elg_vol",
                "sell_elg_amount",
                "net_mf_vol",
                "net_mf_amount"
            ],
            schema=RAW_MONEYFLOW_SCHEMA,
            mappings=FIELD_MAP_MONEYFLOW,
            desc="Money Flows",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        validate_table(df, RAW_MONEYFLOW_SCHEMA)
        self.vlog("Moneyflow fetched.")
        return df

    def get_limit(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.stk_limit,
            fields=[
                "trade_date",
                "ts_code",
                "up_limit",
                "down_limit"
            ],
            schema=RAW_LIMIT_SCHEMA,
            mappings=FIELD_MAP_LIMIT,
            desc="Price Limits",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        validate_table(df, RAW_LIMIT_SCHEMA)
        self.vlog("Limit Fetched.")
        return df

    def get_st(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.stock_st,
            fields=[
                "ts_code",
                "trade_date",
            ],
            schema=RAW_ST_SCHEMA,
            mappings=FIELD_MAP_ST,
            desc="ST Lists",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        df = df.with_columns(
           is_st=pl.lit(True)
        )

        validate_table(df, RAW_ST_SCHEMA)
        self.vlog("St-List Fetched.")
        return df

    def get_suspend(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        df = self._get_data(
            self.pro.suspend_d,
            fields=[
                "ts_code",
                "trade_date",
                "suspend_type"
            ],
            schema=RAW_SUSPEND_SCHEMA,
            mappings=FIELD_MAP_SUSPEND,
            desc="Suspend Lists",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        df = df.filter(pl.col("suspend_type") == "S").drop("suspend_type").with_columns(
            is_suspend=pl.lit(True)
        )

        validate_table(df, RAW_SUSPEND_SCHEMA)
        self.vlog("Suspend-List Fetched.")
        return df
