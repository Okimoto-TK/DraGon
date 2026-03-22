from __future__ import annotations

from typing import Sequence

import polars as pl
import tushare as ts
from tqdm import tqdm
from typing import List, Callable

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
from src.data.providers.api.mapping import (
    FIELD_MAP_CAL,
    FIELD_MAP_DAILY,
    FIELD_MAP_MONEYFLOW,
    FIELD_MAP_ST,
    FIELD_MAP_LIMIT,
    FIELD_MAP_SUSPEND
)
from src.data.utils.raw import parse_calendar
from src.data.types import DailyDF, Map


class TushareApi(RawProvider):
    def __init__(self, config:TushareConfig | None = None):
        self.token = config.token
        self.timeout = config.timeout
        self.mode = config.mode
        self.http_url = config.http_url
        self.pro = self._get_pro()

    def _get_pro(self):
        ts.set_token(self.token)
        pro = ts.pro_api(self.token)

        if self.mode == "private":
            pro._DataApi__token = self.token
            pro._DataApi__http_url = self.http_url

        return pro

    def get_calendar(self) -> pl.DataFrame:
        df = pl.from_pandas(self.pro.trade_cal(**{
            "exchange": "",
            "cal_date": "",
            "start_date": "",
            "end_date": "",
            "is_open": "",
            "limit": "",
            "offset": ""
        }, fields=[
            "cal_date",
            "is_open",
        ]))
        df.rename(FIELD_MAP_CAL)
        df = df.with_columns(
            pl.col("is_open").cast(pl.Boolean)
        )
        validate_table(df, CALENDAR_SCHEMA)
        return df

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
        self._validate_query_args(trade_date=trade_date, start_date=start_date, end_date=end_date, codes=codes)
        results = {}
        if trade_date is not None:
            df = pl.from_pandas(interface(**{
                "trade_date": trade_date,
            }, fields=fields))

            df.rename(mappings)
            validate_table(df, schema)
            results[trade_date] = df

        else:
            dates = parse_calendar(calendar, start_date=start_date, end_date=end_date)
            for date in tqdm(dates, desc=f"Fetching {desc}"):
                df = pl.from_pandas(interface(**{
                    "trade_date": date,
                }, fields=fields))

                df.rename(mappings)
                validate_table(df, schema)
                results[(date,)] = df

        return results

    def get_daily(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        codes: Sequence[str] | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> DailyDF:
        return self._get_data(
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

    def get_moneyflow(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: List[str] | None = None) -> DailyDF:
        return self._get_data(
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

    def get_limit(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> DailyDF:
        return self._get_data(
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

    def get_st(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> DailyDF:
        results = self._get_data(
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

        for (date, ), df in results.items():
            results[date] = df.with_columns(
                is_st=pl.lit(True)
            )

        return results

    def get_suspend(
            self,
            trade_date: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            codes: Sequence[str] | None = None,
            calendar: pl.DataFrame | None = None,
    ) -> DailyDF:
        results = self._get_data(
            self.pro.suspend_d,
            fields=[
                "ts_code",
                "trade_date",
                "suspend_type"
            ],
            schema=RAW_ST_SCHEMA,
            mappings=FIELD_MAP_ST,
            desc="Suspend Lists",
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            codes=codes,
            calendar=calendar,
        )

        for (date, ), df in results.items():
            df = df.filter(pl.col("suspend_type") == "S").drop("suspend_type")
            results[date] = df.with_columns(
                is_suspend=pl.lit(True)
            )

        return results
