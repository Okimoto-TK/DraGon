"""Tushare API client for fetching various stock market data."""
from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Sequence

import config.config as config
import polars as pl
import tushare as ts
from config.api import TushareConfig
from tqdm import tqdm

from src.data.models import Query
from src.data.registry.api import (
    FETCH_FIELD_ADJ_FACTOR,
    FETCH_FIELD_CAL,
    FETCH_FIELD_DAILY,
    FETCH_FIELD_LIMIT,
    FETCH_FIELD_MONEYFLOW,
    FETCH_FIELD_NAMECHANGE,
    FETCH_FIELD_SUSPEND,
    FETCH_FIELD_UNIVERSE,
    FIELD_MAP_ADJ_FACTOR,
    FIELD_MAP_CAL,
    FIELD_MAP_DAILY,
    FIELD_MAP_LIMIT,
    FIELD_MAP_MONEYFLOW,
    FIELD_MAP_NAMECHANGE,
    FIELD_MAP_SUSPEND,
    FIELD_MAP_UNIVERSE,
)
from src.data.providers.base import RawProvider
from src.data.schemas.raw import (
    CALENDAR_SCHEMA,
    RAW_ADJ_FACTOR_SCHEMA,
    RAW_DAILY_SCHEMA,
    RAW_LIMIT_SCHEMA,
    RAW_MONEYFLOW_SCHEMA,
    RAW_NAMECHANGE_SCHEMA,
    RAW_SUSPEND_SCHEMA,
    UNIVERSE_SCHEMA,
    TableSchema,
)
from src.data.types import Map
from src.data.utils.raw import align_df, get_grid, parse_calendar
from src.data.validators import validate_table
from src.utils.log import vlog


class TushareApi(RawProvider):
    """API client for Tushare, supporting multiple data types."""

    def __init__(self, api_config: TushareConfig) -> None:
        """Initialize the Tushare API client.

        Args:
            api_config: Configuration including token, mode, and timeout.
        """
        super().__init__("Tushare")
        vlog(self.api, "Creating Tushare Instance...")

        self.token = api_config.token
        self.timeout = api_config.timeout
        self.mode = api_config.mode
        self.http_url = api_config.http_url
        self.pro = self._get_pro()

    def _get_pro(self) -> ts.pro_api:
        """Initialize and configure the Tushare pro API client."""
        ts.set_token(self.token)
        pro = ts.pro_api(self.token)

        if self.mode == "private":
            self.vlog("Using private mode.")
            pro._DataApi__token = self.token

            if self.http_url is None:
                self.vlog("No private http_url provided.")
                raise Exception("No private http_url provided.")

            pro._DataApi__http_url = self.http_url
        else:
            self.vlog("Using official mode.")

        return pro

    def _get_data(
        self,
        interface: Callable,
        fields: Sequence[str],
        schema: TableSchema,
        mappings: Map,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
        query_interval: bool = False,
    ) -> pl.DataFrame:
        """Fetch data from Tushare API with date-based iteration.

        Args:
            interface: Tushare API method to call.
            fields: List of field names to fetch.
            schema: Target table schema for empty results.
            mappings: Column name mappings from source to canonical.
            query: Query parameters with date range.
            codes: Optional stock codes to filter.
            calendar: Optional trading calendar.
            query_interval: If True, fetch by interval; otherwise by date.

        Returns:
            DataFrame with fetched data.
        """
        self.vlog(f"Fetching {query.desc}...")
        results = []

        if query.desc != "calendar" and calendar is None:
            self.vlog(
                "No calendar provided while daily fetching.", level="ERROR"
            )
            raise ValueError("No calendar provided while daily fetching.")

        if query.desc == "calendar":
            self.vlog("Using start/end_date as default.")
            self.vlog("Requesting through Tushare...")
            _df = pl.from_pandas(
                interface(
                    **{"start_date": query.start_date, "end_date": query.end_date},
                    fields=fields,
                )
            ).rename(mappings)
            results.append(_df)

        elif query_interval:
            self.vlog("Fetching data by interval...")
            df = pl.from_pandas(
                interface(
                    **{
                        "start_date": query.start_date,
                        "end_date": query.end_date,
                    },
                    fields=fields,
                )
            )

            if df.is_empty():
                df = pl.DataFrame(schema=schema.column_names_and_types)
            else:
                df = df.rename(mappings)
                if codes is not None:
                    df = df.filter(
                        pl.col("code").is_in(codes.get_column("code"))
                    )
                df = df.with_columns(
                    pl.col("trade_date").str.to_date(
                        schema.get_column("trade_date").fmt
                    )
                )
            self.vlog(f"{query.start_date} - {query.end_date} received.")
            return df

        else:
            self.vlog("Fetching data by date...")
            dates = parse_calendar(
                calendar, start_date=query.start_date, end_date=query.end_date
            )
            for date in tqdm(
                dates, desc=f"Fetching {query.desc}", disable=config.debug
            ):
                self.vlog(f"Requesting data for {date}...")
                _df = pl.from_pandas(
                    interface(**{"trade_date": date}, fields=fields)
                )

                if _df.is_empty():
                    _df = pl.DataFrame(schema=schema.column_names_and_types)
                else:
                    _df = _df.rename(mappings)
                self.vlog(f"{date} received.")
                results.append(_df)

        if not results:
            self.vlog("No data received, building empty dataframe...", level="WARNING")
            df = pl.DataFrame(schema=schema.column_names_and_types)
        else:
            df = pl.concat(results)
            if codes is not None:
                df = df.filter(pl.col("code").is_in(codes.get_column("code")))
            df = df.with_columns(
                pl.col("trade_date").str.to_date(
                    schema.get_column("trade_date").fmt
                )
            )

        return df

    def get_universe(self, query: Query, **_kwargs) -> pl.DataFrame:
        """Fetch stock universe data from Tushare."""
        self.vlog(f"Fetching {query.desc}...")
        df = pl.from_pandas(
            self.pro.stock_basic(fields=FETCH_FIELD_UNIVERSE)
        )
        df = (
            df.rename(FIELD_MAP_UNIVERSE)
            .cast(UNIVERSE_SCHEMA.column_names_and_types)
            .sort(["code"])
        )
        validate_table(df, UNIVERSE_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_calendar(self, query: Query, **_kwargs) -> pl.DataFrame:
        """Fetch trading calendar from Tushare."""
        df = self._get_data(
            self.pro.trade_cal,
            fields=FETCH_FIELD_CAL,
            schema=CALENDAR_SCHEMA,
            mappings=FIELD_MAP_CAL,
            query=query,
        )
        df = (
            df.filter(pl.col("is_open") == 1)
            .drop("is_open")
            .cast(CALENDAR_SCHEMA.column_names_and_types)
        )
        df = df.sort(["trade_date"])
        validate_table(df, CALENDAR_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_daily(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch daily stock price data."""
        df = (
            self._get_data(
                self.pro.daily,
                fields=FETCH_FIELD_DAILY,
                schema=RAW_DAILY_SCHEMA,
                mappings=FIELD_MAP_DAILY,
                query=query,
                codes=codes,
                calendar=calendar,
            )
            .with_columns(pl.col("vol") * 100)
            .cast(RAW_DAILY_SCHEMA.column_names_and_types)
        )
        if codes is not None and calendar is not None:
            df = align_df(
                df,
                codes=codes,
                calendar=calendar,
                start_date=query.start_date,
                end_date=query.end_date,
            )
        df = df.sort(["code", "trade_date"])
        validate_table(df, RAW_DAILY_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_adj_factor(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch adjustment factor data."""
        df = self._get_data(
            self.pro.adj_factor,
            fields=FETCH_FIELD_ADJ_FACTOR,
            schema=RAW_ADJ_FACTOR_SCHEMA,
            mappings=FIELD_MAP_ADJ_FACTOR,
            query=query,
            codes=codes,
            calendar=calendar,
        ).cast(RAW_ADJ_FACTOR_SCHEMA.column_names_and_types)

        if codes is not None and calendar is not None:
            df = align_df(
                df,
                codes=codes,
                calendar=calendar,
                start_date=query.start_date,
                end_date=query.end_date,
            ).sort(["code", "trade_date"])

        df = df.sort(["code", "trade_date"])
        validate_table(df, RAW_ADJ_FACTOR_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_5min(self, **_kwargs) -> None:
        """Not supported by Tushare."""
        self._raise_not_implemented(inspect.currentframe().f_code.co_name)

    def get_moneyflow(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch daily money flow data."""
        df = self._get_data(
            self.pro.moneyflow,
            fields=FETCH_FIELD_MONEYFLOW,
            schema=RAW_MONEYFLOW_SCHEMA,
            mappings=FIELD_MAP_MONEYFLOW,
            query=query,
            codes=codes,
            calendar=calendar,
        ).cast(RAW_MONEYFLOW_SCHEMA.column_names_and_types)

        if codes is not None and calendar is not None:
            df = align_df(
                df,
                codes=codes,
                calendar=calendar,
                start_date=query.start_date,
                end_date=query.end_date,
            )
        df = df.sort(["code", "trade_date"])
        validate_table(df, RAW_MONEYFLOW_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_limit(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch daily limit price data."""
        df = self._get_data(
            self.pro.stk_limit,
            fields=FETCH_FIELD_LIMIT,
            schema=RAW_LIMIT_SCHEMA,
            mappings=FIELD_MAP_LIMIT,
            query=query,
            codes=codes,
            calendar=calendar,
        ).cast(RAW_LIMIT_SCHEMA.column_names_and_types)

        if codes is not None and calendar is not None:
            df = align_df(
                df,
                codes=codes,
                calendar=calendar,
                start_date=query.start_date,
                end_date=query.end_date,
            )
        df = df.sort(["code", "trade_date"])
        validate_table(df, RAW_LIMIT_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_namechange(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch stock name change history, processed by year."""
        start_year = int(query.start_date[:4])
        end_year = int(query.end_date[:4])
        results = []

        for year in range(start_year, end_year + 1):
            _query = copy.deepcopy(query)
            _query.start_date = max(query.start_date, f"{year}0101")
            _query.end_date = min(query.end_date, f"{year}1231")
            _df = self._get_data(
                self.pro.namechange,
                fields=FETCH_FIELD_NAMECHANGE,
                schema=RAW_NAMECHANGE_SCHEMA,
                mappings=FIELD_MAP_NAMECHANGE,
                query=_query,
                codes=codes,
                calendar=calendar,
                query_interval=True,
            ).sort(["trade_date", "code", "ann_date"]).unique(
                subset=RAW_NAMECHANGE_SCHEMA.primary_key, keep="last"
            ).drop("ann_date")
            results.append(_df)

        if not results:
            df = pl.DataFrame(schema=RAW_NAMECHANGE_SCHEMA.column_names_and_types)
        else:
            df = (
                pl.concat(results)
                .sort(["trade_date", "code"])
                .unique()
            )
        validate_table(df, RAW_NAMECHANGE_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df

    def get_suspend(
        self,
        query: Query,
        codes: pl.DataFrame | None = None,
        calendar: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fetch stock suspension status, processed by year."""
        start_year = int(query.start_date[:4])
        end_year = int(query.end_date[:4])
        results = []

        for year in range(start_year, end_year + 1):
            _query = copy.deepcopy(query)
            _query.start_date = max(query.start_date, f"{year}0101")
            _query.end_date = min(query.end_date, f"{year}1231")
            _df = self._get_data(
                self.pro.suspend_d,
                fields=FETCH_FIELD_SUSPEND,
                schema=RAW_SUSPEND_SCHEMA,
                mappings=FIELD_MAP_SUSPEND,
                query=_query,
                codes=codes,
                calendar=calendar,
                query_interval=True,
            )
            results.append(_df)

        if not results:
            df = pl.DataFrame(schema=RAW_SUSPEND_SCHEMA.column_names_and_types)
        else:
            df = (
                pl.concat(results)
                .sort(["trade_date", "code"])
                .filter(pl.col("suspend_type") == "S")
                .drop("suspend_type")
                .with_columns(is_suspend=pl.lit(True))
                .join(
                    get_grid(
                        codes=codes,
                        calendar=calendar,
                        start_date=query.start_date,
                        end_date=query.end_date,
                    ),
                    on=["code", "trade_date"],
                    how="outer",
                    coalesce=True,
                )
                .fill_null(False)
            )
        validate_table(df, RAW_SUSPEND_SCHEMA)
        self.vlog(f"{query.desc} fetched.")
        return df
