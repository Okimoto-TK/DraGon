"""Raw data pipeline orchestration for fetching, validating, and storing data."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import polars as pl
from config.api import MairuiConfig, TushareConfig
from config.config import DEFAULT_EXCHANGE, DEFAULT_STATUS

from src.data.models import Query, TableSchema
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.registry import PARAM_MAP
from src.data.types import Action, Exchange, Status
from src.data.validators import validate_table


class RawPipeline:
    """Orchestrates data fetching, validation, and storage for raw data tables.

    On initialization, loads the stock universe and trading calendar,
    then filters codes based on exchange and status criteria.
    """

    def __init__(self, no_init: bool = False) -> None:
        """Initialize the pipeline with API clients and reference data.

        Args:
            no_init: Skip loading universe and calendar data if True.
        """
        self.codes: pl.DataFrame | None = None
        self.calendar: pl.DataFrame | None = None
        self.tushare = TushareApi(TushareConfig())
        self.mairui = MairuiApi(MairuiConfig())

        if not no_init:
            self.universe = self.run(
                action={"fetch", "load"},
                query=Query(desc="universe"),
            )
            self.calendar = self.run(
                action={"fetch", "load"},
                query=Query(desc="calendar"),
            )
            self.codes = self._code_filter()

    def _fetch_data(
        self,
        api: Callable,
        provider: Callable,
        writer: Callable,
        path,
        schema: TableSchema,
        query: Query,
        **_kwargs,
    ) -> None:
        """Fetch data from API, then write to storage."""
        df = provider(
            api(self), query=query, codes=self.codes, calendar=self.calendar
        )
        writer(df=df, path=path, schema=schema, desc=query.desc)

    @staticmethod
    def _load_data(
        reader: Callable,
        path,
        schema: TableSchema,
        query: Query,
        **_kwargs,
    ) -> pl.DataFrame:
        """Load data from storage and return."""
        return reader(path=path, schema=schema, desc=query.desc)

    @staticmethod
    def _validate_data(
        sreader: Callable,
        path,
        schema: TableSchema,
        desc: str,
        **_kwargs,
    ) -> None:
        """Load data and validate against schema."""
        df = sreader(path=path, desc=desc)
        validate_table(df=df, schema=schema)

    def _code_filter(
        self,
        exchange: Exchange | tuple[Exchange, ...] | None = DEFAULT_EXCHANGE,
        status: Status | tuple[Status, ...] | None = DEFAULT_STATUS,
        method: Literal["blacklist", "whitelist"] = "whitelist",
    ) -> pl.DataFrame:
        """Filter stock codes by exchange and status.

        Args:
            exchange: Exchange(s) to filter on.
            status: Status code(s) to filter on.
            method: "whitelist" keeps matching codes, "blacklist" removes them.

        Returns:
            Filtered DataFrame with matching stock codes.
        """
        if method == "blacklist":
            df = self.universe.filter(
                ~pl.col("exchange").is_in(exchange)
                | ~pl.col("status").is_in(status)
            )
        else:
            df = self.universe.filter(
                pl.col("exchange").is_in(exchange)
                & pl.col("status").is_in(status)
            )
        return df

    def run(
        self,
        action: Action | set[Action],
        query: Query,
    ) -> pl.DataFrame | None:
        """Execute pipeline actions for the given query.

        Args:
            action: Single action or set of actions to perform.
            query: Query parameters specifying the data type and date range.

        Returns:
            DataFrame if "load" action is performed, otherwise None.
        """
        result = None

        if "fetch" in action:
            params = PARAM_MAP[query.desc]
            self._fetch_data(query=query, **vars(params))

        if "validate" in action:
            params = PARAM_MAP[query.desc]
            self._validate_data(**vars(params))

        if "load" in action:
            params = PARAM_MAP[query.desc]
            result = self._load_data(query=query, **vars(params))

        return result
