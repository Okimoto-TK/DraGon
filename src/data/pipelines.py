from __future__ import annotations

from typing import Callable, Literal
from pathlib import Path
import polars as pl

from config.api import MairuiConfig, TushareConfig
from config.config import DEFAULT_EXCHANGE, DEFAULT_STATUS
from src.data.schemas.raw import TableSchema
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.models import Query
from src.data.registry import PARAM_MAP
from src.data.validators.raw import validate_table
from src.data.types import Action, Exchange, Status


class RawPipeline:
    def __init__(self, no_init=False):
        self.codes = None
        self.calendar = None
        self.tushare = TushareApi(TushareConfig())
        self.mairui = MairuiApi(MairuiConfig())
        if not no_init:
            self.universe = self.run(
                action={"fetch", "load"},
                query=Query(desc="universe")
            )
            self.calendar = self.run(
                action={"fetch", "load"},
                query=Query(desc="calendar")
            )
            self.codes = self._code_filter()

    def _fetch_data(
            self,
            api: Callable,
            provider: Callable,
            writer: Callable,
            path: Path,
            schema: TableSchema,
            query: Query,
            **_kwargs
    ):
        df = provider(api(self), query=query, codes=self.codes, calendar=self.calendar)
        writer(df=df, path=path, schema=schema, desc=query.desc)

    @staticmethod
    def _load_data(
            reader: Callable,
            path: Path,
            schema: TableSchema,
            query: Query,
            **_kwargs
    ):
        df = reader(path=path, schema=schema, desc=query.desc)
        return df

    @staticmethod
    def _validate_date(
            reader: Callable,
            path: Path,
            schema: TableSchema,
            desc: str,
            **_kwargs
    ):
        df = reader(path=path, schema=schema, desc=desc)
        validate_table(df=df, schema=schema)

    def _code_filter(
            self,
            exchange: Exchange | (Exchange, ...) | None = DEFAULT_EXCHANGE,
            status: Status | (Status, ...) | None = DEFAULT_STATUS,
            method: Literal["blacklist", "whitelist"] = "whitelist",
    ):
        if method == "blacklist":
            df = self.universe.filter(
                ~pl.col("exchange").is_in(exchange) & ~pl.col("status").is_in(status)
            )

        else:
            df = self.universe.filter(
                pl.col("exchange").is_in(exchange) & pl.col("status").is_in(status)
            )
        return df

    def run(
            self,
            action: Action | set[Action],
            query: Query,
    ) -> pl.DataFrame | None:
        if "fetch" in action:
            params = PARAM_MAP[query.desc]
            self._fetch_data(query=query, **vars(params))

        if "validate" in action:
            params = PARAM_MAP[query.desc]
            self._validate_date(**vars(params))

        if "load" in action:
            params = PARAM_MAP[query.desc]
            return self._load_data(query=query, **vars(params))
