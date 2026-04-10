"""Raw data pipeline orchestration for fetching, validating, and storing data."""
from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import polars as pl
from config.api import MairuiConfig, TushareConfig
from config.config import DEFAULT_EXCHANGE, DEFAULT_STATUS
from tqdm import tqdm

from src.data.models import Query, TableSchema, ProcessedParams
from src.data.registry.processed import PROCESSED_PARAM_MAP
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.registry.raw import PARAM_MAP
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


class ProcessedPipeline:
    """Orchestrates processing of raw data into processed features."""

    def __init__(self, raw_pipe: RawPipeline) -> None:
        self.raw_pipe = raw_pipe

    @staticmethod
    def _process(
        action: Action | set[Action],
        desc: str,
        params: ProcessedParams,
        **proc_kwargs,
    ) -> None:
        """Standard executor for processing."""
        if "process" in action:
            df = params.processor(**proc_kwargs)
            params.writer(df=df, path=params.path, schema=params.schema, desc=desc)

    @staticmethod
    def _validate(
        params: ProcessedParams,
        desc: str,
    ) -> None:
        """Validate processed data against schema."""
        df = params.sreader(path=params.path, desc=desc)
        validate_table(df=df, schema=params.schema)

    def run(
        self,
        action: Action | set[Action],
        desc: str,
        **extra_kwargs,
    ) -> pl.DataFrame | None:
        """Run the pipeline using the executor bound in ProcessedParams."""
        params = PROCESSED_PARAM_MAP[desc]

        # Fetch raw dependencies (eager loaded)
        raw_data: dict[str, pl.DataFrame | None] = {}
        for kwarg_name, raw_type in params.raw_deps.items():
            raw_data[kwarg_name] = self.raw_pipe.run(
                action={"load"},
                query=Query(desc=raw_type),
            )

        # Fetch lazy dependencies (scanned, no memory overhead)
        from src.data.storage.parquet_io import scan_parquets
        lazy_data: dict[str, pl.LazyFrame | None] = {}
        for kwarg_name, raw_type in params.lazy_deps.items():
            raw_params = PARAM_MAP[raw_type]
            lazy_data[kwarg_name] = scan_parquets(
                path=raw_params.path,
                desc=raw_type,
            )

        # Load Index
        if desc != "index":
            index_params = PROCESSED_PARAM_MAP["index"]
            raw_data["index_df"] = index_params.reader(
                path=index_params.path,
                schema=index_params.schema,
                desc="index",
            )

        kwargs = {**raw_data, **lazy_data, **params.processor_kwargs, **extra_kwargs}

        # 1. Execute processing (either standard or chunked)
        if "process" in action:
            executor = getattr(self, params.proc, self._process)
            executor(
                action=action,
                desc=desc,
                params=params,
                **kwargs,
            )

        # 2. Validate (always reads from final path, no chunking needed)
        if "validate" in action:
            self._validate(params=params, desc=desc)

        # 3. Load (always reads from final path)
        result = None
        if "load" in action:
            result = params.reader(path=params.path, schema=params.schema, desc=desc)

        return result


def load_parquet_files(dir_path: Path, dates: list[str]) -> pl.DataFrame:
    """Helper to load specific parquet files by date."""
    paths = [dir_path / f"{d}.parquet" for d in dates]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in existing])
