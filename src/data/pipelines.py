"""Raw data pipeline orchestration for fetching, validating, and storing data."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import polars as pl
from config.api import MairuiConfig, TushareConfig
from config.config import DEFAULT_EXCHANGE, DEFAULT_STATUS

from src.data.models import Query, TableSchema
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
    """Orchestrates processing of raw data into processed features.

    All storage parameters are bound in ProcessedParams.
    Raw data is fetched via an injected RawPipeline instance.
    """

    def __init__(self, raw_pipe: RawPipeline) -> None:
        """Initialize the processed pipeline.

        Args:
            raw_pipe: A RawPipeline instance to fetch raw data from.
        """
        self.raw_pipe = raw_pipe

    @staticmethod
    def _process(
        processor: Callable,
        writer: Callable,
        path: Path,
        schema: TableSchema,
        desc: str,
        **kwargs,
    ) -> None:
        """Process raw data into features, then write to storage.

        Args:
            processor: Callable that returns processed DataFrame.
            writer: Storage writer function.
            path: Storage path.
            schema: Table schema for validation.
            desc: Description for logging.
            **kwargs: Raw DataFrames passed to processor.
        """
        df = processor(**kwargs)
        writer(df=df, path=path, schema=schema, desc=desc)

    @staticmethod
    def _validate(
        sreader: Callable,
        path: Path,
        schema: TableSchema,
        desc: str,
        **_kwargs,
    ) -> None:
        """Load processed data and validate against schema.

        Args:
            reader: Storage reader function.
            path: Storage path.
            schema: Table schema for validation.
            desc: Description for logging.
        """
        df = sreader(path=path, desc=desc)
        validate_table(df=df, schema=schema)

    def run(
        self,
        action: Action | set[Action],
        desc: str,
    ) -> pl.DataFrame | None:
        """Execute pipeline actions for the given feature type.

        Args:
            action: Single action or set of actions to perform.
            desc: Feature type descriptor (e.g., 'macro', 'mask').

        Returns:
            DataFrame if 'load' action is performed, otherwise None.
        """
        result = None
        params = PROCESSED_PARAM_MAP[desc]

        # Fetch raw dependencies from RawPipeline
        raw_data: dict[str, pl.DataFrame | None] = {}
        for kwarg_name, raw_type in params.raw_deps.items():
            raw_data[kwarg_name] = self.raw_pipe.run(
                action={"load"},
                query=Query(desc=raw_type),
            )

        # Build processor kwargs: raw data + pre-bound kwargs
        proc_kwargs = {**raw_data}

        # Load processed index if needed (skip when processing index itself)
        if desc != "index":
            index_params = PROCESSED_PARAM_MAP["index"]
            proc_kwargs["index_df"] = index_params.reader(
                path=index_params.path,
                schema=index_params.schema,
                desc="index",
            )

        proc_kwargs.update(params.processor_kwargs)
        if "process" in action:
            if params.processor is None:
                raise ValueError(f"No processor bound for {desc!r}")
            self._process(
                processor=params.processor,
                writer=params.writer,
                path=params.path,
                schema=params.schema,
                desc=desc,
                **proc_kwargs,
            )

        if "validate" in action:
            self._validate(
                sreader=params.sreader,
                path=params.path,
                schema=params.schema,
                desc=desc,
            )

        if "load" in action:
            result = params.reader(
                path=params.path,
                schema=params.schema,
                desc=desc,
            )

        return result
