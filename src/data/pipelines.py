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

from config.config import assembled_dir, processed_path, debug
from src.data.assembler.assemble import process_single_stock
from src.data.models import Query, TableSchema, ProcessedParams
from src.data.registry.processed import PROCESSED_PARAM_MAP
from src.data.storage.parquet_io import read_parquet_by_dates
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

    @staticmethod
    def _process_chunk(
        action: Action | set[Action],
        desc: str,
        params: "ProcessedParams",
        **proc_kwargs,
    ) -> None:
        """Chunked executor for high-volume data (micro, mezzo).

        Strategy:
        1. Process in chunks, splitting each chunk by Code and saving to cache.
        2. Merge by iterating Codes (low memory usage) to build final parquet files.
        3. Carry over tail bars from each code for continuity.

        Note: This method only handles 'process' action.
        Validate and Load are handled by the caller (run method).
        """
        from glob import glob
        from datetime import date
        import shutil
        from config.config import cache_dir
        from src.data.registry.processor import CHUNK_DAYS, CHUNK_TAIL_BARS, CHUNK_TAIL_DAYS

        chunk_days = CHUNK_DAYS

        # 1. Identify raw data directories
        raw_5min_dir = PARAM_MAP["5min"].path
        raw_adj_dir = PARAM_MAP["adj_factor"].path
        raw_limit_dir = PARAM_MAP["limit"].path

        # 2. Get available dates
        files = sorted(glob("*.parquet", root_dir=raw_5min_dir, recursive=False))
        dates = [Path(f).stem for f in files]
        if not dates:
            return None

        # 3. Load Index once
        index_params = PROCESSED_PARAM_MAP["index"]
        index_df = index_params.reader(
            path=index_params.path,
            schema=index_params.schema,
            desc="index",
        )

        # 3.5 Setup cache directory
        cache_path = cache_dir / desc
        if cache_path.exists():
            shutil.rmtree(cache_path)

        # State for tail-carry: stores last N bars per code from previous chunk
        tail_5min: pl.DataFrame | None = None
        tail_adj: pl.DataFrame | None = None
        tail_limit: pl.DataFrame | None = None

        all_codes = set()

        # 4. Loop through chunks
        for i in tqdm(range(0, len(dates), chunk_days), desc=f"Processing {desc} (chunked)"):
            current_dates = dates[i: i + chunk_days]
            chunk_dir = cache_path / f"chunk_{i:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Load new chunk
            new_5min = read_parquet_by_dates(raw_5min_dir, current_dates)
            new_adj = read_parquet_by_dates(raw_adj_dir, current_dates)
            new_limit = read_parquet_by_dates(raw_limit_dir, current_dates)

            if new_5min.is_empty():
                continue

            # Concat with tail
            if tail_5min is not None:
                chunk_5min = pl.concat([tail_5min, new_5min])
                chunk_adj = pl.concat([tail_adj, new_adj])
                chunk_limit = pl.concat([tail_limit, new_limit])
            else:
                chunk_5min = new_5min
                chunk_adj = new_adj
                chunk_limit = new_limit

            # Process
            proc_kwargs["min5_df"] = chunk_5min
            proc_kwargs["adj_factor_df"] = chunk_adj
            proc_kwargs["index_df"] = index_df
            proc_kwargs["limit_df"] = chunk_limit

            res = params.processor(**proc_kwargs)

            # Filter to current dates only
            date_objs = [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in current_dates]
            final_res = res.filter(pl.col("trade_date").is_in(date_objs))

            # Write to cache split by Code
            codes_in_chunk = final_res["code"].unique().to_list()
            all_codes.update(codes_in_chunk)

            for code in codes_in_chunk:
                code_df = final_res.filter(pl.col("code") == code)
                code_df.write_parquet(chunk_dir / f"{code}.parquet")

            # Update Tail: slice from chunk by code (not reload from files)
            tail_5min = chunk_5min.sort(["code", "trade_date", "time"]).group_by("code", maintain_order=True).tail(CHUNK_TAIL_BARS)
            tail_adj = chunk_adj.sort(["code", "trade_date"]).group_by("code", maintain_order=True).tail(CHUNK_TAIL_DAYS)
            tail_limit = chunk_limit.sort(["code", "trade_date"]).group_by("code", maintain_order=True).tail(CHUNK_TAIL_DAYS)

            del chunk_5min
            del chunk_adj
            del chunk_limit
            del res
            del final_res
            gc.collect()

        # 5. Merge by Code (Low Memory Strategy)
        # Only merge if process action was requested
        if "process" in action:
            # Clean final directory
            if params.path.exists():
                shutil.rmtree(params.path)
            params.path.mkdir(parents=True, exist_ok=True)

            # Sort codes to process deterministically
            sorted_codes = sorted(list(all_codes))

            for code in tqdm(sorted_codes, desc=f"Merging {desc} by code"):
                # Find all chunk files for this specific code
                chunk_files = sorted(cache_path.glob(f"*/{code}.parquet"))

                if not chunk_files:
                    continue

                # Read and concat only for THIS code (Very small memory footprint)
                dfs = [pl.read_parquet(f) for f in chunk_files]
                merged = pl.concat(dfs)

                # Deduplicate
                merged = merged.unique(
                    subset=["trade_date", "time_index"], keep="last"
                ).sort(["trade_date", "time_index"])

                # Write to final destination
                merged.write_parquet(params.path / f"{code}.parquet")

                del dfs
                del merged

        # 6. Cleanup cache
        if cache_path.exists():
            shutil.rmtree(cache_path)

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

        # Load Index
        if desc != "index":
            index_params = PROCESSED_PARAM_MAP["index"]
            raw_data["index_df"] = index_params.reader(
                path=index_params.path,
                schema=index_params.schema,
                desc="index",
            )

        kwargs = {**raw_data, **params.processor_kwargs, **extra_kwargs}

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


class AssembledPipeline:
    @staticmethod
    def run():
        assembled_dir.mkdir(parents=True, exist_ok=True)
        all_codes = [f.stem for f in processed_path.mask_dir.glob("*.parquet")]
        from tqdm import tqdm
        for code in tqdm(all_codes, desc="Assembling", disable=debug):
            process_single_stock(code)
            gc.collect()
