"""Raw data pipeline orchestration for fetching, validating, and storing data."""
from __future__ import annotations

import gc
import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from config.api import MairuiConfig, TushareConfig
from config.data import DEFAULT_EXCHANGE, DEFAULT_STATUS
from tqdm import tqdm

from config.data import assembled_dir, latest_dir
from src.data.assembler.assemble import (
    LABEL_COLS,
    MACRO_FEATURES,
    MACRO_FLOAT_FEATURES,
    MACRO_INT8_FEATURES,
    MEZZO_BARS_PER_DAY,
    MEZZO_FEATURES,
    MEZZO_FLOAT_FEATURES,
    MEZZO_INT8_FEATURES,
    MICRO_BARS_PER_DAY,
    MICRO_FEATURES,
    MICRO_FLOAT_FEATURES,
    MICRO_INT8_FEATURES,
    SIDECHAIN_FEATURES,
    _LABEL_NAMES_ARRAY,
    _build_packed_payload,
    _compute_sample_valid,
    _list_finite_expr,
    _list_non_null_expr,
    _safe_list_int8_expr,
    _scalar_finite_expr,
    assemble_all,
)
from src.data.models import Query, TableSchema, ProcessedParams
from src.data.registry.dataset import (
    MACRO_LOOKBACK,
    MEZZO_LOOKBACK,
    MICRO_LOOKBACK,
    WARMUP_BARS,
)
from src.data.registry.processed import PROCESSED_PARAM_MAP
from src.data.storage.parquet_io import read_parquet_by_dates
from src.data.providers.api.mairui import MairuiApi
from src.data.providers.api.tushare import TushareApi
from src.data.registry.raw import PARAM_MAP
from src.data.types import Action, Exchange, Status
from src.data.validators import validate_table


_RAW_UPDATE_FIELDS = tuple(
    name
    for name, params in PARAM_MAP.items()
    if "trade_date" in params.schema.column_names
)

_MACRO_WINDOW_DAYS = MACRO_LOOKBACK + WARMUP_BARS
_MEZZO_WINDOW_DAYS = MEZZO_LOOKBACK // MEZZO_BARS_PER_DAY + (
    WARMUP_BARS + MEZZO_BARS_PER_DAY - 1
) // MEZZO_BARS_PER_DAY
_MICRO_WINDOW_DAYS = MICRO_LOOKBACK // MICRO_BARS_PER_DAY + (
    WARMUP_BARS + MICRO_BARS_PER_DAY - 1
) // MICRO_BARS_PER_DAY
_STEP_WARMUP_DAYS = 5
_LATEST_LOGIC_DAYS = _MACRO_WINDOW_DAYS + _STEP_WARMUP_DAYS
_LATEST_PAYLOAD_KEYS = (
    "date",
    "label",
    "macro",
    "sidechain",
    "mezzo",
    "micro",
    "macro_i8",
    "mezzo_i8",
    "micro_i8",
)
_LATEST_SLICE_STATE: dict[str, object] = {}


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

    @staticmethod
    def _as_date(value: str) -> date:
        return datetime.strptime(value, "%Y%m%d").date()

    @staticmethod
    def _as_yyyymmdd(value: date) -> str:
        return value.strftime("%Y%m%d")

    @staticmethod
    def _latest_partition_date(path: Path) -> date | None:
        if not path.exists() or not path.is_dir():
            return None
        stems = [
            p.stem
            for p in path.glob("*.parquet")
            if len(p.stem) == 8 and p.stem.isdigit()
        ]
        if not stems:
            return None
        return datetime.strptime(max(stems), "%Y%m%d").date()

    @staticmethod
    def _earliest_partition_date(path: Path) -> date | None:
        if not path.exists() or not path.is_dir():
            return None
        stems = [
            p.stem
            for p in path.glob("*.parquet")
            if len(p.stem) == 8 and p.stem.isdigit()
        ]
        if not stems:
            return None
        return datetime.strptime(min(stems), "%Y%m%d").date()

    @staticmethod
    def _latest_trade_date_from_table(path: Path) -> date | None:
        if not path.exists():
            return None
        max_df = pl.scan_parquet(path).select(
            pl.max("trade_date").alias("max_date")
        ).collect()
        if max_df.is_empty() or max_df["max_date"][0] is None:
            return None
        return max_df["max_date"][0]

    def natural_end_date(self) -> str:
        """Return the raw update end date using UTC+8 18:00 cutoff."""
        now_utc8 = datetime.now(timezone(timedelta(hours=8)))
        natural_end = (
            now_utc8.date()
            if now_utc8.time() > time(18, 0, 0)
            else now_utc8.date() - timedelta(days=1)
        )
        return self._as_yyyymmdd(natural_end)

    def previous_trading_day(self, end_date: str | None = None) -> str:
        """Return the latest trading day on or before the natural end date."""
        natural_end = self._as_date(end_date or self.natural_end_date())
        if self.calendar is None:
            self.calendar = self.run(
                action={"fetch", "load"},
                query=Query(desc="calendar"),
            )

        calendar = self.calendar
        if calendar is None or calendar.is_empty():
            raise ValueError("calendar is empty, cannot infer previous trading day")

        candidates = calendar.filter(pl.col("trade_date") <= natural_end)
        if candidates.is_empty():
            raise ValueError(
                f"no trading day found on or before {natural_end:%Y%m%d}"
            )

        latest_trade_date = candidates.select(
            pl.max("trade_date").alias("trade_date")
        )["trade_date"][0]
        return self._as_yyyymmdd(latest_trade_date)

    def latest_raw_dates(
        self,
        fields: Sequence[str] | None = None,
    ) -> dict[str, str | None]:
        """Return latest persisted trade_date per raw field."""
        target_fields = tuple(fields) if fields is not None else _RAW_UPDATE_FIELDS
        latest_map: dict[str, str | None] = {}

        for field in target_fields:
            params = PARAM_MAP[field]
            latest: date | None = None

            if params.path.is_dir():
                latest = self._latest_partition_date(params.path)
            elif "trade_date" in params.schema.column_names:
                latest = self._latest_trade_date_from_table(params.path)

            latest_map[field] = (
                self._as_yyyymmdd(latest) if latest is not None else None
            )

        return latest_map

    def load_range(
        self,
        desc: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Load raw data limited to [start_date, end_date]."""
        params = PARAM_MAP[desc]

        if params.path.is_dir():
            start_dt = self._as_date(start_date)
            end_dt = self._as_date(end_date)
            dates = sorted(
                p.stem
                for p in params.path.glob("*.parquet")
                if (
                    len(p.stem) == 8
                    and p.stem.isdigit()
                    and start_dt <= self._as_date(p.stem) <= end_dt
                )
            )
            if not dates:
                return pl.DataFrame(schema=params.schema.column_names_and_types)
            df = read_parquet_by_dates(params.path, dates)
        else:
            df = params.reader(path=params.path, schema=params.schema, desc=desc)

        if "trade_date" in df.columns:
            start_dt = self._as_date(start_date)
            end_dt = self._as_date(end_date)
            df = df.filter(
                (pl.col("trade_date") >= start_dt) & (pl.col("trade_date") <= end_dt)
            )
        return df

    def update(
        self,
        fields: Sequence[str] | None = None,
        end_date: str | None = None,
    ) -> dict[str, tuple[str, str] | None]:
        """Fetch and validate raw fields from latest+1 to the natural end date."""
        target_fields = tuple(fields) if fields is not None else _RAW_UPDATE_FIELDS
        target_end = end_date or self.natural_end_date()
        end_dt = self._as_date(target_end)

        latest_map = self.latest_raw_dates(target_fields)
        updated: dict[str, tuple[str, str] | None] = {}

        for field in target_fields:
            latest = latest_map[field]
            start_dt = (
                self._as_date(latest) + timedelta(days=1)
                if latest is not None
                else end_dt
            )
            if start_dt > end_dt:
                updated[field] = None
                continue

            start = self._as_yyyymmdd(start_dt)
            self.run(
                action={"fetch", "validate"},
                query=Query(desc=field, start_date=start, end_date=target_end),
            )
            updated[field] = (start, target_end)

        return updated

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

    _MERGE_BUCKETS = 64

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
        from collections import defaultdict
        from glob import glob
        from datetime import date
        import shutil
        from config.data import cache_dir
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
        index_dates = index_df.select(["code", "trade_date"]).unique()

        # 3.5 Setup cache directory
        cache_path = cache_dir / desc
        if cache_path.exists():
            shutil.rmtree(cache_path)

        # State for tail-carry: stores last N bars per code from previous chunk
        tail_5min: pl.DataFrame | None = None
        tail_adj: pl.DataFrame | None = None
        tail_limit: pl.DataFrame | None = None

        bucket_chunk_files: dict[int, list[Path]] = defaultdict(list)

        # 4. Loop through chunks
        for i in tqdm(range(0, len(dates), chunk_days), desc=f"Processing {desc} (chunked)"):
            current_dates = dates[i: i + chunk_days]

            # Load new chunk
            new_5min = read_parquet_by_dates(raw_5min_dir, current_dates)
            new_adj = read_parquet_by_dates(raw_adj_dir, current_dates)
            new_limit = read_parquet_by_dates(raw_limit_dir, current_dates)

            if not new_5min.is_empty():
                new_5min = new_5min.join(
                    index_dates,
                    on=["code", "trade_date"],
                    how="semi",
                )
            if not new_adj.is_empty():
                new_adj = new_adj.join(
                    index_dates,
                    on=["code", "trade_date"],
                    how="semi",
                )
            if not new_limit.is_empty():
                new_limit = new_limit.join(
                    index_dates,
                    on=["code", "trade_date"],
                    how="semi",
                )

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

            # Write cache split by merge bucket to avoid generating huge numbers of tiny files.
            if not final_res.is_empty():
                chunk_dir = cache_path / f"chunk_{i:04d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                bucketed_res = final_res.with_columns(
                    (
                        pl.col("code").str.slice(0, 6).cast(pl.Int32) % ProcessedPipeline._MERGE_BUCKETS
                    ).alias("_merge_bucket")
                )
                partitions = bucketed_res.partition_by("_merge_bucket", as_dict=True, maintain_order=True)
                for key, bucket_df in partitions.items():
                    bucket = int(key[0] if isinstance(key, tuple) else key)
                    chunk_file = chunk_dir / f"bucket_{bucket:02d}.parquet"
                    bucket_df.drop("_merge_bucket").write_parquet(
                        chunk_file,
                        compression="uncompressed",
                    )
                    bucket_chunk_files[bucket].append(chunk_file)
                del bucketed_res

            tail_5min = (
                chunk_5min
                .sort(["code", "trade_date", "time"])
                .group_by("code", maintain_order=True)
                .tail(CHUNK_TAIL_BARS)
            )
            tail_adj = (
                chunk_adj
                .sort(["code", "trade_date"])
                .group_by("code", maintain_order=True)
                .tail(CHUNK_TAIL_DAYS)
            )
            tail_limit = (
                chunk_limit
                .sort(["code", "trade_date"])
                .group_by("code", maintain_order=True)
                .tail(CHUNK_TAIL_DAYS)
            )

            del chunk_5min
            del chunk_adj
            del chunk_limit
            del res
            del final_res
            if ((i // chunk_days) + 1) % 4 == 0:
                gc.collect()

        # 5. Merge by Code (Low Memory Strategy)
        # Only merge if process action was requested
        if "process" in action:
            # Clean final directory
            if params.path.exists():
                shutil.rmtree(params.path)
            params.path.mkdir(parents=True, exist_ok=True)

            # Sort codes to process deterministically
            sorted_buckets = sorted(bucket_chunk_files)

            for bucket in tqdm(sorted_buckets, desc=f"Merging {desc} by bucket"):
                chunk_files = bucket_chunk_files[bucket]
                merged_bucket = pl.read_parquet(chunk_files)
                code_partitions = merged_bucket.partition_by("code", as_dict=True, maintain_order=True)

                for key, code_df in code_partitions.items():
                    code = key[0] if isinstance(key, tuple) else key
                    merged = code_df.unique(
                        subset=["trade_date", "time_index"],
                        keep="last",
                        maintain_order=True,
                    ).sort(["trade_date", "time_index"])
                    merged.write_parquet(params.path / f"{code}.parquet")
                    del merged

                del code_partitions
                del merged_bucket
                gc.collect()

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
        assemble_all()

def _build_latest_code_matrices(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    df = df.with_columns([
        pl.col("trade_date")
        .cast(pl.String)
        .str.replace_all("-", "")
        .cast(pl.Float32)
        .alias("date_val"),
        pl.lit(np.nan, dtype=pl.Float32).alias("label_ret"),
        pl.lit(np.nan, dtype=pl.Float32).alias("label_rv"),
    ])

    scalar_valid_expr = _scalar_finite_expr(
        MACRO_FLOAT_FEATURES + SIDECHAIN_FEATURES
    )
    macro_int8_valid_expr = pl.all_horizontal(
        [pl.col(f).is_not_null() for f in MACRO_INT8_FEATURES]
    )
    mezzo_list_valid_expr = _list_finite_expr(
        MEZZO_FLOAT_FEATURES,
        expected_len=MEZZO_BARS_PER_DAY,
    )
    mezzo_int8_list_valid_expr = _list_non_null_expr(
        MEZZO_INT8_FEATURES,
        expected_len=MEZZO_BARS_PER_DAY,
    )
    micro_list_valid_expr = _list_finite_expr(
        MICRO_FLOAT_FEATURES,
        expected_len=MICRO_BARS_PER_DAY,
    )
    micro_int8_list_valid_expr = _list_non_null_expr(
        MICRO_INT8_FEATURES,
        expected_len=MICRO_BARS_PER_DAY,
    )

    df = df.with_columns(
        (
            scalar_valid_expr
            & macro_int8_valid_expr
            & mezzo_list_valid_expr
            & mezzo_int8_list_valid_expr
            & micro_list_valid_expr
            & micro_int8_list_valid_expr
        ).cast(pl.Float32).alias("is_valid_step")
    ).sort("trade_date")

    float_select: list[pl.Expr] = [pl.col("date_val"), pl.col("is_valid_step")]
    float_select.extend([pl.col(f).cast(pl.Float32) for f in LABEL_COLS])
    float_select.extend([pl.col(f).cast(pl.Float32) for f in MACRO_FLOAT_FEATURES])
    float_select.extend([pl.col(f).cast(pl.Float32) for f in SIDECHAIN_FEATURES])

    for feature in MEZZO_FLOAT_FEATURES:
        float_select.append(
            pl.col(feature)
            .list.to_struct(
                fields=[f"{feature}_{i}" for i in range(MEZZO_BARS_PER_DAY)]
            )
            .struct.field("*")
        )
    for feature in MICRO_FLOAT_FEATURES:
        float_select.append(
            pl.col(feature)
            .list.to_struct(
                fields=[f"{feature}_{i}" for i in range(MICRO_BARS_PER_DAY)]
            )
            .struct.field("*")
        )

    float_df = (
        df.select(float_select)
        .select(pl.col(pl.NUMERIC_DTYPES))
    )
    float_numpy = float_df.to_numpy().astype(np.float32, copy=False)

    int8_select: list[pl.Expr] = [
        pl.col(feature).fill_null(0).cast(pl.Int8)
        for feature in MACRO_INT8_FEATURES
    ]
    for feature in MEZZO_INT8_FEATURES:
        int8_select.append(
            _safe_list_int8_expr(
                feature,
                expected_len=MEZZO_BARS_PER_DAY,
            )
            .list.to_struct(
                fields=[f"{feature}_{i}" for i in range(MEZZO_BARS_PER_DAY)]
            )
            .struct.field("*")
        )
    for feature in MICRO_INT8_FEATURES:
        int8_select.append(
            _safe_list_int8_expr(
                feature,
                expected_len=MICRO_BARS_PER_DAY,
            )
            .list.to_struct(
                fields=[f"{feature}_{i}" for i in range(MICRO_BARS_PER_DAY)]
            )
            .struct.field("*")
        )

    int8_df = df.select(int8_select)
    int8_cols = [
        int8_df[col].to_numpy().astype(np.int8, copy=False)
        for col in int8_df.columns
    ]
    int8_numpy = (
        np.column_stack(int8_cols)
        if int8_cols
        else np.empty((float_numpy.shape[0], 0), dtype=np.int8)
    )

    return float_numpy, int8_numpy


def _latest_sample_dates(df: pl.DataFrame, float_numpy: np.ndarray) -> np.ndarray:
    start_idx = max(
        _MACRO_WINDOW_DAYS,
        _MEZZO_WINDOW_DAYS,
        _MICRO_WINDOW_DAYS,
    ) - 1
    if float_numpy.shape[0] <= start_idx:
        return np.empty((0,), dtype=np.int32)

    date_ints = (
        df
        .sort("trade_date")
        .select(
            pl.col("trade_date")
            .dt.strftime("%Y%m%d")
            .cast(pl.Int32)
            .alias("trade_date")
        )["trade_date"]
        .to_numpy()
        .astype(np.int32, copy=False)
    )
    keep = _compute_sample_valid(float_numpy[:, 1] > 0.5)
    return date_ints[start_idx:][keep]


def _slice_latest_worker(code_batch: Sequence[str], asof_int: int) -> dict[str, np.ndarray] | None:
    index_by_code = _LATEST_SLICE_STATE["index_by_code"]
    macro_by_code = _LATEST_SLICE_STATE["macro_by_code"]
    sidechain_by_code = _LATEST_SLICE_STATE["sidechain_by_code"]
    mezzo_by_code = _LATEST_SLICE_STATE["mezzo_by_code"]
    micro_by_code = _LATEST_SLICE_STATE["micro_by_code"]

    payload_parts: dict[str, list[np.ndarray]] = {
        "code": [],
        **{key: [] for key in _LATEST_PAYLOAD_KEYS},
    }

    for code in code_batch:
        daily_base = index_by_code.get(code)
        macro_part = macro_by_code.get(code)
        sidechain_part = sidechain_by_code.get(code)
        mezzo_part = mezzo_by_code.get(code)
        micro_part = micro_by_code.get(code)
        if (
            daily_base is None
            or macro_part is None
            or sidechain_part is None
            or mezzo_part is None
            or micro_part is None
        ):
            continue

        code_df = (
            daily_base
            .join(
                macro_part,
                on=["code", "trade_date"],
                how="left",
            )
            .join(
                sidechain_part,
                on=["code", "trade_date"],
                how="left",
            )
            .join(
                mezzo_part,
                on=["code", "trade_date"],
                how="left",
            )
            .join(
                micro_part,
                on=["code", "trade_date"],
                how="left",
            )
        )

        float_numpy, int8_numpy = _build_latest_code_matrices(code_df)
        sample_dates = _latest_sample_dates(code_df, float_numpy)
        payload = _build_packed_payload(float_numpy, int8_numpy)
        if payload["date"].size == 0:
            continue
        if sample_dates.shape[0] != payload["date"].shape[0]:
            raise ValueError(
                f"latest sample date alignment mismatch for {code}: "
                f"{sample_dates.shape[0]} != {payload['date'].shape[0]}"
            )

        sample_mask = sample_dates == asof_int
        if not np.any(sample_mask):
            continue

        count = int(sample_mask.sum())
        payload_parts["code"].append(np.asarray([code] * count, dtype="<U16"))
        payload_parts["date"].append(sample_dates[sample_mask])
        for payload_key in _LATEST_PAYLOAD_KEYS:
            if payload_key == "date":
                continue
            payload_parts[payload_key].append(payload[payload_key][sample_mask])

    if not payload_parts["code"]:
        return None

    return {
        payload_key: (
            np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        )
        for payload_key, parts in payload_parts.items()
    }


def _chunk_codes(codes: Sequence[str], chunk_size: int) -> list[list[str]]:
    return [
        list(codes[idx: idx + chunk_size])
        for idx in range(0, len(codes), chunk_size)
    ]


def slice_latest(end_date: str | None = None) -> Path:
    """Update raw data and build the latest inference-ready sample bundle."""
    raw_pipe = RawPipeline()
    raw_pipe.update(end_date=end_date)
    asof_date = raw_pipe.previous_trading_day(end_date=end_date)

    suspend_start = raw_pipe._earliest_partition_date(PARAM_MAP["suspend"].path)
    if suspend_start is None:
        raise ValueError("suspend raw data is empty, cannot build latest slice")

    suspend_df = raw_pipe.load_range(
        "suspend",
        raw_pipe._as_yyyymmdd(suspend_start),
        asof_date,
    )
    index_df = PROCESSED_PARAM_MAP["index"].processor(suspend_df=suspend_df)

    full_index_by_code = {
        code: part.select(["code", "trade_date"]).sort("trade_date")
        for code, part in (
            {
                (key[0] if isinstance(key, tuple) else key): value
                for key, value in index_df.partition_by(
                    "code",
                    as_dict=True,
                    maintain_order=True,
                ).items()
            }
        ).items()
    }

    index_by_code: dict[str, pl.DataFrame] = {}
    slice_starts: list[date] = []
    for code, part in tqdm(
        full_index_by_code.items(),
        total=len(full_index_by_code),
        desc="Preparing latest windows",
    ):
        if part.height < _LATEST_LOGIC_DAYS:
            continue
        recent = part.tail(_LATEST_LOGIC_DAYS)
        index_by_code[code] = recent
        slice_starts.append(recent["trade_date"][0])

    if not index_by_code:
        raise ValueError(f"no latest samples were built for {asof_date}")

    raw_start = raw_pipe._as_yyyymmdd(min(slice_starts))

    daily_df = raw_pipe.load_range("daily", raw_start, asof_date)
    adj_factor_df = raw_pipe.load_range("adj_factor", raw_start, asof_date)
    limit_df = raw_pipe.load_range("limit", raw_start, asof_date)
    moneyflow_df = raw_pipe.load_range("moneyflow", raw_start, asof_date)
    min5_df = raw_pipe.load_range("5min", raw_start, asof_date)

    macro_df = PROCESSED_PARAM_MAP["macro"].processor(
        index_df=index_df,
        daily_df=daily_df,
        adj_factor_df=adj_factor_df,
        limit_df=limit_df,
        **PROCESSED_PARAM_MAP["macro"].processor_kwargs,
    )
    sidechain_df = PROCESSED_PARAM_MAP["sidechain"].processor(
        index_df=index_df,
        daily_df=daily_df,
        adj_factor_df=adj_factor_df,
        moneyflow_df=moneyflow_df,
        **PROCESSED_PARAM_MAP["sidechain"].processor_kwargs,
    )
    mezzo_df = PROCESSED_PARAM_MAP["mezzo"].processor(
        index_df=index_df,
        min5_df=min5_df,
        adj_factor_df=adj_factor_df,
        limit_df=limit_df,
        **PROCESSED_PARAM_MAP["mezzo"].processor_kwargs,
    )
    micro_df = PROCESSED_PARAM_MAP["micro"].processor(
        index_df=index_df,
        min5_df=min5_df,
        adj_factor_df=adj_factor_df,
        limit_df=limit_df,
        **PROCESSED_PARAM_MAP["micro"].processor_kwargs,
    )

    mezzo_daily_df = (
        mezzo_df
        .sort(["code", "trade_date", "time_index"])
        .group_by(["code", "trade_date"], maintain_order=True)
        .agg([pl.col(feature) for feature in MEZZO_FEATURES])
    )
    micro_daily_df = (
        micro_df
        .sort(["code", "trade_date", "time_index"])
        .group_by(["code", "trade_date"], maintain_order=True)
        .agg([pl.col(feature) for feature in MICRO_FEATURES])
    )

    def _partition_by_code(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        partitions = df.partition_by("code", as_dict=True, maintain_order=True)
        return {
            (key[0] if isinstance(key, tuple) else key): value
            for key, value in partitions.items()
        }

    macro_by_code = _partition_by_code(macro_df)
    sidechain_by_code = _partition_by_code(sidechain_df)
    mezzo_by_code = _partition_by_code(mezzo_daily_df)
    micro_by_code = _partition_by_code(micro_daily_df)

    codes = sorted(
        set(index_by_code)
        & set(macro_by_code)
        & set(sidechain_by_code)
        & set(mezzo_by_code)
        & set(micro_by_code)
    )
    if not codes:
        raise ValueError(f"no latest samples were built for {asof_date}")

    _LATEST_SLICE_STATE.clear()
    _LATEST_SLICE_STATE.update(
        index_by_code=index_by_code,
        macro_by_code=macro_by_code,
        sidechain_by_code=sidechain_by_code,
        mezzo_by_code=mezzo_by_code,
        micro_by_code=micro_by_code,
    )

    worker_count = min(max(1, (os.cpu_count() or 1) // 2), len(codes))
    chunk_size = max(16, min(128, len(codes) // max(worker_count * 4, 1) or 16))
    code_batches = _chunk_codes(codes, chunk_size)

    batch_payloads: list[dict[str, np.ndarray]] = []
    if worker_count <= 1:
        for code_batch in tqdm(code_batches, desc="Slicing latest"):
            payload = _slice_latest_worker(code_batch, int(asof_date))
            if payload is not None:
                batch_payloads.append(payload)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            for payload in tqdm(
                executor.map(
                    _slice_latest_worker,
                    code_batches,
                    [int(asof_date)] * len(code_batches),
                    chunksize=1,
                ),
                total=len(code_batches),
                desc="Slicing latest",
            ):
                if payload is not None:
                    batch_payloads.append(payload)

    _LATEST_SLICE_STATE.clear()

    latest_dir.mkdir(parents=True, exist_ok=True)
    out_path = latest_dir / f"{asof_date}.npz"

    stacked_payload: dict[str, np.ndarray] = {}
    if not batch_payloads:
        raise ValueError(f"no latest samples were built for {asof_date}")
    for payload_key in ("code", *_LATEST_PAYLOAD_KEYS):
        parts = [payload[payload_key] for payload in batch_payloads]
        stacked_payload[payload_key] = (
            np.concatenate(parts, axis=0)
            if len(parts) > 1
            else parts[0]
        )

    np.savez(
        out_path,
        asof_date=np.asarray(asof_date),
        label_names=_LABEL_NAMES_ARRAY,
        **stacked_payload,
    )
    return out_path
