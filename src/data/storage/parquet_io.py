"""Parquet file I/O operations with schema validation."""
from __future__ import annotations

from glob import glob
from pathlib import Path

import config.config as config
import polars as pl
from tqdm import tqdm

from src.data.models import TableSchema
from src.data.utils.raw import partition_by
from src.data.validators import validate_table
from src.utils.log import vlog

_SRC = "Storage"


def read_parquet_schema(path: Path, desc: str = "") -> pl.DataFrame:
    """Read parquet schema (column names and types) from a single file."""
    vlog(_SRC, f"Reading {desc} schema from {path}...")
    _schema = pl.read_parquet_schema(path)
    df = pl.DataFrame(schema=_schema)
    vlog(_SRC, f"Reading {desc} schema done.")
    return df


def read_parquets_schema(path: Path, desc: str = "") -> pl.DataFrame:
    """Read combined schema from all parquet files in a directory."""
    vlog(_SRC, f"Reading {desc} schema from {path}...")
    files = glob("*.parquet", root_dir=path, recursive=False)
    results = []

    for file in tqdm(files, desc=f"Reading {desc}:", disable=config.debug):
        vlog(_SRC, f"Reading {file}...")
        file_path = path / file
        _schema = pl.read_parquet_schema(file_path)
        results.append(pl.DataFrame(schema=_schema))

    if not results:
        return pl.DataFrame()

    df = pl.concat(results)
    vlog(_SRC, f"Reading {desc} schema done.")
    return df


def read_parquet(path: Path, schema: TableSchema, desc: str = "") -> pl.DataFrame:
    """Read a single parquet file and validate against schema."""
    vlog(_SRC, f"Reading {desc} from {path}...")
    df = pl.read_parquet(path)
    validate_table(df, schema)
    vlog(_SRC, f"Reading {desc} done.")
    return df


def read_parquets(path: Path, schema: TableSchema, desc: str = "") -> pl.DataFrame:
    """Read all parquet files from a directory and concatenate."""
    vlog(_SRC, f"Reading {desc} from {path}...")
    files = glob("*.parquet", root_dir=path, recursive=False)
    results = []

    for file in tqdm(files, desc=f"Reading {desc}:", disable=config.debug):
        vlog(_SRC, f"Reading {file}...")
        file_path = path / file
        results.append(pl.read_parquet(file_path))

    if not results:
        return pl.DataFrame(schema=schema.column_names_and_types)

    df = pl.concat(results)
    validate_table(df, schema)
    vlog(_SRC, f"Reading {desc} done.")
    return df


def write_parquet(df: pl.DataFrame, path: Path, schema: TableSchema, desc: str = "") -> None:
    """Write DataFrame to a single parquet file after schema validation."""
    vlog(_SRC, f"Writing {desc} to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_table(df, schema)
    df = df.select(schema.column_names)
    df.write_parquet(path)
    vlog(_SRC, f"Writing {desc} done.")


def write_parquets(df: pl.DataFrame | pl.LazyFrame, path: Path, schema: TableSchema, desc: str = "") -> None:
    """Integrated Eager and Lazy partitioned writer.

    For LazyFrame: sinks the entire LazyFrame to a single parquet file,
    streaming data to disk without full collect.
    For DataFrame: uses eager partition_by.
    """
    vlog(_SRC, f"Writing {desc} to {path}...")
    path.mkdir(parents=True, exist_ok=True)

    if not schema.partition_by:
        raise ValueError(f"Schema {schema} must define 'partition_by' for partitioned writing.")

    partition_col = schema.partition_by[0]
    is_date_col = partition_col == "trade_date"
    date_fmt = schema.get_column("trade_date").fmt if is_date_col else "%s"

    # --- LazyFrame: single file sink ---
    if isinstance(df, pl.LazyFrame):
        vlog(_SRC, f"Detected LazyFrame, sinking to single file for {desc}...")

        # Select columns to match schema
        lf = df.select(schema.column_names)

        # Sink entire LazyFrame to a single parquet file (streaming, no full collect)
        output_path = path / "cache.parquet"
        lf.sink_parquet(output_path)

    # --- DataFrame: eager path ---
    else:
        vlog(_SRC, f"Detected DataFrame, using eager partitioning for {desc}...")
        validate_table(df, schema)
        df = df.select(schema.column_names)

        results = partition_by(df, schema.partition_by)

        for keys, _df in tqdm(results.items(), desc=f"Writing {desc}:", disable=config.debug):
            val = keys[0]
            filename = f"{val.strftime(date_fmt)}.parquet" if is_date_col else f"{val}.parquet"

            vlog(_SRC, f"Writing {filename}...")
            _df.write_parquet(path / filename)

    vlog(_SRC, f"Writing {desc} done.")


def scan_parquets(path: Path, desc: str = "") -> pl.LazyFrame:
    """Scan all parquet files from a directory lazily (no memory overhead)."""
    vlog(_SRC, f"Scanning {desc} from {path}...")
    lf = pl.scan_parquet(path / "*.parquet")
    vlog(_SRC, f"Scanning {desc} done.")
    return lf
