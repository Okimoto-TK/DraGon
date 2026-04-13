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


def read_parquet_by_dates(dir_path: Path, dates: list[str]) -> pl.DataFrame:
    """Helper to load specific parquet files by date."""
    paths = [dir_path / f"{d}.parquet" for d in dates]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return pl.DataFrame()
    return pl.read_parquet(existing)


def write_parquet(df: pl.DataFrame, path: Path, schema: TableSchema, desc: str = "") -> None:
    """Write DataFrame to a single parquet file after schema validation."""
    vlog(_SRC, f"Writing {desc} to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_table(df, schema)
    df = df.select(schema.column_names)
    df.write_parquet(path)
    vlog(_SRC, f"Writing {desc} done.")


def write_parquets(df: pl.DataFrame, path: Path, schema: TableSchema, desc: str = "") -> None:
    """Partition DataFrame and write to separate parquet files based on schema.partition_by."""
    vlog(_SRC, f"Writing {desc} to {path}...")
    path.mkdir(parents=True, exist_ok=True)
    validate_table(df, schema)
    df = df.select(schema.column_names)

    results = partition_by(df, schema.partition_by)

    partition_col = schema.partition_by[0]
    is_date_col = partition_col == "trade_date"
    date_fmt = schema.get_column("trade_date").fmt if is_date_col else "%s"

    for keys, _df in tqdm(results.items(), desc=f"Writing {desc}:", disable=config.debug):
        val = keys[0]
        if is_date_col:
            filename = f"{val.strftime(date_fmt)}.parquet"
        else:
            filename = f"{val}.parquet"

        vlog(_SRC, f"Writing {filename}...")
        file_path = path / filename
        _df.write_parquet(file_path)

    vlog(_SRC, f"Writing {desc} done.")
