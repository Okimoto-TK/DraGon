from __future__ import annotations

import polars as pl
from pathlib import Path
from tqdm import tqdm
from glob import glob

from src.data.schemas.raw import TableSchema
from src.data.validators.raw import validate_table
from src.data.utils.raw import partition_by
import config.config as config
from src.utils.log import vlog


src = "Storage"


def read_parquet_schema(path: Path, desc: str = "") -> pl.DataFrame:
    vlog(src, f"Reading {desc} schema from {path}...")

    _schema = pl.read_parquet_schema(path)
    df = pl.DataFrame(schema=_schema)

    vlog(src, f"Reading {desc} schema done.")
    return df


def read_parquets_schema(path: Path, desc: str = "") -> pl.DataFrame:
    vlog(src, f"Reading {desc} schema from {path}...")

    files = glob("*.parquet", root_dir=path, recursive=False)
    results = []

    for file in tqdm(files, desc=f"Reading {desc}:", disable=config.debug):
        vlog(src, f"Reading {file}...")

        file_path = path / file
        _schema = pl.read_parquet_schema(file_path)
        _df = pl.DataFrame(schema=_schema)
        results.append(_df)

    if len(results) == 0:
        return pl.DataFrame()

    df = pl.concat(results)

    vlog(src, f"Reading {desc} schema done.")
    return df


def read_parquet(path: Path, schema: TableSchema, desc: str = "") -> pl.DataFrame:
    vlog(src, f"Reading {desc} from {path}...")

    df = pl.read_parquet(path)
    validate_table(df, schema)

    vlog(src, f"Reading {desc} done.")
    return df


def read_parquets(path: Path, schema: TableSchema, desc: str = "") -> pl.DataFrame:
    vlog(src, f"Reading {desc} from {path}...")

    files = glob("*.parquet", root_dir=path, recursive=False)
    results = []

    for file in tqdm(files, desc=f"Reading {desc}:", disable=config.debug):
        vlog(src, f"Reading {file}...")

        file_path = path / file
        _df = pl.read_parquet(file_path)
        results.append(_df)

    if len(results) == 0:
        return pl.DataFrame(schema=schema.column_names_and_types)

    df = pl.concat(results)
    validate_table(df, schema)

    vlog(src, f"Reading {desc} done.")
    return df


def write_parquet(df: pl.DataFrame, path: Path, schema: TableSchema, desc: str = ""):
    vlog(src, f"Writing {desc} to {path}...")

    path.parent.mkdir(parents=True, exist_ok=True)
    validate_table(df, schema)
    df.write_parquet(path)

    vlog(src, f"Writing {desc} done.")


def write_parquets(df: pl.DataFrame, path: Path, schema: TableSchema, desc: str = ""):
    vlog(src, f"Writing {desc} to {path}...")

    path.mkdir(parents=True, exist_ok=True)
    validate_table(df, schema)
    results = partition_by(df, schema.partition_by)
    for (date_val, ), _df in tqdm(results.items(), desc=f"Writing {desc}:", disable=config.debug):
        vlog(src, f"Writing {date_val.strftime(schema.get_column("trade_date").fmt)}...")

        file_name = f'{date_val.strftime(schema.get_column("trade_date").fmt)}.parquet'
        file_path = path / file_name

        _df.write_parquet(file_path)

    vlog(src, f"Writing {desc} done.")
    return df
