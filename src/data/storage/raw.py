from __future__ import annotations

import polars as pl
from pathlib import Path
from tqdm import tqdm
from glob import glob

from src.data.schemas.raw import TableSchema
from src.data.validators.raw import validate_table
from src.data.utils.raw import partition_by
import config.conf as conf
from src.utils.log import vlog


src = "Storage"


def read_parquet(parquet_path: Path, table_schema: TableSchema) -> pl.DataFrame:
    vlog(src, f"Reading {parquet_path}...")

    df = pl.read_parquet(parquet_path)
    validate_table(df, table_schema)

    vlog(src, f"Read {parquet_path} done.")
    return df


def read_parquets(dir_path: Path, table_schema: TableSchema, desc: str = "") -> pl.DataFrame:
    vlog(src, f"Reading {desc} .parquet under {dir_path}...")

    files = glob("*.parquet", root_dir=dir_path, recursive=False)
    results = []

    for file in tqdm(files, desc=f"Reading {desc}:", disable=conf.debug):
        vlog(src, f"Reading {file}...")

        file_path = dir_path / file
        _df = pl.read_parquet(file_path)
        results.append(_df)

    df = pl.concat(results)
    validate_table(df, table_schema)

    vlog(src, f"Read {desc} done.")
    return df


def write_parquet(df: pl.DataFrame, parquet_path: Path, table_schema: TableSchema):
    vlog(src, f"Writing {parquet_path}...")

    validate_table(df, table_schema)
    df.write_parquet(parquet_path)

    vlog(src, f"Writing {parquet_path} done.")


def write_by_date(df: pl.DataFrame, dir_path: Path, table_schema: TableSchema, desc: str = ""):
    vlog(src, f"Writing {desc} .parquet under {dir_path}...")

    validate_table(df, table_schema)
    results = partition_by(df, "trade_date")
    for (date_val, ), _df in tqdm(results.items(), desc=f"Writing {desc}:", disable=conf.debug):
        vlog(src, f"Writing {date_val}...")

        file_name = f'{date_val}.parquet'
        file_path = dir_path / file_name

        _df.write_parquet(file_path)

    vlog(src, f"Writing {desc} done.")
    return df
