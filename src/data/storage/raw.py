from __future__ import annotations

import polars as pl
from pathlib import Path
from tqdm import tqdm
from glob import glob
from typing import Dict, Tuple

from src.data.schemas.raw import TableSchema
from src.data.validators.raw import validate_table


def read_parquet(parquet_path: Path, table_schema: TableSchema) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    validate_table(df, table_schema)
    return df


def read_parquets(dir_path: Path, table_schema: TableSchema, desc: str = "") -> Dict[Tuple, pl.DataFrame]:
    files = glob("*.parquet", root_dir=dir_path, recursive=False)
    results = {}

    for file in tqdm(files, desc=f"Reading {desc}:"):
        df = pl.read_parquet(file)
        validate_table(df, table_schema)
        results[(file, )] = df

    return results


def write_parquet(df: pl.DataFrame, parquet_path: Path, table_schema: TableSchema):
    validate_table(df, table_schema)
    df.write_parquet(parquet_path)


def write_by_date(dfs: Dict[Tuple, pl.DataFrame], dir_path: Path, table_schema: TableSchema, desc: str = ""):
    for (date_val, ), _df in tqdm(dfs.items(), desc=f"Writing {desc}:"):
        file_name = f'{date_val}.parquet'
        file_path = dir_path / file_name

        validate_table(_df, table_schema)
        _df.write_parquet(file_path)

