import warnings
import polars as pl

import config.conf as conf
from src.data.schemas.raw import TableSchema


def _validate_required_columns(df: pl.DataFrame, schema: TableSchema):
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {", ".join(missing)}')


def _validate_extra_columns(df: pl.DataFrame, schema: TableSchema):
    extra = set(df.columns) - set(schema.column_names)
    if extra:
        if conf.debug:
            raise ValueError(f'Extra columns: {", ".join(extra)}')
        else:
            warnings.warn(f'Extra columns: {", ".join(extra)}')


def _validate_data_types(df: pl.DataFrame, schema: TableSchema):
    if not df.schema.items() <= schema.column_names_and_types.items():
        diff = df.schema.items() - schema.column_names_and_types.items()
        raise ValueError(f'Schema mismatch: {", ".join(diff)}')


def _validate_primary_key(df: pl.DataFrame, schema: TableSchema):
    if df.is_empty():
        return

    key_cols = list(schema.primary_key)
    if not key_cols:
        return

    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{schema.name}: missing primary key columns for validation: {missing}"
        )

    duplicated = (
        df.group_by(key_cols)
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicated.height > 0:
        sample = duplicated.head(10).to_dicts()
        raise ValueError(
            f"{schema.name}: duplicated primary key rows found for key columns "
            f"{key_cols}. Sample duplicates: {sample}"
        )


def validate_table(df: pl.DataFrame, schema: TableSchema):
    _validate_required_columns(df, schema)
    _validate_extra_columns(df, schema)
    _validate_primary_key(df, schema)
    _validate_data_types(df, schema)
