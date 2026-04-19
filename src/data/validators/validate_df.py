"""Validation functions for raw data tables against schema definitions."""
from __future__ import annotations

import warnings

import config.data as config
import polars as pl

from src.data.models import TableSchema


def _validate_required_columns(df: pl.DataFrame | pl.LazyFrame, schema: TableSchema) -> None:
    """Check that all required columns are present."""
    missing = [c for c in schema.required_columns if c not in df.collect_schema().names()]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def _validate_extra_columns(df: pl.DataFrame | pl.LazyFrame, schema: TableSchema) -> None:
    """Check for unexpected columns not defined in schema."""
    extra = set(df.collect_schema().names()) - set(schema.column_names)
    if extra:
        if config.debug:
            raise ValueError(f"Extra columns: {', '.join(extra)}")
        else:
            warnings.warn(f"Extra columns: {', '.join(extra)}", stacklevel=2)


def _validate_data_types(df: pl.DataFrame | pl.LazyFrame, schema: TableSchema) -> None:
    """Validate column data types match schema definitions."""
    current_schema = df.collect_schema()
    if not current_schema.items() <= schema.column_names_and_types.items():
        diff = current_schema.items() - schema.column_names_and_types.items()
        for name, dtype in diff:
            if name in schema.column_names:
                raise ValueError(
                    f"Wrong dtype: {name} has type {dtype}, expected {schema.get_column(name).dtype}"
                )


def _validate_primary_key(df: pl.DataFrame, schema: TableSchema) -> None:
    """Check for duplicate primary key rows."""
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

    duplicated = df.group_by(key_cols).len().filter(pl.col("len") > 1)

    if duplicated.height > 0:
        sample = duplicated.head(10).to_dicts()
        raise ValueError(
            f"{schema.name}: duplicated primary key rows found for key columns "
            f"{key_cols}. Sample duplicates: {sample}"
        )


def validate_table(df: pl.DataFrame | pl.LazyFrame, schema: TableSchema) -> None:
    """Validate a DataFrame/LazyFrame against a table schema.

    For LazyFrame: only validates schema (column names and types), skips data-level checks.
    For DataFrame: checks required columns, extra columns, data types, and primary key uniqueness.
    """
    _validate_required_columns(df, schema)
    _validate_extra_columns(df, schema)
    _validate_data_types(df, schema)

    # Primary key uniqueness requires eager evaluation
    if isinstance(df, pl.DataFrame):
        _validate_primary_key(df, schema)
