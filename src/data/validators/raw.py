import polars as pl
from src.data.schemas.raw import *
import warnings


def validate_required_columns(df: pl.DataFrame, schema: TableSchema):
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {", ".join(missing)}')


def validate_extra_columns(df: pl.DataFrame, schema: TableSchema):
    extra = set(df.columns) - set(schema.required_columns)
    if extra:
        warnings.warn(f'Extra columns: {", ".join(extra)}')


def validate_date_format(df: pl.DataFrame, date_schema: ColumnSchema):
    check_df = df.with_columns(
        is_real_date=pl.col("trade_date").str.to_date(date_schema.fmt, strict=False).is_not_null()
    )

    if not check_df['is_real_date'].any():
        bad_data = df.filter(
            pl.col("trade_date").str.to_date(date_schema.fmt, strict=False).is_null()
        )
        raise ValueError(f'Invalid date format: {bad_data}')


def validate_time_format(df: pl.DataFrame, time_schema: ColumnSchema):
    check_df = df.with_columns(
        is_real_time=pl.col("time").str.to_time(time_schema.fmt, strict=False)
    )

    if not check_df['is_real_time'].any():
        bad_data = df.filter(
            pl.col("time").str.to_time(time_schema.fmt, strict=False).is_null()
        )
        raise ValueError(f'Invalid time format: {bad_data}')


def validate_table(df: pl.DataFrame, schema: TableSchema):
    validate_required_columns(df, schema)
    validate_extra_columns(df, schema)
    validate_date_format(df, schema.get_column("trade_date"))