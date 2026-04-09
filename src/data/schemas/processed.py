"""Schema definitions for processed data tables.

Defines column structures and validation rules
for all processed data tables in the pipeline.
"""
from __future__ import annotations

import polars as pl

from ..models import ColumnSchema, TableSchema
from ..types import DType


# === Helper: Generate feature column schemas ===

def _generate_feature_columns(prefix: str, count: int, dtype: DType = pl.Float64) -> tuple[ColumnSchema, ...]:
    """Generate feature column schemas with sequential numbering.

    Args:
        prefix: Column name prefix (e.g., "mcr_f", "mzo_f", "mic_f").
        count: Number of features to generate.
        dtype: Data type for the columns.

    Returns:
        Tuple of ColumnSchema objects.
    """
    return tuple(
        ColumnSchema(
            name=f"{prefix}{i}",
            dtype=dtype,
            required=True,
            nullable=True,
            description=f"Feature {i} ({prefix.rstrip('_f')})",
        )
        for i in range(1, count + 1)
    )


# === Table Schema Definitions ===

# Index columns: logical unique identifier (stitch suspension gaps)
PROCESSED_INDEX_SCHEMA = TableSchema(
    name="processed_index",
    layer="processed",
    description="Logical unique identifier index with suspension gaps stitched.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="logic_index",
            dtype=pl.Int32,
            required=True,
            nullable=False,
            description="Logical index that stitches suspension gaps.",
        ),
    ),
)

# Mask column: filter condition
PROCESSED_MASK_SCHEMA = TableSchema(
    name="processed_mask",
    layer="processed",
    description="Filter mask for non-ST stocks with no physical suspension in next H days.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="filter_mask",
            dtype=pl.Boolean,
            required=True,
            nullable=False,
            description="Whether the stock is non-ST and has no physical suspension in the next H days.",
        ),
    ),
)

# Backbone (Macro): daily-scale features
PROCESSED_MACRO_SCHEMA = TableSchema(
    name="processed_macro",
    layer="processed",
    description="Daily-scale backbone features (NormalRank or Raw).",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        *_generate_feature_columns("mcr_f", 10, pl.Float64),
    ),
)

# Backbone (Mezzo): 30-minute scale features
PROCESSED_MEZZO_SCHEMA = TableSchema(
    name="processed_mezzo",
    layer="processed",
    description="30-minute scale backbone features (NormalRank or Raw).",
    primary_key=("code", "trade_date", "time_index"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="time_index",
            dtype=pl.Int32,
            required=True,
            nullable=False,
            description="Time index for 30-minute bars within the trading day.",
        ),
        *_generate_feature_columns("mzo_f", 10, pl.Float64),
    ),
)

# Backbone (Micro): 5-minute scale features
PROCESSED_MICRO_SCHEMA = TableSchema(
    name="processed_micro",
    layer="processed",
    description="5-minute scale backbone features (NormalRank or Raw).",
    primary_key=("code", "trade_date", "time_index"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="time_index",
            dtype=pl.Int32,
            required=True,
            nullable=False,
            description="Time index for 5-minute bars within the trading day.",
        ),
        *_generate_feature_columns("mic_f", 10, pl.Float64),
    ),
)

# Sidechain: energy modulation features
PROCESSED_SIDECHAIN_SCHEMA = TableSchema(
    name="processed_sidechain",
    layer="processed",
    description="Energy modulation sidechain features (preserve absolute magnitude information).",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="mf_abs_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Money flow absolute rank.",
        ),
        ColumnSchema(
            name="mf_impact",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Money flow impact.",
        ),
        ColumnSchema(
            name="mf_conviction",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Money flow conviction.",
        ),
        ColumnSchema(
            name="energy_factor",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Energy factor.",
        ),
        ColumnSchema(
            name="mkt_vola_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Market volatility rank.",
        ),
    ),
)

# Target: prediction labels
PROCESSED_LABEL_SCHEMA = TableSchema(
    name="processed_label",
    layer="processed",
    description="Prediction labels (PowerRank label).",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype=pl.String,
            required=True,
            nullable=False,
            fmt=r"^\d{6}\.[A-Z]{2}$",
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype=pl.Date,
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="label_final",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="PowerRank label.",
        ),
    ),
)

# Registry of all processed schemas by name
PROCESSED_SCHEMAS: dict[str, TableSchema] = {
    PROCESSED_INDEX_SCHEMA.name: PROCESSED_INDEX_SCHEMA,
    PROCESSED_MASK_SCHEMA.name: PROCESSED_MASK_SCHEMA,
    PROCESSED_MACRO_SCHEMA.name: PROCESSED_MACRO_SCHEMA,
    PROCESSED_MEZZO_SCHEMA.name: PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA.name: PROCESSED_MICRO_SCHEMA,
    PROCESSED_SIDECHAIN_SCHEMA.name: PROCESSED_SIDECHAIN_SCHEMA,
    PROCESSED_LABEL_SCHEMA.name: PROCESSED_LABEL_SCHEMA,
}
