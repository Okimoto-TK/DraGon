"""Schema definitions for processed data tables.

Defines column structures and validation rules
for all processed data tables in the pipeline.
"""
from __future__ import annotations

import polars as pl

from ..models import ColumnSchema, TableSchema
from ..types import DType


# === Helper: Generate feature column schemas ===

def _generate_feature_columns(prefix: str, count: int, dtype: DType = pl.Float64, start_from: int = 1) -> tuple[ColumnSchema, ...]:
    """Generate feature column schemas with sequential numbering.

    Args:
        prefix: Column name prefix (e.g., "mcr_f", "mzo_f", "mic_f").
        count: Number of features to generate.
        dtype: Data type for the columns.
        start_from: Starting index for feature numbering.

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
        for i in range(start_from, start_from + count)
    )


# === Table Schema Definitions ===

# Index columns: logical unique identifier (stitch suspension gaps)
PROCESSED_INDEX_SCHEMA = TableSchema(
    name="processed_index",
    layer="processed",
    description="Logical unique identifier index with suspension gaps stitched.",
    primary_key=("code", "trade_date"),
    partition_by=("code",),
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
    partition_by=("code",),
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
    partition_by=("code",),
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
        *_generate_feature_columns("mcr_f", 6, pl.Float64, start_from=0),
        ColumnSchema(
            name="mcr_f6",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Limit-hit bitmask: hit_up*8 + hit_down*4 + close_up*2 + close_down (Int8)",
        ),
        ColumnSchema(
            name="mcr_f7",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Day of week: Monday=1 .. Friday=5 (Int8)",
        ),
        *_generate_feature_columns("mcr_f", 3, pl.Float64, start_from=8),
    ),
)

# Backbone (Mezzo): 30-minute scale features
PROCESSED_MEZZO_SCHEMA = TableSchema(
    name="processed_mezzo",
    layer="processed",
    description="30-minute scale backbone features (NormalRank or Raw).",
    primary_key=("code", "trade_date", "time_index"),
    partition_by=("code",),
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
        *_generate_feature_columns("mzo_f", 6, pl.Float64, start_from=0),
        ColumnSchema(
            name="mzo_f6",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Limit-hit bitmask: hit_up*8 + hit_down*4 + close_up*2 + close_down (Int8)",
        ),
        ColumnSchema(
            name="mzo_f7",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Time index within trading day (1-8 for 30-min bars, Int8)",
        ),
        *_generate_feature_columns("mzo_f", 3, pl.Float64, start_from=8),
    ),
)

# Backbone (Micro): 5-minute scale features
PROCESSED_MICRO_SCHEMA = TableSchema(
    name="processed_micro",
    layer="processed",
    description="5-minute scale backbone features (NormalRank or Raw).",
    primary_key=("code", "trade_date", "time_index"),
    partition_by=("code",),
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
        *_generate_feature_columns("mic_f", 6, pl.Float64, start_from=0),
        ColumnSchema(
            name="mic_f6",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Limit-hit bitmask: hit_up*8 + hit_down*4 + close_up*2 + close_down (Int8)",
        ),
        ColumnSchema(
            name="mic_f7",
            dtype=pl.Int8,
            required=True,
            nullable=True,
            description="Time index within trading day (1-48 for 5-min bars, Int8)",
        ),
        *_generate_feature_columns("mic_f", 3, pl.Float64, start_from=8),
    ),
)

# Sidechain: energy modulation features
PROCESSED_SIDECHAIN_SCHEMA = TableSchema(
    name="processed_sidechain",
    layer="processed",
    description="Sidechain energy modulation features (gap, moneyflow, volume surge).",
    primary_key=("code", "trade_date"),
    partition_by=("code",),
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
            name="gap",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Overnight gap: ln(Open_t / Close_{t-1})",
        ),
        ColumnSchema(
            name="gap_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Gap cross-sectional normal rank",
        ),
        ColumnSchema(
            name="mf_net_ratio",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Main force net ratio: (buy_main - sell_main) / Amount",
        ),
        ColumnSchema(
            name="mf_concentration",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Main force concentration: (buy_main + sell_main) / Amount",
        ),
        ColumnSchema(
            name="mf_concentration_diff",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Fractional difference of mf_concentration raw value over code timeline.",
        ),
        ColumnSchema(
            name="mf_concentration_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Main force concentration rank: NormalRank(mf_concentration)",
        ),
        ColumnSchema(
            name="mf_main_amount_log",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Raw log of main-flow amount: ln(buy_lg+buy_elg+sell_lg+sell_elg).",
        ),
        ColumnSchema(
            name="mf_main_amount_log_diff",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Fractional difference of mf_main_amount_log over code timeline.",
        ),
        ColumnSchema(
            name="mf_main_amount_log_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Cross-sectional NormalRank of mf_main_amount_log.",
        ),
        ColumnSchema(
            name="mf_net_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Main force net rank: NormalRank(mf_net_ratio)",
        ),
        ColumnSchema(
            name="amount_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Cross-sectional normal rank of raw amount.",
        ),
        ColumnSchema(
            name="velocity_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Velocity cross-sectional normal rank: NormalRank(ln(Close_t / Close_{t-1}))",
        ),
        ColumnSchema(
            name="amihud_rank",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Signed Amihud cross-sectional normal rank: NormalRank(Velocity / Amount)",
        ),
    ),
)

# Target: prediction labels
PROCESSED_LABEL_SCHEMA = TableSchema(
    name="processed_label",
    layer="processed",
    description="Processed labels for ret / rv dual-target modeling.",
    primary_key=("code", "trade_date"),
    partition_by=("code",),
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
            name="label_ret",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Mean adjusted open over T+2..T+LABEL_WINDOW divided by adjusted open at T+1, minus 1.",
        ),
        ColumnSchema(
            name="label_rv",
            dtype=pl.Float64,
            required=True,
            nullable=True,
            description="Future Parkinson volatility over the next LABEL_WINDOW trading days.",
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
