"""Type aliases and literals for the data pipeline."""
from __future__ import annotations

from typing import Literal, TypeAlias

import polars as pl

# Mapping from source column names to canonical names
Map = dict[str, str]

# Stock exchange identifiers
Exchange = Literal["SSE", "SZSE", "BSE"]

# Stock status codes: L=listed, D=delisted, P=pre-IPO, G=growth
Status = Literal["L", "D", "P", "G"]

# Pipeline actions
Action = Literal["fetch", "load", "validate"]

# Supported Polars data types for schema columns
DType: TypeAlias = (
    type[pl.String]
    | type[pl.Float64]
    | type[pl.Float32]
    | type[pl.Int64]
    | type[pl.Int32]
    | type[pl.Boolean]
    | type[pl.Date]
    | type[pl.Time]
)
