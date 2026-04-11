"""Data models for queries, pipeline parameters, and schema definitions."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, model_validator

from src.data.types import DType

# Suppress pydantic warning about shadowing 'schema' field name
warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "Params" shadows an attribute in parent "BaseModel"',
)


# === Schema Definition Classes ===

@dataclass(frozen=True)
class ColumnSchema:
    """Schema definition for a single column in a table."""

    name: str
    dtype: DType
    required: bool = True
    nullable: bool = True
    fmt: str | None = None  # Regex pattern or date/time format
    unit: str | None = None  # Unit of measurement
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Schema definition for an entire table."""

    name: str
    layer: str
    description: str
    primary_key: tuple[str, ...]
    partition_by: tuple[str, ...]
    columns: tuple[ColumnSchema, ...]
    allow_extra_columns: bool = True
    provider_select_only_schema_columns: bool = True

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return tuple of all column names."""
        return tuple(col.name for col in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return tuple of required column names."""
        return tuple(col.name for col in self.columns if col.required)

    @property
    def column_names_and_types(self) -> dict[str, type]:
        """Return dict mapping column names to their data types."""
        return {col.name: col.dtype for col in self.columns}

    def get_column(self, name: str) -> ColumnSchema:
        """Get column schema by name.

        Args:
            name: Column name to look up.

        Returns:
            ColumnSchema for the requested column.

        Raises:
            KeyError: If column name not found.
        """
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{name!r} not found in schema {self.name}")


# === Query and Pipeline Parameter Models ===

class Query(BaseModel):
    """Query parameters for data fetching with date range validation."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    desc: str
    start_date: str | None = None
    end_date: str | None = None

    @model_validator(mode="after")
    def check_dates(self) -> Query:
        """Validate that start_date and end_date are both provided or neither."""
        if (self.start_date is None) ^ (self.end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        return self


class Params(BaseModel):
    """Base pipeline parameters for data type handlers."""

    model_config = ConfigDict(extra="forbid")

    reader: Callable
    writer: Callable
    sreader: Callable
    path: Path
    schema: TableSchema


class RawParams(Params):
    """Pipeline parameters binding API, storage, and schema for raw data."""

    api: Callable
    provider: Callable
    desc: Literal[
        "universe", "calendar", "daily", "adj_factor",
        "5min", "moneyflow", "limit", "namechange", "suspend",
    ]


class ProcessedParams(Params):
    """Pipeline parameters for processed data storage and retrieval."""

    processor: Callable
    proc: str = "_process"  # Method name to dispatch (e.g., "_process", "_process_chunk")
    desc: Literal[
        "index", "mask", "macro", "mezzo", "micro", "sidechain", "label",
    ]
    raw_deps: dict[str, str] = {}  # kwarg_name -> raw_type (eager loaded)
    processor_kwargs: dict[str, Any] = {}
