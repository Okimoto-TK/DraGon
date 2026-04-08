"""Data models for queries and pipeline parameters."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from src.data.schemas.raw import TableSchema

# Suppress pydantic warning about shadowing 'schema' field name
warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "Params" shadows an attribute in parent "BaseModel"',
)


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
    """Pipeline parameters binding API, storage, and schema for a data type."""

    model_config = ConfigDict(extra="forbid")

    api: Callable
    provider: Callable
    reader: Callable
    writer: Callable
    sreader: Callable
    path: Path
    schema: TableSchema
    desc: Literal[
        "universe", "calendar", "daily", "adj_factor",
        "5min", "moneyflow", "limit", "namechange", "suspend",
    ]
