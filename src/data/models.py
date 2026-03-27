from __future__ import annotations

import warnings
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Optional, Callable, Literal
from pathlib import Path

from src.data.schemas.raw import TableSchema


warnings.filterwarnings("ignore", message='Field name "schema" in "Params" shadows an attribute in parent "BaseModel"')


class Query(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='allow')

    desc: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @model_validator(mode="after")
    def check(self):

        if (self.start_date is None) ^ (self.end_date is None):
            raise ValueError("start_date and end_date must be provided together")

        return self


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api: Callable
    provider: Callable
    reader: Callable
    writer: Callable
    sreader: Callable
    path: Path
    schema: TableSchema
    desc: Literal["universe", "calendar", "daily", "adj_factor", "5min", "moneyflow", "limit", "namechange", "suspend"]
