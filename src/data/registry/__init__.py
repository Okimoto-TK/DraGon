"""Registry module for raw and processed pipeline parameters."""
from __future__ import annotations

from importlib import import_module

from .api import (
    FETCH_FIELD_5MIN,
    FETCH_FIELD_ADJ_FACTOR,
    FETCH_FIELD_CAL,
    FETCH_FIELD_DAILY,
    FETCH_FIELD_LIMIT,
    FETCH_FIELD_MONEYFLOW,
    FETCH_FIELD_NAMECHANGE,
    FETCH_FIELD_SUSPEND,
    FETCH_FIELD_UNIVERSE,
    FIELD_MAP_5MIN,
    FIELD_MAP_ADJ_FACTOR,
    FIELD_MAP_CAL,
    FIELD_MAP_DAILY,
    FIELD_MAP_LIMIT,
    FIELD_MAP_MONEYFLOW,
    FIELD_MAP_NAMECHANGE,
    FIELD_MAP_SUSPEND,
    FIELD_MAP_UNIVERSE,
)
from .dataset import (
    MACRO_LOOKBACK,
    MEZZO_LOOKBACK,
    MICRO_LOOKBACK,
    WARMUP_BARS,
)
from .processor import (
    LABEL_WINDOW,
)

__all__ = [
    # API field mappings
    "FETCH_FIELD_UNIVERSE",
    "FETCH_FIELD_CAL",
    "FETCH_FIELD_DAILY",
    "FETCH_FIELD_ADJ_FACTOR",
    "FETCH_FIELD_MONEYFLOW",
    "FETCH_FIELD_LIMIT",
    "FETCH_FIELD_NAMECHANGE",
    "FETCH_FIELD_SUSPEND",
    "FETCH_FIELD_5MIN",
    "FIELD_MAP_UNIVERSE",
    "FIELD_MAP_CAL",
    "FIELD_MAP_DAILY",
    "FIELD_MAP_ADJ_FACTOR",
    "FIELD_MAP_MONEYFLOW",
    "FIELD_MAP_LIMIT",
    "FIELD_MAP_NAMECHANGE",
    "FIELD_MAP_SUSPEND",
    "FIELD_MAP_5MIN",
    # Processor constants
    "MACRO_LOOKBACK",
    "MEZZO_LOOKBACK",
    "MICRO_LOOKBACK",
    "WARMUP_BARS",
    "LABEL_WINDOW",
    # Parameter maps
    "PARAM_MAP",
    "PROCESSED_PARAM_MAP",
]


def __getattr__(name: str):
    """Lazily load heavy registry maps to avoid circular imports."""
    if name == "PARAM_MAP":
        return import_module("src.data.registry.raw").PARAM_MAP
    if name == "PROCESSED_PARAM_MAP":
        return import_module("src.data.registry.processed").PROCESSED_PARAM_MAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
