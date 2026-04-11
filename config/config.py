"""Global configuration and path settings."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from src.data.types import Exchange, Status

# Debug & validation settings
strict = True
debug = False

# Default filter criteria for stock universe
DEFAULT_EXCHANGE: Sequence[Exchange] = ["SSE", "SZSE"]
DEFAULT_STATUS: Sequence[Status] = ["L", "D"]

# Directory paths
base_dir = Path(__file__).resolve().parent.parent
log_dir = base_dir / "logs"

data_dir = base_dir / "data"
cache_dir = base_dir / "cache"

raw_dir = data_dir / "raw"

# Raw parquet file paths
raw_path = SimpleNamespace(
    universe_path=raw_dir / "universe.parquet",
    calendar_path=raw_dir / "calendar.parquet",
    daily_dir=raw_dir / "daily",
    adj_factor_dir=raw_dir / "adj_factor",
    r5min_dir=raw_dir / "5min",
    moneyflow_dir=raw_dir / "moneyflow",
    limit_dir=raw_dir / "limit",
    namechange_dir=raw_dir / "namechange",
    suspend_dir=raw_dir / "suspend",
)

processed_dir = data_dir / "processed"

# Processed parquet file paths
processed_path = SimpleNamespace(
    index_path=processed_dir / "index.parquet",
    mask_dir=processed_dir / "mask",
    macro_dir=processed_dir / "macro",
    mezzo_dir=processed_dir / "mezzo",
    micro_dir=processed_dir / "micro",
    sidechain_dir=processed_dir / "sidechain",
    label_dir=processed_dir / "label",
)

assembled_dir = data_dir / "assembled"