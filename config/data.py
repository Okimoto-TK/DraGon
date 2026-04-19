"""Data-side configuration and path settings (frozen)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from src.data.types import Exchange, Status

# ---------------------------------------------------------------------------
# Debug & validation
# ---------------------------------------------------------------------------
strict = True
debug = False

# ---------------------------------------------------------------------------
# Stock universe defaults
# ---------------------------------------------------------------------------
DEFAULT_EXCHANGE: Sequence[Exchange] = ["SSE", "SZSE"]
DEFAULT_STATUS: Sequence[Status] = ["L", "D"]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent.parent
log_dir = base_dir / "log"
data_dir = base_dir / "data"
cache_dir = base_dir / "cache"
models_dir = base_dir / "models"
checkpoint_dir = models_dir / "checkpoints"

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

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------
label_schema_version = 8
quantile_set = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# ---------------------------------------------------------------------------
# Assembler defaults
# ---------------------------------------------------------------------------
packed_min_files_per_code = 32
train_seed = 114514

# ---------------------------------------------------------------------------
# Feature engineering defaults
# ---------------------------------------------------------------------------
diff_d = 0.5
