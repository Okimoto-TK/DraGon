from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from src.data.types import Exchange, Status

strict = True
debug = False

DEFAULT_EXCHANGE: Sequence[Exchange] = ["SSE", "SZSE"]
DEFAULT_STATUS: Sequence[Status] = ["L", "D"]

base_dir = Path(__file__).resolve().parent.parent
log_dir = base_dir / 'logs'

data_dir = base_dir / 'data'
cache_dir = base_dir / 'cache'

raw_dir = data_dir / 'raw'
raw_path = SimpleNamespace(**{
    "universe_path": raw_dir / "universe.parquet",
    "calendar_path": raw_dir / 'calendar.parquet',
    "daily_dir": raw_dir / 'daily',
    "adj_factor_dir": raw_dir / "adj_factor",
    "r5min_dir": raw_dir / '5min',
    "moneyflow_dir": raw_dir / 'moneyflow',
    "limit_dir": raw_dir / 'limit',
    "st_dir": raw_dir / 'st',
    "suspend_dir": raw_dir / 'suspend',
})
