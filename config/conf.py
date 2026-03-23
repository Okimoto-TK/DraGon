from pathlib import Path
from types import SimpleNamespace
strict = True
debug = True

base_dir = Path(__file__).resolve().parent.parent
log_dir = base_dir / 'logs'

data_dir = base_dir / 'data'
calendar_path = data_dir / 'calendar.parquet'

raw_dir = data_dir / 'raw'
raw = SimpleNamespace(**{
    "daily_dir": raw_dir / 'daily',
    "r5min_dir": raw_dir / '5min',
    "moneyflow_dir": raw_dir / 'moneyflow',
    "limit_dir": raw_dir / 'limit',
    "st_dir": base_dir / 'st',
    "suspend_dir": base_dir / 'suspend',
})
