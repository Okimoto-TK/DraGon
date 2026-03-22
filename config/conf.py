from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

data_dir = base_dir / 'data'
raw_dir = data_dir / 'raw'
r5min_dir = raw_dir / '5min'
