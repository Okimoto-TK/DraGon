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
log_dir = base_dir / "log"
out_dir = base_dir / "out"
mlflow_dir = log_dir / "mlruns"
checkpoint_dir = out_dir / "checkpoints"

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

# Model defaults
hidden_dim = 64
side_hidden_dim = 32
lmf_dim = 32
token_dim = 24
summary_dim = 16
lmf_rank = 6
latent_token = 12
macro_decomp_level = 3
mezzo_decomp_level = 3
micro_decomp_level = 3
macro_wno_num_blocks = 2
mezzo_wno_num_blocks = 2
micro_wno_num_blocks = 3
jointnet_23_blocks = 4

batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 300
train_val_split_date: float | None = None
val_ratio = 0.1
memory_mode = "lazy_packed"
lazy_cache_codes = 16
packed_min_files_per_code = 32
train_samples_per_epoch = 2_000_000
val_samples_per_epoch = None
trend_ema_alpha = 0.2
diagnostics_every_steps = 10
log_every = 50
mlflow_enabled = True
save_every = 5
freeze_scale_s0_S = 0.018
freeze_scale_s0_M = 0.027
freeze_scale_s0_MDD = 0.045
freeze_scale_s0_RV = 0.018
freeze_min_steps = 1000
freeze_patience_steps = 100
freeze_ema_beta = 0.98
grad_clip = 1.0
early_stopping_patience = 5
num_workers = 16
prefetch_factor = 4
amp_enabled = True
use_cuda_graph = True
cuda_graph_warmup_steps = 8

scheduler_name = "plateau"
scheduler_factor = 0.5
scheduler_patience = 2
scheduler_min_lr = 1e-6

train_seed = 114514
