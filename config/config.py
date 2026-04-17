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
latent_dim = 64
num_attention_heads = 4
path_token_count = 4
liquid_token_count = 3
joint_token_count = 4
side_token_count = 4
cross_scale_topk = 4
macro_wno_num_blocks = 2
mezzo_wno_num_blocks = 2
micro_wno_num_blocks = 2
macro_wno_levels = 2
mezzo_wno_levels = 2
micro_wno_levels = 2
path_local_kernel = 3
liquidity_local_kernel = 3
state_local_kernel = 3
cross_scale_window_macro_to_mezzo = 8
cross_scale_window_mezzo_to_micro = 8
cross_scale_window_micro_to_mezzo = 8
label_schema_version = 7

active_target = "ret"
active_quantile = 0.90
quantile_set = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
variance_eps = 1e-6
variance_log_clamp_min = -12.0
variance_log_clamp_max = 8.0
student_t_nu = 5.0
# Auxiliary scale-loss weights.
# ret: main and auxiliary are both Student-T family losses, so keep the extra
# scale emphasis small to avoid the model "learning scale first".
ret_nll_weight = 0.05
# rv / quantile: main loss does not directly train the uncertainty scale branch,
# so keep a moderate auxiliary weight.
rv_nll_weight = 0.10
quantile_nll_weight = 0.10

batch_size = 384
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 300
train_val_split_date: float | None = None
val_ratio = 0.1
memory_mode = "lazy_packed"
lazy_cache_codes = 128
packed_min_files_per_code = 32
train_samples_per_epoch = 2_000_000
val_samples_per_epoch = None
trend_ema_alpha = 0.2
diagnostics_every_steps = 10
log_every = 50
mlflow_enabled = True
save_every = 5
grad_clip = 1.0
early_stopping_patience = 5
num_workers = 16
prefetch_factor = 32
preload_index_into_memory = True
amp_enabled = True
compile_mode = "reduce-overhead"  # "max-autotune" (with cudagraph) or "reduce-overhead" (without)

scheduler_name = "plateau"
scheduler_factor = 0.5
scheduler_patience = 2
scheduler_min_lr = 1e-6

train_seed = 114514
