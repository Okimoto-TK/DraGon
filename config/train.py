"""Training configuration exposed separately from model architecture config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_num_workers() -> int:
    """Pick a worker count that better matches multi-core training hosts."""

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        cpu_count = os.cpu_count() or 8
    return max(1, min(8, cpu_count))


def _default_val_num_workers() -> int:
    """Use a larger worker pool for validation dataloading."""

    return max(1, min(8, _default_num_workers() * 2))


@dataclass(frozen=True)
class TrainingConfig:
    """Open tuning parameters for the training stack."""

    batch_size: int = 3072
    val_batch_size: int = 3072
    num_workers: int = field(default_factory=_default_num_workers)
    val_num_workers: int = field(default_factory=_default_val_num_workers)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor_train: int = 8
    prefetch_factor_val: int = 4
    device_prefetch: bool = True
    device_prefetch_batches: int = 4
    validate_shapes: bool = False
    mmap_mode: str | None = None
    max_open_archives: int = 16
    lr: float = 3e-4
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    early_stop_patience: int = 10
    task: str = "ret"
    log_every: int = 500
    enable_tensorboard: bool = True
    tensorboard_root: str = "models/tensorboard"
    tensorboard_host: str = "0.0.0.0"
    tensorboard_port: int = 11111
    tensorboard_flush_secs: int = 30
    num_epochs: int = 100
    save_every: int = 1
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    dynamo_recompile_limit: int = 64
    dynamo_accumulated_recompile_limit: int = 2048
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True


training = TrainingConfig()
