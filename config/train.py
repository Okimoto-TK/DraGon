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
    return max(4, min(16, cpu_count // 2))


@dataclass(frozen=True)
class TrainingConfig:
    """Open tuning parameters for the training stack."""

    batch_size: int = 2048
    val_batch_size: int = 2048
    num_workers: int = field(default_factory=_default_num_workers)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor_train: int = 4
    prefetch_factor_val: int = 2
    mmap_mode: str | None = None
    max_open_archives: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    log_every: int = 50
    hist_every: int = 500
    viz_every: int = 1000
    num_epochs: int = 20
    save_every: int = 1
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    dynamo_recompile_limit: int = 64
    dynamo_accumulated_recompile_limit: int = 2048
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = False
    enable_wandb: bool = True
    wandb_project: str | None = "dragon"
    wandb_run_group: str | None = "dev"
    wandb_base_url: str | None = None


training = TrainingConfig()
