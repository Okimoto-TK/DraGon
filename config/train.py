"""Training configuration exposed separately from model architecture config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Open tuning parameters for the training stack."""

    batch_size: int = 512
    val_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor_train: int = 2
    prefetch_factor_val: int = 1
    mmap_mode: str | None = None
    max_open_archives: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    log_every: int = 50
    num_epochs: int = 20
    save_every: int = 1
    compile_model: bool = True
    amp_dtype: str = "tf32"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True


training = TrainingConfig()
