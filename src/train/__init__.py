"""Training entrypoints and trainer orchestration."""

from .train_entry import run_training
from .trainer import Trainer

__all__ = ["Trainer", "run_training"]
