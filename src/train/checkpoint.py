"""Checkpoint save and load helpers for training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _resolve_checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "latest.pt"
    return resolved


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int,
    global_step: int,
) -> Path:
    """Persist model and optimizer runtime state."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap_model(model)
    payload: dict[str, Any] = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    load_training_state: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into the provided runtime objects."""

    checkpoint_path = _resolve_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location=map_location)

    if model is not None:
        _unwrap_model(model).load_state_dict(payload["model"])

    if load_training_state:
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])

    payload["checkpoint_path"] = str(checkpoint_path)
    return payload
