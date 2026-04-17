"""Training loop for one epoch."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
import time

import torch
from torch.amp import GradScaler, autocast
from config.config import log_every as DEFAULT_LOG_EVERY
from torch import nn
from torch.utils.data import DataLoader

from src.train.visualize_strict import MLflowVisualizer
from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    move_batch_to_device,
)


def _cuda_amp_autocast_kwargs(amp_enabled: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "device_type": "cuda",
        "enabled": amp_enabled,
        "cache_enabled": False,
    }
    if amp_enabled:
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    *,
    grad_clip: float | None = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
    visualizer: MLflowVisualizer | None = None,
    epoch: int | None = None,
    global_step_offset: int = 0,
    log_every: int = DEFAULT_LOG_EVERY,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
) -> tuple[dict[str, float], None]:
    """Run one training epoch and return averaged metrics."""
    model.train()
    criterion.train()

    tracker = MetricTracker(device=device)
    window_tracker = MetricTracker(device=device)
    total_steps = max(len(dataloader), 1)
    last_sync_time = time.perf_counter()

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )

        with autocast(device_type="cuda", enabled=amp_enabled):
            loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        diag_metrics: dict[str, torch.Tensor | float] = {}
        if visualizer is not None:
            diag_metrics = visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            visualizer.update_epoch_buffer("train", model, outputs, batch, metrics=diag_metrics)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if grad_clip is not None:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        current_grad_norm = None

        if amp_enabled and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_size = int(batch["macro"].shape[0])
        step_metrics = {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}
        tracker.update(step_metrics, weight=batch_size)
        window_tracker.update(step_metrics, weight=batch_size)

        should_sync = (
            log_every <= 0
            or step_idx % log_every == 0
            or step_idx == total_steps
        )
        if should_sync:
            now = time.perf_counter()
            elapsed = max(0.0, now - last_sync_time)
            steps_in_window = log_every if log_every > 0 and step_idx % log_every == 0 else 1
            step_time_seconds = elapsed / max(steps_in_window, 1)
            last_sync_time = now
            synced_metrics = window_tracker.compute_and_reset(reset=True)
            if visualizer is not None:
                lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
                realtime_metrics = visualizer.realtime_metrics(
                    "train",
                    synced_metrics,
                    lr=lr,
                    grad_global_norm_value=current_grad_norm,
                )
                visualizer.track(
                    "train",
                    realtime_metrics,
                    step=global_step_offset + step_idx,
                    epoch=epoch,
                    subset="realtime",
                )
            if step_callback is not None:
                step_callback(
                    {
                        "step": float(step_idx),
                        "total_steps": float(total_steps),
                        "metrics": synced_metrics,
                        "step_time_seconds": float(step_time_seconds),
                    }
                )

    return tracker.compute(), None


__all__ = ["train_one_epoch"]
