"""Training loop for one epoch."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.amp import GradScaler, autocast
from config.config import diagnostics_every_steps as DEFAULT_DIAGNOSTICS_EVERY_STEPS
from config.config import log_every_steps as DEFAULT_LOG_EVERY_STEPS
from torch import nn
from torch.utils.data import DataLoader

from src.train.visualize import MLflowVisualizer
from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    move_batch_to_device,
)


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
    diagnostics_every_steps: int = DEFAULT_DIAGNOSTICS_EVERY_STEPS,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    profiler: object | None = None,
    log_every_steps: int = DEFAULT_LOG_EVERY_STEPS,
) -> dict[str, float]:
    """Run one training epoch and return averaged metrics."""
    model.train()
    criterion.train()
    tracker = MetricTracker()
    total_steps = max(len(dataloader), 1)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        should_collect_diag = (
            visualizer is not None
            and (
                step_idx == 1
                or step_idx == total_steps
                or (diagnostics_every_steps > 0 and step_idx % diagnostics_every_steps == 0)
            )
        )

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
        if visualizer is not None:
            visualizer.update_epoch_buffer("train", model, outputs, batch)
        diag_metrics = (
            visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            if should_collect_diag
            else {}
        )
        if visualizer is not None and step_idx == total_steps:
            visualizer.capture_epoch_snapshot("train", model, outputs, batch)

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
        tracker.update(
            {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics},
            weight=batch_size,
        )

        if should_collect_diag and visualizer is not None:
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
            realtime_metrics = visualizer.realtime_metrics(
                "train",
                diag_metrics,
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

        should_log_metrics = (
            step_idx == 1
            or step_idx == total_steps
            or (log_every_steps > 0 and step_idx % log_every_steps == 0)
        )
        if step_callback is not None:
            metrics_payload = tracker.compute(as_python=True) if should_log_metrics else {}
            step_callback(
                {
                    "step": float(step_idx),
                    "total_steps": float(total_steps),
                    "metrics": metrics_payload,
                }
            )

        if profiler is not None:
            profiler.step()

    return tracker.compute(as_python=True)  # type: ignore[return-value]


__all__ = ["train_one_epoch"]
