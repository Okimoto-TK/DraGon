"""Validation loop."""
from __future__ import annotations

import torch
from torch.amp import autocast
from config.config import diagnostics_every_steps as DEFAULT_DIAGNOSTICS_EVERY_STEPS
from torch import nn
from torch.utils.data import DataLoader

from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    move_batch_to_device,
)
from src.train.visualize import MLflowVisualizer


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    visualizer: MLflowVisualizer | None = None,
    diagnostics_every_steps: int = DEFAULT_DIAGNOSTICS_EVERY_STEPS,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Run one validation epoch and return averaged metrics."""
    model.eval()
    criterion.eval()
    tracker = MetricTracker()

    total_steps = max(len(dataloader), 1)
    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        should_capture_snapshot = visualizer is not None and step_idx == total_steps
        set_debug_capture = getattr(model, "set_debug_capture", None)
        if callable(set_debug_capture):
            set_debug_capture(should_capture_snapshot)

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        if visualizer is not None:
            visualizer.update_epoch_buffer("val", model, outputs, batch)
        should_collect_diag = (
            visualizer is not None
            and (
                step_idx == 1
                or step_idx == total_steps
                or (diagnostics_every_steps > 0 and step_idx % diagnostics_every_steps == 0)
            )
        )
        diag_metrics = (
            visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            if should_collect_diag
            else {}
        )
        if visualizer is not None and step_idx == total_steps:
            visualizer.capture_epoch_snapshot("val", model, outputs, batch)
        if callable(set_debug_capture):
            set_debug_capture(False)

        batch_size = int(batch["macro"].shape[0])
        tracker.update(
            {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics},
            weight=batch_size,
        )

    return tracker.compute()


__all__ = ["validate"]
