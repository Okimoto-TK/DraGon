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
from src.train.visualize_strict import MLflowVisualizer


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
) -> dict[str, float]:
    """Run one validation epoch and return averaged metrics."""
    del diagnostics_every_steps
    model.eval()
    criterion.eval()
    tracker = MetricTracker(device=device)

    total_steps = max(len(dataloader), 1)
    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

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
            visualizer.update_epoch_buffer("val", model, outputs, batch, metrics=diag_metrics)
        if visualizer is not None and step_idx == total_steps:
            visualizer.capture_epoch_snapshot("val", model, outputs, batch)

        batch_size = int(batch["macro"].shape[0])
        tracker.update(
            {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics},
            weight=batch_size,
        )

    return tracker.compute()


__all__ = ["validate"]
