"""Validation loop."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train.utils import MetricTracker, batch_prediction_metrics, move_batch_to_device
from src.train.visualize import MLflowVisualizer


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    visualizer: MLflowVisualizer | None = None,
) -> dict[str, float]:
    """Run one validation epoch and return averaged metrics."""
    model.eval()
    criterion.eval()
    tracker = MetricTracker()

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["macro"],
            batch["mezzo"],
            batch["micro"],
            batch["sidechain"],
        )
        loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        diag_metrics = (
            visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            if visualizer is not None
            else {}
        )

        batch_size = int(batch["macro"].shape[0])
        tracker.update({"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}, weight=batch_size)

    return tracker.compute()


__all__ = ["validate"]
