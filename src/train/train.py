"""Training loop for one epoch."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train.visualize import MLflowVisualizer
from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    grad_norm,
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
) -> dict[str, float]:
    """Run one training epoch and return averaged metrics."""
    model.train()
    criterion.train()
    tracker = MetricTracker()
    total_steps = max(len(dataloader), 1)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
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
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        current_grad_norm = grad_norm(model.parameters()) if visualizer is not None else None

        optimizer.step()

        batch_size = int(batch["macro"].shape[0])
        tracker.update({"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}, weight=batch_size)

        if visualizer is not None:
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
                step=step_idx,
                epoch=epoch,
                subset="realtime",
            )

        if step_callback is not None:
            step_callback(
                {
                    "step": float(step_idx),
                    "total_steps": float(total_steps),
                    "metrics": tracker.compute(),
                }
            )

    return tracker.compute()


__all__ = ["train_one_epoch"]
