from __future__ import annotations

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.train.tensorboard_logger import TensorBoardLogger


def test_tensorboard_logger_writes_grouped_scalars_without_sidechain_merge(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "unit",
        task="ret",
        enabled=True,
    )

    logger.log_step(
        phase="train",
        global_step=8,
        losses={"loss_total": 1.5, "loss_task": 1.5, "loss_ret_weighted_nll": 1.5},
        lr=3e-4,
        grad_norm=0.8,
        param_norm=10.0,
        step_time_ms=12.0,
        samples_per_sec=2048.0,
    )
    logger.log_epoch_metrics(
        phase="train",
        global_step=8,
        epoch=0,
        metrics={"loss_total": 1.2, "loss_task": 1.2, "loss_ret_weighted_nll": 1.2},
    )
    logger.log_prediction_plot(
        phase="train",
        global_step=8,
        predictions=torch.randn(128, 1),
        targets=torch.randn(128, 1),
    )
    logger.close()

    event_files = list((tmp_path / "tb" / "unit").glob("events.out.tfevents.*"))
    assert event_files

    accumulator = EventAccumulator(str(tmp_path / "tb" / "unit"))
    accumulator.Reload()
    scalar_tags = accumulator.Tags()["scalars"]
    image_tags = accumulator.Tags()["images"]
    all_tags = scalar_tags + image_tags

    assert "step_train/loss_total" in scalar_tags
    assert "step_train/loss_ret_weighted_nll" in scalar_tags
    assert "epoch_train/loss_total" in scalar_tags
    assert "epoch_train/loss_ret_weighted_nll" in scalar_tags
    assert "epoch_train/pred_vs_target_heatmap" in image_tags
    assert all("debug_" not in tag for tag in all_tags)
    assert all("sidechain_merge" not in tag for tag in all_tags)
