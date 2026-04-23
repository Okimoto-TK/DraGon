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
        uncertainties=torch.exp(torch.randn(128, 1)),
    )
    logger.close()

    event_files = list((tmp_path / "tb" / "unit").glob("events.out.tfevents.*"))
    assert event_files

    accumulator = EventAccumulator(str(tmp_path / "tb" / "unit"))
    accumulator.Reload()
    scalar_tags = accumulator.Tags()["scalars"]
    image_tags = accumulator.Tags()["images"]
    all_tags = scalar_tags + image_tags

    assert "epoch_train/loss_total" in scalar_tags
    assert "epoch_train/loss_ret_weighted_nll" in scalar_tags
    assert "epoch_train/pred_vs_target_heatmap_confident" in image_tags
    assert "epoch_train/pred_vs_target_heatmap_moderately_confident" in image_tags
    assert "epoch_train/pred_vs_target_heatmap_unconfident" in image_tags
    assert all("debug_" not in tag for tag in all_tags)
    assert all("sidechain_merge" not in tag for tag in all_tags)


def test_tensorboard_logger_keeps_prediction_state_on_input_device(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "device",
        task="ret",
        enabled=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = torch.randn(32, 1, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    targets = torch.randn(32, 1, device=device, dtype=predictions.dtype)
    uncertainties = torch.exp(torch.randn(32, 1, device=device, dtype=predictions.dtype))

    logger.update_prediction_state(
        phase="train",
        predictions=predictions,
        targets=targets,
        uncertainties=uncertainties,
    )

    state = logger._pred_heatmap_state["train"]
    assert state["predictions"][0].device == predictions.device
    assert state["targets"][0].device == targets.device
    assert state["uncertainties"][0].device == uncertainties.device
    logger.close()


def test_tensorboard_logger_clones_prediction_state_storage(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "clone",
        task="ret",
        enabled=True,
    )
    predictions = torch.randn(8, 1)
    targets = torch.randn(8, 1)
    uncertainties = torch.exp(torch.randn(8, 1))

    logger.update_prediction_state(
        phase="train",
        predictions=predictions,
        targets=targets,
        uncertainties=uncertainties,
    )

    state = logger._pred_heatmap_state["train"]
    pred_cached = state["predictions"][0]
    trg_cached = state["targets"][0]
    unc_cached = state["uncertainties"][0]

    assert pred_cached.data_ptr() != predictions.reshape(-1).data_ptr()
    assert trg_cached.data_ptr() != targets.reshape(-1).data_ptr()
    assert unc_cached.data_ptr() != uncertainties.reshape(-1).data_ptr()
    logger.close()
