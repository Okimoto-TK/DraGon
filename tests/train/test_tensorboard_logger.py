from __future__ import annotations

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.train.tensorboard_logger import TensorBoardLogger


def test_mu_tensorboard_logger_writes_epoch_loss_and_hexbin(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "mu",
        task="mu",
        field="ret",
        enabled=True,
    )

    logger.update_mu_state(
        phase="train",
        predictions=torch.randn(128, 1),
        targets=torch.randn(128, 1),
    )
    logger.log_epoch_metrics(
        phase="train",
        epoch=0,
        metrics={"loss_total": 1.2, "loss_mu": 1.2},
    )
    logger.log_epoch_plots(
        phase="train",
        epoch=0,
    )
    logger.close()

    accumulator = EventAccumulator(str(tmp_path / "tb" / "mu"))
    accumulator.Reload()
    scalar_tags = accumulator.Tags()["scalars"]
    image_tags = accumulator.Tags()["images"]
    all_tags = scalar_tags + image_tags

    assert "epoch_train/loss_mu" in scalar_tags
    assert "epoch_train/mu_pred_vs_target_hexbin" in image_tags
    assert "epoch_train/epoch_index" not in scalar_tags
    assert all("confident" not in tag for tag in all_tags)
    assert all("uncertainty_" not in tag for tag in all_tags)


def test_sigma_tensorboard_logger_writes_nll_attention_and_hexbin(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "sigma",
        task="sigma",
        field="ret",
        enabled=True,
    )

    logger.update_sigma_state(
        phase="val",
        mu_input=torch.randn(96, 1),
        targets=torch.randn(96, 1),
        sigmas=torch.rand(96, 1).clamp_min(1e-3),
        attn_entropy=torch.tensor(1.3),
        attn_max_weight=torch.tensor(0.42),
    )
    logger.log_epoch_metrics(
        phase="val",
        epoch=2,
        metrics={"loss_total": 0.4, "loss_nll": 0.4},
    )
    logger.log_epoch_plots(
        phase="val",
        epoch=2,
    )
    logger.close()

    accumulator = EventAccumulator(str(tmp_path / "tb" / "sigma"))
    accumulator.Reload()
    scalar_tags = accumulator.Tags()["scalars"]
    image_tags = accumulator.Tags()["images"]
    all_tags = scalar_tags + image_tags

    assert "epoch_val/loss_nll" in scalar_tags
    assert "epoch_val/conf_attn_entropy_mean" in scalar_tags
    assert "epoch_val/conf_attn_max_weight_mean" in scalar_tags
    assert "epoch_val/residual_vs_sigma_hexbin" in image_tags
    assert "epoch_val/epoch_index" not in scalar_tags
    assert all("confident" not in tag for tag in all_tags)
    assert all("uncertainty_" not in tag for tag in all_tags)


def test_mu_tensorboard_logger_keeps_state_on_input_device(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "device",
        task="mu",
        field="ret",
        enabled=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    predictions = torch.randn(32, 1, device=device, dtype=dtype)
    targets = torch.randn(32, 1, device=device, dtype=dtype)

    logger.update_mu_state(
        phase="train",
        predictions=predictions,
        targets=targets,
    )

    state = logger._phase_state["train"]
    assert state["predictions"][0].device == predictions.device
    assert state["targets"][0].device == targets.device
    logger.close()


def test_sigma_tensorboard_logger_clones_prediction_state_storage(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "clone",
        task="sigma",
        field="ret",
        enabled=True,
    )
    mu_input = torch.randn(8, 1)
    targets = torch.randn(8, 1)
    sigmas = torch.rand(8, 1).clamp_min(1e-3)

    logger.update_sigma_state(
        phase="train",
        mu_input=mu_input,
        targets=targets,
        sigmas=sigmas,
        attn_entropy=torch.tensor(1.0),
        attn_max_weight=torch.tensor(0.5),
    )

    state = logger._phase_state["train"]
    sigma_cached = state["sigmas"][0]

    assert sigma_cached.data_ptr() != sigmas.reshape(-1).data_ptr()

    logger.log_epoch_plots(
        phase="train",
        epoch=0,
    )
    assert "train" not in logger._phase_state
    logger.close()
