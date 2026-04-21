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
        debug_every=1,
    )
    output = {
        "pred_primary": torch.randn(4, 1),
        "pred_aux_raw": torch.randn(4, 1),
        "pred_mu_ret": torch.randn(4, 1),
        "pred_scale_ret_raw": torch.randn(4, 1),
        "sigma_pred": torch.rand(4, 1).clamp_min(1e-3),
        "nu_ret": torch.full((1,), 8.0),
        "feature_rms_macro_pre": torch.rand(22),
        "feature_rms_macro_post": torch.rand(22),
        "feature_rms_mezzo_pre": torch.rand(9),
        "feature_rms_mezzo_post": torch.rand(9),
        "feature_rms_micro_pre": torch.rand(9),
        "feature_rms_micro_post": torch.rand(9),
        "_debug": {
            "wavelet_macro_energy_raw": 1.0,
            "wavelet_macro_energy_denoised": 0.8,
            "encoder_macro_final_block_act_mean": 0.1,
            "within_scale_macro_feature_cosdist_post": 0.3,
            "cross_scale_macro_ctx_l2_mean": 2.0,
            "cross_scale_macro_to_mezzo_gate_mean": 0.0,
            "head_head_context_l2_mean": 3.0,
            "head_task_attn_entropy_mean": 1.2,
        },
    }

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
    logger.log_debug_snapshot(
        phase="train",
        global_step=8,
        output=output,
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
    histogram_tags = accumulator.Tags()["histograms"]
    image_tags = accumulator.Tags()["images"]
    all_tags = scalar_tags + histogram_tags

    assert "step_train/loss_total" in scalar_tags
    assert "step_train/loss_ret_weighted_nll" in scalar_tags
    assert "epoch_train/loss_total" in scalar_tags
    assert "epoch_train/loss_ret_weighted_nll" in scalar_tags
    assert "debug_train/cross_scale_macro_ctx_l2_mean" in scalar_tags
    assert "debug_train/head_head_context_l2_mean" in scalar_tags
    assert "debug_train/loss_sigma_pred_mean" in scalar_tags
    assert "debug_train/head_pred_primary_hist" in histogram_tags
    assert "debug_train/feature_macro_activation_hist" in image_tags
    assert all("sidechain_merge" not in tag for tag in all_tags)
