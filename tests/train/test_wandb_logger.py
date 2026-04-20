from __future__ import annotations

import pytest
import torch

from src.train.wandb_logger import WandbLoggerConfig, WandbVisualizationLogger


def _make_disabled_logger() -> WandbVisualizationLogger:
    return WandbVisualizationLogger(
        config=WandbLoggerConfig(enabled=False),
        run_name="test",
        run_config={},
    )


def test_add_stats_records_mean_std_and_minmax() -> None:
    logger = _make_disabled_logger()
    payload: dict[str, object] = {}
    value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    logger._add_stats(  # pyright: ignore[reportPrivateUsage]
        payload,
        prefix="train/example",
        value=value,
        include_minmax=True,
    )

    assert payload["train/example/mean"] == pytest.approx(2.5)
    assert payload["train/example/std"] == pytest.approx(float(value.std(unbiased=False)))
    assert payload["train/example/min"] == pytest.approx(1.0)
    assert payload["train/example/max"] == pytest.approx(4.0)


def test_log_debug_scalars_uses_feature_cosdist_metric_names() -> None:
    logger = _make_disabled_logger()
    payload: dict[str, object] = {}
    z_pre = torch.randn(2, 3, 4, 5)
    z_post = torch.randn(2, 3, 4, 5)

    logger._log_debug_scalars(  # pyright: ignore[reportPrivateUsage]
        split="train",
        payload=payload,
        output={
            "_debug": {
                "within_scale": {
                    "macro_pre": z_pre,
                    "macro_post": z_post,
                }
            }
        },
    )

    assert "train_within_scale_macro/feature_cosdist_pre" in payload
    assert "train_within_scale_macro/feature_cosdist_post" in payload
    assert "train_within_scale_macro/feature_cosdist_ratio" in payload
    assert "train_within_scale_macro/feature_var_pre" not in payload
    assert "train_within_scale_macro/feature_var_post" not in payload
    assert "train_within_scale_macro/feature_diversity_ratio" not in payload
