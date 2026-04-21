from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import Dataset

from src.train.wandb_logger import WandbLoggerConfig, WandbVisualizationLogger


def _make_disabled_logger() -> WandbVisualizationLogger:
    return WandbVisualizationLogger(
        config=WandbLoggerConfig(enabled=False),
        run_name="test",
        run_config={},
    )


class _FakeWandbModule:
    class Image:
        def __init__(self, value) -> None:
            self.value = value

    class Histogram:
        def __init__(self, np_histogram) -> None:
            self.np_histogram = np_histogram

    def init(self, **_: object) -> None:
        return None

    def define_metric(self, *_: object, **__: object) -> None:
        return None

    def log(self, *_: object, **__: object) -> None:
        return None

    def finish(self) -> None:
        return None


def _make_enabled_logger(*, monkeypatch: pytest.MonkeyPatch, task: str) -> WandbVisualizationLogger:
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule())
    return WandbVisualizationLogger(
        config=WandbLoggerConfig(
            enabled=True,
            task=task,
            project="unit-test",
        ),
        run_name="test",
        run_config={},
    )


class _FixedValDataset(Dataset):
    def __init__(self, *, ret_values: list[float], rv_values: list[float]) -> None:
        if len(ret_values) != len(rv_values):
            raise ValueError("ret_values and rv_values must have the same length.")
        self.ret_values = ret_values
        self.rv_values = rv_values

    def __len__(self) -> int:
        return len(self.ret_values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ret = torch.tensor([self.ret_values[index]], dtype=torch.float32)
        rv = torch.tensor([self.rv_values[index]], dtype=torch.float32)
        return {
            "target_ret": ret,
            "target_rv": rv,
            "target_q": ret.clone(),
        }


@dataclass
class _FixedValLoader:
    dataset: Dataset
    batch_size: int

    @staticmethod
    def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = samples[0].keys()
        return {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in keys
        }


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


@pytest.mark.parametrize("task", ["ret", "q"])
def test_capture_fixed_val_batch_uses_ret_buckets_for_ret_and_q(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    logger = _make_enabled_logger(monkeypatch=monkeypatch, task=task)
    dataset = _FixedValDataset(
        ret_values=[-0.12, -0.105, -0.07, -0.055, 0.06, 0.08, 0.11, 0.14, 0.01],
        rv_values=[0.01] * 9,
    )
    loader = _FixedValLoader(dataset=dataset, batch_size=2)

    logger.capture_fixed_val_batch(loader)

    batches = logger.get_fixed_val_batches()
    assert set(batches) == {
        "ret_le_neg10",
        "ret_neg10_to_neg5",
        "ret_5_to_10",
        "ret_ge_10",
    }
    assert torch.all(batches["ret_le_neg10"]["target_ret"] * 100.0 <= -10.0)
    assert torch.all((batches["ret_neg10_to_neg5"]["target_ret"] * 100.0 > -10.0))
    assert torch.all((batches["ret_neg10_to_neg5"]["target_ret"] * 100.0 <= -5.0))
    assert torch.all((batches["ret_5_to_10"]["target_ret"] * 100.0 >= 5.0))
    assert torch.all((batches["ret_5_to_10"]["target_ret"] * 100.0 < 10.0))
    assert torch.all(batches["ret_ge_10"]["target_ret"] * 100.0 >= 10.0)


def test_capture_fixed_val_batch_uses_rv_buckets_for_rv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _make_enabled_logger(monkeypatch=monkeypatch, task="rv")
    dataset = _FixedValDataset(
        ret_values=[0.0] * 9,
        rv_values=[0.009, 0.012, 0.016, 0.020, 0.022, 0.028, 0.032, 0.050, 0.018],
    )
    loader = _FixedValLoader(dataset=dataset, batch_size=2)

    logger.capture_fixed_val_batch(loader)

    batches = logger.get_fixed_val_batches()
    assert set(batches) == {
        "rv_0_to_1_5",
        "rv_1_5_to_2_1",
        "rv_2_1_to_3_0",
        "rv_ge_3_0",
    }
    assert torch.all((batches["rv_0_to_1_5"]["target_rv"] * 100.0 >= 0.0))
    assert torch.all((batches["rv_0_to_1_5"]["target_rv"] * 100.0 < 1.5))
    assert torch.all((batches["rv_1_5_to_2_1"]["target_rv"] * 100.0 >= 1.5))
    assert torch.all((batches["rv_1_5_to_2_1"]["target_rv"] * 100.0 < 2.1))
    assert torch.all((batches["rv_2_1_to_3_0"]["target_rv"] * 100.0 >= 2.1))
    assert torch.all((batches["rv_2_1_to_3_0"]["target_rv"] * 100.0 < 3.0))
    assert torch.all(batches["rv_ge_3_0"]["target_rv"] * 100.0 >= 3.0)
