from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.arch.networks import MultiScaleForecastNetwork


class _TailCropDenoise(nn.Module):
    def __init__(self, target_len: int) -> None:
        super().__init__()
        self.target_len = int(target_len)

    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        return x_long[..., -self.target_len :]


def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    macro_state = torch.randint(0, 16, (batch_size, 1, 112), dtype=torch.int64)
    macro_pos = torch.randint(0, 8, (batch_size, 1, 112), dtype=torch.int64)
    mezzo_state = torch.randint(0, 16, (batch_size, 1, 144), dtype=torch.int64)
    mezzo_pos = torch.randint(0, 16, (batch_size, 1, 144), dtype=torch.int64)
    micro_state = torch.randint(0, 16, (batch_size, 1, 192), dtype=torch.int64)
    micro_pos = torch.randint(0, 64, (batch_size, 1, 192), dtype=torch.int64)
    return {
        "macro_float_long": torch.randn(batch_size, 9, 112),
        "macro_i8_long": torch.cat([macro_state, macro_pos], dim=1),
        "mezzo_float_long": torch.randn(batch_size, 9, 144),
        "mezzo_i8_long": torch.cat([mezzo_state, mezzo_pos], dim=1),
        "micro_float_long": torch.randn(batch_size, 9, 192),
        "micro_i8_long": torch.cat([micro_state, micro_pos], dim=1),
        "sidechain_cond": torch.randn(batch_size, 13, 64),
        "target_ret": torch.randn(batch_size, 1),
        "target_rv": torch.rand(batch_size, 1).clamp_min(1e-4),
        "target_q": torch.randn(batch_size, 1),
    }


def _make_model(task: str = "ret") -> MultiScaleForecastNetwork:
    model = MultiScaleForecastNetwork(task=task)
    model.denoise_macro = _TailCropDenoise(target_len=64)
    model.denoise_mezzo = _TailCropDenoise(target_len=96)
    model.denoise_micro = _TailCropDenoise(target_len=144)
    return model


@pytest.mark.parametrize(
    ("task", "expected_keys"),
    [
        ("ret", {"pred_mu_ret", "pred_scale_ret_raw"}),
        ("rv", {"pred_mean_rv_raw", "pred_shape_rv_raw"}),
        ("q", {"pred_mu_q", "pred_scale_q_raw"}),
    ],
)
def test_forward_smoke_outputs_and_shapes(task: str, expected_keys: set[str]) -> None:
    model = _make_model(task=task)
    out = model(_make_batch())
    assert expected_keys.issubset(set(out.keys()))
    assert out["pred_primary"].shape == (2, 1)
    assert out["pred_aux_raw"].shape == (2, 1)
    assert out["fused_latents"].shape == (2, 128, 36)
    assert out["fused_global"].shape == (2, 128)

@pytest.mark.parametrize("task", ["ret", "rv", "q"])
def test_forward_loss_smoke_outputs(task: str) -> None:
    model = _make_model(task=task)
    out = model.forward_loss(_make_batch())
    assert "loss_total" in out
    assert "loss_task" in out
    assert "sigma_pred" in out
    if task == "ret":
        assert "nu_ret" in out
    if task == "rv":
        assert "shape_rv" in out


def test_forward_aux_outputs_present() -> None:
    model = _make_model(task="ret")
    out = model(_make_batch(), return_aux=True)
    assert "s1" in out and "s2" in out and "s3" in out
    assert "macro_fused" in out and "mezzo_fused" in out and "micro_fused" in out
    assert "macro_ctx" in out and "mezzo_ctx" in out and "micro_ctx" in out


def test_forward_loss_debug_outputs_present() -> None:
    model = _make_model(task="ret")
    out = model.forward_loss(_make_batch(), return_debug=True)
    assert "_debug" in out
    debug = out["_debug"]
    assert "wavelet" in debug
    assert "bridge" in debug
    assert "cross_scale" in debug
    assert "heads" in debug


def test_forward_missing_key_raises_value_error() -> None:
    model = _make_model(task="ret")
    batch = _make_batch()
    batch.pop("micro_i8_long")
    with pytest.raises(ValueError, match="Missing required batch key"):
        _ = model(batch)


def test_forward_invalid_shape_raises_value_error() -> None:
    model = _make_model(task="ret")
    batch = _make_batch()
    batch["macro_float_long"] = torch.randn(2, 9, 111)
    with pytest.raises(ValueError, match="macro_float_long shape mismatch"):
        _ = model(batch)


def test_forward_train_and_eval_both_run() -> None:
    model = _make_model(task="ret")
    batch = _make_batch()
    model.train()
    out_train = model(batch)
    model.eval()
    out_eval = model(batch)
    assert out_train["fused_latents"].shape == (2, 128, 36)
    assert out_eval["fused_latents"].shape == (2, 128, 36)
