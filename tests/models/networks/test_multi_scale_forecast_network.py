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

    def forward_features(
        self,
        x_long: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        y = self.forward(x_long)
        return y, (y, y, y)


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


def _make_model(
    *,
    mode: str = "mu",
    field: str = "ret",
) -> MultiScaleForecastNetwork:
    model = MultiScaleForecastNetwork(mode=mode, field=field)
    model.denoise_macro = _TailCropDenoise(target_len=64)
    model.denoise_mezzo = _TailCropDenoise(target_len=96)
    model.denoise_micro = _TailCropDenoise(target_len=144)
    return model


@pytest.mark.parametrize(
    "field",
    [
        "ret",
        "rv",
        "p10",
    ],
)
def test_forward_mu_smoke_outputs_and_shapes(field: str) -> None:
    model = _make_model(mode="mu", field=field)
    out = model(_make_batch())
    assert "mu_raw" in out
    assert "task_repr" in out
    assert "sigma_raw" not in out
    assert out["mu_raw"].shape == (2, 1)
    assert out["mezzo_head_tokens"].shape == (2, 128, 24)
    assert out["mezzo_head_context"].shape == (2, 128)


@pytest.mark.parametrize("field", ["ret", "rv", "p10"])
def test_forward_sigma_smoke_outputs_and_shapes(field: str) -> None:
    model = _make_model(mode="sigma", field=field)
    batch = _make_batch()
    batch["mu_input"] = torch.randn(2, 1)
    out = model(batch)
    assert "sigma_raw" in out
    assert "confidence_query" in out
    assert "confidence_repr" in out
    assert "conf_attn_entropy_mean" in out
    assert "conf_attn_max_weight_mean" in out
    assert "mu_raw" not in out
    assert out["sigma_raw"].shape == (2, 1)
    assert out["confidence_query"].shape == (2, 128)
    assert out["confidence_repr"].shape == (2, 128)


@pytest.mark.parametrize("field", ["ret", "rv", "p10"])
def test_forward_loss_mu_smoke_outputs(field: str) -> None:
    model = _make_model(mode="mu", field=field)
    out = model.forward_loss(_make_batch())
    assert "loss_total" in out
    assert "loss_task" in out
    assert "loss_mu" in out
    assert "mu_pred" in out
    assert "sigma_pred" not in out
    assert "mezzo_head_tokens" not in out
    assert "mezzo_head_context" not in out
    assert "macro_dual_summary" not in out
    assert "micro_dual_summary" not in out


@pytest.mark.parametrize("field", ["ret", "rv", "p10"])
def test_forward_loss_sigma_smoke_outputs(field: str) -> None:
    model = _make_model(mode="sigma", field=field)
    batch = _make_batch()
    batch["mu_input"] = torch.randn(2, 1) if field != "rv" else torch.rand(2, 1).clamp_min(1e-3)
    if field == "rv":
        batch["target_rv"] = torch.rand(2, 1).clamp_min(1e-3)
    out = model.forward_loss(batch)
    assert "loss_total" in out
    assert "loss_task" in out
    assert "loss_nll" in out
    assert "sigma_pred" in out
    assert "mu_pred" not in out
    assert "conf_attn_entropy_mean" in out
    assert "conf_attn_max_weight_mean" in out
    if field == "ret":
        assert "nu_ret" in out
    if field == "rv":
        assert "shape_rv" in out


def test_forward_aux_outputs_present() -> None:
    model = _make_model(mode="mu", field="ret")
    out = model(_make_batch(), return_aux=True)
    assert "macro_input" in out
    assert "time_tokens_macro" in out and "wavelet_tokens_macro" in out
    assert "time_tokens_mezzo" in out and "wavelet_tokens_mezzo" in out
    assert "time_tokens_micro" in out and "wavelet_tokens_micro" in out
    assert "macro_dual_summary" in out
    assert "mezzo_head_context" in out
    assert "micro_dual_summary" in out
    assert "macro_wavelet_sidechain_input" in out
    assert "feature_rms_macro_pre" in out
    assert "feature_rms_mezzo_pre" in out
    assert "feature_rms_micro_pre" in out
    assert out["macro_input"].shape == (2, 22, 64)
    assert out["time_tokens_macro"].shape == (2, 128, 16)
    assert out["wavelet_tokens_macro"].shape == (2, 128, 16)
    assert out["time_tokens_mezzo"].shape == (2, 128, 24)
    assert out["wavelet_tokens_mezzo"].shape == (2, 128, 24)
    assert out["time_tokens_micro"].shape == (2, 128, 36)
    assert out["wavelet_tokens_micro"].shape == (2, 128, 36)
    assert out["macro_dual_summary"].shape == (2, 128)
    assert out["mezzo_head_context"].shape == (2, 128)
    assert out["micro_dual_summary"].shape == (2, 128)
    assert out["macro_wavelet_sidechain_input"].shape == (2, 13, 64)
    assert out["feature_rms_macro_pre"].shape == (22,)
    assert out["feature_rms_mezzo_pre"].shape == (9,)
    assert out["feature_rms_micro_pre"].shape == (9,)


def test_forward_loss_debug_outputs_present() -> None:
    model = _make_model(mode="mu", field="ret")
    out = model.forward_loss(_make_batch(), return_debug=True)
    assert "_debug" in out
    debug = out["_debug"]
    assert "wavelet_macro_energy_raw" in debug
    assert "time_macro_summary_l2_mean" in debug
    assert "head_task_repr_l2_mean" in debug


def test_forward_missing_key_raises_value_error() -> None:
    model = _make_model(mode="mu", field="ret")
    batch = _make_batch()
    batch.pop("micro_i8_long")
    with pytest.raises(ValueError, match="Missing required batch key"):
        _ = model(batch)


def test_forward_invalid_shape_raises_value_error() -> None:
    model = _make_model(mode="mu", field="ret")
    batch = _make_batch()
    batch["macro_float_long"] = torch.randn(2, 9, 111)
    with pytest.raises(ValueError, match="macro_float_long shape mismatch"):
        _ = model(batch)


def test_forward_train_and_eval_both_run() -> None:
    model = _make_model(mode="mu", field="ret")
    batch = _make_batch()
    model.train()
    out_train = model(batch)
    model.eval()
    out_eval = model(batch)
    assert out_train["mezzo_head_tokens"].shape == (2, 128, 24)
    assert out_eval["mezzo_head_tokens"].shape == (2, 128, 24)


def test_macro_targeted_feature_dropout_configuration_present() -> None:
    model = _make_model(mode="mu", field="ret")
    assert model.dropout_macro_input.channel_ps.shape == (22,)
    assert float(model.dropout_macro_input.channel_ps[16]) > float(model.dropout_macro_input.channel_ps[15])
