from __future__ import annotations

import pytest
import torch

from src.models.arch.losses import SingleTaskDistributionLoss, StudentTNLLLoss


@pytest.mark.parametrize(
    ("field", "target"),
    [
        ("ret", torch.randn(4, 1)),
        ("rv", torch.rand(4, 1).clamp_min(1e-4)),
        ("p10", torch.randn(4, 1)),
    ],
)
def test_single_task_distribution_loss_mu_smoke(
    field: str,
    target: torch.Tensor,
) -> None:
    prediction_raw = torch.randn(4, 1, requires_grad=True)
    loss_fn = SingleTaskDistributionLoss(mode="mu", field=field, q_tau=0.05)
    out = loss_fn(
        target=target,
        prediction_raw=prediction_raw,
    )

    assert "loss_total" in out
    assert "loss_task" in out
    assert "loss_mu" in out
    assert "mu_pred" in out
    out["loss_total"].backward()
    assert prediction_raw.grad is not None
    if field == "rv":
        assert torch.all(out["mu_pred"] > 0)


def test_single_task_distribution_loss_sigma_ret_exposes_nu() -> None:
    prediction_raw = torch.randn(3, 1, requires_grad=True)
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="ret", q_tau=0.05)
    out = loss_fn(
        target=torch.randn(3, 1),
        prediction_raw=prediction_raw,
        mu_input=torch.randn(3, 1),
    )

    assert "loss_nll" in out
    assert "sigma_pred" in out
    assert "nu_ret" in out
    out["loss_total"].backward()
    assert prediction_raw.grad is not None


def test_single_task_distribution_loss_sigma_ret_matches_student_t_nll() -> None:
    prediction_raw = torch.zeros(2, 1, dtype=torch.float32, requires_grad=True)
    mu_input = torch.zeros(2, 1, dtype=torch.float32)
    target = torch.tensor([[0.08], [0.01]], dtype=torch.float32)
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="ret", q_tau=0.05)
    out = loss_fn(
        target=target,
        prediction_raw=prediction_raw,
        mu_input=mu_input,
    )

    nu_ret = out["nu_ret"].detach()
    sigma_pred = out["sigma_pred"].detach()
    expected = StudentTNLLLoss(_nu_min=2.01, _eps=1e-6)(
        target=target,
        mu=mu_input,
        scale=sigma_pred * torch.sqrt((nu_ret - 2.0) / nu_ret),
        nu=nu_ret,
    )

    assert torch.allclose(out["loss_total"].detach(), expected)
    out["loss_total"].backward()
    assert prediction_raw.grad is not None


def test_single_task_distribution_loss_sigma_rv_exposes_shape() -> None:
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="rv", q_tau=0.05)
    out = loss_fn(
        target=torch.rand(3, 1).clamp_min(1e-4),
        prediction_raw=torch.randn(3, 1, requires_grad=True),
        mu_input=torch.rand(3, 1).clamp_min(1e-3),
    )

    assert "loss_nll" in out
    assert "sigma_pred" in out
    assert "shape_rv" in out


def test_single_task_distribution_loss_sigma_quantile_smoke() -> None:
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="p10", q_tau=0.05)
    out = loss_fn(
        target=torch.randn(3, 1),
        prediction_raw=torch.randn(3, 1, requires_grad=True),
        mu_input=torch.randn(3, 1),
    )

    assert "loss_nll" in out
    assert "sigma_pred" in out
    assert "shape_rv" not in out
    assert "nu_ret" not in out


def test_single_task_distribution_loss_sigma_requires_mu_input() -> None:
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="ret", q_tau=0.05)
    with pytest.raises(ValueError, match="mu_input is required"):
        loss_fn(
            target=torch.randn(2, 1),
            prediction_raw=torch.randn(2, 1),
        )


def test_single_task_distribution_loss_sigma_ret_uses_fp32_loss_math_only() -> None:
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="ret", q_tau=0.05)
    out = loss_fn(
        target=torch.tensor([[0.08], [0.01]], dtype=torch.bfloat16),
        prediction_raw=torch.tensor([[-3.5], [-3.5]], dtype=torch.bfloat16, requires_grad=True),
        mu_input=torch.tensor([[0.00], [0.00]], dtype=torch.bfloat16),
    )

    assert out["loss_total"].dtype == torch.float32
    assert out["loss_task"].dtype == torch.float32
    assert out["loss_nll"].dtype == torch.float32
    assert out["sigma_pred"].dtype == torch.float32
    assert out["nu_ret"].dtype == torch.float32


def test_single_task_distribution_loss_keeps_nu_ret_param_fp32_after_bf16_cast() -> None:
    loss_fn = SingleTaskDistributionLoss(mode="sigma", field="ret", q_tau=0.05)
    loss_fn = loss_fn.to(dtype=torch.bfloat16)
    assert loss_fn._nu_ret_raw is not None
    assert loss_fn._nu_ret_raw.dtype == torch.float32


def test_single_task_distribution_loss_mu_ret_matches_fixed_student_t_nll() -> None:
    loss_fn = SingleTaskDistributionLoss(
        mode="mu",
        field="ret",
        q_tau=0.05,
        ret_mu_fixed_scale=0.02335,
        ret_mu_fixed_nu=2.82,
    )
    target = torch.tensor([[0.08], [0.01]], dtype=torch.float32)
    prediction_raw = torch.zeros(2, 1, dtype=torch.float32, requires_grad=True)

    out = loss_fn(
        target=target,
        prediction_raw=prediction_raw,
    )
    expected = StudentTNLLLoss(_nu_min=2.000001, _eps=1e-6)(
        target=target,
        mu=prediction_raw,
        scale=torch.full_like(target, 0.02335),
        nu=target.new_tensor(2.82, dtype=torch.float32),
    )

    assert out["loss_total"] == out["loss_task"]
    assert out["loss_mu"] == out["loss_total"]
    assert torch.allclose(out["loss_total"], expected)
    out["loss_total"].backward()
    assert prediction_raw.grad is not None
