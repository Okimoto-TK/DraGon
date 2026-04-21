from __future__ import annotations

import pytest
import torch

from src.models.arch.losses import SingleTaskDistributionLoss


@pytest.mark.parametrize("task", ["ret", "rv", "q"])
def test_single_task_distribution_loss_smoke(task: str) -> None:
    loss_fn = SingleTaskDistributionLoss(task=task, q_tau=0.05)
    out = loss_fn(
        target=torch.rand(4, 1).clamp_min(1e-4) if task == "rv" else torch.randn(4, 1),
        pred_primary=torch.randn(4, 1, requires_grad=True),
        pred_aux_raw=torch.randn(4, 1, requires_grad=True),
    )

    assert "loss_total" in out
    assert "loss_task" in out
    assert "sigma_pred" in out
    out["loss_total"].backward()


def test_single_task_distribution_loss_ret_exposes_nu() -> None:
    loss_fn = SingleTaskDistributionLoss(task="ret", q_tau=0.05)
    out = loss_fn(
        target=torch.randn(3, 1),
        pred_primary=torch.randn(3, 1),
        pred_aux_raw=torch.randn(3, 1),
    )

    assert "nu_ret" in out


def test_single_task_distribution_loss_uses_fp32_loss_math_only() -> None:
    loss_fn = SingleTaskDistributionLoss(task="ret", q_tau=0.05)
    out = loss_fn(
        target=torch.tensor([[0.08], [0.01]], dtype=torch.bfloat16),
        pred_primary=torch.tensor([[0.00], [0.00]], dtype=torch.bfloat16, requires_grad=True),
        pred_aux_raw=torch.tensor([[-3.5], [-3.5]], dtype=torch.bfloat16, requires_grad=True),
    )

    assert out["loss_total"].dtype == torch.float32
    assert out["loss_task"].dtype == torch.float32
    assert out["sigma_pred"].dtype == torch.float32
    assert out["nu_ret"].dtype == torch.float32
    assert out["loss_ret_weighted_nll"].dtype == torch.float32


def test_single_task_distribution_loss_ret_weighted_student_t_emphasizes_tail_samples() -> None:
    loss_fn = SingleTaskDistributionLoss(
        task="ret",
        q_tau=0.05,
        ret_tail_weight_threshold=0.05,
        ret_tail_weight_alpha=2.0,
        ret_tail_weight_max=4.0,
    )
    base_loss_fn = SingleTaskDistributionLoss(
        task="ret",
        q_tau=0.05,
        ret_tail_weight_threshold=0.05,
        ret_tail_weight_alpha=0.0,
        ret_tail_weight_max=1.0,
    )
    target = torch.tensor([[0.08], [0.01]], dtype=torch.float32)
    pred_primary = torch.zeros(2, 1, dtype=torch.float32, requires_grad=True)
    pred_aux_raw = torch.full((2, 1), -3.5, dtype=torch.float32, requires_grad=True)
    out = loss_fn(
        target=target,
        pred_primary=pred_primary,
        pred_aux_raw=pred_aux_raw,
    )
    out_base = base_loss_fn(
        target=torch.tensor([[0.08], [0.01]], dtype=torch.float32),
        pred_primary=torch.zeros(2, 1, dtype=torch.float32),
        pred_aux_raw=torch.full((2, 1), -3.5, dtype=torch.float32),
    )

    assert out["loss_total"] == out["loss_task"]
    assert out["loss_ret_weighted_nll"] == out["loss_total"]
    assert out["loss_total"] > out_base["loss_total"]


def test_single_task_distribution_loss_weighting_strengthens_nu_gradient_on_tail_samples() -> None:
    target = torch.tensor([[0.20], [-0.18], [0.01], [0.00]], dtype=torch.float32)
    pred_primary = torch.zeros(4, 1, dtype=torch.float32, requires_grad=True)
    pred_aux_raw = torch.full((4, 1), -3.5, dtype=torch.float32, requires_grad=True)

    base_loss_fn = SingleTaskDistributionLoss(
        task="ret",
        q_tau=0.05,
        ret_tail_weight_alpha=0.0,
        ret_tail_weight_max=1.0,
    )
    weighted_loss_fn = SingleTaskDistributionLoss(
        task="ret",
        q_tau=0.05,
        ret_tail_weight_threshold=0.05,
        ret_tail_weight_alpha=2.0,
        ret_tail_weight_max=4.0,
    )

    base_out = base_loss_fn(
        target=target,
        pred_primary=pred_primary.detach().clone().requires_grad_(True),
        pred_aux_raw=pred_aux_raw.detach().clone().requires_grad_(True),
    )
    base_out["loss_total"].backward()
    base_grad = float(base_loss_fn._nu_ret_raw.grad.detach().abs().cpu())

    weighted_out = weighted_loss_fn(
        target=target,
        pred_primary=pred_primary.detach().clone().requires_grad_(True),
        pred_aux_raw=pred_aux_raw.detach().clone().requires_grad_(True),
    )
    weighted_out["loss_total"].backward()
    weighted_grad = float(weighted_loss_fn._nu_ret_raw.grad.detach().abs().cpu())

    assert weighted_grad > base_grad


def test_single_task_distribution_loss_rv_exposes_shape() -> None:
    loss_fn = SingleTaskDistributionLoss(task="rv", q_tau=0.05)
    out = loss_fn(
        target=torch.rand(3, 1).clamp_min(1e-4),
        pred_primary=torch.randn(3, 1),
        pred_aux_raw=torch.randn(3, 1),
    )

    assert "shape_rv" in out
