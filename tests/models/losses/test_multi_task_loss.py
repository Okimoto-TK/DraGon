from __future__ import annotations

import pytest
import torch

from src.models.arch.losses import MultiTaskDistributionLoss


def test_multi_task_distribution_loss_forward() -> None:
    batch_size = 3
    loss_fn = MultiTaskDistributionLoss(q_tau=0.05)

    out = loss_fn(
        target_ret=torch.randn(batch_size, 1),
        pred_mu_ret=torch.randn(batch_size, 1),
        pred_scale_ret_raw=torch.randn(batch_size, 1),
        target_rv=torch.rand(batch_size, 1) + 0.1,
        pred_mean_rv_raw=torch.randn(batch_size, 1),
        pred_shape_rv_raw=torch.randn(batch_size, 1),
        target_q=torch.randn(batch_size, 1),
        pred_mu_q=torch.randn(batch_size, 1),
        pred_scale_q_raw=torch.randn(batch_size, 1),
    )

    assert set(out) >= {
        "loss_total",
        "loss_ret",
        "loss_rv",
        "loss_q",
        "nu_ret",
        "sigma_ret_pred",
        "sigma_rv_pred",
        "sigma_q_pred",
    }
    assert out["loss_total"].ndim == 0


def test_multi_task_distribution_loss_nu_ret_receives_gradient() -> None:
    batch_size = 4
    loss_fn = MultiTaskDistributionLoss(q_tau=0.05)

    out = loss_fn(
        target_ret=torch.randn(batch_size, 1),
        pred_mu_ret=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_ret_raw=torch.randn(batch_size, 1, requires_grad=True),
        target_rv=torch.rand(batch_size, 1) + 0.1,
        pred_mean_rv_raw=torch.randn(batch_size, 1, requires_grad=True),
        pred_shape_rv_raw=torch.randn(batch_size, 1, requires_grad=True),
        target_q=torch.randn(batch_size, 1),
        pred_mu_q=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_q_raw=torch.randn(batch_size, 1, requires_grad=True),
    )
    out["loss_total"].backward()

    assert loss_fn._nu_ret_raw.grad is not None


def test_multi_task_distribution_loss_rv_shape_receives_gradient() -> None:
    batch_size = 4
    loss_fn = MultiTaskDistributionLoss(q_tau=0.05)
    pred_shape_rv_raw = torch.randn(batch_size, 1, requires_grad=True)

    out = loss_fn(
        target_ret=torch.randn(batch_size, 1),
        pred_mu_ret=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_ret_raw=torch.randn(batch_size, 1, requires_grad=True),
        target_rv=torch.rand(batch_size, 1) + 0.1,
        pred_mean_rv_raw=torch.randn(batch_size, 1, requires_grad=True),
        pred_shape_rv_raw=pred_shape_rv_raw,
        target_q=torch.randn(batch_size, 1),
        pred_mu_q=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_q_raw=torch.randn(batch_size, 1, requires_grad=True),
    )
    out["loss_total"].backward()

    assert pred_shape_rv_raw.grad is not None


def test_multi_task_distribution_loss_q_scale_receives_gradient() -> None:
    batch_size = 4
    loss_fn = MultiTaskDistributionLoss(q_tau=0.05)
    pred_scale_q_raw = torch.randn(batch_size, 1, requires_grad=True)

    out = loss_fn(
        target_ret=torch.randn(batch_size, 1),
        pred_mu_ret=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_ret_raw=torch.randn(batch_size, 1, requires_grad=True),
        target_rv=torch.rand(batch_size, 1) + 0.1,
        pred_mean_rv_raw=torch.randn(batch_size, 1, requires_grad=True),
        pred_shape_rv_raw=torch.randn(batch_size, 1, requires_grad=True),
        target_q=torch.randn(batch_size, 1),
        pred_mu_q=torch.randn(batch_size, 1, requires_grad=True),
        pred_scale_q_raw=pred_scale_q_raw,
    )
    out["loss_total"].backward()

    assert pred_scale_q_raw.grad is not None


def test_multi_task_distribution_loss_invalid_q_tau_raises_value_error() -> None:
    with pytest.raises(ValueError, match="q_tau must satisfy 0 < q_tau < 1"):
        _ = MultiTaskDistributionLoss(q_tau=0.0)

    with pytest.raises(ValueError, match="q_tau must satisfy 0 < q_tau < 1"):
        _ = MultiTaskDistributionLoss(q_tau=1.0)


def test_multi_task_distribution_loss_gamma_handles_tiny_positive_target() -> None:
    batch_size = 2
    loss_fn = MultiTaskDistributionLoss(q_tau=0.05)

    out = loss_fn(
        target_ret=torch.randn(batch_size, 1),
        pred_mu_ret=torch.randn(batch_size, 1),
        pred_scale_ret_raw=torch.randn(batch_size, 1),
        target_rv=torch.full((batch_size, 1), 1e-12),
        pred_mean_rv_raw=torch.randn(batch_size, 1),
        pred_shape_rv_raw=torch.randn(batch_size, 1),
        target_q=torch.randn(batch_size, 1),
        pred_mu_q=torch.randn(batch_size, 1),
        pred_scale_q_raw=torch.randn(batch_size, 1),
    )

    assert torch.isfinite(out["loss_rv"])


def test_multi_task_distribution_loss_rv_tail_weighting_emphasizes_large_targets() -> None:
    weighted_loss_fn = MultiTaskDistributionLoss(
        q_tau=0.05,
        rv_tail_weight_threshold=0.03,
        rv_tail_weight_alpha=2.0,
        rv_tail_weight_max=4.0,
    )
    base_loss_fn = MultiTaskDistributionLoss(
        q_tau=0.05,
        rv_tail_weight_threshold=0.03,
        rv_tail_weight_alpha=0.0,
        rv_tail_weight_max=1.0,
    )
    target_rv = torch.tensor([[0.08], [0.01]], dtype=torch.float32)
    kwargs = dict(
        target_ret=torch.zeros(2, 1),
        pred_mu_ret=torch.zeros(2, 1),
        pred_scale_ret_raw=torch.zeros(2, 1),
        target_rv=target_rv,
        pred_mean_rv_raw=torch.zeros(2, 1),
        pred_shape_rv_raw=torch.zeros(2, 1),
        target_q=torch.zeros(2, 1),
        pred_mu_q=torch.zeros(2, 1),
        pred_scale_q_raw=torch.zeros(2, 1),
    )

    weighted_out = weighted_loss_fn(**kwargs)
    base_out = base_loss_fn(**kwargs)

    assert weighted_out["loss_rv"] > base_out["loss_rv"]
