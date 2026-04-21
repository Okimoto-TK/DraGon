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


def test_single_task_distribution_loss_rv_exposes_shape() -> None:
    loss_fn = SingleTaskDistributionLoss(task="rv", q_tau=0.05)
    out = loss_fn(
        target=torch.rand(3, 1).clamp_min(1e-4),
        pred_primary=torch.randn(3, 1),
        pred_aux_raw=torch.randn(3, 1),
    )

    assert "shape_rv" in out
