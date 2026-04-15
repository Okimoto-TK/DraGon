"""Single-task training losses for Edge / Persist / DownRisk."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.task_labels import canonical_task_label


class SingleTaskLoss(nn.Module):
    """Criterion that selects the correct single-task loss by label."""

    def __init__(
        self,
        *,
        task_label: str,
        eps: float = 1e-6,
        nu_floor: float = 2.0,
        initial_nu: float = 5.0,
    ) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if nu_floor <= 0.0:
            raise ValueError(f"nu_floor must be positive, got {nu_floor}")
        if initial_nu <= nu_floor:
            raise ValueError(f"initial_nu must be greater than nu_floor, got {initial_nu} <= {nu_floor}")

        self.task_label = canonical_task_label(task_label)
        self.eps = float(eps)
        self.nu_floor = float(nu_floor)
        initial_raw_nu = math.log(math.expm1(initial_nu - nu_floor))

        if self.task_label in {"Edge", "DownRisk"}:
            self.raw_nu = nn.Parameter(torch.tensor(initial_raw_nu, dtype=torch.float32))
        else:
            self.register_parameter("raw_nu", None)

    def _nu(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.raw_nu is None:
            raise RuntimeError(f"Task {self.task_label} does not use Student-t degrees of freedom.")
        return self.nu_floor + F.softplus(self.raw_nu).to(device=device, dtype=dtype)

    def _student_t_nll(self, pred: Tensor, scale: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        pred = pred.reshape(-1)
        scale = scale.reshape(-1).clamp_min(self.eps)
        target = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
        nu = self._nu(device=pred.device, dtype=pred.dtype)
        residual = (target - pred) / scale
        nll = (
            torch.lgamma((nu + 1.0) * 0.5)
            - torch.lgamma(nu * 0.5)
            + 0.5 * (torch.log(nu) + math.log(math.pi))
            + torch.log(scale)
            + 0.5 * (nu + 1.0) * torch.log1p((residual * residual) / nu)
        )
        return nll, nu

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        *,
        return_metrics: bool = True,
        update_state: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del update_state
        metrics: dict[str, Tensor] = {}

        if self.task_label == "Edge":
            per_sample, nu = self._student_t_nll(
                preds["pred_Edge"],
                preds["unc_Edge"],
                targets["label_Edge"],
            )
            loss = per_sample.mean()
            if return_metrics:
                metrics["loss_Edge"] = loss.detach()
                metrics["loss_task"] = loss.detach()
                metrics["loss_total"] = loss.detach()
                metrics["nu_Edge"] = nu.detach()
            return loss, metrics

        if self.task_label == "Persist":
            target = targets["label_Persist"].reshape(-1).to(device=preds["logit_Persist"].device, dtype=preds["logit_Persist"].dtype)
            loss = F.binary_cross_entropy_with_logits(preds["logit_Persist"].reshape(-1), target)
            if return_metrics:
                metrics["loss_Persist"] = loss.detach()
                metrics["loss_task"] = loss.detach()
                metrics["loss_total"] = loss.detach()
            return loss, metrics

        target = torch.log(targets["label_DownRisk"].reshape(-1).to(device=preds["pred_log_DownRisk"].device, dtype=preds["pred_log_DownRisk"].dtype) + self.eps)
        per_sample, nu = self._student_t_nll(
            preds["pred_log_DownRisk"],
            preds["unc_DownRisk"],
            target,
        )
        loss = per_sample.mean()
        if return_metrics:
            metrics["loss_DownRisk"] = loss.detach()
            metrics["loss_task"] = loss.detach()
            metrics["loss_total"] = loss.detach()
            metrics["nu_DownRisk"] = nu.detach()
        return loss, metrics


__all__ = ["SingleTaskLoss"]
