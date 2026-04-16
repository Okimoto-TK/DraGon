"""Single-task training losses for Edge / Persist / DownRisk."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.config import downrisk_log_huber_beta as DEFAULT_DOWNRISK_LOG_HUBER_BETA
from config.config import edge_huber_beta as DEFAULT_EDGE_HUBER_BETA
from config.config import persist_logit_huber_beta as DEFAULT_PERSIST_LOGIT_HUBER_BETA
from config.config import persist_probability_loss_weight as DEFAULT_PERSIST_PROBABILITY_LOSS_WEIGHT
from config.config import student_t_nu as DEFAULT_STUDENT_T_NU
from config.config import uncertainty_loss_weight as DEFAULT_UNCERTAINTY_LOSS_WEIGHT
from src.task_labels import canonical_task_label


class SingleTaskLoss(nn.Module):
    """Criterion that selects the correct single-task loss by label."""

    def __init__(
        self,
        *,
        task_label: str,
        eps: float = 1e-6,
        student_t_nu: float = DEFAULT_STUDENT_T_NU,
        uncertainty_loss_weight: float = DEFAULT_UNCERTAINTY_LOSS_WEIGHT,
        persist_probability_loss_weight: float = DEFAULT_PERSIST_PROBABILITY_LOSS_WEIGHT,
        edge_huber_beta: float = DEFAULT_EDGE_HUBER_BETA,
        persist_logit_huber_beta: float = DEFAULT_PERSIST_LOGIT_HUBER_BETA,
        downrisk_log_huber_beta: float = DEFAULT_DOWNRISK_LOG_HUBER_BETA,
    ) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if student_t_nu <= 0.0:
            raise ValueError(f"student_t_nu must be positive, got {student_t_nu}")
        if uncertainty_loss_weight < 0.0:
            raise ValueError(
                f"uncertainty_loss_weight must be non-negative, got {uncertainty_loss_weight}"
            )
        if persist_probability_loss_weight < 0.0:
            raise ValueError(
                "persist_probability_loss_weight must be non-negative, "
                f"got {persist_probability_loss_weight}"
            )
        if edge_huber_beta <= 0.0:
            raise ValueError(f"edge_huber_beta must be positive, got {edge_huber_beta}")
        if persist_logit_huber_beta <= 0.0:
            raise ValueError(
                f"persist_logit_huber_beta must be positive, got {persist_logit_huber_beta}"
            )
        if downrisk_log_huber_beta <= 0.0:
            raise ValueError(
                f"downrisk_log_huber_beta must be positive, got {downrisk_log_huber_beta}"
            )

        self.task_label = canonical_task_label(task_label)
        self.eps = float(eps)
        self.uncertainty_loss_weight = float(uncertainty_loss_weight)
        self.persist_probability_loss_weight = float(persist_probability_loss_weight)
        self.edge_huber_beta = float(edge_huber_beta)
        self.persist_logit_huber_beta = float(persist_logit_huber_beta)
        self.downrisk_log_huber_beta = float(downrisk_log_huber_beta)
        self.register_buffer("student_t_nu", torch.tensor(float(student_t_nu), dtype=torch.float32))

    def _student_t_nll_from_error(self, error: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        error = error.reshape(-1)
        scale = scale.reshape(-1).clamp_min(self.eps)
        error = error.to(device=scale.device, dtype=scale.dtype)
        nu = self.student_t_nu.to(device=scale.device, dtype=scale.dtype)
        residual = error / scale
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
            pred = preds["pred_Edge"].reshape(-1)
            target = targets["label_Edge"].reshape(-1).to(device=pred.device, dtype=pred.dtype)
            scale = preds["unc_Edge"].reshape(-1)
            point_per_sample = F.smooth_l1_loss(
                pred,
                target,
                reduction="none",
                beta=self.edge_huber_beta,
            )
            unc_per_sample, nu = self._student_t_nll_from_error((target - pred).detach(), scale)
            point_loss = point_per_sample.mean()
            unc_loss = unc_per_sample.mean()
            loss = point_loss + self.uncertainty_loss_weight * unc_loss
            if return_metrics:
                metrics["loss_Edge"] = loss.detach()
                metrics["loss_task"] = loss.detach()
                metrics["loss_total"] = loss.detach()
                metrics["loss_mu"] = point_loss.detach()
                metrics["loss_unc"] = unc_loss.detach()
                metrics["nu_Edge"] = nu.detach()
            return loss, metrics

        if self.task_label == "Persist":
            pred_logit = preds["logit_Persist"].reshape(-1)
            target_prob = targets["label_Persist"].reshape(-1).to(
                device=pred_logit.device,
                dtype=pred_logit.dtype,
            ).clamp(self.eps, 1.0 - self.eps)
            target_logit = torch.logit(target_prob, eps=self.eps)
            scale = preds["unc_logit_Persist"].reshape(-1)
            point_per_sample = F.smooth_l1_loss(
                pred_logit,
                target_logit,
                reduction="none",
                beta=self.persist_logit_huber_beta,
            )
            unc_per_sample, nu = self._student_t_nll_from_error(
                (target_logit - pred_logit).detach(),
                scale,
            )
            point_loss = point_per_sample.mean()
            unc_loss = unc_per_sample.mean()
            prob = torch.sigmoid(pred_logit)
            prob_loss = torch.mean((prob - target_prob) * (prob - target_prob))
            loss = (
                point_loss
                + self.uncertainty_loss_weight * unc_loss
                + self.persist_probability_loss_weight * prob_loss
            )
            if return_metrics:
                metrics["loss_Persist"] = loss.detach()
                metrics["loss_task"] = loss.detach()
                metrics["loss_total"] = loss.detach()
                metrics["loss_mu"] = point_loss.detach()
                metrics["loss_unc"] = unc_loss.detach()
                metrics["loss_prob"] = prob_loss.detach()
                metrics["nu_Persist"] = nu.detach()
            return loss, metrics

        pred = preds["pred_log_DownRisk"].reshape(-1)
        target = torch.log(
            targets["label_DownRisk"].reshape(-1).to(device=pred.device, dtype=pred.dtype) + self.eps
        )
        scale = preds["unc_DownRisk"].reshape(-1)
        point_per_sample = F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=self.downrisk_log_huber_beta,
        )
        unc_per_sample, nu = self._student_t_nll_from_error((target - pred).detach(), scale)
        point_loss = point_per_sample.mean()
        unc_loss = unc_per_sample.mean()
        loss = point_loss + self.uncertainty_loss_weight * unc_loss
        if return_metrics:
            metrics["loss_DownRisk"] = loss.detach()
            metrics["loss_task"] = loss.detach()
            metrics["loss_total"] = loss.detach()
            metrics["loss_mu"] = point_loss.detach()
            metrics["loss_unc"] = unc_loss.detach()
            metrics["nu_DownRisk"] = nu.detach()
        return loss, metrics


__all__ = ["SingleTaskLoss"]
