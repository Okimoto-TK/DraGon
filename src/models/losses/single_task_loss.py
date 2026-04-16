"""Single-target loss orchestration for ret / rv / quantile tasks."""
from __future__ import annotations

from torch import Tensor, nn

from config.config import quantile_nll_weight as DEFAULT_QUANTILE_NLL_WEIGHT
from config.config import ret_nll_weight as DEFAULT_RET_NLL_WEIGHT
from config.config import rv_nll_weight as DEFAULT_RV_NLL_WEIGHT
from config.config import variance_eps as DEFAULT_VARIANCE_EPS
from config.config import variance_log_clamp_max as DEFAULT_LOG_CLAMP_MAX
from config.config import variance_log_clamp_min as DEFAULT_LOG_CLAMP_MIN
from src.models.losses.objectives import (
    corr_loss,
    gaussian_nll_from_logvar,
    log_ald_scale_loss,
    pinball_loss,
    qlike_from_log_variance,
)
from src.task_labels import is_quantile_task, quantile_level


class SingleTaskLoss(nn.Module):
    def __init__(self, *, task_label: str) -> None:
        super().__init__()
        self.task_label = task_label
        self.eps = float(DEFAULT_VARIANCE_EPS)
        self.log_clamp_min = float(DEFAULT_LOG_CLAMP_MIN)
        self.log_clamp_max = float(DEFAULT_LOG_CLAMP_MAX)
        self.ret_nll_weight = float(DEFAULT_RET_NLL_WEIGHT)
        self.rv_nll_weight = float(DEFAULT_RV_NLL_WEIGHT)
        self.quantile_nll_weight = float(DEFAULT_QUANTILE_NLL_WEIGHT)
        self.q = quantile_level(task_label) if is_quantile_task(task_label) else None

    def _ret_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        y = batch["label_ret"].clamp_min(self.eps)
        z = y.log()
        mu = outputs["ret_mu"]
        log_sigma2 = outputs["ret_log_sigma2"].clamp(self.log_clamp_min, self.log_clamp_max)
        loss_main = corr_loss(mu, z, self.eps)
        loss_unc = gaussian_nll_from_logvar(mu.detach(), z, log_sigma2).mean()
        total = loss_main + self.ret_nll_weight * loss_unc
        metrics = {
            "loss_total": total.detach(),
            "loss_task": loss_main.detach(),
            "loss_mu": loss_main.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_ret": loss_main.detach(),
        }
        return total, metrics

    def _rv_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        y = batch["label_rv"].clamp_min(self.eps)
        variance_target = y.square().clamp_min(self.eps)
        log_variance_target = variance_target.log()
        log_var_hat = outputs["rv_log_var"].clamp(self.log_clamp_min, self.log_clamp_max)
        log_sigma2 = outputs["rv_log_sigma2"].clamp(self.log_clamp_min, self.log_clamp_max)
        loss_main = qlike_from_log_variance(log_var_hat, variance_target).mean()
        loss_unc = gaussian_nll_from_logvar(log_var_hat.detach(), log_variance_target, log_sigma2).mean()
        total = loss_main + self.rv_nll_weight * loss_unc
        metrics = {
            "loss_total": total.detach(),
            "loss_task": loss_main.detach(),
            "loss_mu": loss_main.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_rv": loss_main.detach(),
        }
        return total, metrics

    def _quantile_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.q is not None
        y = batch["label_ret"].clamp_min(self.eps)
        z = y.log()
        mu = outputs[f"mu_{self.task_label}"].clamp_min(self.eps)
        log_mu = mu.log()
        log_b = outputs[f"log_b_{self.task_label}"].clamp(self.log_clamp_min, self.log_clamp_max)
        residual = z - log_mu
        loss_main = pinball_loss(residual, self.q).mean()
        detached_residual = z - log_mu.detach()
        detached_pinball = pinball_loss(detached_residual, self.q)
        loss_unc = log_ald_scale_loss(log_b, detached_pinball).mean()
        total = loss_main + self.quantile_nll_weight * loss_unc
        metrics = {
            "loss_total": total.detach(),
            "loss_task": loss_main.detach(),
            "loss_mu": loss_main.detach(),
            "loss_unc": loss_unc.detach(),
            f"loss_{self.task_label}": loss_main.detach(),
        }
        return total, metrics

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        *,
        return_metrics: bool = True,
        update_state: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del update_state
        if self.task_label == "ret":
            loss, metrics = self._ret_loss(outputs, batch)
        elif self.task_label == "rv":
            loss, metrics = self._rv_loss(outputs, batch)
        else:
            loss, metrics = self._quantile_loss(outputs, batch)
        if not return_metrics:
            return loss, {}
        return loss, metrics


__all__ = ["SingleTaskLoss"]
