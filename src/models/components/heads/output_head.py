"""Task-specific output head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.config import variance_log_clamp_max as DEFAULT_LOG_CLAMP_MAX
from config.config import variance_log_clamp_min as DEFAULT_LOG_CLAMP_MIN
from src.models.components.trunks.common import SwiGLUFFN


class OutputHead(nn.Module):
    def __init__(self, dim: int, *, task_label: str) -> None:
        super().__init__()
        self.task_label = task_label
        self.log_min = float(DEFAULT_LOG_CLAMP_MIN)
        self.log_max = float(DEFAULT_LOG_CLAMP_MAX)
        self.log_center = 0.5 * (self.log_max + self.log_min)
        self.log_radius = 0.5 * (self.log_max - self.log_min)
        self.stem = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            SwiGLUFFN(dim),
        )
        self.ret_mu = nn.Linear(dim, 1)
        self.ret_log_sigma2 = nn.Linear(dim, 1)
        self.rv_log_var = nn.Linear(dim, 1)
        self.rv_log_sigma2 = nn.Linear(dim, 1)
        self.quantile_mu_raw = nn.Linear(dim, 1)
        self.quantile_log_b = nn.Linear(dim, 1)

    def _bounded_log_param(self, raw: Tensor) -> Tensor:
        return self.log_center + self.log_radius * torch.tanh(raw)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        head_out = self.stem(h)
        outputs: dict[str, Tensor] = {"head_out": head_out}
        if self.task_label == "ret":
            ret_mu = self.ret_mu(head_out).squeeze(-1)
            ret_log_sigma2 = self._bounded_log_param(self.ret_log_sigma2(head_out).squeeze(-1))
            outputs.update(
                {
                    "ret_mu": ret_mu,
                    "ret_log_sigma2": ret_log_sigma2,
                    "pred_ret": torch.exp(ret_mu),
                    "unc_ret": torch.exp(ret_log_sigma2),
                }
            )
        elif self.task_label == "rv":
            rv_log_var = self._bounded_log_param(self.rv_log_var(head_out).squeeze(-1))
            rv_log_sigma2 = self._bounded_log_param(self.rv_log_sigma2(head_out).squeeze(-1))
            outputs.update(
                {
                    "rv_log_var": rv_log_var,
                    "rv_log_sigma2": rv_log_sigma2,
                    "pred_rv": torch.exp(0.5 * rv_log_var),
                    "unc_rv": torch.exp(rv_log_sigma2),
                }
            )
        else:
            mu_raw = self.quantile_mu_raw(head_out).squeeze(-1)
            log_b = self._bounded_log_param(self.quantile_log_b(head_out).squeeze(-1))
            pred = F.softplus(mu_raw) + 1e-6
            outputs.update(
                {
                    f"mu_{self.task_label}": pred,
                    f"log_b_{self.task_label}": log_b,
                    f"pred_{self.task_label}": pred,
                    f"unc_{self.task_label}": torch.exp(log_b),
                }
            )
        return outputs


__all__ = ["OutputHead"]
