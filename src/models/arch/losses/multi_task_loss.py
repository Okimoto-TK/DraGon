from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asymmetric_laplace_nll import AsymmetricLaplaceNLLLoss
from .gamma_nll import GammaNLLLoss
from .student_t_nll import StudentTNLLLoss


class MultiTaskDistributionLoss(nn.Module):
    """Distributional multi-task loss for ret, rv, and q targets."""

    def __init__(
        self,
        q_tau: float,
        ret_loss_weight: float = 1.0,
        rv_loss_weight: float = 1.0,
        q_loss_weight: float = 1.0,
        rv_tail_weight_threshold: float = 0.03,
        rv_tail_weight_alpha: float = 2.0,
        rv_tail_weight_max: float = 4.0,
        _eps: float = 1e-6,
        _nu_ret_init: float = 8.0,
        _nu_ret_min: float = 2.01,
        _gamma_shape_min: float = 1e-4,
        _ald_scale_min: float = 1e-6,
    ) -> None:
        super().__init__()
        if ret_loss_weight < 0:
            raise ValueError(
                f"ret_loss_weight must be >= 0, got {ret_loss_weight}. Valid range: [0, +inf)."
            )
        if rv_loss_weight < 0:
            raise ValueError(
                f"rv_loss_weight must be >= 0, got {rv_loss_weight}. Valid range: [0, +inf)."
            )
        if q_loss_weight < 0:
            raise ValueError(
                f"q_loss_weight must be >= 0, got {q_loss_weight}. Valid range: [0, +inf)."
            )
        if q_tau <= 0 or q_tau >= 1:
            raise ValueError(
                f"q_tau must satisfy 0 < q_tau < 1, got {q_tau}. Valid range: (0, 1)."
            )
        if rv_tail_weight_threshold <= 0:
            raise ValueError(
                "rv_tail_weight_threshold must be > 0, "
                f"got {rv_tail_weight_threshold}. Valid range: (0, +inf)."
            )
        if rv_tail_weight_alpha < 0:
            raise ValueError(
                "rv_tail_weight_alpha must be >= 0, "
                f"got {rv_tail_weight_alpha}. Valid range: [0, +inf)."
            )
        if rv_tail_weight_max < 1:
            raise ValueError(
                "rv_tail_weight_max must be >= 1, "
                f"got {rv_tail_weight_max}. Valid range: [1, +inf)."
            )
        if _eps <= 0:
            raise ValueError(
                f"_eps must be > 0, got {_eps}. Valid range: (0, +inf)."
            )
        if _nu_ret_min <= 2:
            raise ValueError(
                f"_nu_ret_min must be > 2, got {_nu_ret_min}. Valid range: (2, +inf)."
            )
        if _nu_ret_init <= _nu_ret_min:
            raise ValueError(
                f"_nu_ret_init must be > _nu_ret_min, got _nu_ret_init={_nu_ret_init}, "
                f"_nu_ret_min={_nu_ret_min}. Valid range: (_nu_ret_min, +inf)."
            )
        if _gamma_shape_min <= 0:
            raise ValueError(
                f"_gamma_shape_min must be > 0, got {_gamma_shape_min}. Valid range: (0, +inf)."
            )
        if _ald_scale_min <= 0:
            raise ValueError(
                f"_ald_scale_min must be > 0, got {_ald_scale_min}. Valid range: (0, +inf)."
            )

        self.q_tau = float(q_tau)
        self.ret_loss_weight = float(ret_loss_weight)
        self.rv_loss_weight = float(rv_loss_weight)
        self.q_loss_weight = float(q_loss_weight)
        self.rv_tail_weight_threshold = float(rv_tail_weight_threshold)
        self.rv_tail_weight_alpha = float(rv_tail_weight_alpha)
        self.rv_tail_weight_max = float(rv_tail_weight_max)
        self._eps = float(_eps)
        self._nu_ret_init = float(_nu_ret_init)
        self._nu_ret_min = float(_nu_ret_min)
        self._gamma_shape_min = float(_gamma_shape_min)
        self._ald_scale_min = float(_ald_scale_min)

        init_gap = self._nu_ret_init - self._nu_ret_min
        self._nu_ret_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_gap), dtype=torch.float32)
        )

        self.ret_nll = StudentTNLLLoss(_nu_min=self._nu_ret_min, _eps=self._eps)
        self.rv_nll = GammaNLLLoss(_eps=self._eps)
        self.q_nll = AsymmetricLaplaceNLLLoss(tau=self.q_tau, _eps=self._eps)
        self._q_sigma_factor = math.sqrt(
            (1.0 - 2.0 * self.q_tau + 2.0 * self.q_tau * self.q_tau)
            / (self.q_tau * self.q_tau * (1.0 - self.q_tau) * (1.0 - self.q_tau))
        )

    def forward(
        self,
        target_ret: torch.Tensor,
        pred_mu_ret: torch.Tensor,
        pred_scale_ret_raw: torch.Tensor,
        target_rv: torch.Tensor,
        pred_mean_rv_raw: torch.Tensor,
        pred_shape_rv_raw: torch.Tensor,
        target_q: torch.Tensor,
        pred_mu_q: torch.Tensor,
        pred_scale_q_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensors = {
            "target_ret": target_ret,
            "pred_mu_ret": pred_mu_ret,
            "pred_scale_ret_raw": pred_scale_ret_raw,
            "target_rv": target_rv,
            "pred_mean_rv_raw": pred_mean_rv_raw,
            "pred_shape_rv_raw": pred_shape_rv_raw,
            "target_q": target_q,
            "pred_mu_q": pred_mu_q,
            "pred_scale_q_raw": pred_scale_q_raw,
        }
        for name, value in tensors.items():
            _validate_column_tensor(name, value)

        batch_sizes = {value.shape[0] for value in tensors.values()}
        if len(batch_sizes) != 1:
            batch_desc = ", ".join(
                f"{name}={value.shape[0]}" for name, value in tensors.items()
            )
            raise ValueError(
                f"Batch size mismatch across multi-task loss inputs: {batch_desc}. "
                "Valid range: all batch sizes must be equal."
            )

        nu_ret = (
            self._nu_ret_min + F.softplus(self._nu_ret_raw)
        ).to(device=pred_mu_ret.device, dtype=pred_mu_ret.dtype)
        scale_ret = F.softplus(pred_scale_ret_raw) + self._eps
        mean_rv = F.softplus(pred_mean_rv_raw) + self._eps
        shape_rv = F.softplus(pred_shape_rv_raw) + self._gamma_shape_min
        scale_q = F.softplus(pred_scale_q_raw) + self._ald_scale_min

        loss_ret = self.ret_nll(
            target=target_ret,
            mu=pred_mu_ret,
            scale=scale_ret,
            nu=nu_ret,
        )
        loss_rv = self.rv_nll(
            target=target_rv,
            mean=mean_rv,
            shape=shape_rv,
            sample_weight=self._rv_sample_weight(target=target_rv),
        )
        loss_q = self.q_nll(
            target=target_q,
            mu=pred_mu_q,
            scale=scale_q,
        )

        loss_total = (
            self.ret_loss_weight * loss_ret
            + self.rv_loss_weight * loss_rv
            + self.q_loss_weight * loss_q
        )

        sigma_ret_pred = scale_ret * torch.sqrt(nu_ret / (nu_ret - 2.0))
        sigma_rv_pred = mean_rv / torch.sqrt(shape_rv)
        sigma_q_pred = scale_q * scale_q.new_tensor(self._q_sigma_factor)

        return {
            "loss_total": loss_total,
            "loss_ret": loss_ret,
            "loss_rv": loss_rv,
            "loss_q": loss_q,
            "nu_ret": nu_ret,
            "sigma_ret_pred": sigma_ret_pred,
            "sigma_rv_pred": sigma_rv_pred,
            "sigma_q_pred": sigma_q_pred,
        }

    def _rv_sample_weight(
        self,
        *,
        target: torch.Tensor,
    ) -> torch.Tensor:
        tail_excess = torch.relu(target / self.rv_tail_weight_threshold - 1.0)
        weights = 1.0 + self.rv_tail_weight_alpha * tail_excess
        return weights.clamp_max(self.rv_tail_weight_max)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0:
            raise ValueError(
                f"value must be > 0 for inverse_softplus, got {value}. Valid range: (0, +inf)."
            )
        return value + math.log(-math.expm1(-value))


def _validate_column_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B, 1], got shape={tuple(value.shape)}. Valid shape: [B, 1]."
        )
