from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task_labels import canonical_task_label, canonical_training_task, field_domain, is_quantile_task, quantile_level

from .asymmetric_laplace_nll import AsymmetricLaplaceNLLLoss
from .gamma_nll import GammaNLLLoss
from .student_t_nll import StudentTNLLLoss


class SingleTaskDistributionLoss(nn.Module):
    """Field-aware mu/sigma objective wrapper with fixed semantics per stage."""

    def __init__(
        self,
        *,
        mode: str,
        field: str,
        q_tau: float,
        ret_mu_fixed_scale: float = 0.02335,
        ret_mu_fixed_nu: float = 2.82,
        _eps: float = 1e-6,
        _nu_ret_init: float = 8.0,
        _nu_ret_min: float = 2.01,
        _gamma_shape_min: float = 1e-4,
        _ald_scale_min: float = 1e-6,
        _loss_compute_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if q_tau <= 0 or q_tau >= 1:
            raise ValueError(
                f"q_tau must satisfy 0 < q_tau < 1, got {q_tau}. Valid range: (0, 1)."
            )
        if ret_mu_fixed_scale <= 0:
            raise ValueError(
                "ret_mu_fixed_scale must be > 0, "
                f"got {ret_mu_fixed_scale}. Valid range: (0, +inf)."
            )
        if ret_mu_fixed_nu <= 2:
            raise ValueError(
                f"ret_mu_fixed_nu must be > 2, got {ret_mu_fixed_nu}. Valid range: (2, +inf)."
            )
        if _eps <= 0:
            raise ValueError(f"_eps must be > 0, got {_eps}. Valid range: (0, +inf).")
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
        if _loss_compute_dtype not in {"float32"}:
            raise ValueError(
                f"_loss_compute_dtype must be 'float32', got {_loss_compute_dtype!r}."
            )

        self.mode = canonical_training_task(mode)
        self.field = canonical_task_label(field)
        self.domain = field_domain(self.field)
        self.q_tau = quantile_level(self.field) if is_quantile_task(self.field) else float(q_tau)
        self.ret_mu_fixed_scale = float(ret_mu_fixed_scale)
        self.ret_mu_fixed_nu = float(ret_mu_fixed_nu)
        self._eps = float(_eps)
        self._nu_ret_init = float(_nu_ret_init)
        self._nu_ret_min = float(_nu_ret_min)
        self._gamma_shape_min = float(_gamma_shape_min)
        self._ald_scale_min = float(_ald_scale_min)
        self._loss_compute_dtype = _loss_compute_dtype

        if self.mode == "mu" and self.domain == "ret":
            self.ret_mu_nll = StudentTNLLLoss(_nu_min=2.000001, _eps=self._eps)
        else:
            self.ret_mu_nll = None
        if self.mode == "sigma" and self.domain == "ret":
            init_gap = self._nu_ret_init - self._nu_ret_min
            self._nu_ret_raw = nn.Parameter(
                torch.tensor(self._inverse_softplus(init_gap), dtype=torch.float32)
            )
            self.ret_sigma_nll = StudentTNLLLoss(_nu_min=self._nu_ret_min, _eps=self._eps)
        else:
            self._nu_ret_raw = None
            self.ret_sigma_nll = None
        self.rv_nll = GammaNLLLoss(_eps=self._eps)
        self.q_nll = AsymmetricLaplaceNLLLoss(tau=self.q_tau, _eps=self._eps)
        self._q_sigma_factor = math.sqrt(
            (1.0 - 2.0 * self.q_tau + 2.0 * self.q_tau * self.q_tau)
            / (self.q_tau * self.q_tau * (1.0 - self.q_tau) * (1.0 - self.q_tau))
        )

    def _apply(self, fn):
        super()._apply(fn)
        if self._nu_ret_raw is not None:
            self._nu_ret_raw.data = self._nu_ret_raw.data.to(dtype=torch.float32)
            if self._nu_ret_raw.grad is not None:
                self._nu_ret_raw.grad.data = self._nu_ret_raw.grad.data.to(dtype=torch.float32)
        return self

    def forward(
        self,
        *,
        target: torch.Tensor,
        prediction_raw: torch.Tensor,
        mu_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_column_tensor("target", target)
        _validate_column_tensor("prediction_raw", prediction_raw)
        if target.shape != prediction_raw.shape:
            raise ValueError(
                "target and prediction_raw must share the same shape, "
                f"got target={tuple(target.shape)}, prediction_raw={tuple(prediction_raw.shape)}."
            )
        if mu_input is not None:
            _validate_column_tensor("mu_input", mu_input)
            if mu_input.shape != target.shape:
                raise ValueError(
                    "mu_input must share the same shape as target, "
                    f"got mu_input={tuple(mu_input.shape)}, target={tuple(target.shape)}."
                )

        target_f32 = target.float()
        prediction_raw_f32 = prediction_raw.float()
        mu_input_f32 = None if mu_input is None else mu_input.float()

        if self.mode == "mu":
            return self._forward_mu(
                target=target_f32,
                prediction_raw=prediction_raw_f32,
            )
        return self._forward_sigma(
            target=target_f32,
            prediction_raw=prediction_raw_f32,
            mu_input=mu_input_f32,
        )

    def _forward_mu(
        self,
        *,
        target: torch.Tensor,
        prediction_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.domain == "ret":
            if self.ret_mu_nll is None:
                raise RuntimeError("ret_mu_nll must be initialized for mu-ret.")
            mu_pred = prediction_raw
            scale = torch.full_like(target, self.ret_mu_fixed_scale)
            nu = target.new_tensor(self.ret_mu_fixed_nu, dtype=torch.float32)
            loss = self.ret_mu_nll(
                target=target,
                mu=prediction_raw,
                scale=scale,
                nu=nu,
            )
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_mu": loss,
                "mu_pred": mu_pred,
            }

        if self.domain == "rv":
            mu_pred = F.softplus(prediction_raw) + self._eps
            target_safe = target.clamp_min(self._eps)
            ratio = target_safe / mu_pred
            loss = (ratio - torch.log(ratio) - 1.0).mean()
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_mu": loss,
                "mu_pred": mu_pred,
            }

        mu_pred = prediction_raw
        residual = target - mu_pred
        loss = torch.where(
            residual >= 0,
            self.q_tau * residual,
            (self.q_tau - 1.0) * residual,
        ).mean()
        return {
            "loss_total": loss,
            "loss_task": loss,
            "loss_mu": loss,
            "mu_pred": mu_pred,
        }

    def _forward_sigma(
        self,
        *,
        target: torch.Tensor,
        prediction_raw: torch.Tensor,
        mu_input: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if mu_input is None:
            raise ValueError("mu_input is required when mode='sigma'.")

        sigma_pred = F.softplus(prediction_raw) + self._eps
        if self.domain == "ret":
            if self.ret_sigma_nll is None or self._nu_ret_raw is None:
                raise RuntimeError("ret_nll and nu parameter must be initialized for sigma-ret.")
            nu_ret = (
                self._nu_ret_min + F.softplus(self._nu_ret_raw)
            ).to(device=target.device, dtype=torch.float32)
            scale = sigma_pred * torch.sqrt((nu_ret - 2.0) / nu_ret)
            loss = self.ret_sigma_nll(
                target=target,
                mu=mu_input,
                scale=scale,
                nu=nu_ret,
            )
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_nll": loss,
                "sigma_pred": sigma_pred,
                "nu_ret": nu_ret,
            }

        if self.domain == "rv":
            mean = mu_input.clamp_min(self._eps)
            shape = (mean / sigma_pred.clamp_min(self._eps)).pow(2).clamp_min(self._gamma_shape_min)
            loss = self.rv_nll(
                target=target,
                mean=mean,
                shape=shape,
            )
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_nll": loss,
                "sigma_pred": sigma_pred,
                "shape_rv": shape,
            }

        scale = (sigma_pred / sigma_pred.new_tensor(self._q_sigma_factor)).clamp_min(self._ald_scale_min)
        loss = self.q_nll(
            target=target,
            mu=mu_input,
            scale=scale,
        )
        return {
            "loss_total": loss,
            "loss_task": loss,
            "loss_nll": loss,
            "sigma_pred": sigma_pred,
        }

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
