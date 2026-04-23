from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.arch.heads.single_task_head import VALID_SINGLE_TASKS

from .asymmetric_laplace_nll import AsymmetricLaplaceNLLLoss
from .gamma_nll import GammaNLLLoss
from .student_t_nll import StudentTNLLLoss


class SingleTaskDistributionLoss(nn.Module):
    """Distributional loss wrapper for one selected forecasting task."""

    def __init__(
        self,
        task: str,
        q_tau: float,
        ret_tail_weight_threshold: float = 0.05,
        ret_tail_weight_alpha: float = 2.0,
        ret_tail_weight_max: float = 4.0,
        rv_tail_weight_threshold: float = 0.03,
        rv_tail_weight_alpha: float = 2.0,
        rv_tail_weight_max: float = 4.0,
        _eps: float = 1e-6,
        _nu_ret_init: float = 8.0,
        _nu_ret_min: float = 2.01,
        _gamma_shape_min: float = 1e-4,
        _ald_scale_min: float = 1e-6,
        _loss_compute_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if task not in VALID_SINGLE_TASKS:
            raise ValueError(
                f"task must be one of {VALID_SINGLE_TASKS}, got {task!r}."
            )
        if q_tau <= 0 or q_tau >= 1:
            raise ValueError(
                f"q_tau must satisfy 0 < q_tau < 1, got {q_tau}. Valid range: (0, 1)."
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
        if ret_tail_weight_threshold <= 0:
            raise ValueError(
                "ret_tail_weight_threshold must be > 0, "
                f"got {ret_tail_weight_threshold}. Valid range: (0, +inf)."
            )
        if ret_tail_weight_alpha < 0:
            raise ValueError(
                "ret_tail_weight_alpha must be >= 0, "
                f"got {ret_tail_weight_alpha}. Valid range: [0, +inf)."
            )
        if ret_tail_weight_max < 1:
            raise ValueError(
                "ret_tail_weight_max must be >= 1, "
                f"got {ret_tail_weight_max}. Valid range: [1, +inf)."
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
        if _loss_compute_dtype not in {"float32"}:
            raise ValueError(
                f"_loss_compute_dtype must be 'float32', got {_loss_compute_dtype!r}."
            )

        self.task = task
        self.q_tau = float(q_tau)
        self.ret_tail_weight_threshold = float(ret_tail_weight_threshold)
        self.ret_tail_weight_alpha = float(ret_tail_weight_alpha)
        self.ret_tail_weight_max = float(ret_tail_weight_max)
        self.rv_tail_weight_threshold = float(rv_tail_weight_threshold)
        self.rv_tail_weight_alpha = float(rv_tail_weight_alpha)
        self.rv_tail_weight_max = float(rv_tail_weight_max)
        self._eps = float(_eps)
        self._nu_ret_init = float(_nu_ret_init)
        self._nu_ret_min = float(_nu_ret_min)
        self._gamma_shape_min = float(_gamma_shape_min)
        self._ald_scale_min = float(_ald_scale_min)
        self._loss_compute_dtype = _loss_compute_dtype

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

    def _apply(self, fn):
        super()._apply(fn)
        # Keep the global Student-T degrees-of-freedom parameter in FP32 so
        # tiny AdamW updates are not quantized away when the rest of the model
        # runs in BF16.
        self._nu_ret_raw.data = self._nu_ret_raw.data.to(dtype=torch.float32)
        if self._nu_ret_raw.grad is not None:
            self._nu_ret_raw.grad.data = self._nu_ret_raw.grad.data.to(dtype=torch.float32)
        return self

    def forward(
        self,
        *,
        target: torch.Tensor,
        pred_primary: torch.Tensor,
        pred_aux_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        for name, value in (
            ("target", target),
            ("pred_primary", pred_primary),
            ("pred_aux_raw", pred_aux_raw),
        ):
            _validate_column_tensor(name, value)
        batch_sizes = {target.shape[0], pred_primary.shape[0], pred_aux_raw.shape[0]}
        if len(batch_sizes) != 1:
            raise ValueError(
                "Batch size mismatch across single-task loss inputs: "
                f"target={target.shape[0]}, pred_primary={pred_primary.shape[0]}, pred_aux_raw={pred_aux_raw.shape[0]}. "
                "Valid range: all batch sizes must be equal."
            )

        target_f32 = target.float()
        pred_primary_f32 = pred_primary.float()
        pred_aux_raw_f32 = pred_aux_raw.float()

        if self.task == "ret":
            nu_ret = (
                self._nu_ret_min + F.softplus(self._nu_ret_raw)
            ).to(device=pred_primary.device, dtype=torch.float32)
            scale = F.softplus(pred_aux_raw_f32) + self._eps
            sample_weight = self._ret_sample_weight(target=target_f32)
            loss = self.ret_nll(
                target=target_f32,
                mu=pred_primary_f32,
                scale=scale,
                nu=nu_ret,
                sample_weight=sample_weight,
            )
            sigma = scale * torch.sqrt(nu_ret / (nu_ret - 2.0))
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_ret_weighted_nll": loss,
                "sigma_pred": sigma,
                "nu_ret": nu_ret,
            }

        if self.task == "rv":
            mean = F.softplus(pred_primary_f32) + self._eps
            shape = F.softplus(pred_aux_raw_f32) + self._gamma_shape_min
            sample_weight = self._rv_sample_weight(target=target_f32)
            loss = self.rv_nll(
                target=target_f32,
                mean=mean,
                shape=shape,
                sample_weight=sample_weight,
            )
            sigma = mean / torch.sqrt(shape)
            return {
                "loss_total": loss,
                "loss_task": loss,
                "loss_rv_weighted_nll": loss,
                "sigma_pred": sigma,
                "shape_rv": shape,
            }

        scale = F.softplus(pred_aux_raw_f32) + self._ald_scale_min
        loss = self.q_nll(
            target=target_f32,
            mu=pred_primary_f32,
            scale=scale,
        )
        sigma = scale * scale.new_tensor(self._q_sigma_factor)
        return {
            "loss_total": loss,
            "loss_task": loss,
            "sigma_pred": sigma,
        }

    def _ret_sample_weight(
        self,
        *,
        target: torch.Tensor,
    ) -> torch.Tensor:
        abs_target = target.abs()
        tail_excess = torch.relu(abs_target / self.ret_tail_weight_threshold - 1.0)
        weights = 1.0 + self.ret_tail_weight_alpha * tail_excess
        return weights.clamp_max(self.ret_tail_weight_max)

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
