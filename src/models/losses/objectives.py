"""Shared loss primitives."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def pearson_corr(pred: Tensor, target: Tensor, eps: float) -> Tensor:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = torch.mean(pred_centered * target_centered)
    denominator = torch.sqrt(torch.mean(pred_centered.square()) * torch.mean(target_centered.square()) + eps)
    return numerator / denominator


def student_t_nll_from_logvar(mean: Tensor, target: Tensor, log_sigma2: Tensor, nu: float) -> Tensor:
    nu_value = float(nu)
    if nu_value <= 0.0:
        raise ValueError(f"Student-T degrees of freedom must be positive, got {nu_value}.")
    squared_error = (target - mean).square()
    inv_sigma2 = torch.exp(-log_sigma2)
    scaled_error = squared_error * inv_sigma2 / nu_value
    log_norm = (
        torch.lgamma(torch.tensor((nu_value + 1.0) * 0.5, device=mean.device, dtype=mean.dtype))
        - torch.lgamma(torch.tensor(nu_value * 0.5, device=mean.device, dtype=mean.dtype))
        - 0.5 * math.log(nu_value * math.pi)
    )
    return -log_norm + 0.5 * log_sigma2 + 0.5 * (nu_value + 1.0) * torch.log1p(scaled_error)


def gaussian_nll_from_logvar(mean: Tensor, target: Tensor, log_sigma2: Tensor) -> Tensor:
    return 0.5 * log_sigma2 + 0.5 * torch.exp(-log_sigma2) * (target - mean).square()


def qlike_from_log_variance(log_var_hat: Tensor, variance_target: Tensor) -> Tensor:
    return log_var_hat + variance_target * torch.exp(-log_var_hat)


def pinball_loss(residual: Tensor, q: float) -> Tensor:
    q_tensor = torch.full_like(residual, q)
    return torch.maximum(q_tensor * residual, (q_tensor - 1.0) * residual)


def log_ald_scale_loss(log_b: Tensor, detached_pinball: Tensor) -> Tensor:
    return log_b + torch.exp(-log_b) * detached_pinball


__all__ = [
    "gaussian_nll_from_logvar",
    "log_ald_scale_loss",
    "pearson_corr",
    "pinball_loss",
    "qlike_from_log_variance",
    "student_t_nll_from_logvar",
]
