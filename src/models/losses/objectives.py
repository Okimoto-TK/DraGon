"""Shared loss primitives."""
from __future__ import annotations

import torch
from torch import Tensor


def corr_loss(pred: Tensor, target: Tensor, eps: float) -> Tensor:
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = torch.mean(pred_centered * target_centered)
    denominator = torch.sqrt(torch.mean(pred_centered.square()) * torch.mean(target_centered.square()) + eps)
    corr = numerator / denominator
    return 1.0 - corr


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
    "corr_loss",
    "gaussian_nll_from_logvar",
    "log_ald_scale_loss",
    "pinball_loss",
    "qlike_from_log_variance",
]
