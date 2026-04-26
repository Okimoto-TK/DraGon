from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._runtime_checks import tensor_value_checks_enabled


class AsymmetricLaplaceNLLLoss(nn.Module):
    """Asymmetric Laplace negative log-likelihood with fixed tau."""

    def __init__(
        self,
        tau: float,
        _eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if tau <= 0 or tau >= 1:
            raise ValueError(
                f"tau must satisfy 0 < tau < 1, got {tau}. Valid range: (0, 1)."
            )
        if _eps <= 0:
            raise ValueError(
                f"_eps must be > 0, got {_eps}. Valid range: (0, +inf)."
            )

        self.tau = float(tau)
        self._eps = float(_eps)
        self._log_tau_term = math.log(self.tau * (1.0 - self.tau))

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        _validate_column_tensor("target", target)
        _validate_column_tensor("mu", mu)
        _validate_column_tensor("scale", scale)

        if target.shape != mu.shape or target.shape != scale.shape:
            raise ValueError(
                "target, mu, and scale must share the same shape, "
                f"got target={tuple(target.shape)}, mu={tuple(mu.shape)}, scale={tuple(scale.shape)}. "
                "Valid shape: [B, 1] with a shared batch size."
            )
        if tensor_value_checks_enabled() and torch.any(scale <= 0):
            raise ValueError(
                f"scale must be strictly positive, got min={scale.min().item()}. Valid range: (0, +inf)."
            )

        residual = (target - mu) / scale
        rho = torch.where(
            residual >= 0,
            self.tau * residual,
            (self.tau - 1.0) * residual,
        )
        nll = rho + torch.log(scale) - self._log_tau_term
        return nll.mean()


def _validate_column_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B, 1], got shape={tuple(value.shape)}. Valid shape: [B, 1]."
        )
