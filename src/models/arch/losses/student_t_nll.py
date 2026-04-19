from __future__ import annotations

import torch
import torch.nn as nn


class StudentTNLLLoss(nn.Module):
    """Student-t negative log-likelihood with mean reduction."""

    def __init__(
        self,
        _nu_min: float = 2.01,
        _eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if _nu_min <= 2:
            raise ValueError(
                f"_nu_min must be > 2, got {_nu_min}. Valid range: (2, +inf)."
            )
        if _eps <= 0:
            raise ValueError(
                f"_eps must be > 0, got {_eps}. Valid range: (0, +inf)."
            )

        self._nu_min = float(_nu_min)
        self._eps = float(_eps)

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
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
        if nu.ndim > 2:
            raise ValueError(
                f"nu must be scalar-like or broadcastable, got ndim={nu.ndim}, shape={tuple(nu.shape)}. "
                "Valid shape: [] or [1] or [1, 1]."
            )
        if nu.numel() != 1:
            raise ValueError(
                f"nu must contain exactly one value, got numel={nu.numel()}, shape={tuple(nu.shape)}. "
                "Valid shape: [] or [1] or [1, 1]."
            )
        if torch.any(scale <= 0):
            raise ValueError(
                f"scale must be strictly positive, got min={scale.min().item()}. Valid range: (0, +inf)."
            )
        if torch.any(nu <= self._nu_min):
            raise ValueError(
                f"nu must be > {self._nu_min}, got value={nu.detach().reshape(-1)[0].item()}. "
                f"Valid range: ({self._nu_min}, +inf)."
            )

        nu = nu.to(device=target.device, dtype=target.dtype)
        pi = target.new_tensor(torch.pi)
        squared_error = (target - mu).pow(2)
        nll = (
            torch.log(scale)
            + 0.5 * torch.log(nu * pi)
            + torch.lgamma(nu / 2.0)
            - torch.lgamma((nu + 1.0) / 2.0)
            + ((nu + 1.0) / 2.0)
            * torch.log1p(squared_error / (nu * scale.pow(2)))
        )
        return nll.mean()


def _validate_column_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B, 1], got shape={tuple(value.shape)}. Valid shape: [B, 1]."
        )
