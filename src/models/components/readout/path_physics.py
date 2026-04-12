from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.data.registry.processor import LABEL_WEIGHTS, LABEL_WINDOW


class PathPhysicsReadout(nn.Module):
    """Deterministic physics layer.

    Inputs
    ------
    c0   : [B, 1] or [B]
    c123 : [B, 3]

    Outputs
    -------
    dict with:
        mu, sigma_c, kappa, v_jump,
        sigma_total, sigma_up, sigma_down,
        pred_S, pred_M, pred_MDD, pred_RV,
        scale_S, scale_M, scale_MDD, scale_RV
    """

    def __init__(
        self,
        eps: float = 1e-6,
        use_effective_horizon_for_S: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.use_effective_horizon_for_S = bool(use_effective_horizon_for_S)

        T = float(LABEL_WINDOW)
        T_eff = float(sum((k + 1) * float(w) for k, w in enumerate(LABEL_WEIGHTS)))

        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))
        self.register_buffer("T_eff", torch.tensor(T_eff, dtype=torch.float32))

        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        self.sqrt_pi_over_2 = math.sqrt(math.pi / 2.0)

    def forward(self, c0: Tensor, c123: Tensor) -> dict[str, Tensor]:
        if c0.ndim == 2 and c0.size(-1) == 1:
            mu = c0.squeeze(-1)
        elif c0.ndim == 1:
            mu = c0
        else:
            raise ValueError(f"Expected c0 shape [B,1] or [B], got {tuple(c0.shape)}")

        if c123.ndim != 2 or c123.size(-1) != 3:
            raise ValueError(f"Expected c123 shape [B,3], got {tuple(c123.shape)}")

        raw_sigma_c = c123[:, 0]
        raw_kappa = c123[:, 1]
        raw_v_jump = c123[:, 2]

        sigma_c = F.softplus(raw_sigma_c) + self.eps
        kappa = torch.tanh(raw_kappa)
        v_jump = F.softplus(raw_v_jump) + self.eps

        sigma_total = torch.sqrt(sigma_c.square() + v_jump)

        sigma_up = sigma_total * (1.0 + kappa)
        sigma_down = sigma_total * (1.0 - kappa)

        sigma_up = sigma_up.clamp_min(self.eps)
        sigma_down = sigma_down.clamp_min(self.eps)

        T = self.T.to(device=mu.device, dtype=mu.dtype)
        T_eff = self.T_eff.to(device=mu.device, dtype=mu.dtype)
        sqrt_T = torch.sqrt(T)

        pred_S = mu * (T_eff if self.use_effective_horizon_for_S else T)
        pred_M = mu * T + sigma_up * (self.sqrt_2_over_pi * sqrt_T)
        pred_MDD = F.softplus(
            sigma_down * (self.sqrt_pi_over_2 * sqrt_T) - 0.5 * mu * T
        )
        pred_RV = sigma_total * sqrt_T

        scale_S = (sigma_total * sqrt_T).clamp_min(self.eps)
        scale_M = (sigma_up * sqrt_T).clamp_min(self.eps)
        scale_MDD = (sigma_down * sqrt_T).clamp_min(self.eps)
        scale_RV = (sigma_total * sqrt_T).clamp_min(self.eps)

        return {
            "mu": mu,
            "sigma_c": sigma_c,
            "kappa": kappa,
            "v_jump": v_jump,
            "sigma_total": sigma_total,
            "sigma_up": sigma_up,
            "sigma_down": sigma_down,
            "pred_S": pred_S,
            "pred_M": pred_M,
            "pred_MDD": pred_MDD,
            "pred_RV": pred_RV,
            "scale_S": scale_S,
            "scale_M": scale_M,
            "scale_MDD": scale_MDD,
            "scale_RV": scale_RV,
        }