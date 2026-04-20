"""Fixed-wavelet denoise front-end for time series inputs."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.config.hparams import WAVELET_DENOISE_HPARAMS


def _require_ptwt():
    try:
        import ptwt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in env setup only
        raise ImportError(
            "WaveletDenoise1D requires 'ptwt'. Install it with `pip install ptwt`."
        ) from exc
    return ptwt


def _disable_for_compile(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable(fn)
    return fn


class WaveletDenoise1D(nn.Module):
    """Denoise [B, C, T_total] with fixed DWT and detail-band shrinkage."""

    def __init__(
        self,
        n_channels: int,
        target_len: int,
        warmup_len: int,
        wavelet: str = WAVELET_DENOISE_HPARAMS._wavelet,
        level: int = WAVELET_DENOISE_HPARAMS._level,
        eps: float = WAVELET_DENOISE_HPARAMS._eps,
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError(
                f"n_channels must be > 0, got {n_channels}. Expected positive integer."
            )
        if target_len <= 0:
            raise ValueError(
                f"target_len must be > 0, got {target_len}. Expected positive integer."
            )
        if warmup_len < 0:
            raise ValueError(
                f"warmup_len must be >= 0, got {warmup_len}. Expected non-negative integer."
            )
        if level <= 0:
            raise ValueError(
                f"level must be > 0, got {level}. Expected positive integer."
            )

        self.n_channels = int(n_channels)
        self.target_len = int(target_len)
        self.warmup_len = int(warmup_len)
        self._wavelet = str(wavelet)
        self._level = int(level)
        self._eps = float(eps)

        self.theta_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))
        self.phi_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))
        self.psi_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))

    @staticmethod
    def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(x.abs() - thr)

    @_disable_for_compile
    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        if x_long.ndim != 3:
            raise ValueError(
                f"x_long must have shape [B, C, T], got ndim={x_long.ndim}, "
                f"shape={tuple(x_long.shape)}."
            )

        batch, channels, t_total = x_long.shape
        expected_t = self.target_len + self.warmup_len
        expected_shape = f"[{batch}, {self.n_channels}, {expected_t}]"
        actual_shape = f"[{batch}, {channels}, {t_total}]"

        if channels != self.n_channels or t_total != expected_t:
            raise ValueError(
                "x_long shape mismatch: expected "
                f"{expected_shape}, got {actual_shape}."
            )

        ptwt = _require_ptwt()

        # Keep wavelet math in fp32/TF32 while the rest of the model stays on bf16.
        autocast_context = (
            torch.autocast(device_type=x_long.device.type, enabled=False)
            if x_long.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            x_long_fp32 = x_long.float()
            coeffs = ptwt.wavedec(
                x_long_fp32,
                self._wavelet,
                mode="zero",
                level=self._level,
                axis=-1,
            )
            approximation = coeffs[0]
            details = list(coeffs[1:])

            if len(details) != self._level:
                raise ValueError(
                    f"DWT detail level mismatch: expected {self._level}, got {len(details)}."
                )

            new_details: list[torch.Tensor] = []
            for level_idx, detail in enumerate(details):
                sigma = self._rms(detail, self._eps)

                tau = F.softplus(self.theta_detail[level_idx]).view(
                    1, self.n_channels, 1
                )
                alpha = torch.sigmoid(self.phi_detail[level_idx]).view(
                    1, self.n_channels, 1
                )
                rho = torch.sigmoid(self.psi_detail[level_idx]).view(
                    1, self.n_channels, 1
                )

                threshold = tau * sigma
                detail_shrunk = self._soft_threshold(detail, threshold)
                detail_out = rho * detail + (1.0 - rho) * (alpha * detail_shrunk)
                new_details.append(detail_out)

            y_long = ptwt.waverec([approximation] + new_details, self._wavelet, axis=-1)
            if y_long.shape[-1] < expected_t:
                raise ValueError(
                    "Reconstructed length is shorter than expected: "
                    f"got {y_long.shape[-1]}, expected at least {expected_t}."
                )

            y_long = y_long[..., -expected_t:]
            y = y_long[..., -self.target_len:]
        return y.to(device=x_long.device, dtype=x_long.dtype)
