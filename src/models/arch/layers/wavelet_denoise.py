"""Fixed-wavelet denoise front-end for time series inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.config.hparams import WAVELET_DENOISE_HPARAMS

_DB4_DEC_LO = (
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
    0.6308807679298589,
    0.7148465705529157,
    0.2303778133088965,
)
_DB4_DEC_HI = (
    -0.2303778133088965,
    0.7148465705529157,
    -0.6308807679298589,
    -0.027983769416859854,
    0.18703481171909309,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
)
_DB4_REC_LO = (
    0.2303778133088965,
    0.7148465705529157,
    0.6308807679298589,
    -0.027983769416859854,
    -0.18703481171909309,
    0.030841381835560764,
    0.0328830116668852,
    -0.010597401785069032,
)
_DB4_REC_HI = (
    -0.010597401785069032,
    -0.0328830116668852,
    0.030841381835560764,
    0.18703481171909309,
    -0.027983769416859854,
    -0.6308807679298589,
    0.7148465705529157,
    -0.2303778133088965,
)


class WaveletDenoise1D(nn.Module):
    """Denoise [B, C, T_total] with fixed db4 DWT and detail-band shrinkage."""

    def __init__(
        self,
        n_channels: int,
        target_len: int,
        warmup_len: int,
        wavelet: str = WAVELET_DENOISE_HPARAMS._wavelet,
        level: int = WAVELET_DENOISE_HPARAMS._level,
        eps: float = WAVELET_DENOISE_HPARAMS._eps,
        allow_backward: bool = True,
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
        self.allow_backward = bool(allow_backward)
        if self._wavelet != "db4":
            raise ValueError(
                f"Only 'db4' is currently supported by the torch-only frontend, got {self._wavelet!r}."
            )

        self.theta_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))
        self.phi_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))
        self.psi_detail = nn.Parameter(torch.zeros(self._level, self.n_channels))
        self.register_buffer("_dec_lo", torch.tensor(_DB4_DEC_LO, dtype=torch.float32), persistent=False)
        self.register_buffer("_dec_hi", torch.tensor(_DB4_DEC_HI, dtype=torch.float32), persistent=False)
        self.register_buffer("_rec_lo", torch.tensor(_DB4_REC_LO, dtype=torch.float32), persistent=False)
        self.register_buffer("_rec_hi", torch.tensor(_DB4_REC_HI, dtype=torch.float32), persistent=False)
        self._filter_len = int(self._dec_lo.numel())
        self._analysis_pad = self._filter_len - 1

    @staticmethod
    def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(x.abs() - thr)

    def _grouped_filter(
        self,
        filt: torch.Tensor,
        *,
        channels: int,
        dtype: torch.dtype,
        device: torch.device,
        flip: bool,
    ) -> torch.Tensor:
        weight = filt.to(device=device, dtype=dtype)
        weight = weight.flip(0) if flip else weight
        return weight.view(1, 1, -1).repeat(channels, 1, 1)

    def _analysis_step(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels = x.shape[1]
        lo_weight = self._grouped_filter(
            self._dec_lo,
            channels=channels,
            dtype=x.dtype,
            device=x.device,
            flip=True,
        )
        hi_weight = self._grouped_filter(
            self._dec_hi,
            channels=channels,
            dtype=x.dtype,
            device=x.device,
            flip=True,
        )
        padded = F.pad(x, (self._analysis_pad, self._analysis_pad))
        approx = F.conv1d(padded, lo_weight, stride=2, groups=channels)
        detail = F.conv1d(padded, hi_weight, stride=2, groups=channels)
        return approx, detail

    def _synthesis_step(
        self,
        approx: torch.Tensor,
        detail: torch.Tensor,
        *,
        out_len: int,
    ) -> torch.Tensor:
        channels = approx.shape[1]
        lo_weight = self._grouped_filter(
            self._rec_lo,
            channels=channels,
            dtype=approx.dtype,
            device=approx.device,
            flip=False,
        )
        hi_weight = self._grouped_filter(
            self._rec_hi,
            channels=channels,
            dtype=approx.dtype,
            device=approx.device,
            flip=False,
        )
        recon = F.conv_transpose1d(approx, lo_weight, stride=2, groups=channels)
        recon = recon + F.conv_transpose1d(detail, hi_weight, stride=2, groups=channels)
        start = self._analysis_pad
        end = start + out_len
        return recon[..., start:end]

    def _shrink_detail(
        self,
        detail: torch.Tensor,
        *,
        level_idx: int,
    ) -> torch.Tensor:
        sigma = self._rms(detail, self._eps)
        tau = F.softplus(self.theta_detail[level_idx].to(dtype=detail.dtype)).view(
            1,
            self.n_channels,
            1,
        )
        alpha = torch.sigmoid(self.phi_detail[level_idx].to(dtype=detail.dtype)).view(
            1,
            self.n_channels,
            1,
        )
        rho = torch.sigmoid(self.psi_detail[level_idx].to(dtype=detail.dtype)).view(
            1,
            self.n_channels,
            1,
        )
        threshold = tau * sigma
        detail_shrunk = self._soft_threshold(detail, threshold)
        return rho * detail + (1.0 - rho) * (alpha * detail_shrunk)

    def _forward_features_impl(
        self,
        x_long: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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

        compute_dtype = self.theta_detail.dtype
        wavelet_dtype = torch.float32
        x_compute = x_long.to(device=x_long.device, dtype=wavelet_dtype)

        approximations = [x_compute]
        details_low_to_high: list[torch.Tensor] = []
        approx = x_compute
        for _ in range(self._level):
            approx, detail = self._analysis_step(approx)
            approximations.append(approx)
            details_low_to_high.append(detail)

        details_high_to_low = list(reversed(details_low_to_high))
        shrunk_high_to_low = [
            self._shrink_detail(detail, level_idx=level_idx)
            for level_idx, detail in enumerate(details_high_to_low)
        ]
        shrunk_low_to_high = list(reversed(shrunk_high_to_low))

        recon = approximations[-1]
        for level_idx in range(self._level - 1, -1, -1):
            recon = self._synthesis_step(
                recon,
                shrunk_low_to_high[level_idx],
                out_len=approximations[level_idx].shape[-1],
            )

        y_long = recon[..., -expected_t:]
        y = y_long[..., -self.target_len:]
        wavelet_coeffs = (
            approximations[-1].to(device=x_long.device, dtype=compute_dtype),
            *[
                detail.to(device=x_long.device, dtype=compute_dtype)
                for detail in shrunk_high_to_low
            ],
        )
        return y.to(device=x_long.device, dtype=compute_dtype), wavelet_coeffs

    def _forward_impl(self, x_long: torch.Tensor) -> torch.Tensor:
        y, _ = self._forward_features_impl(x_long)
        return y

    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        if self.allow_backward:
            return self._forward_impl(x_long)
        with torch.no_grad():
            return self._forward_impl(x_long).detach()

    def forward_features(
        self,
        x_long: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.allow_backward:
            return self._forward_features_impl(x_long)
        with torch.no_grad():
            y, coeffs = self._forward_features_impl(x_long)
        return y.detach(), tuple(coeff.detach() for coeff in coeffs)
