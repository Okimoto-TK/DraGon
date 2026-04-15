"""Wavelet neural operator style 1D blocks and encoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from config.config import hidden_dim as DEFAULT_HIDDEN_DIM
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import macro_wno_num_blocks as DEFAULT_WNO_NUM_BLOCKS
from torch import Tensor, nn

from src.models.components.normalization import LayerNorm1d
from src.models.components.encoders.res_conv_1d import ResConv1dBlock

_DB4_DEC_LO = (
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171888114,
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
    0.18703481171888114,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
)
_DB4_REC_LO = (
    0.2303778133088965,
    0.7148465705529157,
    0.6308807679298589,
    -0.027983769416859854,
    -0.18703481171888114,
    0.030841381835560764,
    0.0328830116668852,
    -0.010597401785069032,
)
_DB4_REC_HI = (
    -0.010597401785069032,
    -0.0328830116668852,
    0.030841381835560764,
    0.18703481171888114,
    -0.027983769416859854,
    -0.6308807679298589,
    0.7148465705529157,
    -0.2303778133088965,
)


class WNOBlock(nn.Module):
    """A single-level db4 wavelet residual block."""

    def __init__(self, channels: int = DEFAULT_HIDDEN_DIM, decomp_level: int = 3) -> None:
        super().__init__()

        if channels <= 0:
            msg = f"channels must be positive, got {channels}"
            raise ValueError(msg)

        if decomp_level <= 0:
            msg = f"decomp_level must be positive, got {decomp_level}"
            raise ValueError(msg)

        self.channels = channels
        self.decomp_level = decomp_level
        self.analysis_pad = 3
        self.skip = nn.Identity()
        self.approx_block = ResConv1dBlock(channels, kernel_size=1)
        self.detail_block = ResConv1dBlock(channels, kernel_size=1)
        self.output_norm = LayerNorm1d(channels)

        self.register_buffer("dec_lo", torch.tensor(_DB4_DEC_LO, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("dec_hi", torch.tensor(_DB4_DEC_HI, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("rec_lo", torch.tensor(_DB4_REC_LO, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("rec_hi", torch.tensor(_DB4_REC_HI, dtype=torch.float32).view(1, 1, -1))

    def _expand_filter(self, kernel: Tensor) -> Tensor:
        return kernel.expand(self.channels, 1, -1)

    def _dwt(self, x: Tensor) -> tuple[Tensor, Tensor, int]:
        original_length = x.shape[-1]
        if original_length % 2 != 0:
            x = F.pad(x, (0, 1))

        dec_lo = self._expand_filter(self.dec_lo.to(dtype=x.dtype, device=x.device))
        dec_hi = self._expand_filter(self.dec_hi.to(dtype=x.dtype, device=x.device))
        approx = F.conv1d(x, dec_lo, stride=2, padding=self.analysis_pad, groups=self.channels)
        detail = F.conv1d(x, dec_hi, stride=2, padding=self.analysis_pad, groups=self.channels)
        return approx, detail, original_length

    def _idwt(self, approx: Tensor, detail: Tensor, output_length: int) -> Tensor:
        rec_lo = self._expand_filter(self.rec_lo.to(dtype=approx.dtype, device=approx.device))
        rec_hi = self._expand_filter(self.rec_hi.to(dtype=detail.dtype, device=detail.device))

        y_lo = F.conv_transpose1d(
            approx,
            rec_lo,
            stride=2,
            padding=self.analysis_pad,
            groups=self.channels,
        )
        y_hi = F.conv_transpose1d(
            detail,
            rec_hi,
            stride=2,
            padding=self.analysis_pad,
            groups=self.channels,
        )
        y = y_lo + y_hi
        return y[..., :output_length]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, C, L], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.channels:
            msg = f"Expected {self.channels} channels, got {x.shape[1]}"
            raise ValueError(msg)

        skip = self.skip(x)
        approx = x
        details: list[Tensor] = []
        output_lengths: list[int] = []

        for _ in range(self.decomp_level):
            approx, detail, output_length = self._dwt(approx)
            details.append(self.detail_block(detail))
            output_lengths.append(output_length)

        approx = self.approx_block(approx)

        y = approx
        for detail, output_length in zip(reversed(details), reversed(output_lengths), strict=False):
            y = self._idwt(y, detail, output_length)

        return self.output_norm(y + skip)


class WNOEncoder(nn.Module):
    """Encode [B, C, L] into [B, lmf_dim, L] with stacked WNO blocks."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
        decomp_level: int = 3,
        num_blocks: int = DEFAULT_WNO_NUM_BLOCKS,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            msg = f"in_channels must be positive, got {in_channels}"
            raise ValueError(msg)

        if hidden_dim <= 0:
            msg = f"hidden_dim must be positive, got {hidden_dim}"
            raise ValueError(msg)

        if lmf_dim <= 0:
            msg = f"lmf_dim must be positive, got {lmf_dim}"
            raise ValueError(msg)

        if decomp_level <= 0:
            msg = f"decomp_level must be positive, got {decomp_level}"
            raise ValueError(msg)
        if num_blocks <= 0:
            msg = f"num_blocks must be positive, got {num_blocks}"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lmf_dim = lmf_dim
        self.decomp_level = decomp_level
        self.num_blocks = num_blocks
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, stride=1)
        self.input_norm = LayerNorm1d(hidden_dim)
        self.input_act = nn.GELU()
        self.blocks = nn.Sequential(*[WNOBlock(hidden_dim, decomp_level=decomp_level) for _ in range(num_blocks)])
        self.output_proj = nn.Conv1d(hidden_dim, lmf_dim, kernel_size=1, stride=1)
        self.output_norm = LayerNorm1d(lmf_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            msg = f"Expected input shape [B, C, L], got {tuple(x.shape)}"
            raise ValueError(msg)

        if x.shape[1] != self.in_channels:
            msg = f"Expected {self.in_channels} channels, got {x.shape[1]}"
            raise ValueError(msg)

        x = self.input_proj(x)
        x = self.input_act(self.input_norm(x))
        x = self.blocks(x)
        x = self.output_proj(x)
        return self.output_norm(x)


__all__ = ["WNOBlock", "WNOEncoder"]
