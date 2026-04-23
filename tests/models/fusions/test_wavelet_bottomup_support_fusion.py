from __future__ import annotations

import pytest
import torch

from src.models.arch.fusions import WaveletBottomUpSupportFusion


def _support(intervals: list[tuple[float, float]]) -> torch.Tensor:
    return torch.tensor(intervals, dtype=torch.float32)


def test_wavelet_bottomup_support_fusion_smoke_shapes() -> None:
    fusion = WaveletBottomUpSupportFusion(hidden_dim=16, num_layers=1)
    macro = torch.randn(2, 16, 4)
    mezzo = torch.randn(2, 16, 3)
    micro = torch.randn(2, 16, 2)

    macro_out, mezzo_out, micro_out = fusion(
        macro,
        mezzo,
        micro,
        macro_support=_support([(-4.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0)]),
        mezzo_support=_support([(-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.0)]),
        micro_support=_support([(-0.75, -0.25), (-0.25, 0.0)]),
    )

    assert macro_out.shape == macro.shape
    assert mezzo_out.shape == mezzo.shape
    assert micro_out.shape == micro.shape


def test_wavelet_bottomup_support_fusion_zero_overlap_rows_stay_finite() -> None:
    fusion = WaveletBottomUpSupportFusion(hidden_dim=16, num_layers=1)
    macro = torch.randn(2, 16, 2)
    mezzo = torch.randn(2, 16, 2)
    micro = torch.randn(2, 16, 2)

    macro_out, mezzo_out, micro_out = fusion(
        macro,
        mezzo,
        micro,
        macro_support=_support([(-10.0, -9.0), (-9.0, -8.0)]),
        mezzo_support=_support([(-2.0, -1.0), (-1.0, 0.0)]),
        micro_support=_support([(-0.5, -0.25), (-0.25, 0.0)]),
    )

    assert torch.isfinite(macro_out).all()
    assert torch.isfinite(mezzo_out).all()
    assert torch.isfinite(micro_out).all()


def test_wavelet_bottomup_support_fusion_invalid_support_shape_raises() -> None:
    fusion = WaveletBottomUpSupportFusion(hidden_dim=16, num_layers=1)
    macro = torch.randn(2, 16, 2)
    mezzo = torch.randn(2, 16, 2)
    micro = torch.randn(2, 16, 2)

    with pytest.raises(ValueError, match="macro_support must have shape \\[N, 2\\]"):
        _ = fusion(
            macro,
            mezzo,
            micro,
            macro_support=torch.randn(2, 3),
            mezzo_support=_support([(-2.0, -1.0), (-1.0, 0.0)]),
            micro_support=_support([(-0.5, -0.25), (-0.25, 0.0)]),
        )
