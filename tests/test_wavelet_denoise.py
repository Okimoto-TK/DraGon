from __future__ import annotations

import pytest
import src.models.arch.layers.wavelet_denoise as wd
import torch


class _FakePtwt:
    @staticmethod
    def wavedec(
        x: torch.Tensor,
        wavelet: str,
        mode: str,
        level: int,
        axis: int,
    ) -> list[torch.Tensor]:
        del wavelet, mode, axis
        return [x] + [torch.zeros_like(x) for _ in range(level)]

    @staticmethod
    def waverec(
        coeffs: list[torch.Tensor],
        wavelet: str,
        axis: int,
    ) -> torch.Tensor:
        del wavelet, axis
        return coeffs[0]


@pytest.fixture
def patch_ptwt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wd, "_require_ptwt", lambda: _FakePtwt)


def test_wavelet_denoise_macro_shape_and_crop(patch_ptwt: None) -> None:
    module = wd.WaveletDenoise1D(n_channels=5, target_len=64, warmup_len=48)
    x = torch.randn(2, 5, 112)
    y = module(x)

    assert y.shape == (2, 5, 64)
    assert torch.allclose(y, x[..., -64:])


def test_wavelet_denoise_mezzo_shape(patch_ptwt: None) -> None:
    module = wd.WaveletDenoise1D(n_channels=7, target_len=96, warmup_len=48)
    x = torch.randn(2, 7, 144)
    y = module(x)

    assert y.shape == (2, 7, 96)


def test_wavelet_denoise_dtype_and_device_consistent(patch_ptwt: None) -> None:
    module = wd.WaveletDenoise1D(n_channels=4, target_len=16, warmup_len=8).to(
        dtype=torch.bfloat16
    )
    x = torch.randn(2, 4, 24, dtype=torch.bfloat16)
    y = module(x)

    assert y.dtype == torch.bfloat16
    assert y.device == x.device


def test_wavelet_denoise_uses_explicit_bf16(patch_ptwt: None) -> None:
    module = wd.WaveletDenoise1D(n_channels=4, target_len=16, warmup_len=8).to(
        dtype=torch.bfloat16
    )
    x = torch.randn(2, 4, 24, dtype=torch.bfloat16)
    y = module(x)

    assert y.dtype == torch.bfloat16


def test_wavelet_denoise_three_scales_forward(patch_ptwt: None) -> None:
    denoise_macro = wd.WaveletDenoise1D(
        n_channels=3,
        target_len=64,
        warmup_len=48,
        wavelet="db4",
        level=2,
    )
    denoise_mezzo = wd.WaveletDenoise1D(
        n_channels=5,
        target_len=96,
        warmup_len=48,
        wavelet="db4",
        level=2,
    )
    denoise_micro = wd.WaveletDenoise1D(
        n_channels=7,
        target_len=144,
        warmup_len=48,
        wavelet="db4",
        level=2,
    )

    y_macro = denoise_macro(torch.randn(2, 3, 112))
    y_mezzo = denoise_mezzo(torch.randn(2, 5, 144))
    y_micro = denoise_micro(torch.randn(2, 7, 192))

    assert y_macro.shape == (2, 3, 64)
    assert y_mezzo.shape == (2, 5, 96)
    assert y_micro.shape == (2, 7, 144)


def test_wavelet_denoise_invalid_input_length_raises_value_error(
    patch_ptwt: None,
) -> None:
    module = wd.WaveletDenoise1D(n_channels=5, target_len=64, warmup_len=48)
    x_bad = torch.randn(2, 5, 111)

    with pytest.raises(ValueError, match="shape mismatch"):
        _ = module(x_bad)
