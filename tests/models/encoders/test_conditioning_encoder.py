from __future__ import annotations

import pytest
import torch
from src.models.arch.encoders import ConditioningEncoder


def test_conditioning_encoder_default_shape() -> None:
    x = torch.randn(2, 13, 64)
    encoder = ConditioningEncoder()
    cond_seq, cond_global = encoder(x)

    assert cond_seq.shape == (2, 32, 64)
    assert cond_global.shape == (2, 32)


def test_conditioning_encoder_dtype_and_device_consistency() -> None:
    x = torch.randn(2, 13, 64, dtype=torch.float64)
    encoder = ConditioningEncoder().to(dtype=x.dtype, device=x.device)
    cond_seq, cond_global = encoder(x)

    assert cond_seq.dtype == x.dtype
    assert cond_global.dtype == x.dtype
    assert cond_seq.device == x.device
    assert cond_global.device == x.device


@pytest.mark.parametrize(
    "shape",
    [
        (2, 13, 63),
        (2, 12, 64),
        (2, 13),
    ],
)
def test_conditioning_encoder_invalid_shape_raises_value_error(shape: tuple[int, ...]) -> None:
    x = torch.randn(*shape)
    encoder = ConditioningEncoder()
    with pytest.raises(ValueError):
        _ = encoder(x)


def test_conditioning_encoder_different_d_cond_runs() -> None:
    x = torch.randn(2, 13, 64)
    encoder = ConditioningEncoder(d_cond=64)
    cond_seq, cond_global = encoder(x)

    assert cond_seq.shape == (2, 64, 64)
    assert cond_global.shape == (2, 64)


def test_conditioning_encoder_different_num_blocks_runs() -> None:
    x = torch.randn(2, 13, 64)
    encoder = ConditioningEncoder(num_blocks=2)
    cond_seq, cond_global = encoder(x)

    assert cond_seq.shape == (2, 32, 64)
    assert cond_global.shape == (2, 32)


def test_conditioning_encoder_dtype_mismatch_raises_without_mutating_parameters() -> None:
    x = torch.randn(2, 13, 64, dtype=torch.float64)
    encoder = ConditioningEncoder()
    before_dtype = next(encoder.parameters()).dtype

    with pytest.raises(ValueError, match="x_cond dtype mismatch"):
        _ = encoder(x)

    assert next(encoder.parameters()).dtype == before_dtype
