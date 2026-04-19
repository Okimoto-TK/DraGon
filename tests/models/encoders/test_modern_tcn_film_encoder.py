from __future__ import annotations

import pytest
import torch
from src.models.arch.embeddings import ConditionEmbedding1D
from src.models.arch.encoders.modern_tcn_film_encoder import ModernTCNFiLMEncoder
from src.models.arch.layers import Patch1D


def _num_patches(seq_len: int, patch_len: int = 8, patch_stride: int = 4) -> int:
    pad_right = patch_len - patch_stride
    return ((seq_len + pad_right - patch_len) // patch_stride) + 1


def test_modern_tcn_film_encoder_macro_forward_shape() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=64, pos_vocab_size=8)
    x_float = torch.randn(2, 9, 64)
    x_state = torch.randint(0, 16, (2, 64))
    x_pos = torch.randint(0, 8, (2, 64))

    y = encoder(x_float, x_state, x_pos)
    assert y.shape == (2, 9, 128, _num_patches(64))


def test_modern_tcn_film_encoder_mezzo_forward_shape() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=96, pos_vocab_size=16)
    x_float = torch.randn(2, 9, 96)
    x_state = torch.randint(0, 16, (2, 96))
    x_pos = torch.randint(0, 16, (2, 96))

    y = encoder(x_float, x_state, x_pos)
    assert y.shape == (2, 9, 128, _num_patches(96))


def test_modern_tcn_film_encoder_micro_forward_shape() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=144, pos_vocab_size=64)
    x_float = torch.randn(2, 9, 144)
    x_state = torch.randint(0, 16, (2, 144))
    x_pos = torch.randint(0, 64, (2, 144))

    y = encoder(x_float, x_state, x_pos)
    assert y.shape == (2, 9, 128, _num_patches(144))


def test_modern_tcn_film_encoder_dtype_and_device_consistency() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=64, pos_vocab_size=8)
    x_float = torch.randn(2, 9, 64, dtype=torch.float64)
    x_state = torch.randint(0, 16, (2, 64))
    x_pos = torch.randint(0, 8, (2, 64))
    encoder = encoder.to(dtype=x_float.dtype, device=x_float.device)

    y = encoder(x_float, x_state, x_pos)
    assert y.dtype == x_float.dtype
    assert y.device == x_float.device


def test_modern_tcn_film_encoder_invalid_input_length_raises_value_error() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=64, pos_vocab_size=8)
    x_float = torch.randn(2, 9, 63)
    x_state = torch.randint(0, 16, (2, 63))
    x_pos = torch.randint(0, 8, (2, 63))

    with pytest.raises(ValueError, match="sequence length mismatch"):
        _ = encoder(x_float, x_state, x_pos)


def test_modern_tcn_film_encoder_feature_mismatch_raises_value_error() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=64, pos_vocab_size=8)
    x_float = torch.randn(2, 8, 64)
    x_state = torch.randint(0, 16, (2, 64))
    x_pos = torch.randint(0, 8, (2, 64))

    with pytest.raises(ValueError, match="feature dimension mismatch"):
        _ = encoder(x_float, x_state, x_pos)


def test_condition_alignment_patch_count_matches_patch1d() -> None:
    batch_size = 2
    num_features = 9
    seq_len = 96
    patch = Patch1D(patch_len=8, patch_stride=4, hidden_dim=128)
    cond_embed = ConditionEmbedding1D(
        state_vocab_size=16,
        pos_vocab_size=16,
        cond_dim=64,
        patch_len=8,
        patch_stride=4,
        num_features=num_features,
    )

    x_float = torch.randn(batch_size * num_features, 1, seq_len)
    x_state = torch.randint(0, 16, (batch_size, seq_len))
    x_pos = torch.randint(0, 16, (batch_size, seq_len))

    z = patch(x_float)
    c = cond_embed(x_state, x_pos)

    assert z.shape[-1] == c.shape[-1]


def test_modern_tcn_film_encoder_dtype_mismatch_raises_without_mutating_parameters() -> None:
    encoder = ModernTCNFiLMEncoder(seq_len=64, pos_vocab_size=8)
    before_dtype = next(encoder.parameters()).dtype
    x_float = torch.randn(2, 9, 64, dtype=torch.float64)
    x_state = torch.randint(0, 16, (2, 64))
    x_pos = torch.randint(0, 8, (2, 64))

    with pytest.raises(ValueError, match="x_float dtype mismatch"):
        _ = encoder(x_float, x_state, x_pos)

    assert next(encoder.parameters()).dtype == before_dtype
