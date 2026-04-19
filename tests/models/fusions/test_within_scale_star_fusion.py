from __future__ import annotations

import pytest
import torch
from src.models.arch.fusions import WithinScaleSTARFusion


def test_within_scale_star_fusion_macro_shape() -> None:
    x = torch.randn(2, 9, 128, 16)
    fusion = WithinScaleSTARFusion()
    z_fused, scale_seq = fusion(x)

    assert z_fused.shape == (2, 9, 128, 16)
    assert scale_seq.shape == (2, 128, 16)


def test_within_scale_star_fusion_mezzo_shape() -> None:
    x = torch.randn(2, 9, 128, 24)
    fusion = WithinScaleSTARFusion()
    z_fused, scale_seq = fusion(x)

    assert z_fused.shape == (2, 9, 128, 24)
    assert scale_seq.shape == (2, 128, 24)


def test_within_scale_star_fusion_micro_shape() -> None:
    x = torch.randn(2, 9, 128, 36)
    fusion = WithinScaleSTARFusion()
    z_fused, scale_seq = fusion(x)

    assert z_fused.shape == (2, 9, 128, 36)
    assert scale_seq.shape == (2, 128, 36)


def test_within_scale_star_fusion_dtype_and_device_consistency() -> None:
    x = torch.randn(2, 9, 128, 16, dtype=torch.float64)
    fusion = WithinScaleSTARFusion().to(dtype=x.dtype, device=x.device)
    z_fused, scale_seq = fusion(x)

    assert z_fused.dtype == x.dtype
    assert scale_seq.dtype == x.dtype
    assert z_fused.device == x.device
    assert scale_seq.device == x.device


def test_within_scale_star_fusion_invalid_feature_count_raises_value_error() -> None:
    x = torch.randn(2, 8, 128, 16)
    fusion = WithinScaleSTARFusion()
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        _ = fusion(x)


def test_within_scale_star_fusion_invalid_hidden_dim_raises_value_error() -> None:
    x = torch.randn(2, 9, 127, 16)
    fusion = WithinScaleSTARFusion()
    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        _ = fusion(x)


def test_within_scale_star_fusion_train_and_eval_forward() -> None:
    x = torch.randn(2, 9, 128, 16)
    fusion = WithinScaleSTARFusion()

    fusion.train()
    z_fused_train, scale_seq_train = fusion(x)
    assert z_fused_train.shape == (2, 9, 128, 16)
    assert scale_seq_train.shape == (2, 128, 16)

    fusion.eval()
    z_fused_eval, scale_seq_eval = fusion(x)
    assert z_fused_eval.shape == (2, 9, 128, 16)
    assert scale_seq_eval.shape == (2, 128, 16)


def test_within_scale_star_fusion_patch_dim_is_preserved() -> None:
    x = torch.randn(2, 9, 128, 24)
    fusion = WithinScaleSTARFusion()
    z_fused, scale_seq = fusion(x)

    assert z_fused.shape[-1] == 24
    assert scale_seq.shape[-1] == 24


def test_within_scale_star_fusion_score_projection_receives_gradients() -> None:
    x = torch.randn(2, 9, 128, 16, requires_grad=True)
    fusion = WithinScaleSTARFusion()

    z_fused, scale_seq = fusion(x)
    (z_fused.sum() + scale_seq.sum()).backward()

    first_block = fusion.blocks[0]
    assert first_block.score_proj.weight.grad is not None
    assert first_block.score_proj.bias.grad is not None
    assert torch.count_nonzero(first_block.score_proj.weight.grad) > 0
    assert torch.count_nonzero(first_block.score_proj.bias.grad) > 0


def test_within_scale_star_fusion_dtype_mismatch_raises_without_mutating_parameters() -> None:
    x = torch.randn(2, 9, 128, 16, dtype=torch.float64)
    fusion = WithinScaleSTARFusion()
    before_dtype = next(fusion.parameters()).dtype

    with pytest.raises(ValueError, match="z_scale dtype mismatch"):
        _ = fusion(x)

    assert next(fusion.parameters()).dtype == before_dtype
