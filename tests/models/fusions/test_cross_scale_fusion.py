from __future__ import annotations

import pytest
import torch
from src.models.arch.fusions import CrossScaleFusion


def test_cross_scale_fusion_shape() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)

    fusion = CrossScaleFusion()
    fused_latents, fused_global = fusion(macro, mezzo, micro)

    assert fused_latents.shape == (2, 128, 8)
    assert fused_global.shape == (2, 128)


def test_cross_scale_fusion_dtype_and_device_consistency() -> None:
    macro = torch.randn(2, 128, 16, dtype=torch.float64)
    mezzo = torch.randn(2, 128, 24, dtype=torch.float64)
    micro = torch.randn(2, 128, 36, dtype=torch.float64)

    fusion = CrossScaleFusion().to(dtype=macro.dtype, device=macro.device)
    fused_latents, fused_global = fusion(macro, mezzo, micro)

    assert fused_latents.dtype == macro.dtype
    assert fused_global.dtype == macro.dtype
    assert fused_latents.device == macro.device
    assert fused_global.device == macro.device


def test_cross_scale_fusion_invalid_macro_length_raises_value_error() -> None:
    macro = torch.randn(2, 128, 15)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)
    fusion = CrossScaleFusion()

    with pytest.raises(ValueError, match="macro_seq length mismatch"):
        _ = fusion(macro, mezzo, micro)


def test_cross_scale_fusion_invalid_mezzo_length_raises_value_error() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 25)
    micro = torch.randn(2, 128, 36)
    fusion = CrossScaleFusion()

    with pytest.raises(ValueError, match="mezzo_seq length mismatch"):
        _ = fusion(macro, mezzo, micro)


def test_cross_scale_fusion_invalid_micro_length_raises_value_error() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 35)
    fusion = CrossScaleFusion()

    with pytest.raises(ValueError, match="micro_seq length mismatch"):
        _ = fusion(macro, mezzo, micro)


def test_cross_scale_fusion_batch_mismatch_raises_value_error() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(3, 128, 24)
    micro = torch.randn(2, 128, 36)
    fusion = CrossScaleFusion()

    with pytest.raises(ValueError, match="Batch size mismatch"):
        _ = fusion(macro, mezzo, micro)


def test_cross_scale_fusion_train_and_eval_forward() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)
    fusion = CrossScaleFusion()

    fusion.train()
    fused_latents_train, fused_global_train = fusion(macro, mezzo, micro)
    assert fused_latents_train.shape == (2, 128, 8)
    assert fused_global_train.shape == (2, 128)

    fusion.eval()
    fused_latents_eval, fused_global_eval = fusion(macro, mezzo, micro)
    assert fused_latents_eval.shape == (2, 128, 8)
    assert fused_global_eval.shape == (2, 128)
