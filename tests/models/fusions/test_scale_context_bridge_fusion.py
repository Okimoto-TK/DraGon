from __future__ import annotations

import pytest
import torch

from src.models.arch.fusions import ScaleContextBridgeFusion


def test_scale_context_bridge_fusion_shapes() -> None:
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)

    fusion = ScaleContextBridgeFusion()
    micro_td, macro_ctx, mezzo_ctx, micro_ctx = fusion(macro, mezzo, micro)

    assert micro_td.shape == (2, 128, 36)
    assert macro_ctx.shape == (2, 128)
    assert mezzo_ctx.shape == (2, 128)
    assert micro_ctx.shape == (2, 128)


def test_scale_context_bridge_fusion_debug_outputs_present() -> None:
    fusion = ScaleContextBridgeFusion()
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)

    _, _, _, _, debug = fusion(macro, mezzo, micro, return_debug=True)

    assert "macro_to_mezzo" in debug
    assert "mezzo_to_micro" in debug
    assert "macro_ctx" in debug


def test_scale_context_bridge_fusion_invalid_micro_length_raises_value_error() -> None:
    fusion = ScaleContextBridgeFusion()
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 35)

    with pytest.raises(ValueError, match="micro_seq length mismatch"):
        _ = fusion(macro, mezzo, micro)
