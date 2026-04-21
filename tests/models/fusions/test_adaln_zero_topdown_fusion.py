from __future__ import annotations

import pytest
import torch

from src.models.arch.fusions import AdaLNZeroTopDownFusion
from src.models.arch.fusions.adaln_zero_topdown_fusion import AdaLNZeroTopDownBlock


def test_adaln_zero_topdown_fusion_shapes() -> None:
    fusion = AdaLNZeroTopDownFusion()
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)

    micro_td, macro_ctx, mezzo_ctx, micro_ctx = fusion(macro, mezzo, micro)

    assert micro_td.shape == (2, 128, 36)
    assert macro_ctx.shape == (2, 128)
    assert mezzo_ctx.shape == (2, 128)
    assert micro_ctx.shape == (2, 128)


def test_adaln_zero_topdown_fusion_debug_outputs_present() -> None:
    fusion = AdaLNZeroTopDownFusion()
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 36)

    _, _, _, _, debug = fusion(macro, mezzo, micro, return_debug=True)

    assert "cross_scale_macro_ctx_l2_mean" in debug
    assert "cross_scale_mezzo_ctx_l2_mean" in debug
    assert "cross_scale_micro_ctx_l2_mean" in debug
    assert "cross_scale_macro_to_mezzo_gate_mean" in debug
    assert "cross_scale_mezzo_to_micro_delta_l2_mean" in debug


def test_adaln_zero_topdown_fusion_invalid_micro_length_raises_value_error() -> None:
    fusion = AdaLNZeroTopDownFusion()
    macro = torch.randn(2, 128, 16)
    mezzo = torch.randn(2, 128, 24)
    micro = torch.randn(2, 128, 35)

    with pytest.raises(ValueError, match="micro_seq length mismatch"):
        fusion(macro, mezzo, micro)


def test_adaln_zero_topdown_block_uses_dit_style_zero_gate_semantics() -> None:
    torch.manual_seed(0)
    block = AdaLNZeroTopDownBlock(hidden_dim=8, ffn_ratio=2.0)
    child = torch.randn(2, 8, 5)
    parent = torch.randn(2, 8)

    initial_out, initial_debug = block(child, parent, return_debug=True)
    assert torch.allclose(initial_out, child)
    assert initial_debug["gate_abs_mean"] == pytest.approx(0.0)

    loss = initial_out.square().mean()
    loss.backward()
    assert block.modulation.weight.grad is not None
    assert block.modulation.weight.grad.abs().sum().item() > 0.0
    assert block.out_proj.weight.grad is not None
    assert block.out_proj.weight.grad.abs().sum().item() == pytest.approx(0.0)

    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    stepped_out, stepped_debug = block(child, parent, return_debug=True)
    assert stepped_debug["gate_abs_mean"] > 0.0
    assert not torch.allclose(stepped_out, child)
