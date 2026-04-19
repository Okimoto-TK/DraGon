from __future__ import annotations

import pytest
import torch
from src.models.arch.fusions import ExogenousBridgeFusion


def test_exogenous_bridge_fusion_macro_shape() -> None:
    endo = torch.randn(2, 128, 16)
    exo = torch.randn(2, 32, 64)
    exo_g = torch.randn(2, 32)

    fusion = ExogenousBridgeFusion()
    endo_fused, bridge_global = fusion(endo, exo, exo_g)

    assert endo_fused.shape == (2, 128, 16)
    assert bridge_global.shape == (2, 128)


def test_exogenous_bridge_fusion_mezzo_shape_variable_exogenous_length() -> None:
    endo = torch.randn(2, 128, 24)
    exo = torch.randn(2, 32, 12)
    exo_g = torch.randn(2, 32)

    fusion = ExogenousBridgeFusion()
    endo_fused, bridge_global = fusion(endo, exo, exo_g)

    assert endo_fused.shape == (2, 128, 24)
    assert bridge_global.shape == (2, 128)


def test_exogenous_bridge_fusion_micro_shape_short_exogenous_length() -> None:
    endo = torch.randn(2, 128, 36)
    exo = torch.randn(2, 32, 3)
    exo_g = torch.randn(2, 32)

    fusion = ExogenousBridgeFusion()
    endo_fused, bridge_global = fusion(endo, exo, exo_g)

    assert endo_fused.shape == (2, 128, 36)
    assert bridge_global.shape == (2, 128)


def test_exogenous_bridge_fusion_dtype_and_device_consistency() -> None:
    endo = torch.randn(2, 128, 16, dtype=torch.float64)
    exo = torch.randn(2, 32, 8, dtype=torch.float64)
    exo_g = torch.randn(2, 32, dtype=torch.float64)

    fusion = ExogenousBridgeFusion().to(dtype=endo.dtype, device=endo.device)
    endo_fused, bridge_global = fusion(endo, exo, exo_g)

    assert endo_fused.dtype == endo.dtype
    assert bridge_global.dtype == endo.dtype
    assert endo_fused.device == endo.device
    assert bridge_global.device == endo.device


def test_exogenous_bridge_fusion_invalid_hidden_dim_raises_value_error() -> None:
    endo = torch.randn(2, 127, 16)
    exo = torch.randn(2, 32, 8)
    exo_g = torch.randn(2, 32)
    fusion = ExogenousBridgeFusion(hidden_dim=128, exogenous_dim=32)

    with pytest.raises(ValueError, match="endogenous_seq hidden_dim mismatch"):
        _ = fusion(endo, exo, exo_g)


def test_exogenous_bridge_fusion_invalid_exogenous_dim_raises_value_error() -> None:
    endo = torch.randn(2, 128, 16)
    fusion = ExogenousBridgeFusion(hidden_dim=128, exogenous_dim=32)

    exo_bad = torch.randn(2, 31, 8)
    exo_g = torch.randn(2, 32)
    with pytest.raises(ValueError, match="exogenous_seq exogenous_dim mismatch"):
        _ = fusion(endo, exo_bad, exo_g)

    exo = torch.randn(2, 32, 8)
    exo_g_bad = torch.randn(2, 31)
    with pytest.raises(ValueError, match="exogenous_global exogenous_dim mismatch"):
        _ = fusion(endo, exo, exo_g_bad)


def test_exogenous_bridge_fusion_batch_mismatch_raises_value_error() -> None:
    endo = torch.randn(2, 128, 16)
    exo = torch.randn(3, 32, 8)
    exo_g = torch.randn(2, 32)
    fusion = ExogenousBridgeFusion()

    with pytest.raises(ValueError, match="Batch size mismatch"):
        _ = fusion(endo, exo, exo_g)


def test_exogenous_bridge_fusion_train_and_eval_forward() -> None:
    endo = torch.randn(2, 128, 16)
    exo = torch.randn(2, 32, 8)
    exo_g = torch.randn(2, 32)
    fusion = ExogenousBridgeFusion()

    fusion.train()
    endo_fused_train, bridge_global_train = fusion(endo, exo, exo_g)
    assert endo_fused_train.shape == (2, 128, 16)
    assert bridge_global_train.shape == (2, 128)

    fusion.eval()
    endo_fused_eval, bridge_global_eval = fusion(endo, exo, exo_g)
    assert endo_fused_eval.shape == (2, 128, 16)
    assert bridge_global_eval.shape == (2, 128)
