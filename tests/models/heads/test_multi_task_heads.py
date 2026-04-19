from __future__ import annotations

import pytest
import torch

from src.models.arch.heads import MultiTaskHeads


def test_multi_task_heads_forward() -> None:
    fused_latents = torch.randn(2, 128, 8)
    fused_global = torch.randn(2, 128)

    model = MultiTaskHeads()
    out = model(fused_latents, fused_global)

    assert out["pred_mu_ret"].shape == (2, 1)
    assert out["pred_scale_ret_raw"].shape == (2, 1)
    assert out["pred_mean_rv_raw"].shape == (2, 1)
    assert out["pred_shape_rv_raw"].shape == (2, 1)
    assert out["pred_mu_q"].shape == (2, 1)
    assert out["pred_scale_q_raw"].shape == (2, 1)


def test_multi_task_heads_task_tokens_are_not_shared() -> None:
    model = MultiTaskHeads()

    assert model.ret_tower.task_token is not model.rv_tower.task_token
    assert model.ret_tower.task_token is not model.q_tower.task_token
    assert model.rv_tower.task_token is not model.q_tower.task_token


def test_multi_task_heads_value_heads_do_not_share_parameters() -> None:
    model = MultiTaskHeads()

    ret_params = list(model.ret_value_head.parameters())
    rv_params = list(model.rv_value_head.parameters())
    q_params = list(model.q_value_head.parameters())

    assert ret_params[0] is not rv_params[0]
    assert ret_params[0] is not q_params[0]
    assert rv_params[0] is not q_params[0]


def test_multi_task_heads_dtype_and_device_consistency() -> None:
    fused_latents = torch.randn(2, 128, 8, dtype=torch.float64)
    fused_global = torch.randn(2, 128, dtype=torch.float64)

    model = MultiTaskHeads().to(dtype=fused_latents.dtype, device=fused_latents.device)
    out = model(fused_latents, fused_global)

    for value in out.values():
        assert value.dtype == fused_latents.dtype
        assert value.device == fused_latents.device


def test_multi_task_heads_hidden_dim_mismatch_raises_value_error() -> None:
    fused_latents = torch.randn(2, 64, 8)
    fused_global = torch.randn(2, 128)
    model = MultiTaskHeads()

    with pytest.raises(ValueError, match="fused_latents hidden_dim mismatch"):
        _ = model(fused_latents, fused_global)


def test_multi_task_heads_num_latents_mismatch_raises_value_error() -> None:
    fused_latents = torch.randn(2, 128, 7)
    fused_global = torch.randn(2, 128)
    model = MultiTaskHeads()

    with pytest.raises(ValueError, match="fused_latents num_latents mismatch"):
        _ = model(fused_latents, fused_global)


def test_multi_task_heads_batch_mismatch_raises_value_error() -> None:
    fused_latents = torch.randn(2, 128, 8)
    fused_global = torch.randn(3, 128)
    model = MultiTaskHeads()

    with pytest.raises(ValueError, match="Batch size mismatch"):
        _ = model(fused_latents, fused_global)


def test_multi_task_heads_gradients_flow() -> None:
    fused_latents = torch.randn(2, 128, 8, requires_grad=True)
    fused_global = torch.randn(2, 128)
    model = MultiTaskHeads()

    out = model(fused_latents, fused_global)
    loss = sum(value.sum() for value in out.values())
    loss.backward()

    heads = [
        model.ret_value_head,
        model.ret_uncertainty_head,
        model.rv_value_head,
        model.rv_uncertainty_head,
        model.q_value_head,
        model.q_uncertainty_head,
    ]
    for head in heads:
        assert any(param.grad is not None for param in head.parameters())

    assert model.ret_tower.task_token.grad is not None
    assert model.rv_tower.task_token.grad is not None
    assert model.q_tower.task_token.grad is not None
