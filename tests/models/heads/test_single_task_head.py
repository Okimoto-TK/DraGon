from __future__ import annotations

import pytest
import torch

from src.models.arch.heads import SingleTaskHead


@pytest.mark.parametrize(
    ("task", "expected_keys"),
    [
        ("ret", {"pred_mu_ret", "pred_scale_ret_raw"}),
        ("rv", {"pred_mean_rv_raw", "pred_shape_rv_raw"}),
        ("q", {"pred_mu_q", "pred_scale_q_raw"}),
    ],
)
def test_single_task_head_forward(task: str, expected_keys: set[str]) -> None:
    head = SingleTaskHead(task=task)
    out = head(
        micro_td=torch.randn(2, 128, 36),
        mezzo_ctx=torch.randn(2, 128),
        macro_ctx=torch.randn(2, 128),
    )

    assert expected_keys.issubset(set(out.keys()))
    assert out["pred_primary"].shape == (2, 1)
    assert out["pred_aux_raw"].shape == (2, 1)
    assert out["head_context"].shape == (2, 128)


def test_single_task_head_debug_contains_query_outputs() -> None:
    head = SingleTaskHead(task="ret")
    out = head(
        micro_td=torch.randn(2, 128, 36),
        mezzo_ctx=torch.randn(2, 128),
        macro_ctx=torch.randn(2, 128),
        return_debug=True,
    )

    assert "_debug" in out
    assert "task_repr" in out["_debug"]
    assert "task_attn_weights" in out["_debug"]


def test_single_task_head_invalid_task_raises_value_error() -> None:
    with pytest.raises(ValueError, match="task must be one of"):
        _ = SingleTaskHead(task="foo")
