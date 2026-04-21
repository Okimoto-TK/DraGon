from __future__ import annotations

import torch

from src.models.arch.layers import FeatureChannelDropout1D


def test_feature_channel_dropout_eval_is_identity() -> None:
    layer = FeatureChannelDropout1D(num_channels=4, p=0.5, special_channel_ps={2: 0.75})
    layer.eval()
    x = torch.randn(3, 4, 8)

    out = layer(x)

    assert torch.allclose(out, x)


def test_feature_channel_dropout_drops_whole_channel_over_time() -> None:
    torch.manual_seed(0)
    layer = FeatureChannelDropout1D(num_channels=3, p=0.0, special_channel_ps={1: 0.999})
    layer.train()
    x = torch.ones(2, 3, 5)

    out = layer(x)

    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.all(out[:, 1] == 0)
    assert torch.allclose(out[:, 2], x[:, 2])


def test_feature_channel_dropout_preserves_expected_scale() -> None:
    torch.manual_seed(0)
    layer = FeatureChannelDropout1D(num_channels=1, p=0.5)
    layer.train()
    x = torch.ones(4096, 1, 4)

    out = layer(x)

    mean_value = float(out.mean())
    assert abs(mean_value - 1.0) < 0.08
