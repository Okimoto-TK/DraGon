"""Model configuration exposed at root config package."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.config.hparams import (
    CROSS_SCALE_FUSION_HPARAMS,
    EXOGENOUS_BRIDGE_FUSION_HPARAMS,
    MULTI_SCALE_FORECAST_NETWORK_HPARAMS,
    MULTI_TASK_HEADS_HPARAMS,
    MULTI_TASK_LOSS_HPARAMS,
    MODERN_TCN_FILM_ENCODER_HPARAMS,
    WITHIN_SCALE_STAR_FUSION_HPARAMS,
)

modern_tcn_film_encoder = MODERN_TCN_FILM_ENCODER_HPARAMS


@dataclass(frozen=True)
class ConditioningEncoderConfig:
    """Open tuning parameters for ConditioningEncoder."""

    d_cond: int = 32
    input_features: int = 13
    num_blocks: int = 1
    dropout: float = 0.0


conditioning_encoder = ConditioningEncoderConfig()


@dataclass(frozen=True)
class WithinScaleSTARFusionConfig:
    """Open tuning parameters for WithinScaleSTARFusion."""

    hidden_dim: int = 128
    num_features: int = 9
    core_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


within_scale_star_fusion = WithinScaleSTARFusionConfig()
within_scale_star_fusion_hparams = WITHIN_SCALE_STAR_FUSION_HPARAMS


@dataclass(frozen=True)
class ExogenousBridgeFusionConfig:
    """Open tuning parameters for ExogenousBridgeFusion."""

    hidden_dim: int = 128
    exogenous_dim: int = 32
    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


exogenous_bridge_fusion = ExogenousBridgeFusionConfig()
exogenous_bridge_fusion_hparams = EXOGENOUS_BRIDGE_FUSION_HPARAMS


@dataclass(frozen=True)
class CrossScaleFusionConfig:
    """Open tuning parameters for CrossScaleFusion."""

    hidden_dim: int = 128
    num_latents: int = 8
    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


cross_scale_fusion = CrossScaleFusionConfig()
cross_scale_fusion_hparams = CROSS_SCALE_FUSION_HPARAMS


@dataclass(frozen=True)
class MultiTaskHeadsConfig:
    """Open tuning parameters for MultiTaskHeads."""

    hidden_dim: int = 128
    num_latents: int = 8
    tower_num_heads: int = 4
    tower_ffn_ratio: float = 2.0
    tower_dropout: float = 0.0


multi_task_heads = MultiTaskHeadsConfig()
multi_task_heads_hparams = MULTI_TASK_HEADS_HPARAMS


@dataclass(frozen=True)
class MultiTaskLossConfig:
    """Open tuning parameters for MultiTaskDistributionLoss."""

    ret_loss_weight: float = 1.0
    rv_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    q_tau: float = 0.05


multi_task_loss = MultiTaskLossConfig()
multi_task_loss_hparams = MULTI_TASK_LOSS_HPARAMS


@dataclass(frozen=True)
class MultiScaleForecastNetworkConfig:
    """Open tuning parameters for MultiScaleForecastNetwork."""

    hidden_dim: int = 128
    cond_dim: int = 32
    num_latents: int = 8


multi_scale_forecast_network = MultiScaleForecastNetworkConfig()
multi_scale_forecast_network_hparams = MULTI_SCALE_FORECAST_NETWORK_HPARAMS
