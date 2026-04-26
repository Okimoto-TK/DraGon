"""Model configuration exposed at root config package."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.config.hparams import (
    CROSS_SCALE_FUSION_HPARAMS,
    DUAL_DOMAIN_CONCAT_HEAD_HPARAMS,
    DUAL_DOMAIN_MUTUAL_ATTENTION_HPARAMS,
    EXOGENOUS_BRIDGE_FUSION_HPARAMS,
    MULTI_SCALE_FORECAST_NETWORK_HPARAMS,
    MULTI_TASK_HEADS_HPARAMS,
    MULTI_TASK_LOSS_HPARAMS,
    MODERN_TCN_FILM_ENCODER_HPARAMS,
    TIME_TOPDOWN_FUSION_HPARAMS,
    WAVELET_BOTTOMUP_FUSION_HPARAMS,
    WAVELET_BRANCH_ENCODER_HPARAMS,
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
class WaveletDenoiseConfig:
    """Open tuning parameters for the wavelet front-end."""

    backprop: bool = True


wavelet_denoise = WaveletDenoiseConfig()


@dataclass(frozen=True)
class WithinScaleSTARFusionConfig:
    """Open tuning parameters for WithinScaleSTARFusion."""

    core_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


within_scale_star_fusion = WithinScaleSTARFusionConfig()
within_scale_star_fusion_hparams = WITHIN_SCALE_STAR_FUSION_HPARAMS


@dataclass(frozen=True)
class WaveletBranchEncoderConfig:
    """Open tuning parameters for the wavelet-domain branch encoder."""

    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


wavelet_branch_encoder = WaveletBranchEncoderConfig()
wavelet_branch_encoder_hparams = WAVELET_BRANCH_ENCODER_HPARAMS


@dataclass(frozen=True)
class DualDomainMutualAttentionConfig:
    """Open tuning parameters for scale-local mutual attention."""

    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


dual_domain_mutual_attention = DualDomainMutualAttentionConfig()
dual_domain_mutual_attention_hparams = DUAL_DOMAIN_MUTUAL_ATTENTION_HPARAMS


@dataclass(frozen=True)
class TimeTopDownFusionConfig:
    """Open tuning parameters for time-domain top-down hierarchical fusion."""

    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


time_topdown_fusion = TimeTopDownFusionConfig()
time_topdown_fusion_hparams = TIME_TOPDOWN_FUSION_HPARAMS


@dataclass(frozen=True)
class WaveletBottomUpFusionConfig:
    """Open tuning parameters for wavelet-domain support-aware fusion."""

    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


wavelet_bottomup_fusion = WaveletBottomUpFusionConfig()
wavelet_bottomup_fusion_hparams = WAVELET_BOTTOMUP_FUSION_HPARAMS


@dataclass(frozen=True)
class DualDomainConcatHeadConfig:
    """Open tuning parameters for the final dual-domain concat head."""

    num_heads: int = 4
    ffn_ratio: float = 2.0
    dropout: float = 0.0


dual_domain_concat_head = DualDomainConcatHeadConfig()
dual_domain_concat_head_hparams = DUAL_DOMAIN_CONCAT_HEAD_HPARAMS


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
class ScaleContextBridgeFusionConfig:
    """Open tuning parameters for macro -> mezzo -> micro context bridging."""

    hidden_dim: int = 128
    num_heads: int = 4
    ffn_ratio: float = 2.0
    num_layers: int = 1
    dropout: float = 0.0


scale_context_bridge_fusion = ScaleContextBridgeFusionConfig()


@dataclass(frozen=True)
class AdaLNZeroTopDownFusionConfig:
    """Open tuning parameters for AdaLN-Zero macro -> mezzo -> micro modulation."""

    hidden_dim: int = 128
    ffn_ratio: float = 2.0
    dropout: float = 0.0


adaln_zero_topdown_fusion = AdaLNZeroTopDownFusionConfig()


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
class SingleTaskHeadConfig:
    """Open tuning parameters for the selected task head."""

    hidden_dim: int = 128
    num_heads: int = 4
    ffn_ratio: float = 2.0
    dropout: float = 0.0


single_task_head = SingleTaskHeadConfig()


@dataclass(frozen=True)
class MultiTaskLossConfig:
    """Open tuning parameters for MultiTaskDistributionLoss."""

    ret_loss_weight: float = 1.0
    rv_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    rv_tail_weight_threshold: float = 0.03
    rv_tail_weight_alpha: float = 2.0
    rv_tail_weight_max: float = 4.0
    q_tau: float = 0.10


multi_task_loss = MultiTaskLossConfig()
multi_task_loss_hparams = MULTI_TASK_LOSS_HPARAMS


@dataclass(frozen=True)
class SingleTaskLossConfig:
    """Open tuning parameters for the single-field mu/sigma objectives."""

    q_tau: float = 0.10
    ret_mu_fixed_scale: float = 0.02335
    ret_mu_fixed_nu: float = 2.82
    rv_tail_weight_threshold: float = 0.03
    rv_tail_weight_alpha: float = 2.0
    rv_tail_weight_max: float = 4.0


single_task_loss = SingleTaskLossConfig()


@dataclass(frozen=True)
class FeatureChannelDropoutConfig:
    """Open tuning parameters for input feature-channel dropout."""

    macro_p: float = 0.05
    mezzo_p: float = 0.03
    micro_p: float = 0.03
    macro_mf_main_amount_log_p: float = 0.25


feature_channel_dropout = FeatureChannelDropoutConfig()


@dataclass(frozen=True)
class MultiScaleForecastNetworkConfig:
    """Open tuning parameters for MultiScaleForecastNetwork."""

    hidden_dim: int = 128
    cond_dim: int = 32
    sidechain_features: int = 13
    field: str = "ret"


multi_scale_forecast_network = MultiScaleForecastNetworkConfig()
multi_scale_forecast_network_hparams = MULTI_SCALE_FORECAST_NETWORK_HPARAMS
