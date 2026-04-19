"""Model hyper-parameters for ModernTCN-FiLM encoders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveletDenoiseHParams:
    """Hidden hyper-parameters for WaveletDenoise1D."""

    _wavelet: str = "db4"
    _level: int = 2
    _eps: float = 1e-6


@dataclass(frozen=True)
class ModernTCNFiLMEncoderHParams:
    """Per-scale hyper-parameters for ModernTCNFiLMEncoder."""

    seq_len: int
    num_features: int = 9
    patch_len: int = 8
    patch_stride: int = 4
    hidden_dim: int = 128
    cond_dim: int = 64
    kernel_size: int = 7
    ffn_ratio: float = 2.0
    num_layers: int = 2
    state_vocab_size: int = 16
    pos_vocab_size: int = 64
    dropout: float = 0.0

    @property
    def _expected_seq_len(self) -> int:
        return self.seq_len

    @property
    def _num_patches(self) -> int:
        pad_right = self.patch_len - self.patch_stride
        return ((self.seq_len + pad_right - self.patch_len) // self.patch_stride) + 1


@dataclass(frozen=True)
class ConditioningEncoderHParams:
    """Hidden hyper-parameters for ConditioningEncoder."""

    _temporal_mlp_mult: int = 2
    _feature_mlp_mult: int = 2
    _norm_eps: float = 1e-6
    _pool_type: str = "mean"


@dataclass(frozen=True)
class WithinScaleSTARFusionHParams:
    """Hidden hyper-parameters for WithinScaleSTARFusion."""

    _norm_eps: float = 1e-6
    _pool_temperature: float = 1.0


@dataclass(frozen=True)
class ExogenousBridgeFusionHParams:
    """Hidden hyper-parameters for ExogenousBridgeFusion."""

    _norm_eps: float = 1e-6


@dataclass(frozen=True)
class CrossScaleFusionHParams:
    """Hidden hyper-parameters for CrossScaleFusion."""

    _norm_eps: float = 1e-6


@dataclass(frozen=True)
class MultiTaskHeadsHParams:
    """Hidden hyper-parameters for MultiTaskHeads."""

    _tower_norm_eps: float = 1e-6


@dataclass(frozen=True)
class MultiTaskLossHParams:
    """Hidden hyper-parameters for MultiTaskDistributionLoss."""

    _eps: float = 1e-6
    _nu_ret_init: float = 8.0
    _nu_ret_min: float = 2.01
    _gamma_shape_min: float = 1e-4
    _ald_scale_min: float = 1e-6


@dataclass(frozen=True)
class MultiScaleForecastNetworkHParams:
    """Hidden hyper-parameters for network assembly."""

    _macro_target_len: int = 64
    _macro_warmup_len: int = 48
    _mezzo_target_len: int = 96
    _mezzo_warmup_len: int = 48
    _micro_target_len: int = 144
    _micro_warmup_len: int = 48
    _side_len: int = 64
    _mezzo_days: int = 12
    _micro_days: int = 3
    _norm_eps: float = 1e-6


@dataclass(frozen=True)
class TrainRuntimeHParams:
    """Hidden runtime knobs for training and dataloading."""

    _pin_memory: bool = True
    _prefetch_factor_train: int = 4
    _prefetch_factor_val: int = 2
    _persistent_workers: bool = True
    _drop_last_train: bool = True
    _drop_last_val: bool = False
    _compile_mode: str = "max-autotune"


MODERN_TCN_FILM_ENCODER_HPARAMS = {
    "macro": ModernTCNFiLMEncoderHParams(seq_len=64, pos_vocab_size=8),
    "mezzo": ModernTCNFiLMEncoderHParams(seq_len=96, pos_vocab_size=16),
    "micro": ModernTCNFiLMEncoderHParams(seq_len=144, pos_vocab_size=64),
}

WAVELET_DENOISE_HPARAMS = WaveletDenoiseHParams()
CONDITIONING_ENCODER_HPARAMS = ConditioningEncoderHParams()
WITHIN_SCALE_STAR_FUSION_HPARAMS = WithinScaleSTARFusionHParams()
EXOGENOUS_BRIDGE_FUSION_HPARAMS = ExogenousBridgeFusionHParams()
CROSS_SCALE_FUSION_HPARAMS = CrossScaleFusionHParams()
MULTI_TASK_HEADS_HPARAMS = MultiTaskHeadsHParams()
MULTI_TASK_LOSS_HPARAMS = MultiTaskLossHParams()
MULTI_SCALE_FORECAST_NETWORK_HPARAMS = MultiScaleForecastNetworkHParams()
TRAIN_RUNTIME_HPARAMS = TrainRuntimeHParams()
