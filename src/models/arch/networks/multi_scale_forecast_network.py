"""Full multi-scale forecast network assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from config.models import (
    adaln_zero_topdown_fusion,
    feature_channel_dropout,
    modern_tcn_film_encoder,
    multi_scale_forecast_network,
    single_task_head,
    single_task_loss,
    wavelet_denoise,
    within_scale_star_fusion,
)
from src.models.arch.encoders.modern_tcn_film_encoder import ModernTCNFiLMEncoder
from src.models.arch.fusions import (
    AdaLNZeroTopDownFusion,
    WithinScaleSTARFusion,
)
from src.models.arch.heads import SingleTaskHead
from src.models.arch.layers import FeatureChannelDropout1D
from src.models.arch.layers.wavelet_denoise import WaveletDenoise1D
from src.models.arch.losses import SingleTaskDistributionLoss
from src.models.config.hparams import (
    MULTI_SCALE_FORECAST_NETWORK_HPARAMS,
    MULTI_TASK_LOSS_HPARAMS,
)


class MultiScaleForecastNetwork(nn.Module):
    """Assemble denoise -> encoder -> fusion -> heads into one forward chain."""

    def __init__(
        self,
        hidden_dim: int = multi_scale_forecast_network.hidden_dim,
        cond_dim: int = multi_scale_forecast_network.cond_dim,
        num_latents: int = multi_scale_forecast_network.num_latents,
        task: str = multi_scale_forecast_network.task,
        return_aux_default: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {cond_dim}.")
        if num_latents <= 0:
            raise ValueError(f"num_latents must be > 0, got {num_latents}.")

        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.num_latents = int(num_latents)
        self.task = task
        self.return_aux_default = bool(return_aux_default)
        self._hparams = MULTI_SCALE_FORECAST_NETWORK_HPARAMS
        self._macro_input_features = (
            modern_tcn_film_encoder["macro"].num_features
            + multi_scale_forecast_network.sidechain_features
        )

        self.denoise_macro = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["macro"].num_features,
            target_len=self._hparams._macro_target_len,
            warmup_len=self._hparams._macro_warmup_len,
            allow_backward=wavelet_denoise.backprop,
        )
        self.denoise_mezzo = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["mezzo"].num_features,
            target_len=self._hparams._mezzo_target_len,
            warmup_len=self._hparams._mezzo_warmup_len,
            allow_backward=wavelet_denoise.backprop,
        )
        self.denoise_micro = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["micro"].num_features,
            target_len=self._hparams._micro_target_len,
            warmup_len=self._hparams._micro_warmup_len,
            allow_backward=wavelet_denoise.backprop,
        )
        self.dropout_macro_input = FeatureChannelDropout1D(
            num_channels=self._macro_input_features,
            p=feature_channel_dropout.macro_p,
            special_channel_ps={
                16: feature_channel_dropout.macro_mf_main_amount_log_p,
            },
        )
        self.dropout_mezzo_input = FeatureChannelDropout1D(
            num_channels=modern_tcn_film_encoder["mezzo"].num_features,
            p=feature_channel_dropout.mezzo_p,
        )
        self.dropout_micro_input = FeatureChannelDropout1D(
            num_channels=modern_tcn_film_encoder["micro"].num_features,
            p=feature_channel_dropout.micro_p,
        )

        macro_encoder_kwargs = modern_tcn_film_encoder["macro"].__dict__.copy()
        macro_encoder_kwargs["num_features"] = self._macro_input_features
        self.encoder_macro = ModernTCNFiLMEncoder(**macro_encoder_kwargs)
        self.encoder_mezzo = ModernTCNFiLMEncoder(
            **modern_tcn_film_encoder["mezzo"].__dict__
        )
        self.encoder_micro = ModernTCNFiLMEncoder(
            **modern_tcn_film_encoder["micro"].__dict__
        )

        self.star_macro = WithinScaleSTARFusion(
            hidden_dim=self.hidden_dim,
            num_features=self._macro_input_features,
            core_dim=within_scale_star_fusion.core_dim,
            num_layers=within_scale_star_fusion.num_layers,
            dropout=within_scale_star_fusion.dropout,
        )
        self.star_mezzo = WithinScaleSTARFusion(
            hidden_dim=self.hidden_dim,
            num_features=within_scale_star_fusion.num_features,
            core_dim=within_scale_star_fusion.core_dim,
            num_layers=within_scale_star_fusion.num_layers,
            dropout=within_scale_star_fusion.dropout,
        )
        self.star_micro = WithinScaleSTARFusion(
            hidden_dim=self.hidden_dim,
            num_features=within_scale_star_fusion.num_features,
            core_dim=within_scale_star_fusion.core_dim,
            num_layers=within_scale_star_fusion.num_layers,
            dropout=within_scale_star_fusion.dropout,
        )

        self.cross_scale_fusion = AdaLNZeroTopDownFusion(
            hidden_dim=self.hidden_dim,
            ffn_ratio=adaln_zero_topdown_fusion.ffn_ratio,
            dropout=adaln_zero_topdown_fusion.dropout,
        )
        self.single_task_head = SingleTaskHead(
            task=self.task,
            hidden_dim=self.hidden_dim,
            num_heads=single_task_head.num_heads,
            ffn_ratio=single_task_head.ffn_ratio,
            dropout=single_task_head.dropout,
        )
        self.loss_fn = SingleTaskDistributionLoss(
            task=self.task,
            q_tau=single_task_loss.q_tau,
            ret_tail_weight_threshold=single_task_loss.ret_tail_weight_threshold,
            ret_tail_weight_alpha=single_task_loss.ret_tail_weight_alpha,
            ret_tail_weight_max=single_task_loss.ret_tail_weight_max,
            _eps=MULTI_TASK_LOSS_HPARAMS._eps,
            _nu_ret_init=MULTI_TASK_LOSS_HPARAMS._nu_ret_init,
            _nu_ret_min=MULTI_TASK_LOSS_HPARAMS._nu_ret_min,
            _gamma_shape_min=MULTI_TASK_LOSS_HPARAMS._gamma_shape_min,
            _ald_scale_min=MULTI_TASK_LOSS_HPARAMS._ald_scale_min,
            _loss_compute_dtype=MULTI_TASK_LOSS_HPARAMS._loss_compute_dtype,
        )

    def _expect_shape(
        self, name: str, tensor: torch.Tensor, expected: tuple[int, ...]
    ) -> None:
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}."
            )

    def _require_key(self, batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise ValueError(f"Missing required batch key: {key}.")
        return batch[key]

    @staticmethod
    def _energy_mean(value: torch.Tensor) -> float:
        return float(value.detach().float().square().mean().cpu())

    @staticmethod
    def _mean(value: torch.Tensor) -> float:
        return float(value.detach().float().mean().cpu())

    @staticmethod
    def _std(value: torch.Tensor) -> float:
        return float(value.detach().float().std(unbiased=False).cpu())

    @staticmethod
    def _abs_mean(value: torch.Tensor) -> float:
        return float(value.detach().float().abs().mean().cpu())

    @staticmethod
    def _feature_cosdist(value: torch.Tensor) -> float:
        if value.ndim != 4 or value.shape[1] < 2:
            return 0.0
        feats = value.detach().float().reshape(value.shape[0], value.shape[1], -1)
        feats = torch.nn.functional.normalize(feats, dim=-1, eps=1e-6)
        cos = torch.matmul(feats, feats.transpose(1, 2))
        feature_count = value.shape[1]
        mask = ~torch.eye(feature_count, device=cos.device, dtype=torch.bool).unsqueeze(0)
        dist = (1.0 - cos).masked_select(mask)
        if dist.numel() == 0:
            return 0.0
        return float(dist.mean().cpu())

    @staticmethod
    def _feature_channel_rms(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"value must have shape [B, F, D, N], got shape={tuple(value.shape)}."
            )
        return value.detach().float().square().mean(dim=(0, 2, 3)).sqrt().cpu()

    def _validate_forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = [
            "macro_float_long",
            "macro_i8_long",
            "mezzo_float_long",
            "mezzo_i8_long",
            "micro_float_long",
            "micro_i8_long",
            "sidechain_cond",
        ]
        values = {k: self._require_key(batch, k) for k in required}

        bsz = values["macro_float_long"].shape[0]
        self._expect_shape("macro_float_long", values["macro_float_long"], (bsz, 9, 112))
        self._expect_shape("macro_i8_long", values["macro_i8_long"], (bsz, 2, 112))
        self._expect_shape("mezzo_float_long", values["mezzo_float_long"], (bsz, 9, 144))
        self._expect_shape("mezzo_i8_long", values["mezzo_i8_long"], (bsz, 2, 144))
        self._expect_shape("micro_float_long", values["micro_float_long"], (bsz, 9, 192))
        self._expect_shape("micro_i8_long", values["micro_i8_long"], (bsz, 2, 192))
        self._expect_shape("sidechain_cond", values["sidechain_cond"], (bsz, 13, 64))

        return (
            values["macro_float_long"],
            values["macro_i8_long"],
            values["mezzo_float_long"],
            values["mezzo_i8_long"],
            values["micro_float_long"],
            values["micro_i8_long"],
            values["sidechain_cond"],
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        (
            macro_float_long,
            macro_i8_long,
            mezzo_float_long,
            mezzo_i8_long,
            micro_float_long,
            micro_i8_long,
            sidechain_cond,
        ) = self._validate_forward_batch(batch)

        macro_float = self.denoise_macro(macro_float_long)
        mezzo_float = self.denoise_mezzo(mezzo_float_long)
        micro_float = self.denoise_micro(micro_float_long)
        macro_input = torch.cat([macro_float, sidechain_cond], dim=1)
        macro_input = self.dropout_macro_input(macro_input)
        mezzo_float = self.dropout_mezzo_input(mezzo_float)
        micro_float = self.dropout_micro_input(micro_float)

        macro_state = macro_i8_long[:, 0, -64:].long()
        macro_pos = macro_i8_long[:, 1, -64:].long()
        mezzo_state = mezzo_i8_long[:, 0, -96:].long()
        mezzo_pos = mezzo_i8_long[:, 1, -96:].long()
        micro_state = micro_i8_long[:, 0, -144:].long()
        micro_pos = micro_i8_long[:, 1, -144:].long()

        z_macro = self.encoder_macro(macro_input, macro_state, macro_pos)
        z_mezzo = self.encoder_mezzo(mezzo_float, mezzo_state, mezzo_pos)
        z_micro = self.encoder_micro(micro_float, micro_state, micro_pos)

        z_macro_fused, scale_seq_macro = self.star_macro(z_macro)
        z_mezzo_fused, scale_seq_mezzo = self.star_mezzo(z_mezzo)
        z_micro_fused, scale_seq_micro = self.star_micro(z_micro)

        debug: dict[str, float] = {}
        if return_debug:
            macro_fused = scale_seq_macro
            mezzo_fused = scale_seq_mezzo
            micro_fused = scale_seq_micro
            macro_raw_tail = macro_float_long[:, :, -64:]
            mezzo_raw_tail = mezzo_float_long[:, :, -96:]
            micro_raw_tail = micro_float_long[:, :, -144:]
            macro_raw_energy = self._energy_mean(macro_raw_tail)
            mezzo_raw_energy = self._energy_mean(mezzo_raw_tail)
            micro_raw_energy = self._energy_mean(micro_raw_tail)
            macro_denoised_energy = self._energy_mean(macro_float)
            mezzo_denoised_energy = self._energy_mean(mezzo_float)
            micro_denoised_energy = self._energy_mean(micro_float)
            debug.update(
                {
                    "wavelet_macro_energy_raw": macro_raw_energy,
                    "wavelet_macro_energy_denoised": macro_denoised_energy,
                    "wavelet_macro_energy_ratio_denoised_over_raw": macro_denoised_energy / max(macro_raw_energy, 1e-12),
                    "wavelet_mezzo_energy_raw": mezzo_raw_energy,
                    "wavelet_mezzo_energy_denoised": mezzo_denoised_energy,
                    "wavelet_mezzo_energy_ratio_denoised_over_raw": mezzo_denoised_energy / max(mezzo_raw_energy, 1e-12),
                    "wavelet_micro_energy_raw": micro_raw_energy,
                    "wavelet_micro_energy_denoised": micro_denoised_energy,
                    "wavelet_micro_energy_ratio_denoised_over_raw": micro_denoised_energy / max(micro_raw_energy, 1e-12),
                    "encoder_macro_final_block_act_mean": self._mean(z_macro),
                    "encoder_macro_final_block_act_std": self._std(z_macro),
                    "encoder_macro_final_block_act_abs_mean": self._abs_mean(z_macro),
                    "encoder_mezzo_final_block_act_mean": self._mean(z_mezzo),
                    "encoder_mezzo_final_block_act_std": self._std(z_mezzo),
                    "encoder_mezzo_final_block_act_abs_mean": self._abs_mean(z_mezzo),
                    "encoder_micro_final_block_act_mean": self._mean(z_micro),
                    "encoder_micro_final_block_act_std": self._std(z_micro),
                    "encoder_micro_final_block_act_abs_mean": self._abs_mean(z_micro),
                }
            )
            macro_pre_dist = self._feature_cosdist(z_macro)
            macro_post_dist = self._feature_cosdist(z_macro_fused)
            mezzo_pre_dist = self._feature_cosdist(z_mezzo)
            mezzo_post_dist = self._feature_cosdist(z_mezzo_fused)
            micro_pre_dist = self._feature_cosdist(z_micro)
            micro_post_dist = self._feature_cosdist(z_micro_fused)
            debug.update(
                {
                    "within_scale_macro_feature_cosdist_pre": macro_pre_dist,
                    "within_scale_macro_feature_cosdist_post": macro_post_dist,
                    "within_scale_macro_feature_cosdist_ratio": macro_post_dist / max(macro_pre_dist, 1e-12),
                    "within_scale_mezzo_feature_cosdist_pre": mezzo_pre_dist,
                    "within_scale_mezzo_feature_cosdist_post": mezzo_post_dist,
                    "within_scale_mezzo_feature_cosdist_ratio": mezzo_post_dist / max(mezzo_pre_dist, 1e-12),
                    "within_scale_micro_feature_cosdist_pre": micro_pre_dist,
                    "within_scale_micro_feature_cosdist_post": micro_post_dist,
                    "within_scale_micro_feature_cosdist_ratio": micro_post_dist / max(micro_pre_dist, 1e-12),
                }
            )
        else:
            macro_fused = scale_seq_macro
            mezzo_fused = scale_seq_mezzo
            micro_fused = scale_seq_micro

        if return_debug:
            micro_td, macro_ctx, mezzo_ctx, micro_ctx, cross_scale_debug = self.cross_scale_fusion(
                macro_seq=macro_fused,
                mezzo_seq=mezzo_fused,
                micro_seq=micro_fused,
                return_debug=True,
            )
            pred_dict = self.single_task_head(
                micro_td,
                mezzo_ctx,
                macro_ctx,
                return_debug=True,
            )
            debug.update(cross_scale_debug)
            debug.update(pred_dict.pop("_debug", {}))
        else:
            micro_td, macro_ctx, mezzo_ctx, micro_ctx = self.cross_scale_fusion(
                macro_seq=macro_fused,
                mezzo_seq=mezzo_fused,
                micro_seq=micro_fused,
            )
            pred_dict = self.single_task_head(
                micro_td,
                mezzo_ctx,
                macro_ctx,
            )
        out: dict[str, torch.Tensor] = {
            **pred_dict,
            "fused_latents": micro_td,
            "fused_global": pred_dict["head_context"],
            "macro_ctx": macro_ctx,
            "mezzo_ctx": mezzo_ctx,
            "micro_ctx": micro_ctx,
        }

        use_aux = self.return_aux_default if return_aux is None else bool(return_aux)
        if use_aux:
            out.update(
                {
                    "scale_seq_macro": scale_seq_macro,
                    "scale_seq_mezzo": scale_seq_mezzo,
                    "scale_seq_micro": scale_seq_micro,
                    "macro_input": macro_input,
                    "macro_fused": macro_fused,
                    "mezzo_fused": mezzo_fused,
                    "micro_fused": micro_fused,
                    "micro_td": micro_td,
                    "feature_rms_macro_pre": self._feature_channel_rms(z_macro),
                    "feature_rms_macro_post": self._feature_channel_rms(z_macro_fused),
                    "feature_rms_mezzo_pre": self._feature_channel_rms(z_mezzo),
                    "feature_rms_mezzo_post": self._feature_channel_rms(z_mezzo_fused),
                    "feature_rms_micro_pre": self._feature_channel_rms(z_micro),
                    "feature_rms_micro_post": self._feature_channel_rms(z_micro_fused),
                }
            )
        if return_debug:
            out["_debug"] = debug
        return out

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        pred = self.forward(
            batch,
            return_aux=return_aux,
            return_debug=return_debug,
        )
        target_key = {
            "ret": "target_ret",
            "rv": "target_rv",
            "q": "target_q",
        }[self.task]
        target = self._require_key(batch, target_key)

        loss = self.loss_fn(
            target=target,
            pred_primary=pred["pred_primary"],
            pred_aux_raw=pred["pred_aux_raw"],
        )
        out = {
            **loss,
            "pred_primary": pred["pred_primary"],
            "pred_aux_raw": pred["pred_aux_raw"],
            "fused_latents": pred["fused_latents"],
            "fused_global": pred["fused_global"],
            "macro_ctx": pred["macro_ctx"],
            "mezzo_ctx": pred["mezzo_ctx"],
            "micro_ctx": pred["micro_ctx"],
        }
        for name in (
            "pred_mu_ret",
            "pred_scale_ret_raw",
            "pred_mean_rv_raw",
            "pred_shape_rv_raw",
            "pred_mu_q",
            "pred_scale_q_raw",
            "task_repr",
            "head_context",
        ):
            if name in pred:
                out[name] = pred[name]
        if return_aux:
            out.update(
                {
                    "micro_td": pred["fused_latents"],
                }
            )
        if return_debug and "_debug" in pred:
            out["_debug"] = pred["_debug"]
        return out
