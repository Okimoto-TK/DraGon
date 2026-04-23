"""Full multi-scale forecast network assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from config.models import (
    dual_domain_concat_head,
    dual_domain_concat_head_hparams,
    dual_domain_mutual_attention,
    dual_domain_mutual_attention_hparams,
    feature_channel_dropout,
    modern_tcn_film_encoder,
    multi_scale_forecast_network,
    single_task_loss,
    time_topdown_fusion,
    time_topdown_fusion_hparams,
    wavelet_bottomup_fusion,
    wavelet_bottomup_fusion_hparams,
    wavelet_branch_encoder,
    wavelet_branch_encoder_hparams,
    wavelet_denoise,
    within_scale_star_fusion,
)
from src.models.arch.encoders import DualDomainScaleEncoder
from src.models.arch.fusions import (
    TimeTopDownHierarchicalFusion,
    WaveletBottomUpSupportFusion,
)
from src.models.arch.heads import DualDomainConcatHead
from src.models.arch.layers import FeatureChannelDropout1D
from src.models.arch.layers.wavelet_denoise import WaveletDenoise1D
from src.models.arch.losses import SingleTaskDistributionLoss
from src.models.config.hparams import (
    MULTI_SCALE_FORECAST_NETWORK_HPARAMS,
    MULTI_TASK_LOSS_HPARAMS,
)


class MultiScaleForecastNetwork(nn.Module):
    """Assemble denoise -> dual-domain encoders -> hierarchical fusion -> head."""

    def __init__(
        self,
        hidden_dim: int = multi_scale_forecast_network.hidden_dim,
        cond_dim: int = multi_scale_forecast_network.cond_dim,
        task: str = multi_scale_forecast_network.task,
        return_aux_default: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {cond_dim}.")

        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
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
            special_channel_ps={16: feature_channel_dropout.macro_mf_main_amount_log_p},
        )
        self.dropout_mezzo_input = FeatureChannelDropout1D(
            num_channels=modern_tcn_film_encoder["mezzo"].num_features,
            p=feature_channel_dropout.mezzo_p,
        )
        self.dropout_micro_input = FeatureChannelDropout1D(
            num_channels=modern_tcn_film_encoder["micro"].num_features,
            p=feature_channel_dropout.micro_p,
        )

        self.scale_macro = self._build_scale_encoder(
            scale_name="macro",
            scale_index=0,
            time_num_features=self._macro_input_features,
            wavelet_num_features=modern_tcn_film_encoder["macro"].num_features,
            wavelet_sidechain_features=multi_scale_forecast_network.sidechain_features,
        )
        self.scale_mezzo = self._build_scale_encoder(
            scale_name="mezzo",
            scale_index=1,
            time_num_features=modern_tcn_film_encoder["mezzo"].num_features,
            wavelet_num_features=modern_tcn_film_encoder["mezzo"].num_features,
            wavelet_sidechain_features=0,
        )
        self.scale_micro = self._build_scale_encoder(
            scale_name="micro",
            scale_index=2,
            time_num_features=modern_tcn_film_encoder["micro"].num_features,
            wavelet_num_features=modern_tcn_film_encoder["micro"].num_features,
            wavelet_sidechain_features=0,
        )
        self.register_buffer(
            "_macro_token_support",
            self._build_token_support(
                scale_name="macro",
                total_days=float(self._hparams._macro_days),
            ),
            persistent=False,
        )
        self.register_buffer(
            "_mezzo_token_support",
            self._build_token_support(
                scale_name="mezzo",
                total_days=float(self._hparams._mezzo_days),
            ),
            persistent=False,
        )
        self.register_buffer(
            "_micro_token_support",
            self._build_token_support(
                scale_name="micro",
                total_days=float(self._hparams._micro_days),
            ),
            persistent=False,
        )

        self.time_cross_scale_fusion = TimeTopDownHierarchicalFusion(
            hidden_dim=self.hidden_dim,
            num_heads=time_topdown_fusion.num_heads,
            ffn_ratio=time_topdown_fusion.ffn_ratio,
            num_layers=time_topdown_fusion.num_layers,
            dropout=time_topdown_fusion.dropout,
            _norm_eps=time_topdown_fusion_hparams._norm_eps,
            _gate_floor=time_topdown_fusion_hparams._gate_floor,
        )
        self.wavelet_cross_scale_fusion = WaveletBottomUpSupportFusion(
            hidden_dim=self.hidden_dim,
            ffn_ratio=wavelet_bottomup_fusion.ffn_ratio,
            num_layers=wavelet_bottomup_fusion.num_layers,
            dropout=wavelet_bottomup_fusion.dropout,
            _norm_eps=wavelet_bottomup_fusion_hparams._norm_eps,
            _gate_floor=wavelet_bottomup_fusion_hparams._gate_floor,
        )
        self.prediction_head = DualDomainConcatHead(
            task=self.task,
            hidden_dim=self.hidden_dim,
            num_heads=dual_domain_concat_head.num_heads,
            ffn_ratio=dual_domain_concat_head.ffn_ratio,
            dropout=dual_domain_concat_head.dropout,
            _norm_eps=dual_domain_concat_head_hparams._norm_eps,
        )
        self.loss_fn = SingleTaskDistributionLoss(
            task=self.task,
            q_tau=single_task_loss.q_tau,
            ret_tail_weight_threshold=single_task_loss.ret_tail_weight_threshold,
            ret_tail_weight_alpha=single_task_loss.ret_tail_weight_alpha,
            ret_tail_weight_max=single_task_loss.ret_tail_weight_max,
            rv_tail_weight_threshold=single_task_loss.rv_tail_weight_threshold,
            rv_tail_weight_alpha=single_task_loss.rv_tail_weight_alpha,
            rv_tail_weight_max=single_task_loss.rv_tail_weight_max,
            _eps=MULTI_TASK_LOSS_HPARAMS._eps,
            _nu_ret_init=MULTI_TASK_LOSS_HPARAMS._nu_ret_init,
            _nu_ret_min=MULTI_TASK_LOSS_HPARAMS._nu_ret_min,
            _gamma_shape_min=MULTI_TASK_LOSS_HPARAMS._gamma_shape_min,
            _ald_scale_min=MULTI_TASK_LOSS_HPARAMS._ald_scale_min,
            _loss_compute_dtype=MULTI_TASK_LOSS_HPARAMS._loss_compute_dtype,
        )

    def _build_scale_encoder(
        self,
        *,
        scale_name: str,
        scale_index: int,
        time_num_features: int,
        wavelet_num_features: int,
        wavelet_sidechain_features: int,
    ) -> DualDomainScaleEncoder:
        hp = modern_tcn_film_encoder[scale_name]
        time_encoder_kwargs = hp.__dict__.copy()
        time_encoder_kwargs["num_features"] = int(time_num_features)
        time_encoder_kwargs["hidden_dim"] = self.hidden_dim
        time_encoder_kwargs["cond_dim"] = self.cond_dim

        return DualDomainScaleEncoder(
            time_encoder_kwargs=time_encoder_kwargs,
            wavelet_num_features=wavelet_num_features,
            scale_index=scale_index,
            time_star_core_dim=within_scale_star_fusion.core_dim,
            time_star_num_layers=within_scale_star_fusion.num_layers,
            time_star_dropout=within_scale_star_fusion.dropout,
            wavelet_num_heads=wavelet_branch_encoder.num_heads,
            wavelet_ffn_ratio=wavelet_branch_encoder.ffn_ratio,
            wavelet_num_layers=wavelet_branch_encoder.num_layers,
            wavelet_sidechain_features=wavelet_sidechain_features,
            mutual_num_heads=dual_domain_mutual_attention.num_heads,
            mutual_ffn_ratio=dual_domain_mutual_attention.ffn_ratio,
            mutual_num_layers=dual_domain_mutual_attention.num_layers,
            dropout=dual_domain_mutual_attention.dropout,
            wavelet_norm_eps=wavelet_branch_encoder_hparams._norm_eps,
            wavelet_band_gate_floor=wavelet_branch_encoder_hparams._band_gate_floor,
            wavelet_resample_mode=wavelet_branch_encoder_hparams._resample_mode,
            mutual_norm_eps=dual_domain_mutual_attention_hparams._norm_eps,
            mutual_gate_floor=dual_domain_mutual_attention_hparams._gate_floor,
        )

    def _build_token_support(
        self,
        *,
        scale_name: str,
        total_days: float,
    ) -> torch.Tensor:
        hp = modern_tcn_film_encoder[scale_name]
        seq_len = int(hp.seq_len)
        patch_len = int(hp.patch_len)
        patch_stride = int(hp.patch_stride)
        num_tokens = int(hp._num_patches)
        if total_days <= 0:
            raise ValueError(f"total_days must be > 0, got {total_days}.")
        day_per_step = float(total_days) / float(seq_len)
        starts = []
        ends = []
        for token_idx in range(num_tokens):
            step_start = token_idx * patch_stride
            step_end = min(step_start + patch_len, seq_len)
            start_day = -float(total_days) + float(step_start) * day_per_step
            end_day = -float(total_days) + float(step_end) * day_per_step
            starts.append(start_day)
            ends.append(min(end_day, 0.0))
        return torch.tensor(list(zip(starts, ends)), dtype=torch.float32)

    def _expect_shape(
        self,
        name: str,
        tensor: torch.Tensor,
        expected: tuple[int, ...],
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
    def _l2_mean(value: torch.Tensor) -> float:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return float(flat.norm(dim=1).mean().cpu())

    @staticmethod
    def _feature_channel_rms(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"value must have shape [B, F, D, N], got shape={tuple(value.shape)}."
            )
        return value.detach().float().square().mean(dim=(0, 2, 3)).sqrt().cpu()

    def _forward_wavelet_frontend(
        self,
        module: nn.Module,
        x_long: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if hasattr(module, "forward_features"):
            y, coeffs = module.forward_features(x_long)  # type: ignore[attr-defined]
            return y, coeffs
        y = module(x_long)
        return y, (y, y, y)

    def _validate_forward_batch(
        self,
        batch: dict[str, torch.Tensor],
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

        macro_float, macro_coeffs = self._forward_wavelet_frontend(
            self.denoise_macro,
            macro_float_long,
        )
        mezzo_float, mezzo_coeffs = self._forward_wavelet_frontend(
            self.denoise_mezzo,
            mezzo_float_long,
        )
        micro_float, micro_coeffs = self._forward_wavelet_frontend(
            self.denoise_micro,
            micro_float_long,
        )

        macro_input = self.dropout_macro_input(torch.cat([macro_float, sidechain_cond], dim=1))
        mezzo_input = self.dropout_mezzo_input(mezzo_float)
        micro_input = self.dropout_micro_input(micro_float)

        macro_state = macro_i8_long[:, 0, -64:].long()
        macro_pos = macro_i8_long[:, 1, -64:].long()
        mezzo_state = mezzo_i8_long[:, 0, -96:].long()
        mezzo_pos = mezzo_i8_long[:, 1, -96:].long()
        micro_state = micro_i8_long[:, 0, -144:].long()
        micro_pos = micro_i8_long[:, 1, -144:].long()

        macro_time, macro_wavelet, macro_time_features = self.scale_macro(
            time_x=macro_input,
            x_state=macro_state,
            x_pos=macro_pos,
            wavelet_coeffs=macro_coeffs,
            sidechain=sidechain_cond,
        )
        mezzo_time, mezzo_wavelet, mezzo_time_features = self.scale_mezzo(
            time_x=mezzo_input,
            x_state=mezzo_state,
            x_pos=mezzo_pos,
            wavelet_coeffs=mezzo_coeffs,
        )
        micro_time, micro_wavelet, micro_time_features = self.scale_micro(
            time_x=micro_input,
            x_state=micro_state,
            x_pos=micro_pos,
            wavelet_coeffs=micro_coeffs,
        )

        macro_time, mezzo_time, micro_time = self.time_cross_scale_fusion(
            macro_time,
            mezzo_time,
            micro_time,
        )
        macro_wavelet, mezzo_wavelet, micro_wavelet = self.wavelet_cross_scale_fusion(
            macro_wavelet,
            mezzo_wavelet,
            micro_wavelet,
            macro_support=self._macro_token_support,
            mezzo_support=self._mezzo_token_support,
            micro_support=self._micro_token_support,
        )

        pred_dict = self.prediction_head(
            mezzo_time,
            mezzo_wavelet,
            return_debug=return_debug,
        )

        macro_dual_summary = 0.5 * (macro_time.mean(dim=-1) + macro_wavelet.mean(dim=-1))
        micro_dual_summary = 0.5 * (micro_time.mean(dim=-1) + micro_wavelet.mean(dim=-1))

        out: dict[str, torch.Tensor] = {
            "pred_primary": pred_dict["pred_primary"],
            "pred_aux_raw": pred_dict["pred_aux_raw"],
            "task_repr": pred_dict["task_repr"],
            "mezzo_head_tokens": pred_dict["head_tokens"],
            "mezzo_head_context": pred_dict["head_context"],
            "macro_dual_summary": macro_dual_summary,
            "micro_dual_summary": micro_dual_summary,
        }
        for name in (
            "pred_mu_ret",
            "pred_scale_ret_raw",
            "pred_mean_rv_raw",
            "pred_shape_rv_raw",
            "pred_mu_q",
            "pred_scale_q_raw",
        ):
            if name in pred_dict:
                out[name] = pred_dict[name]

        use_aux = self.return_aux_default if return_aux is None else bool(return_aux)
        if use_aux:
            out.update(
                {
                    "macro_input": macro_input,
                    "time_tokens_macro": macro_time,
                    "time_tokens_mezzo": mezzo_time,
                    "time_tokens_micro": micro_time,
                    "wavelet_tokens_macro": macro_wavelet,
                    "wavelet_tokens_mezzo": mezzo_wavelet,
                    "wavelet_tokens_micro": micro_wavelet,
                    "macro_wavelet_sidechain_input": sidechain_cond,
                    "feature_rms_macro_pre": self._feature_channel_rms(macro_time_features),
                    "feature_rms_mezzo_pre": self._feature_channel_rms(mezzo_time_features),
                    "feature_rms_micro_pre": self._feature_channel_rms(micro_time_features),
                }
            )

        if return_debug:
            debug = {
                "wavelet_macro_energy_raw": self._energy_mean(macro_float_long[:, :, -64:]),
                "wavelet_macro_energy_denoised": self._energy_mean(macro_float),
                "wavelet_mezzo_energy_raw": self._energy_mean(mezzo_float_long[:, :, -96:]),
                "wavelet_mezzo_energy_denoised": self._energy_mean(mezzo_float),
                "wavelet_micro_energy_raw": self._energy_mean(micro_float_long[:, :, -144:]),
                "wavelet_micro_energy_denoised": self._energy_mean(micro_float),
                "time_macro_summary_l2_mean": self._l2_mean(macro_time.mean(dim=-1)),
                "time_mezzo_summary_l2_mean": self._l2_mean(mezzo_time.mean(dim=-1)),
                "time_micro_summary_l2_mean": self._l2_mean(micro_time.mean(dim=-1)),
                "wavelet_macro_summary_l2_mean": self._l2_mean(macro_wavelet.mean(dim=-1)),
                "wavelet_mezzo_summary_l2_mean": self._l2_mean(mezzo_wavelet.mean(dim=-1)),
                "wavelet_micro_summary_l2_mean": self._l2_mean(micro_wavelet.mean(dim=-1)),
            }
            debug.update(pred_dict.pop("_debug", {}))
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
        }
        if return_debug and "_debug" in pred:
            out["_debug"] = pred["_debug"]
        return out
