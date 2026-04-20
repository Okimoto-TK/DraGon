"""Full multi-scale forecast network assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from config.models import (
    conditioning_encoder,
    cross_scale_fusion,
    exogenous_bridge_fusion,
    modern_tcn_film_encoder,
    multi_scale_forecast_network,
    multi_task_heads,
    multi_task_loss,
    within_scale_star_fusion,
)
from src.models.arch.encoders import ConditioningEncoder
from src.models.arch.encoders.modern_tcn_film_encoder import ModernTCNFiLMEncoder
from src.models.arch.fusions import (
    CrossScaleFusion,
    ExogenousBridgeFusion,
    WithinScaleSTARFusion,
)
from src.models.arch.heads import MultiTaskHeads
from src.models.arch.layers.wavelet_denoise import WaveletDenoise1D
from src.models.arch.losses import MultiTaskDistributionLoss
from src.models.arch.networks.side_memory_hierarchy import SideMemoryHierarchy
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
        self.return_aux_default = bool(return_aux_default)
        self._hparams = MULTI_SCALE_FORECAST_NETWORK_HPARAMS

        self.denoise_macro = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["macro"].num_features,
            target_len=self._hparams._macro_target_len,
            warmup_len=self._hparams._macro_warmup_len,
        )
        self.denoise_mezzo = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["mezzo"].num_features,
            target_len=self._hparams._mezzo_target_len,
            warmup_len=self._hparams._mezzo_warmup_len,
        )
        self.denoise_micro = WaveletDenoise1D(
            n_channels=modern_tcn_film_encoder["micro"].num_features,
            target_len=self._hparams._micro_target_len,
            warmup_len=self._hparams._micro_warmup_len,
        )

        self.encoder_macro = ModernTCNFiLMEncoder(
            **modern_tcn_film_encoder["macro"].__dict__
        )
        self.encoder_mezzo = ModernTCNFiLMEncoder(
            **modern_tcn_film_encoder["mezzo"].__dict__
        )
        self.encoder_micro = ModernTCNFiLMEncoder(
            **modern_tcn_film_encoder["micro"].__dict__
        )

        self.star_macro = WithinScaleSTARFusion(
            hidden_dim=self.hidden_dim,
            num_features=within_scale_star_fusion.num_features,
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

        self.conditioning_encoder = ConditioningEncoder(
            d_cond=self.cond_dim,
            input_features=conditioning_encoder.input_features,
            num_blocks=conditioning_encoder.num_blocks,
            dropout=conditioning_encoder.dropout,
        )
        self.side_memory_hierarchy = SideMemoryHierarchy(
            d_cond=self.cond_dim,
            _norm_eps=self._hparams._norm_eps,
        )

        self.bridge_macro = ExogenousBridgeFusion(
            hidden_dim=self.hidden_dim,
            exogenous_dim=self.cond_dim,
            num_heads=exogenous_bridge_fusion.num_heads,
            ffn_ratio=exogenous_bridge_fusion.ffn_ratio,
            num_layers=exogenous_bridge_fusion.num_layers,
            dropout=exogenous_bridge_fusion.dropout,
        )
        self.bridge_mezzo = ExogenousBridgeFusion(
            hidden_dim=self.hidden_dim,
            exogenous_dim=self.cond_dim,
            num_heads=exogenous_bridge_fusion.num_heads,
            ffn_ratio=exogenous_bridge_fusion.ffn_ratio,
            num_layers=exogenous_bridge_fusion.num_layers,
            dropout=exogenous_bridge_fusion.dropout,
        )
        self.bridge_micro = ExogenousBridgeFusion(
            hidden_dim=self.hidden_dim,
            exogenous_dim=self.cond_dim,
            num_heads=exogenous_bridge_fusion.num_heads,
            ffn_ratio=exogenous_bridge_fusion.ffn_ratio,
            num_layers=exogenous_bridge_fusion.num_layers,
            dropout=exogenous_bridge_fusion.dropout,
        )

        self.cross_scale_fusion = CrossScaleFusion(
            hidden_dim=self.hidden_dim,
            num_latents=self.num_latents,
            num_heads=cross_scale_fusion.num_heads,
            ffn_ratio=cross_scale_fusion.ffn_ratio,
            num_layers=cross_scale_fusion.num_layers,
            dropout=cross_scale_fusion.dropout,
        )
        self.multi_task_heads = MultiTaskHeads(
            hidden_dim=self.hidden_dim,
            num_latents=self.num_latents,
            tower_num_heads=multi_task_heads.tower_num_heads,
            tower_ffn_ratio=multi_task_heads.tower_ffn_ratio,
            tower_dropout=multi_task_heads.tower_dropout,
        )
        self.loss_fn = MultiTaskDistributionLoss(
            q_tau=multi_task_loss.q_tau,
            ret_loss_weight=multi_task_loss.ret_loss_weight,
            rv_loss_weight=multi_task_loss.rv_loss_weight,
            q_loss_weight=multi_task_loss.q_loss_weight,
            _eps=MULTI_TASK_LOSS_HPARAMS._eps,
            _nu_ret_init=MULTI_TASK_LOSS_HPARAMS._nu_ret_init,
            _nu_ret_min=MULTI_TASK_LOSS_HPARAMS._nu_ret_min,
            _gamma_shape_min=MULTI_TASK_LOSS_HPARAMS._gamma_shape_min,
            _ald_scale_min=MULTI_TASK_LOSS_HPARAMS._ald_scale_min,
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

        macro_state = macro_i8_long[:, 0, -64:].long()
        macro_pos = macro_i8_long[:, 1, -64:].long()
        mezzo_state = mezzo_i8_long[:, 0, -96:].long()
        mezzo_pos = mezzo_i8_long[:, 1, -96:].long()
        micro_state = micro_i8_long[:, 0, -144:].long()
        micro_pos = micro_i8_long[:, 1, -144:].long()

        z_macro = self.encoder_macro(macro_float, macro_state, macro_pos)
        z_mezzo = self.encoder_mezzo(mezzo_float, mezzo_state, mezzo_pos)
        z_micro = self.encoder_micro(micro_float, micro_state, micro_pos)

        z_macro_fused, scale_seq_macro = self.star_macro(z_macro)
        z_mezzo_fused, scale_seq_mezzo = self.star_mezzo(z_mezzo)
        z_micro_fused, scale_seq_micro = self.star_micro(z_micro)

        cond_seq, cond_global = self.conditioning_encoder(sidechain_cond)
        s1, g1, s2, g2, s3, g3 = self.side_memory_hierarchy(cond_seq, cond_global)

        debug: dict[str, object] = {}
        if return_debug:
            macro_fused, _, bridge_macro_debug = self.bridge_macro(
                scale_seq_macro,
                s1,
                g1,
                return_debug=True,
            )
            mezzo_fused, _, bridge_mezzo_debug = self.bridge_mezzo(
                scale_seq_mezzo,
                s2,
                g2,
                return_debug=True,
            )
            micro_fused, _, bridge_micro_debug = self.bridge_micro(
                scale_seq_micro,
                s3,
                g3,
                return_debug=True,
            )
            debug["wavelet"] = {
                "macro_raw_tail": macro_float_long[:, :, -64:],
                "macro_denoised": macro_float,
                "mezzo_raw_tail": mezzo_float_long[:, :, -96:],
                "mezzo_denoised": mezzo_float,
                "micro_raw_tail": micro_float_long[:, :, -144:],
                "micro_denoised": micro_float,
            }
            debug["encoder"] = {
                "macro": z_macro,
                "mezzo": z_mezzo,
                "micro": z_micro,
            }
            debug["within_scale"] = {
                "macro_pre": z_macro,
                "macro_post": z_macro_fused,
                "macro_seq": scale_seq_macro,
                "mezzo_pre": z_mezzo,
                "mezzo_post": z_mezzo_fused,
                "mezzo_seq": scale_seq_mezzo,
                "micro_pre": z_micro,
                "micro_post": z_micro_fused,
                "micro_seq": scale_seq_micro,
            }
            debug["conditioning"] = {
                "cond_seq": cond_seq,
                "cond_global": cond_global,
            }
            debug["side_memory"] = {
                "s1": s1,
                "g1": g1,
                "s2": s2,
                "g2": g2,
                "s3": s3,
                "g3": g3,
            }
            debug["bridge"] = {
                "macro": bridge_macro_debug,
                "mezzo": bridge_mezzo_debug,
                "micro": bridge_micro_debug,
            }
        else:
            macro_fused, _ = self.bridge_macro(scale_seq_macro, s1, g1)
            mezzo_fused, _ = self.bridge_mezzo(scale_seq_mezzo, s2, g2)
            micro_fused, _ = self.bridge_micro(scale_seq_micro, s3, g3)

        if return_debug:
            fused_latents, fused_global, cross_scale_debug = self.cross_scale_fusion(
                macro_seq=macro_fused,
                mezzo_seq=mezzo_fused,
                micro_seq=micro_fused,
                return_debug=True,
            )
            pred_dict = self.multi_task_heads(
                fused_latents,
                fused_global,
                return_debug=True,
            )
            debug["cross_scale"] = cross_scale_debug
            debug["heads"] = pred_dict.pop("_debug", {})
        else:
            fused_latents, fused_global = self.cross_scale_fusion(
                macro_seq=macro_fused,
                mezzo_seq=mezzo_fused,
                micro_seq=micro_fused,
            )
            pred_dict = self.multi_task_heads(fused_latents, fused_global)
        out: dict[str, torch.Tensor] = {
            **pred_dict,
            "fused_latents": fused_latents,
            "fused_global": fused_global,
        }

        use_aux = self.return_aux_default if return_aux is None else bool(return_aux)
        if use_aux:
            out.update(
                {
                    "scale_seq_macro": scale_seq_macro,
                    "scale_seq_mezzo": scale_seq_mezzo,
                    "scale_seq_micro": scale_seq_micro,
                    "s1": s1,
                    "g1": g1,
                    "s2": s2,
                    "g2": g2,
                    "s3": s3,
                    "g3": g3,
                    "macro_fused": macro_fused,
                    "mezzo_fused": mezzo_fused,
                    "micro_fused": micro_fused,
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
        target_ret = self._require_key(batch, "target_ret")
        target_rv = self._require_key(batch, "target_rv")
        target_q = self._require_key(batch, "target_q")

        loss = self.loss_fn(
            target_ret=target_ret,
            pred_mu_ret=pred["pred_mu_ret"],
            pred_scale_ret_raw=pred["pred_scale_ret_raw"],
            target_rv=target_rv,
            pred_mean_rv_raw=pred["pred_mean_rv_raw"],
            pred_shape_rv_raw=pred["pred_shape_rv_raw"],
            target_q=target_q,
            pred_mu_q=pred["pred_mu_q"],
            pred_scale_q_raw=pred["pred_scale_q_raw"],
        )
        out = {
            **loss,
            "pred_mu_ret": pred["pred_mu_ret"],
            "pred_mean_rv_raw": pred["pred_mean_rv_raw"],
            "pred_mu_q": pred["pred_mu_q"],
            "pred_scale_ret_raw": pred["pred_scale_ret_raw"],
            "pred_shape_rv_raw": pred["pred_shape_rv_raw"],
            "pred_scale_q_raw": pred["pred_scale_q_raw"],
        }
        if return_aux:
            out.update(
                {
                    "fused_latents": pred["fused_latents"],
                    "fused_global": pred["fused_global"],
                }
            )
        if return_debug and "_debug" in pred:
            out["_debug"] = pred["_debug"]
        return out
