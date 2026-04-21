"""Full multi-scale forecast network assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from config.models import (
    conditioning_encoder,
    exogenous_bridge_fusion,
    modern_tcn_film_encoder,
    multi_scale_forecast_network,
    scale_context_bridge_fusion,
    single_task_head,
    single_task_loss,
    within_scale_star_fusion,
)
from src.models.arch.encoders import ConditioningEncoder
from src.models.arch.encoders.modern_tcn_film_encoder import ModernTCNFiLMEncoder
from src.models.arch.fusions import (
    ExogenousBridgeFusion,
    ScaleContextBridgeFusion,
    WithinScaleSTARFusion,
)
from src.models.arch.heads import SingleTaskHead
from src.models.arch.layers.wavelet_denoise import WaveletDenoise1D
from src.models.arch.losses import SingleTaskDistributionLoss
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

        self.cross_scale_fusion = ScaleContextBridgeFusion(
            hidden_dim=self.hidden_dim,
            num_heads=scale_context_bridge_fusion.num_heads,
            ffn_ratio=scale_context_bridge_fusion.ffn_ratio,
            num_layers=scale_context_bridge_fusion.num_layers,
            dropout=scale_context_bridge_fusion.dropout,
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
            debug["cross_scale"] = cross_scale_debug
            debug["heads"] = pred_dict.pop("_debug", {})
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
                    "s1": s1,
                    "g1": g1,
                    "s2": s2,
                    "g2": g2,
                    "s3": s3,
                    "g3": g3,
                    "macro_fused": macro_fused,
                    "mezzo_fused": mezzo_fused,
                    "micro_fused": micro_fused,
                    "micro_td": micro_td,
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
