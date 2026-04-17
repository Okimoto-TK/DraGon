"""Architecture assembly for the rebuilt multi-scale fusion network."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from config.config import cross_scale_window_macro_to_mezzo as DEFAULT_A2M_WINDOW
from config.config import cross_scale_window_mezzo_to_micro as DEFAULT_M2I_WINDOW
from config.config import cross_scale_window_micro_to_mezzo as DEFAULT_I2M_WINDOW
from config.config import joint_token_count as DEFAULT_JOINT_TOKENS
from config.config import latent_dim as DEFAULT_LATENT_DIM
from config.config import local_kernel as DEFAULT_LOCAL_KERNEL
from config.config import macro_wno_levels as DEFAULT_MACRO_WNO_LEVELS
from config.config import macro_wno_num_blocks as DEFAULT_MACRO_WNO_BLOCKS
from config.config import mezzo_wno_levels as DEFAULT_MEZZO_WNO_LEVELS
from config.config import mezzo_wno_num_blocks as DEFAULT_MEZZO_WNO_BLOCKS
from config.config import micro_wno_levels as DEFAULT_MICRO_WNO_LEVELS
from config.config import micro_wno_num_blocks as DEFAULT_MICRO_WNO_BLOCKS
from config.config import num_attention_heads as DEFAULT_NUM_HEADS
from config.config import path_token_count as DEFAULT_PATH_TOKENS
from src.models.components.encoders.sidechain import SideContextEncoder
from src.models.components.heads.output_head import OutputHead
from src.models.components.readout.topk_cross import LocalDenseCrossReadout
from src.models.components.trunks.common import pool_tokens
from src.models.components.trunks.cross_scale import CrossScaleCrossAttention, CrossScaleFFN
from src.models.components.trunks.single_scale import ScaleFrontend


@dataclass
class ScaleOutputs:
    price_tokens: Tensor
    liquid_tokens: Tensor
    joint_tokens: Tensor
    state_tokens: Tensor
    price_relation_tokens: Tensor
    pv_interaction_tokens: Tensor


class MultiScaleFusionNet(nn.Module):
    def __init__(self, *, task_label: str, dim: int = DEFAULT_LATENT_DIM) -> None:
        super().__init__()
        self.task_label = task_label
        self.dim = int(dim)

        self.side_context_encoder = SideContextEncoder(dim, num_heads=DEFAULT_NUM_HEADS)
        self.macro_frontend = ScaleFrontend(
            seq_len=64,
            dim=dim,
            num_heads=DEFAULT_NUM_HEADS,
            path_tokens=DEFAULT_PATH_TOKENS,
            liquid_tokens=DEFAULT_PATH_TOKENS,
            joint_tokens=DEFAULT_JOINT_TOKENS,
            path_levels=DEFAULT_MACRO_WNO_LEVELS,
            path_blocks=DEFAULT_MACRO_WNO_BLOCKS,
            local_kernel=DEFAULT_LOCAL_KERNEL,
        )
        self.mezzo_frontend = ScaleFrontend(
            seq_len=64,
            dim=dim,
            num_heads=DEFAULT_NUM_HEADS,
            path_tokens=DEFAULT_PATH_TOKENS,
            liquid_tokens=DEFAULT_PATH_TOKENS,
            joint_tokens=DEFAULT_JOINT_TOKENS,
            path_levels=DEFAULT_MEZZO_WNO_LEVELS,
            path_blocks=DEFAULT_MEZZO_WNO_BLOCKS,
            local_kernel=DEFAULT_LOCAL_KERNEL,
        )
        self.micro_frontend = ScaleFrontend(
            seq_len=48,
            dim=dim,
            num_heads=DEFAULT_NUM_HEADS,
            path_tokens=DEFAULT_PATH_TOKENS,
            liquid_tokens=DEFAULT_PATH_TOKENS,
            joint_tokens=DEFAULT_JOINT_TOKENS,
            path_levels=DEFAULT_MICRO_WNO_LEVELS,
            path_blocks=DEFAULT_MICRO_WNO_BLOCKS,
            local_kernel=DEFAULT_LOCAL_KERNEL,
        )

        # Monitoring / visualization aliases (指向新路径).
        self.macro_return_encoder = self.macro_frontend.path_branch.feature_encoder
        self.mezzo_return_encoder = self.mezzo_frontend.path_branch.feature_encoder
        self.micro_return_encoder = self.micro_frontend.path_branch.feature_encoder
        self.side_gap_encoder = self.side_context_encoder.gap_encoder
        self.mezzo_pv_interaction_fusion = self.mezzo_frontend.pv_fusion.interaction_fusion
        self.micro_pv_interaction_fusion = self.micro_frontend.pv_fusion.interaction_fusion
        self.mezzo_joint_token_readout = self.mezzo_frontend.pv_fusion.joint_readout
        self.micro_joint_token_readout = self.micro_frontend.pv_fusion.joint_readout

        self.macro_to_mezzo = CrossScaleCrossAttention(dim, num_heads=DEFAULT_NUM_HEADS, num_contexts=3)
        self.mezzo_to_micro = CrossScaleCrossAttention(dim, num_heads=DEFAULT_NUM_HEADS, num_contexts=3)
        self.micro_to_mezzo_sig = LocalDenseCrossReadout(
            dim,
            q_steps=64,
            q_tokens=DEFAULT_JOINT_TOKENS,
            kv_steps=48,
            kv_tokens=DEFAULT_JOINT_TOKENS,
            window=DEFAULT_I2M_WINDOW,
            num_contexts=3,
        )
        self.micro_to_mezzo = CrossScaleCrossAttention(dim, num_heads=DEFAULT_NUM_HEADS, num_contexts=3)
        self.mezzo_ffn = CrossScaleFFN(dim, num_contexts=2)
        self.joint_summary_norm = nn.LayerNorm(dim)
        self.side_summary_norm = nn.LayerNorm(dim)
        self.joint_side_interaction_norm = nn.LayerNorm(dim)
        self.head = OutputHead(dim, task_label=task_label)

        # Keep explicit references for documentation/debug symmetry even if not all are consumed directly.
        self.cross_scale_window_macro_to_mezzo = DEFAULT_A2M_WINDOW
        self.cross_scale_window_mezzo_to_micro = DEFAULT_M2I_WINDOW
        self.cross_scale_window_micro_to_mezzo = DEFAULT_I2M_WINDOW

    def _encode_scale(self, frontend: ScaleFrontend, scale_x: Tensor, e_d: Tensor) -> ScaleOutputs:
        outputs = frontend(scale_x, e_d)
        return ScaleOutputs(
            price_tokens=outputs["price_tokens"],
            liquid_tokens=outputs["liquid_tokens"],
            joint_tokens=outputs["joint_tokens"],
            state_tokens=outputs["state_tokens"],
            price_relation_tokens=outputs["price_relation_tokens"],
            pv_interaction_tokens=outputs["pv_interaction_tokens"],
        )

    def forward(self, macro: Tensor, mezzo: Tensor, micro: Tensor, sidechain: Tensor) -> dict[str, Tensor]:
        e_d, side_debug = self.side_context_encoder(sidechain)

        macro_out = self._encode_scale(self.macro_frontend, macro, e_d)
        mezzo_out = self._encode_scale(self.mezzo_frontend, mezzo, e_d)
        micro_out = self._encode_scale(self.micro_frontend, micro, e_d)

        joint_macro = macro_out.joint_tokens
        joint_mezzo = mezzo_out.joint_tokens
        joint_micro = micro_out.joint_tokens

        macro_conditioned_mezzo_tokens = self.macro_to_mezzo(joint_mezzo, joint_macro, e_d, macro_out.state_tokens, mezzo_out.state_tokens)
        micro_conditioned_tokens = self.mezzo_to_micro(joint_micro, macro_conditioned_mezzo_tokens, e_d, mezzo_out.state_tokens, micro_out.state_tokens)
        micro_signal_tokens = self.micro_to_mezzo_sig(macro_conditioned_mezzo_tokens, micro_conditioned_tokens, e_d, mezzo_out.state_tokens, micro_out.state_tokens)
        micro_refined_mezzo_tokens = self.micro_to_mezzo(
            macro_conditioned_mezzo_tokens,
            micro_signal_tokens,
            e_d,
            micro_out.state_tokens,
            mezzo_out.state_tokens,
        )
        fused_joint_tokens = self.mezzo_ffn(micro_refined_mezzo_tokens, e_d, mezzo_out.state_tokens)

        state_tokens = mezzo_out.state_tokens + self.mezzo_frontend.state_reader(mezzo_out.state_tokens, fused_joint_tokens)

        joint_summary = self.joint_summary_norm(pool_tokens(fused_joint_tokens))
        side_summary = self.side_summary_norm(pool_tokens(e_d))
        joint_side_interaction = self.joint_side_interaction_norm(joint_summary * side_summary)
        head_outputs = self.head(joint_summary)

        outputs: dict[str, Tensor] = {
            "side_context_tokens": e_d,
            "price_tokens": mezzo_out.price_tokens,
            "liquid_tokens": mezzo_out.liquid_tokens,
            "joint_tokens": fused_joint_tokens,
            "state_tokens": state_tokens,
            "macro_joint_tokens": joint_macro,
            "mezzo_joint_tokens": joint_mezzo,
            "micro_joint_tokens": joint_micro,
            "macro_conditioned_mezzo_tokens": macro_conditioned_mezzo_tokens,
            "micro_refined_mezzo_tokens": micro_refined_mezzo_tokens,
            "joint_summary": joint_summary,
            "side_summary": side_summary,
            "joint_side_interaction": joint_side_interaction,
            "price_relation_tokens": mezzo_out.price_relation_tokens,
            "pv_interaction_tokens": mezzo_out.pv_interaction_tokens,
            "micro_signal_tokens": micro_signal_tokens,
            "side/moneyflow_context_tokens": side_debug["z_moneyflow"],
            "side/liquidity_context_tokens": side_debug["z_liquidity_reg"],
            "side/causal_context_tokens": side_debug["z_causal"],
            "side/gap_context_tokens": side_debug["z_gap_context"],
        }
        outputs.update(head_outputs)
        return outputs


__all__ = ["MultiScaleFusionNet"]
