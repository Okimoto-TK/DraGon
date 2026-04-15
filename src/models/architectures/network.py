"""Top-level multiscale fusion network."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.config import hidden_dim as DEFAULT_HIDDEN_DIM
from config.config import jointnet_23_blocks as DEFAULT_JOINTNET_23_BLOCKS
from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import lmf_rank as DEFAULT_LMF_RANK
from config.config import macro_decomp_level as DEFAULT_MACRO_DECOMP_LEVEL
from config.config import macro_wno_num_blocks as DEFAULT_MACRO_WNO_NUM_BLOCKS
from config.config import mezzo_decomp_level as DEFAULT_MEZZO_DECOMP_LEVEL
from config.config import mezzo_wno_num_blocks as DEFAULT_MEZZO_WNO_NUM_BLOCKS
from config.config import micro_decomp_level as DEFAULT_MICRO_DECOMP_LEVEL
from config.config import micro_wno_num_blocks as DEFAULT_MICRO_WNO_NUM_BLOCKS
from config.config import side_hidden_dim as DEFAULT_SIDE_HIDDEN_DIM
from config.config import summary_dim as DEFAULT_SUMMARY_DIM
from config.config import token_dim as DEFAULT_TOKEN_DIM
from src.models.components.encoders import SidechainEncoder, WNOEncoder
from src.models.components.fusion import (
    DualCrossAttentionFusion,
    PairwiseLMFMap,
    SemanticGatedChannelFusion,
    TensorFusion,
)
from src.models.components.heads import DecoderHead, SummaryHead
from src.models.components.pooling import InteractionMapToTokens, PerceiverResampler
from src.models.components.trunks import JointNet2D
from src.task_labels import canonical_task_label


def _channelwise_l2_mean(x: Tensor, *, channel_dim: int) -> Tensor:
    moved = torch.movedim(x.detach(), channel_dim, -1).float()
    flat = moved.reshape(-1, moved.shape[-1])
    return flat.norm(dim=-1).mean()


class MultiScaleFusionNet(nn.Module):
    """Fuse multiscale market features into one label-specific prediction head."""

    def __init__(
        self,
        *,
        task_label: str = "Edge",
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        side_hidden_dim: int = DEFAULT_SIDE_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
        lmf_rank: int = DEFAULT_LMF_RANK,
        latent_token: int = DEFAULT_LATENT_TOKEN,
        token_dim: int = DEFAULT_TOKEN_DIM,
        summary_dim: int = DEFAULT_SUMMARY_DIM,
        macro_decomp_level: int = DEFAULT_MACRO_DECOMP_LEVEL,
        macro_wno_num_blocks: int = DEFAULT_MACRO_WNO_NUM_BLOCKS,
        mezzo_decomp_level: int = DEFAULT_MEZZO_DECOMP_LEVEL,
        mezzo_wno_num_blocks: int = DEFAULT_MEZZO_WNO_NUM_BLOCKS,
        micro_decomp_level: int = DEFAULT_MICRO_DECOMP_LEVEL,
        micro_wno_num_blocks: int = DEFAULT_MICRO_WNO_NUM_BLOCKS,
        jointnet_23_blocks: int = DEFAULT_JOINTNET_23_BLOCKS,
        uncertainty_floor: float = 1e-4,
        downrisk_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if side_hidden_dim <= 0:
            raise ValueError(f"side_hidden_dim must be positive, got {side_hidden_dim}")
        if lmf_dim <= 0:
            raise ValueError(f"lmf_dim must be positive, got {lmf_dim}")
        if lmf_rank <= 0:
            raise ValueError(f"lmf_rank must be positive, got {lmf_rank}")
        if latent_token <= 0:
            raise ValueError(f"latent_token must be positive, got {latent_token}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if summary_dim <= 0:
            raise ValueError(f"summary_dim must be positive, got {summary_dim}")
        if macro_wno_num_blocks <= 0:
            raise ValueError(f"macro_wno_num_blocks must be positive, got {macro_wno_num_blocks}")
        if mezzo_wno_num_blocks <= 0:
            raise ValueError(f"mezzo_wno_num_blocks must be positive, got {mezzo_wno_num_blocks}")
        if micro_wno_num_blocks <= 0:
            raise ValueError(f"micro_wno_num_blocks must be positive, got {micro_wno_num_blocks}")
        if jointnet_23_blocks <= 0:
            raise ValueError(f"jointnet_23_blocks must be positive, got {jointnet_23_blocks}")
        if macro_decomp_level <= 0 or mezzo_decomp_level <= 0 or micro_decomp_level <= 0:
            raise ValueError(
                "decomp levels must be positive, got "
                f"macro={macro_decomp_level}, mezzo={mezzo_decomp_level}, micro={micro_decomp_level}"
            )
        if uncertainty_floor <= 0.0:
            raise ValueError(f"uncertainty_floor must be positive, got {uncertainty_floor}")
        if downrisk_eps <= 0.0:
            raise ValueError(f"downrisk_eps must be positive, got {downrisk_eps}")

        self.task_label = canonical_task_label(task_label)
        self.hidden_dim = hidden_dim
        self.side_hidden_dim = side_hidden_dim
        self.lmf_dim = lmf_dim
        self.lmf_rank = lmf_rank
        self.latent_token = latent_token
        self.token_dim = token_dim
        self.summary_dim = summary_dim
        self.macro_decomp_level = macro_decomp_level
        self.macro_wno_num_blocks = macro_wno_num_blocks
        self.mezzo_decomp_level = mezzo_decomp_level
        self.mezzo_wno_num_blocks = mezzo_wno_num_blocks
        self.micro_decomp_level = micro_decomp_level
        self.micro_wno_num_blocks = micro_wno_num_blocks
        self.jointnet_23_blocks = jointnet_23_blocks
        self.uncertainty_floor = float(uncertainty_floor)
        self.downrisk_eps = float(downrisk_eps)

        self.macro_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=macro_decomp_level,
            num_blocks=macro_wno_num_blocks,
        )
        self.mezzo_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=mezzo_decomp_level,
            num_blocks=mezzo_wno_num_blocks,
        )
        self.micro_encoder = WNOEncoder(
            in_channels=7,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=micro_decomp_level,
            num_blocks=micro_wno_num_blocks,
        )
        self.side_encoder = SidechainEncoder(
            in_channels=8,
            hidden_dim=side_hidden_dim,
            lmf_dim=lmf_dim,
        )

        self.pairwise_lmf_12 = PairwiseLMFMap(dx=lmf_dim, dy=lmf_dim, d_out=lmf_dim, rank=lmf_rank)
        self.pairwise_lmf_23 = PairwiseLMFMap(dx=lmf_dim, dy=lmf_dim, d_out=lmf_dim, rank=lmf_rank)

        self.jointnet_12 = JointNet2D(channels=lmf_dim)
        self.jointnet_23 = JointNet2D(channels=lmf_dim, num_blocks=jointnet_23_blocks)

        self.map_to_tokens_12 = InteractionMapToTokens(latent_token=latent_token, dim=lmf_dim)
        self.map_to_tokens_23 = InteractionMapToTokens(latent_token=latent_token, dim=lmf_dim)
        self.side_resampler = PerceiverResampler(
            latent_token=latent_token,
            input_dim=lmf_dim,
            dim=token_dim,
        )
        self.m1_token_proj = nn.Linear(lmf_dim, token_dim)
        self.m2_token_proj = nn.Linear(lmf_dim, token_dim)

        self.drift_fusion = DualCrossAttentionFusion(dim=token_dim, num_layers=2)
        self.diffusion_fusion = SemanticGatedChannelFusion(dim=token_dim)

        self.drift_summary_head = SummaryHead(dim=token_dim, summary_dim=summary_dim)
        self.diffusion_summary_head = SummaryHead(dim=token_dim, summary_dim=summary_dim)
        self.full_tfn = TensorFusion(dim_x=summary_dim, dim_y=summary_dim)

        head_out_dim = 1 if self.task_label == "Persist" else 2
        self.decoder_head = DecoderHead(in_dim=self.full_tfn.out_dim, out_dim=head_out_dim)

    def forward(
        self,
        macro: Tensor,
        mezzo: Tensor,
        micro: Tensor,
        side: Tensor,
    ) -> dict[str, Tensor]:
        e1 = self.macro_encoder(macro)
        e2 = self.mezzo_encoder(mezzo)
        e3 = self.micro_encoder(micro)
        e4 = self.side_encoder(side)

        u1 = e1.transpose(1, 2)
        u2 = e2.transpose(1, 2)
        u3 = e3.transpose(1, 2)

        t12 = self.pairwise_lmf_12(u1, u2)
        t23 = self.pairwise_lmf_23(u2, u3)

        h12 = self.jointnet_12(t12)
        h23 = self.jointnet_23(t23)

        m1 = self.m1_token_proj(self.map_to_tokens_12(h12))
        m2 = self.m2_token_proj(self.map_to_tokens_23(h23))
        s = self.side_resampler(e4.transpose(1, 2))

        z0 = self.drift_fusion(m1, s)
        z1, diffusion_diag = self.diffusion_fusion(m2, s, return_debug=True)

        z_d = self.drift_summary_head(z0)
        z_v = self.diffusion_summary_head(z1)
        tfn_feat = self.full_tfn(z_d, z_v)
        head_out = self.decoder_head(tfn_feat)

        outputs: dict[str, Tensor] = {
            "H12": h12,
            "H23": h23,
            "M1": m1,
            "M2": m2,
            "S": s,
            "Z0": z0,
            "Z1": z1,
            "z_d": z_d,
            "z_v": z_v,
            "tfn_feat": tfn_feat,
            "head_out": head_out,
            "diag/encoder/E1_norm": _channelwise_l2_mean(e1, channel_dim=1),
            "diag/encoder/E2_norm": _channelwise_l2_mean(e2, channel_dim=1),
            "diag/encoder/E3_norm": _channelwise_l2_mean(e3, channel_dim=1),
            "diag/encoder/E4_norm": _channelwise_l2_mean(e4, channel_dim=1),
            "diag/map/T12_norm": _channelwise_l2_mean(t12, channel_dim=1),
            "diag/map/T23_norm": _channelwise_l2_mean(t23, channel_dim=1),
            "diag/map/H12_norm": _channelwise_l2_mean(h12, channel_dim=1),
            "diag/map/H23_norm": _channelwise_l2_mean(h23, channel_dim=1),
        }
        outputs.update({f"diag/{key}": value for key, value in diffusion_diag.items()})

        if self.task_label == "Edge":
            pred_edge = head_out[:, 0]
            raw_unc_edge = head_out[:, 1]
            unc_edge = F.softplus(raw_unc_edge) + self.uncertainty_floor
            outputs.update(
                {
                    "pred_Edge": pred_edge,
                    "raw_unc_Edge": raw_unc_edge,
                    "unc_Edge": unc_edge,
                    "Edge": pred_edge,
                    "Edge_unc": unc_edge,
                }
            )
            return outputs

        if self.task_label == "Persist":
            logit_persist = head_out[:, 0]
            pred_persist = torch.sigmoid(logit_persist)
            persist_unc = 4.0 * pred_persist * (1.0 - pred_persist)
            outputs.update(
                {
                    "logit_Persist": logit_persist,
                    "pred_Persist": pred_persist,
                    "Persist": pred_persist,
                    "Persist_unc": persist_unc,
                }
            )
            return outputs

        pred_log_downrisk = head_out[:, 0]
        raw_unc_downrisk = head_out[:, 1]
        unc_downrisk = F.softplus(raw_unc_downrisk) + self.uncertainty_floor
        pred_downrisk = (torch.exp(pred_log_downrisk) - self.downrisk_eps).clamp_min(0.0)
        outputs.update(
            {
                "pred_log_DownRisk": pred_log_downrisk,
                "pred_DownRisk": pred_downrisk,
                "raw_unc_DownRisk": raw_unc_downrisk,
                "unc_DownRisk": unc_downrisk,
                "DownRisk": pred_downrisk,
                "DownRisk_unc": unc_downrisk,
            }
        )
        return outputs


def run_smoke_test(batch_size: int = 2, *, task_label: str = "Edge") -> dict[str, object]:
    """Run a minimal forward+loss smoke test."""
    from src.models.losses import SingleTaskLoss

    model = MultiScaleFusionNet(task_label=task_label)
    outputs = model(
        torch.randn(batch_size, 9, 64),
        torch.randn(batch_size, 9, 64),
        torch.randn(batch_size, 7, 48),
        torch.randn(batch_size, 8, 64),
    )
    labels = {
        "label_Edge": torch.randn(batch_size),
        "label_Persist": torch.rand(batch_size),
        "label_DownRisk": torch.rand(batch_size),
    }
    criterion = SingleTaskLoss(task_label=task_label)
    loss, _ = criterion(outputs, labels)
    return {
        "task_label": task_label,
        "M1": tuple(outputs["M1"].shape),
        "M2": tuple(outputs["M2"].shape),
        "S": tuple(outputs["S"].shape),
        "z_d": tuple(outputs["z_d"].shape),
        "z_v": tuple(outputs["z_v"].shape),
        "tfn_feat": tuple(outputs["tfn_feat"].shape),
        "head_out": tuple(outputs["head_out"].shape),
        "loss_scalar": tuple(loss.shape),
    }


__all__ = ["MultiScaleFusionNet", "run_smoke_test"]
