"""Top-level multiscale fusion network."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from config.config import hidden_dim as DEFAULT_HIDDEN_DIM
from config.config import jointnet_23_blocks as DEFAULT_JOINTNET_23_BLOCKS
from config.config import jointnet_23_channels as DEFAULT_JOINTNET_23_CHANNELS
from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import lmf_rank as DEFAULT_LMF_RANK
from config.config import macro_decomp_level as DEFAULT_MACRO_DECOMP_LEVEL
from config.config import mezzo_decomp_level as DEFAULT_MEZZO_DECOMP_LEVEL
from config.config import micro_decomp_level as DEFAULT_MICRO_DECOMP_LEVEL
from config.config import side_hidden_dim as DEFAULT_SIDE_HIDDEN_DIM
from config.config import summary_dim as DEFAULT_SUMMARY_DIM
from config.config import token_dim as DEFAULT_TOKEN_DIM
from config.config import wno_num_blocks as DEFAULT_WNO_NUM_BLOCKS
from torch import Tensor, nn

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


class MultiScaleFusionNet(nn.Module):
    """Fuse multiscale market features into direct task predictions and scales."""

    def __init__(
        self,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        side_hidden_dim: int = DEFAULT_SIDE_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
        lmf_rank: int = DEFAULT_LMF_RANK,
        latent_token: int = DEFAULT_LATENT_TOKEN,
        token_dim: int = DEFAULT_TOKEN_DIM,
        summary_dim: int = DEFAULT_SUMMARY_DIM,
        macro_decomp_level: int = DEFAULT_MACRO_DECOMP_LEVEL,
        mezzo_decomp_level: int = DEFAULT_MEZZO_DECOMP_LEVEL,
        micro_decomp_level: int = DEFAULT_MICRO_DECOMP_LEVEL,
        wno_num_blocks: int = DEFAULT_WNO_NUM_BLOCKS,
        jointnet_23_channels: int = DEFAULT_JOINTNET_23_CHANNELS,
        jointnet_23_blocks: int = DEFAULT_JOINTNET_23_BLOCKS,
        min_scale_S: float = 1e-3,
        min_scale_M: float = 1e-2,
        min_scale_MDD: float = 1e-2,
        min_scale_RV: float = 1e-3,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            msg = f"hidden_dim must be positive, got {hidden_dim}"
            raise ValueError(msg)

        if lmf_dim <= 0:
            msg = f"lmf_dim must be positive, got {lmf_dim}"
            raise ValueError(msg)
        if side_hidden_dim <= 0:
            msg = f"side_hidden_dim must be positive, got {side_hidden_dim}"
            raise ValueError(msg)

        if lmf_rank <= 0:
            msg = f"lmf_rank must be positive, got {lmf_rank}"
            raise ValueError(msg)

        if latent_token <= 0:
            msg = f"latent_token must be positive, got {latent_token}"
            raise ValueError(msg)
        if token_dim <= 0:
            msg = f"token_dim must be positive, got {token_dim}"
            raise ValueError(msg)
        if summary_dim <= 0:
            msg = f"summary_dim must be positive, got {summary_dim}"
            raise ValueError(msg)
        if wno_num_blocks <= 0:
            msg = f"wno_num_blocks must be positive, got {wno_num_blocks}"
            raise ValueError(msg)
        if jointnet_23_channels <= 0:
            msg = f"jointnet_23_channels must be positive, got {jointnet_23_channels}"
            raise ValueError(msg)
        if jointnet_23_blocks <= 0:
            msg = f"jointnet_23_blocks must be positive, got {jointnet_23_blocks}"
            raise ValueError(msg)

        if macro_decomp_level <= 0 or mezzo_decomp_level <= 0 or micro_decomp_level <= 0:
            msg = (
                "decomp levels must be positive, got "
                f"macro={macro_decomp_level}, mezzo={mezzo_decomp_level}, micro={micro_decomp_level}"
            )
            raise ValueError(msg)

        self.hidden_dim = hidden_dim
        self.side_hidden_dim = side_hidden_dim
        self.lmf_dim = lmf_dim
        self.lmf_rank = lmf_rank
        self.latent_token = latent_token
        self.token_dim = token_dim
        self.summary_dim = summary_dim
        self.macro_decomp_level = macro_decomp_level
        self.mezzo_decomp_level = mezzo_decomp_level
        self.micro_decomp_level = micro_decomp_level
        self.wno_num_blocks = wno_num_blocks
        self.jointnet_23_channels = jointnet_23_channels
        self.jointnet_23_blocks = jointnet_23_blocks
        self.min_scale_S = float(min_scale_S)
        self.min_scale_M = float(min_scale_M)
        self.min_scale_MDD = float(min_scale_MDD)
        self.min_scale_RV = float(min_scale_RV)

        self.macro_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=macro_decomp_level,
            num_blocks=wno_num_blocks,
        )
        self.mezzo_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=mezzo_decomp_level,
            num_blocks=wno_num_blocks,
        )
        self.micro_encoder = WNOEncoder(
            in_channels=7,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=micro_decomp_level,
            num_blocks=wno_num_blocks,
        )
        self.side_encoder = SidechainEncoder(
            in_channels=8,
            hidden_dim=side_hidden_dim,
            lmf_dim=lmf_dim,
        )

        self.pairwise_lmf_12 = PairwiseLMFMap(
            dx=lmf_dim,
            dy=lmf_dim,
            d_out=lmf_dim,
            rank=lmf_rank,
        )
        self.pairwise_lmf_23 = PairwiseLMFMap(
            dx=lmf_dim,
            dy=lmf_dim,
            d_out=lmf_dim,
            rank=lmf_rank,
        )

        self.jointnet_12 = JointNet2D(channels=lmf_dim)
        self.jointnet_23_in_proj = nn.Conv2d(lmf_dim, jointnet_23_channels, kernel_size=1, stride=1)
        self.jointnet_23 = JointNet2D(channels=jointnet_23_channels, num_blocks=jointnet_23_blocks)

        self.map_to_tokens_12 = InteractionMapToTokens(
            latent_token=latent_token,
            dim=lmf_dim,
        )
        self.map_to_tokens_23 = InteractionMapToTokens(
            latent_token=latent_token,
            dim=jointnet_23_channels,
        )
        self.side_resampler = PerceiverResampler(
            latent_token=latent_token,
            input_dim=lmf_dim,
            dim=token_dim,
        )
        self.m1_token_proj = nn.Linear(lmf_dim, token_dim)

        self.drift_fusion = DualCrossAttentionFusion(dim=token_dim, num_layers=2)
        self.diffusion_fusion = SemanticGatedChannelFusion(dim=token_dim)

        self.drift_summary_head = SummaryHead(dim=token_dim, summary_dim=summary_dim)
        self.diffusion_summary_head = SummaryHead(dim=token_dim, summary_dim=summary_dim)
        self.full_tfn = TensorFusion(dim_x=summary_dim, dim_y=summary_dim)
        self.decoder_head = DecoderHead(in_dim=self.full_tfn.out_dim, out_dim=8)

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
        h23 = self.jointnet_23(self.jointnet_23_in_proj(t23))

        m1 = self.m1_token_proj(self.map_to_tokens_12(h12))
        m2 = self.map_to_tokens_23(h23)

        s = self.side_resampler(e4.transpose(1, 2))

        z0 = self.drift_fusion(m1, s)
        z1 = self.diffusion_fusion(m2, s)

        z_d = self.drift_summary_head(z0)
        z_v = self.diffusion_summary_head(z1)
        tfn_feat = self.full_tfn(z_d, z_v)
        head_out = self.decoder_head(tfn_feat)

        pred_S = head_out[:, 0]
        pred_M = head_out[:, 1]
        pred_MDD = head_out[:, 2]
        pred_RV = head_out[:, 3]
        raw_scale_S = head_out[:, 4]
        raw_scale_M = head_out[:, 5]
        raw_scale_MDD = head_out[:, 6]
        raw_scale_RV = head_out[:, 7]

        scale_S = F.softplus(raw_scale_S) + self.min_scale_S
        scale_M = F.softplus(raw_scale_M) + self.min_scale_M
        scale_MDD = F.softplus(raw_scale_MDD) + self.min_scale_MDD
        scale_RV = F.softplus(raw_scale_RV) + self.min_scale_RV

        return {
            "M1": m1,
            "M2": m2,
            "S": s,
            "Z0": z0,
            "Z1": z1,
            "z_d": z_d,
            "z_v": z_v,
            "tfn_feat": tfn_feat,
            "head_out": head_out,
            "pred_S": pred_S,
            "pred_M": pred_M,
            "pred_MDD": pred_MDD,
            "pred_RV": pred_RV,
            "raw_scale_S": raw_scale_S,
            "raw_scale_M": raw_scale_M,
            "raw_scale_MDD": raw_scale_MDD,
            "raw_scale_RV": raw_scale_RV,
            "scale_S": scale_S,
            "scale_M": scale_M,
            "scale_MDD": scale_MDD,
            "scale_RV": scale_RV,
        }


def run_smoke_test(batch_size: int = 2) -> dict[str, object]:
    """Run a minimal forward+loss smoke test."""
    from src.models.losses import LaplaceLSELoss

    model = MultiScaleFusionNet()
    outputs = model(
        torch.randn(batch_size, 9, 64),
        torch.randn(batch_size, 9, 64),
        torch.randn(batch_size, 7, 48),
        torch.randn(batch_size, 8, 64),
    )
    labels = {
        "label_S": torch.randn(batch_size),
        "label_M": torch.randn(batch_size),
        "label_MDD": torch.randn(batch_size),
        "label_RV": torch.randn(batch_size),
    }
    criterion = LaplaceLSELoss()
    loss, _ = criterion(outputs, labels)
    return {
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
