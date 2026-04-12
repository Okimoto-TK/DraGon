"""Top-level multiscale fusion network."""
from __future__ import annotations

from config.config import hidden_dim as DEFAULT_HIDDEN_DIM
from config.config import latent_token as DEFAULT_LATENT_TOKEN
from config.config import lmf_dim as DEFAULT_LMF_DIM
from config.config import lmf_rank as DEFAULT_LMF_RANK
from config.config import macro_decomp_level as DEFAULT_MACRO_DECOMP_LEVEL
from config.config import mezzo_decomp_level as DEFAULT_MEZZO_DECOMP_LEVEL
from config.config import micro_decomp_level as DEFAULT_MICRO_DECOMP_LEVEL
from torch import Tensor, nn

from src.models.components.encoders import SidechainEncoder, WNOEncoder
from src.models.components.fusion import GatedFiLM, PairwiseLMFMap, TokenLMF
from src.models.components.heads import UnifiedHead
from src.models.components.pooling import InteractionMapToTokens
from src.models.components.trunks import JointNet2D


class MultiScaleFusionNet(nn.Module):
    """Fuse multiscale market features into two prediction heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        lmf_dim: int = DEFAULT_LMF_DIM,
        lmf_rank: int = DEFAULT_LMF_RANK,
        latent_token: int = DEFAULT_LATENT_TOKEN,
        macro_decomp_level: int = DEFAULT_MACRO_DECOMP_LEVEL,
        mezzo_decomp_level: int = DEFAULT_MEZZO_DECOMP_LEVEL,
        micro_decomp_level: int = DEFAULT_MICRO_DECOMP_LEVEL,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            msg = f"hidden_dim must be positive, got {hidden_dim}"
            raise ValueError(msg)

        if lmf_dim <= 0:
            msg = f"lmf_dim must be positive, got {lmf_dim}"
            raise ValueError(msg)

        if lmf_rank <= 0:
            msg = f"lmf_rank must be positive, got {lmf_rank}"
            raise ValueError(msg)

        if latent_token <= 0:
            msg = f"latent_token must be positive, got {latent_token}"
            raise ValueError(msg)

        if macro_decomp_level <= 0 or mezzo_decomp_level <= 0 or micro_decomp_level <= 0:
            msg = (
                "decomp levels must be positive, got "
                f"macro={macro_decomp_level}, mezzo={mezzo_decomp_level}, micro={micro_decomp_level}"
            )
            raise ValueError(msg)

        self.hidden_dim = hidden_dim
        self.lmf_dim = lmf_dim
        self.lmf_rank = lmf_rank
        self.latent_token = latent_token
        self.macro_decomp_level = macro_decomp_level
        self.mezzo_decomp_level = mezzo_decomp_level
        self.micro_decomp_level = micro_decomp_level

        self.macro_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=macro_decomp_level,
        )
        self.mezzo_encoder = WNOEncoder(
            in_channels=9,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=mezzo_decomp_level,
        )
        self.micro_encoder = WNOEncoder(
            in_channels=7,
            hidden_dim=hidden_dim,
            lmf_dim=lmf_dim,
            decomp_level=micro_decomp_level,
        )
        self.side_encoder = SidechainEncoder(
            in_channels=8,
            hidden_dim=hidden_dim,
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
        self.jointnet_23 = JointNet2D(channels=lmf_dim)

        self.map_to_tokens_12 = InteractionMapToTokens(
            latent_token=latent_token,
            dim=lmf_dim,
        )
        self.map_to_tokens_23 = InteractionMapToTokens(
            latent_token=latent_token,
            dim=lmf_dim,
        )
        self.side_pool = nn.AdaptiveAvgPool1d(latent_token)

        self.token_lmf_c0 = TokenLMF(
            dx=lmf_dim,
            dy=lmf_dim,
            d_out=lmf_dim,
            rank=lmf_rank,
        )
        self.gated_film_c123 = GatedFiLM(dim=lmf_dim)

        self.head_c0 = UnifiedHead(out_dim=1, dim=lmf_dim)
        self.head_c123 = UnifiedHead(out_dim=3, dim=lmf_dim)

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

        m1 = self.map_to_tokens_12(h12)
        m2 = self.map_to_tokens_23(h23)

        s = self.side_pool(e4).transpose(1, 2)

        z0 = self.token_lmf_c0(m1, s)
        z1 = self.gated_film_c123(m2, s)

        c0 = self.head_c0(z0)
        c123 = self.head_c123(z1)

        return {"c0": c0, "c123": c123}


__all__ = ["MultiScaleFusionNet"]
