"""Single-scale frontend trunk."""
from __future__ import annotations

from torch import Tensor, nn

from src.models.components.encoders.liquidity_state import LiquidityBranch, StateQueryEncoder
from src.models.components.encoders.path_encoder import PathEncoder
from src.models.components.fusion.pairwise import PriceLiquidityDualFusion
from src.models.components.fusion.token_attention import SideWriteIntoJoint, StateQueryJointReader


class SingleScaleFrontend(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        dim: int,
        num_heads: int,
        path_tokens: int,
        liquid_tokens: int,
        joint_tokens: int,
        path_levels: int,
        path_blocks: int,
        local_kernel: int,
    ) -> None:
        super().__init__()
        del seq_len
        self.path_branch = PathEncoder(
            dim,
            levels=path_levels,
            num_blocks=path_blocks,
            local_kernel=local_kernel,
            out_tokens=path_tokens,
            num_heads=num_heads,
        )
        self.liquidity_branch = LiquidityBranch(dim, out_tokens=liquid_tokens)
        self.pv_fusion = PriceLiquidityDualFusion(dim, num_heads=num_heads, joint_tokens=joint_tokens)
        self.state_query_encoder = StateQueryEncoder(dim, kernel_size=3)
        self.side_write = SideWriteIntoJoint(dim, num_heads=num_heads)
        self.state_reader = StateQueryJointReader(dim, num_heads=num_heads)

    def forward(self, scale_x: Tensor, e_d: Tensor) -> dict[str, Tensor]:
        x = scale_x.transpose(1, 2).contiguous()
        xy = x[..., 7:9]
        z_price, path_tokens_19 = self.path_branch(scale_x[:, 0:4, :], xy)
        z_liquid, liquid_base_seq = self.liquidity_branch(x[..., 4], x[..., 5], xy)
        z_price_dual, z_liquid_dual, pair_grid, z_joint = self.pv_fusion(z_price, z_liquid)
        s6_ctx = self.state_query_encoder(x[..., 6], xy)
        z_joint_ctx = self.side_write(z_joint, e_d, s6_ctx)
        z_state = self.state_reader(s6_ctx, z_joint_ctx)
        return {
            "Z_price": z_price,
            "Z_liquid": z_liquid,
            "Z_price_dual": z_price_dual,
            "Z_liquid_dual": z_liquid_dual,
            "Z_joint": z_joint_ctx,
            "Z_state": z_state,
            "path_tokens_19": path_tokens_19,
            "pair_grid": pair_grid,
            "liquid_base_seq": liquid_base_seq,
            "s6_ctx": s6_ctx,
        }


__all__ = ["SingleScaleFrontend"]

