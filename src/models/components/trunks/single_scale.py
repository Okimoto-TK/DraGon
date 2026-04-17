"""Single-scale frontend trunk."""
from __future__ import annotations

from torch import Tensor, nn

from src.models.components.encoders.liquidity_state import LiquidityBranch, StateQueryEncoder
from src.models.components.encoders.path_encoder import PathEncoder
from src.models.components.fusion.pairwise import PriceLiquidityDualFusion
from src.models.components.fusion.token_attention import SideToJointWrite, StateJointReader


class ScaleFrontend(nn.Module):
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
        self.liquidity_branch = LiquidityBranch(dim, out_tokens=liquid_tokens, local_kernel=local_kernel)
        self.pv_fusion = PriceLiquidityDualFusion(dim, num_heads=num_heads, joint_tokens=joint_tokens)
        self.state_query_encoder = StateQueryEncoder(dim, kernel_size=3)
        self.side_write = SideToJointWrite(dim, num_heads=num_heads)
        self.state_reader = StateJointReader(dim, num_heads=num_heads)

    def forward(self, scale_x: Tensor, e_d: Tensor) -> dict[str, Tensor]:
        x = scale_x.transpose(1, 2).contiguous()
        xy = x[..., 7:9]
        price_tokens, price_relation_tokens = self.path_branch(scale_x[:, 0:4, :], xy)
        liquid_tokens, liquid_base_sequence = self.liquidity_branch(x[..., 4], x[..., 5], xy)
        price_tokens_dual, liquid_tokens_dual, pv_interaction_tokens, joint_tokens = self.pv_fusion(price_tokens, liquid_tokens)
        hard_state_tokens = self.state_query_encoder(x[..., 6], xy)
        joint_tokens_context = self.side_write(joint_tokens, e_d, hard_state_tokens)
        state_tokens_hard = self.state_reader(hard_state_tokens, joint_tokens_context)
        return {
            "price_tokens": price_tokens,
            "liquid_tokens": liquid_tokens,
            "price_tokens_dual": price_tokens_dual,
            "liquid_tokens_dual": liquid_tokens_dual,
            "joint_tokens": joint_tokens_context,
            "state_tokens": state_tokens_hard,
            "price_relation_tokens": price_relation_tokens,
            "pv_interaction_tokens": pv_interaction_tokens,
            "liquid_base_sequence": liquid_base_sequence,
            "hard_state_tokens": hard_state_tokens,
        }


__all__ = ["ScaleFrontend"]
