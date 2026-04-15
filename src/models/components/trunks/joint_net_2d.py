"""EfficientViT-style lightweight multi-scale attention trunk for 2D joint maps."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.normalization import LayerNorm2d


def _channelwise_l2_mean(x: Tensor) -> Tensor:
    moved = torch.movedim(x.detach(), 1, -1)
    flat = moved.reshape(-1, moved.shape[-1]).float()
    return flat.norm(dim=-1).mean()


class EfficientViTJointBlock(nn.Module):
    """One EfficientViT-style local+global block without any downsampling."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_mult: int = 4,
        include_dilated_local: bool = False,
        ffn_act: str = "gelu",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive, got {ffn_mult}")
        if ffn_act not in {"gelu", "silu"}:
            raise ValueError(f"ffn_act must be 'gelu' or 'silu', got {ffn_act}")

        resolved_head_dim = head_dim if head_dim is not None else max(1, channels // num_heads)
        if resolved_head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {resolved_head_dim}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = int(resolved_head_dim)
        self.attn_dim = self.num_heads * self.head_dim
        self.include_dilated_local = bool(include_dilated_local)
        self.eps = 1e-6

        self.pre_norm = LayerNorm2d(channels)

        # Branch A: multi-scale local aggregation
        self.local_pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        self.local_dw3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.local_dw5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=channels,
            bias=True,
        )
        self.local_dilated3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=channels,
            bias=True,
        )
        self.local_fuse = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        self.local_out_norm = LayerNorm2d(channels)

        # Branch B: lightweight global linear attention
        self.qkv_pw = nn.Conv2d(channels, self.attn_dim * 3, kernel_size=1, stride=1, bias=True)
        self.q_dw3 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.attn_dim,
            bias=True,
        )
        self.q_dw5 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.attn_dim,
            bias=True,
        )
        self.k_dw3 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.attn_dim,
            bias=True,
        )
        self.k_dw5 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.attn_dim,
            bias=True,
        )
        self.v_dw3 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.attn_dim,
            bias=True,
        )
        self.v_dw5 = nn.Conv2d(
            self.attn_dim,
            self.attn_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.attn_dim,
            bias=True,
        )
        self.global_out = nn.Conv2d(self.attn_dim, channels, kernel_size=1, stride=1, bias=True)
        self.global_branch_norm = LayerNorm2d(channels)

        # Fuse A+B, residual
        self.fuse_pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        self.fusion_res_norm = LayerNorm2d(channels)

        # Conv-FFN
        hidden_dim = channels * ffn_mult
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, stride=1, bias=True)
        self.ffn_dw3 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.ffn_act = nn.GELU() if ffn_act == "gelu" else nn.SiLU()
        self.ffn_shrink = nn.Conv2d(hidden_dim, channels, kernel_size=1, stride=1, bias=True)
        self.output_norm = LayerNorm2d(channels)

    def _linear_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q/k/v: [B, H, N, Dh], ReLU linear attention (no softmax).
        q = F.relu(q)
        k = F.relu(k)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1).clamp_min_(self.eps)
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / denom
        return out

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")

        b, _, h, w = x.shape
        residual = x
        x = self.pre_norm(x)

        # Local branch
        local_seed = self.local_pw(x)
        local = self.local_dw3(local_seed) + self.local_dw5(local_seed)
        if self.include_dilated_local:
            local = local + self.local_dilated3(local_seed)
        local = self.local_out_norm(self.local_fuse(local))

        # Global branch
        qkv = self.qkv_pw(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        q = self.q_dw3(q) + self.q_dw5(q)
        k = self.k_dw3(k) + self.k_dw5(k)
        v = self.v_dw3(v) + self.v_dw5(v)

        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

        attn_out = self._linear_attention(q, k, v)
        attn_out = attn_out.transpose(2, 3).contiguous().view(b, self.attn_dim, h, w)
        global_branch = self.global_branch_norm(self.global_out(attn_out))

        # Fuse + residual
        fusion_residual_out = residual + self.fuse_pw(local + global_branch)
        x = self.fusion_res_norm(fusion_residual_out)

        # Conv-FFN + residual
        ffn_residual = x
        x = self.ffn_norm(x)
        x = self.ffn_expand(x)
        x = self.ffn_dw3(x)
        x = self.ffn_act(x)
        x = self.ffn_shrink(x)
        residual_out = ffn_residual + x
        out = self.output_norm(residual_out)
        if not return_debug:
            return out
        return out, {
            "local_norm": _channelwise_l2_mean(local),
            "global_norm": _channelwise_l2_mean(global_branch),
            "fusion_residual_out_norm": _channelwise_l2_mean(fusion_residual_out),
            "fusion_out_norm": _channelwise_l2_mean(ffn_residual),
            "residual_out_norm": _channelwise_l2_mean(residual_out),
            "out_norm": _channelwise_l2_mean(out),
        }


class JointNet2D(nn.Module):
    """EfficientViT-style 2D trunk for pairwise interaction maps."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        num_blocks: int = 2,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_mult: int | None = None,
        include_dilated_local: bool | None = None,
        ffn_act: str = "gelu",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        # jointnet_12: typically num_blocks=2 -> 4C, no dilated local
        # jointnet_23: typically num_blocks>=3 -> 6C, enable dilated local
        resolved_ffn_mult = ffn_mult if ffn_mult is not None else (6 if num_blocks >= 3 else 4)
        resolved_include_dilated = (
            include_dilated_local if include_dilated_local is not None else bool(num_blocks >= 3)
        )

        self.channels = channels
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
        )
        self.pre_norm = LayerNorm2d(channels)
        self.blocks = nn.ModuleList(
            [
                EfficientViTJointBlock(
                    channels=channels,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_mult=resolved_ffn_mult,
                    include_dilated_local=resolved_include_dilated,
                    ffn_act=ffn_act,
                )
                for _ in range(num_blocks)
            ]
        )
        self.post = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
        )
        self.trunk_norm = LayerNorm2d(channels)

    def forward(self, x: Tensor, *, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")

        trunk = self.pre_norm(self.pre(x))
        if not return_debug:
            for block in self.blocks:
                trunk = block(trunk)
            trunk = self.trunk_norm(self.post(trunk))
            return x + trunk

        debug: dict[str, Tensor] = {
            "pre_norm": _channelwise_l2_mean(trunk),
        }
        for block_idx, block in enumerate(self.blocks, start=1):
            trunk, block_debug = block(trunk, return_debug=True)
            for name, value in block_debug.items():
                debug[f"block{block_idx}_{name}"] = value
        trunk = self.trunk_norm(self.post(trunk))
        out = x + trunk
        debug["trunk_norm"] = _channelwise_l2_mean(trunk)
        debug["out_norm"] = _channelwise_l2_mean(out)
        return out, debug


__all__ = ["EfficientViTJointBlock", "JointNet2D"]
