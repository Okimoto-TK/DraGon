"""EfficientViT-style lightweight 2D trunk with multi-scale local + linear attention."""
from __future__ import annotations

from config.config import lmf_dim as DEFAULT_LMF_DIM
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvFFN2d(nn.Module):
    """Depthwise-enhanced Conv-FFN used after fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.norm = LayerNorm2d(channels)
        self.pw_expand = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, bias=True)
        self.dw = nn.Conv2d(
            channels * 4,
            channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 4,
            bias=True,
        )
        self.act = nn.SiLU()
        self.pw_shrink = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw_shrink(x)
        return residual + x


class EfficientViTJointBlock(nn.Module):
    """EfficientViT-style lightweight joint block with local+global fusion."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        num_heads: int = 4,
        head_dim: int = 6,
        ffn_mult: int = 4,
        use_dilated_local: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive, got {ffn_mult}")

        attn_dim = num_heads * head_dim
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = attn_dim

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
        dilation = 2 if use_dilated_local else 1
        padding = 2 if use_dilated_local else 2
        self.local_dw5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5 if not use_dilated_local else 3,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=True,
        )

        # Branch B: lightweight global linear attention
        self.qkv = nn.Conv2d(channels, attn_dim * 3, kernel_size=1, stride=1, bias=True)
        self.q_dw = nn.Conv2d(
            attn_dim,
            attn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=attn_dim,
            bias=True,
        )
        self.k_dw = nn.Conv2d(
            attn_dim,
            attn_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=attn_dim,
            bias=True,
        )
        self.v_dw = nn.Conv2d(
            attn_dim,
            attn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=attn_dim,
            bias=True,
        )
        self.global_proj = nn.Conv2d(attn_dim, channels, kernel_size=1, stride=1, bias=True)

        self.fuse_pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        hidden = channels * ffn_mult
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, bias=True)
        self.ffn_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.ffn_act = nn.SiLU()
        self.ffn_shrink = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
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

        # Global branch with ReLU linear attention
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = self.q_dw(q)
        k = self.k_dw(k)
        v = self.v_dw(v)

        q = F.relu(q)
        k = F.relu(k)

        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # [B, H, N, Dh]
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # [B, H, N, Dh]
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # [B, H, N, Dh]

        kv = torch.matmul(k.transpose(-2, -1), v)  # [B, H, Dh, Dh]
        out = torch.matmul(q, kv)  # [B, H, N, Dh]
        denom = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)).clamp_min_(1e-6)
        out = out / denom
        out = out.transpose(2, 3).contiguous().view(b, self.attn_dim, h, w)
        global_branch = self.global_proj(out)

        fused = self.fuse_pw(local + global_branch)
        x = residual + fused

        # Conv-FFN
        ffn_residual = x
        x = self.ffn_norm(x)
        x = self.ffn_expand(x)
        x = self.ffn_dw(x)
        x = self.ffn_act(x)
        x = self.ffn_shrink(x)
        return ffn_residual + x


class JointNet2D(nn.Module):
    """EfficientViT-style 2D trunk for refining pairwise interaction maps."""

    def __init__(
        self,
        channels: int = DEFAULT_LMF_DIM,
        *,
        num_blocks: int = 2,
        num_heads: int = 4,
        head_dim: int = 6,
        ffn_mult: int = 4,
        use_dilated_local: bool = False,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.channels = channels
        self.use_gradient_checkpoint = bool(use_gradient_checkpoint)
        self.blocks = nn.ModuleList(
            [
                EfficientViTJointBlock(
                    channels=channels,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_mult=ffn_mult,
                    use_dilated_local=use_dilated_local,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
            nn.SiLU(),
        )
        self.post = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        trunk = self.pre(x)
        if self.use_gradient_checkpoint and self.training and x.requires_grad:
            for block in self.blocks:
                trunk = checkpoint(block, trunk, use_reentrant=False)
        else:
            for block in self.blocks:
                trunk = block(trunk)
        trunk = self.post(trunk)
        return x + trunk


__all__ = ["EfficientViTJointBlock", "JointNet2D"]
