"""ModernTCN-FiLM encoder for single-scale per-feature sequence encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.arch.embeddings import ConditionEmbedding1D
from src.models.arch.layers import AdaLayerNorm1DLast, ChannelFFN1DLast, Patch1D


class ModernTCNFiLMBlock(nn.Module):
    """Single ModernTCN-FiLM block with depthwise temporal conv and channel FFN."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        kernel_size: int,
        ffn_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if cond_dim <= 0:
            raise ValueError(
                f"cond_dim must be > 0, got {cond_dim}. Expected positive integer."
            )
        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be > 0, got {kernel_size}. Expected positive integer."
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd, got {kernel_size}. Expected kernel_size % 2 == 1."
            )
        if ffn_ratio <= 0:
            raise ValueError(
                f"ffn_ratio must be > 0, got {ffn_ratio}. Expected positive value."
            )

        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.kernel_size = int(kernel_size)
        self.ffn_ratio = float(ffn_ratio)
        self.dropout = float(dropout)

        self._groups = self.hidden_dim
        self._in_channels = self.hidden_dim
        self._out_channels = self.hidden_dim
        self._padding = self.kernel_size // 2
        self._eps = 1e-5

        self.ada_norm1 = AdaLayerNorm1DLast(
            self.hidden_dim,
            cond_dim=self.cond_dim,
            eps=self._eps,
        )
        self.temporal_conv = nn.Conv1d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self._padding,
            groups=self._groups,
        )

        self.ada_norm2 = AdaLayerNorm1DLast(
            self.hidden_dim,
            cond_dim=self.cond_dim,
            eps=self._eps,
        )
        self.ffn = ChannelFFN1DLast(
            hidden_dim=self.hidden_dim,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [Bf, N, hidden_dim], got ndim={x.ndim}, shape={tuple(x.shape)}."
            )
        if cond.ndim != 3:
            raise ValueError(
                "cond must have shape [Bf, N, cond_dim], "
                f"got ndim={cond.ndim}, shape={tuple(cond.shape)}."
            )
        if x.shape[0] != cond.shape[0] or x.shape[1] != cond.shape[1]:
            raise ValueError(
                "x/cond batch or patch mismatch: "
                f"x shape={tuple(x.shape)}, cond shape={tuple(cond.shape)}."
            )
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x hidden_dim mismatch: expected {self.hidden_dim}, got {x.shape[-1]}."
            )
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond cond_dim mismatch: expected {self.cond_dim}, got {cond.shape[-1]}."
            )

        x = x.contiguous()
        cond = cond.contiguous()

        temporal_branch = self.ada_norm1(x, cond)
        temporal_branch = temporal_branch.transpose(1, 2).contiguous()
        temporal_branch = self.temporal_conv(temporal_branch)
        temporal_branch = temporal_branch.transpose(1, 2)
        x = x + temporal_branch

        ffn_branch = self.ada_norm2(x, cond)
        ffn_branch = self.ffn(ffn_branch)
        x = x + ffn_branch

        return x


class ModernTCNFiLMEncoder(nn.Module):
    """Single-scale per-feature encoder with FiLM-modulated ModernTCN blocks."""

    def __init__(
        self,
        seq_len: int,
        num_features: int = 9,
        patch_len: int = 8,
        patch_stride: int = 4,
        hidden_dim: int = 128,
        cond_dim: int = 64,
        kernel_size: int = 7,
        ffn_ratio: float = 2.0,
        num_layers: int = 2,
        state_vocab_size: int = 16,
        pos_vocab_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(
                f"seq_len must be > 0, got {seq_len}. Expected positive integer."
            )
        if num_features <= 0:
            raise ValueError(
                f"num_features must be > 0, got {num_features}. Expected positive integer."
            )
        if patch_len <= 0:
            raise ValueError(
                f"patch_len must be > 0, got {patch_len}. Expected positive integer."
            )
        if patch_stride <= 0:
            raise ValueError(
                f"patch_stride must be > 0, got {patch_stride}. Expected positive integer."
            )
        if patch_len < patch_stride:
            raise ValueError(
                "patch_len must be >= patch_stride, "
                f"got patch_len={patch_len}, patch_stride={patch_stride}."
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}. Expected positive integer."
            )
        if cond_dim <= 0:
            raise ValueError(
                f"cond_dim must be > 0, got {cond_dim}. Expected positive integer."
            )
        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be > 0, got {kernel_size}. Expected positive integer."
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd, got {kernel_size}. Expected kernel_size % 2 == 1."
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {num_layers}. Expected positive integer."
            )
        if state_vocab_size <= 0:
            raise ValueError(
                "state_vocab_size must be > 0, "
                f"got {state_vocab_size}. Expected positive integer."
            )
        if pos_vocab_size <= 0:
            raise ValueError(
                f"pos_vocab_size must be > 0, got {pos_vocab_size}. Expected positive integer."
            )

        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.kernel_size = int(kernel_size)
        self.ffn_ratio = float(ffn_ratio)
        self.num_layers = int(num_layers)
        self.state_vocab_size = int(state_vocab_size)
        self.pos_vocab_size = int(pos_vocab_size)
        self.dropout = float(dropout)

        self._expected_seq_len = self.seq_len
        pad_right = self.patch_len - self.patch_stride
        self._num_patches = ((self.seq_len + pad_right - self.patch_len) // self.patch_stride) + 1

        self.patch = Patch1D(
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            hidden_dim=self.hidden_dim,
        )
        self.condition_embedding = ConditionEmbedding1D(
            state_vocab_size=self.state_vocab_size,
            pos_vocab_size=self.pos_vocab_size,
            cond_dim=self.cond_dim,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            num_features=self.num_features,
        )
        self.blocks = nn.ModuleList(
            [
                ModernTCNFiLMBlock(
                    hidden_dim=self.hidden_dim,
                    cond_dim=self.cond_dim,
                    kernel_size=self.kernel_size,
                    ffn_ratio=self.ffn_ratio,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        x_float: torch.Tensor,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        if x_float.ndim != 3:
            raise ValueError(
                "x_float must have shape [B, F, T], "
                f"got ndim={x_float.ndim}, shape={tuple(x_float.shape)}."
            )
        if x_state.ndim != 2:
            raise ValueError(
                f"x_state must have shape [B, T], got ndim={x_state.ndim}, shape={tuple(x_state.shape)}."
            )
        if x_pos.ndim != 2:
            raise ValueError(
                f"x_pos must have shape [B, T], got ndim={x_pos.ndim}, shape={tuple(x_pos.shape)}."
            )

        batch_size, num_features, seq_len = x_float.shape
        if x_state.shape != (batch_size, seq_len):
            raise ValueError(
                "x_state shape mismatch: "
                f"expected ({batch_size}, {seq_len}), got {tuple(x_state.shape)}."
            )
        if x_pos.shape != (batch_size, seq_len):
            raise ValueError(
                "x_pos shape mismatch: "
                f"expected ({batch_size}, {seq_len}), got {tuple(x_pos.shape)}."
            )
        if num_features != self.num_features:
            raise ValueError(
                "x_float feature dimension mismatch: "
                f"expected {self.num_features}, got {num_features}."
            )
        if seq_len != self.seq_len:
            raise ValueError(
                f"x_float sequence length mismatch: expected {self.seq_len}, got {seq_len}."
            )
        ref_param = next(self.parameters())
        if x_float.device != ref_param.device:
            raise ValueError(
                "x_float device mismatch: "
                f"input device={x_float.device}, module device={ref_param.device}."
            )
        allowed_amp_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x_float.dtype != ref_param.dtype and not (
            ref_param.dtype == torch.float32 and x_float.dtype in allowed_amp_dtypes
        ):
            raise ValueError(
                "x_float dtype mismatch: "
                f"input dtype={x_float.dtype}, module dtype={ref_param.dtype}. "
                "Expected matching dtypes, or an AMP/autocast input in "
                "{torch.float16, torch.bfloat16, torch.float32} for float32 modules."
            )
        if x_state.device != ref_param.device:
            raise ValueError(
                "x_state device mismatch: "
                f"input device={x_state.device}, module device={ref_param.device}."
            )
        if x_pos.device != ref_param.device:
            raise ValueError(
                "x_pos device mismatch: "
                f"input device={x_pos.device}, module device={ref_param.device}."
            )

        x = x_float.reshape(batch_size * num_features, 1, seq_len)
        z = self.patch(x).transpose(1, 2).contiguous()
        cond = self.condition_embedding(x_state, x_pos).transpose(1, 2).contiguous()

        for block in self.blocks:
            z = block(z, cond)

        z = z.transpose(1, 2).view(batch_size, num_features, self.hidden_dim, self._num_patches)
        return z
