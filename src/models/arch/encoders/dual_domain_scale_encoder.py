"""Dual-domain per-scale encoder with time and wavelet branches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.arch.encoders.modern_tcn_film_encoder import ModernTCNFiLMEncoder
from src.models.arch.fusions.within_scale_star_fusion import WithinScaleSTARFusion


class _CrossWriteBlock(nn.Module):
    """Write source tokens into target tokens through gated cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        *,
        _norm_eps: float,
        _gate_floor: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")
        if _gate_floor < 0 or _gate_floor >= 1:
            raise ValueError(
                f"_gate_floor must satisfy 0 <= _gate_floor < 1, got {_gate_floor}."
            )

        self.hidden_dim = int(hidden_dim)
        self._gate_floor = float(_gate_floor)

        self.target_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.source_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.write_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        ffn_dim = int(self.hidden_dim * float(ffn_ratio))
        self.ffn_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if target_tokens.ndim != 3 or source_tokens.ndim != 3:
            raise ValueError(
                "target_tokens and source_tokens must both have shape [B, D, N]."
            )
        if target_tokens.shape[0] != source_tokens.shape[0]:
            raise ValueError(
                "Batch mismatch between target_tokens and source_tokens: "
                f"{target_tokens.shape[0]} vs {source_tokens.shape[0]}."
            )
        if target_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                f"target_tokens hidden_dim mismatch: expected {self.hidden_dim}, got {target_tokens.shape[1]}."
            )
        if source_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                f"source_tokens hidden_dim mismatch: expected {self.hidden_dim}, got {source_tokens.shape[1]}."
            )

        target_seq = target_tokens.transpose(1, 2)
        source_seq = source_tokens.transpose(1, 2)

        q = self.target_norm(target_seq)
        kv = self.source_norm(source_seq)
        delta, _ = self.cross_attn(q, kv, kv, need_weights=False)

        gate = torch.sigmoid(self.write_gate(kv.mean(dim=1)))
        gate = self._gate_floor + (1.0 - self._gate_floor) * gate
        target_seq = target_seq + gate.unsqueeze(1) * delta
        target_seq = target_seq + self.ffn(self.ffn_norm(target_seq))
        return target_seq.transpose(1, 2)


class DualDomainMutualAttentionBlock(nn.Module):
    """Bidirectional scale-local interaction between time and wavelet tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        *,
        _norm_eps: float,
        _gate_floor: float,
    ) -> None:
        super().__init__()
        self.wavelet_to_time = _CrossWriteBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            _norm_eps=_norm_eps,
            _gate_floor=_gate_floor,
        )
        self.time_to_wavelet = _CrossWriteBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            _norm_eps=_norm_eps,
            _gate_floor=_gate_floor,
        )

    def forward(
        self,
        time_tokens: torch.Tensor,
        wavelet_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_tokens = self.wavelet_to_time(time_tokens, wavelet_tokens)
        wavelet_tokens = self.time_to_wavelet(wavelet_tokens, time_tokens)
        return time_tokens, wavelet_tokens


class WaveletCrossBandHybridBlock(nn.Module):
    """Hybrid cross-band block over subband tokens with gated residual updates."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        *,
        _norm_eps: float,
        _gate_floor: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be > 0, got {ffn_ratio}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        if _norm_eps <= 0:
            raise ValueError(f"_norm_eps must be > 0, got {_norm_eps}.")
        if _gate_floor < 0 or _gate_floor >= 1:
            raise ValueError(
                f"_gate_floor must satisfy 0 <= _gate_floor < 1, got {_gate_floor}."
            )

        self.hidden_dim = int(hidden_dim)
        self._gate_floor = float(_gate_floor)
        self.band_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.band_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.band_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        ffn_dim = int(self.hidden_dim * float(ffn_ratio))
        self.ffn_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B, num_bands, hidden_dim, N], got shape={tuple(x.shape)}."
            )
        batch_size, num_bands, hidden_dim, num_tokens = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"x hidden_dim mismatch: expected {self.hidden_dim}, got {hidden_dim}."
            )

        tokens = x.permute(0, 3, 1, 2).reshape(batch_size * num_tokens, num_bands, hidden_dim)
        attn_in = self.band_norm(tokens)
        attn_delta, _ = self.band_attn(attn_in, attn_in, attn_in, need_weights=False)
        attn_gate = torch.sigmoid(self.band_gate(attn_in.mean(dim=1)))
        attn_gate = self._gate_floor + (1.0 - self._gate_floor) * attn_gate
        tokens = tokens + attn_gate.unsqueeze(1) * attn_delta
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.view(batch_size, num_tokens, num_bands, hidden_dim).permute(0, 2, 3, 1)


class WaveletBranchEncoder(nn.Module):
    """Encode wavelet subbands into one wavelet token sequence per scale."""

    def __init__(
        self,
        *,
        num_features: int,
        num_patches: int,
        hidden_dim: int,
        state_vocab_size: int,
        pos_vocab_size: int,
        scale_index: int,
        num_bands: int,
        num_heads: int,
        ffn_ratio: float,
        num_layers: int,
        dropout: float,
        sidechain_features: int,
        _norm_eps: float,
        _band_gate_floor: float,
        _resample_mode: str,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}.")
        if num_patches <= 0:
            raise ValueError(f"num_patches must be > 0, got {num_patches}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if state_vocab_size <= 0:
            raise ValueError(f"state_vocab_size must be > 0, got {state_vocab_size}.")
        if pos_vocab_size <= 0:
            raise ValueError(f"pos_vocab_size must be > 0, got {pos_vocab_size}.")
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}.")
        if _resample_mode not in {"linear", "nearest"}:
            raise ValueError(
                f"_resample_mode must be one of ['linear', 'nearest'], got {_resample_mode!r}."
            )

        self.num_features = int(num_features)
        self.num_patches = int(num_patches)
        self.hidden_dim = int(hidden_dim)
        self.num_bands = int(num_bands)
        self._resample_mode = _resample_mode
        self.sidechain_features = int(sidechain_features)

        self.state_embedding = nn.Embedding(int(state_vocab_size), self.hidden_dim)
        self.pos_embedding = nn.Embedding(int(pos_vocab_size), self.hidden_dim)
        self.scale_embedding = nn.Embedding(3, self.hidden_dim)
        self.band_type_embedding = nn.Embedding(2, self.hidden_dim)
        self.level_embedding = nn.Embedding(self.num_bands, self.hidden_dim)
        self.sidechain_proj = (
            nn.Conv1d(self.sidechain_features, self.hidden_dim, kernel_size=1)
            if self.sidechain_features > 0
            else None
        )
        self.band_projectors = nn.ModuleList(
            [nn.Conv1d(self.num_features, self.hidden_dim, kernel_size=1) for _ in range(self.num_bands)]
        )
        self.blocks = nn.ModuleList(
            [
                WaveletCrossBandHybridBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_band_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=_norm_eps)
        self.band_score = nn.Linear(self.hidden_dim, 1)
        self.register_buffer(
            "_scale_index_tensor",
            torch.tensor(int(scale_index), dtype=torch.long),
            persistent=False,
        )

    def _resize_band(self, coeff: torch.Tensor) -> torch.Tensor:
        if coeff.ndim != 3:
            raise ValueError(
                f"Each wavelet coefficient tensor must have shape [B, F, T], got {tuple(coeff.shape)}."
            )
        if coeff.shape[1] != self.num_features:
            raise ValueError(
                f"Wavelet coefficient feature mismatch: expected {self.num_features}, got {coeff.shape[1]}."
            )
        if self._resample_mode == "linear":
            return F.interpolate(
                coeff,
                size=self.num_patches,
                mode="linear",
                align_corners=False,
            )
        return F.interpolate(coeff, size=self.num_patches, mode="nearest")

    def _condition_tokens(
        self,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        if x_state.ndim != 2 or x_pos.ndim != 2:
            raise ValueError("x_state and x_pos must both have shape [B, T].")
        if x_state.shape != x_pos.shape:
            raise ValueError(
                f"x_state/x_pos shape mismatch: {tuple(x_state.shape)} vs {tuple(x_pos.shape)}."
            )
        state = self.state_embedding(x_state.long())
        pos = self.pos_embedding(x_pos.long())
        cond = (state + pos).transpose(1, 2)
        cond = F.interpolate(cond, size=self.num_patches, mode="linear", align_corners=False)
        return cond

    def forward(
        self,
        wavelet_coeffs: tuple[torch.Tensor, ...],
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
        sidechain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(wavelet_coeffs) != self.num_bands:
            raise ValueError(
                f"wavelet_coeffs band mismatch: expected {self.num_bands}, got {len(wavelet_coeffs)}."
            )

        condition = self._condition_tokens(x_state, x_pos)
        scale_embed = self.scale_embedding(self._scale_index_tensor).view(1, self.hidden_dim, 1)
        sidechain_embed: torch.Tensor | None = None
        if self.sidechain_proj is not None:
            if sidechain is None:
                raise ValueError("sidechain must be provided when sidechain_features > 0.")
            if sidechain.ndim != 3:
                raise ValueError(
                    f"sidechain must have shape [B, F_side, T], got shape={tuple(sidechain.shape)}."
                )
            if sidechain.shape[1] != self.sidechain_features:
                raise ValueError(
                    "sidechain feature mismatch: "
                    f"expected {self.sidechain_features}, got {sidechain.shape[1]}."
                )
            sidechain_embed = self.sidechain_proj(sidechain)
            sidechain_embed = F.interpolate(
                sidechain_embed,
                size=self.num_patches,
                mode="linear",
                align_corners=False,
            )
        elif sidechain is not None:
            raise ValueError(
                "sidechain was provided but this encoder is configured without sidechain_features."
            )

        band_tokens: list[torch.Tensor] = []
        for band_idx, coeff in enumerate(wavelet_coeffs):
            resized = self._resize_band(coeff)
            token = self.band_projectors[band_idx](resized)
            band_type = 0 if band_idx == 0 else 1
            token = token + condition
            token = token + scale_embed
            if sidechain_embed is not None:
                token = token + sidechain_embed
            token = token + self.band_type_embedding.weight[band_type].view(1, self.hidden_dim, 1)
            token = token + self.level_embedding.weight[band_idx].view(1, self.hidden_dim, 1)
            band_tokens.append(token)

        x = torch.stack(band_tokens, dim=1)
        for block in self.blocks:
            x = block(x)

        pooled = x.permute(0, 3, 1, 2)
        pooled = self.output_norm(pooled)
        weights = torch.softmax(self.band_score(pooled), dim=2)
        tokens = (pooled * weights).sum(dim=2)
        return tokens.transpose(1, 2)


class DualDomainScaleEncoder(nn.Module):
    """One scale encoder that returns aligned time and wavelet token sequences."""

    def __init__(
        self,
        *,
        time_encoder_kwargs: dict[str, int | float],
        wavelet_num_features: int,
        scale_index: int,
        time_star_core_dim: int,
        time_star_num_layers: int,
        time_star_dropout: float,
        wavelet_num_heads: int,
        wavelet_ffn_ratio: float,
        wavelet_num_layers: int,
        wavelet_sidechain_features: int,
        mutual_num_heads: int,
        mutual_ffn_ratio: float,
        mutual_num_layers: int,
        dropout: float,
        wavelet_norm_eps: float,
        wavelet_band_gate_floor: float,
        wavelet_resample_mode: str,
        mutual_norm_eps: float,
        mutual_gate_floor: float,
    ) -> None:
        super().__init__()

        self.time_encoder = ModernTCNFiLMEncoder(**time_encoder_kwargs)
        self.time_star = WithinScaleSTARFusion(
            hidden_dim=int(time_encoder_kwargs["hidden_dim"]),
            num_features=int(time_encoder_kwargs["num_features"]),
            core_dim=int(time_star_core_dim),
            num_layers=int(time_star_num_layers),
            dropout=float(time_star_dropout),
        )
        self.wavelet_encoder = WaveletBranchEncoder(
            num_features=int(wavelet_num_features),
            num_patches=self.time_encoder._num_patches,
            hidden_dim=int(time_encoder_kwargs["hidden_dim"]),
            state_vocab_size=int(time_encoder_kwargs["state_vocab_size"]),
            pos_vocab_size=int(time_encoder_kwargs["pos_vocab_size"]),
            scale_index=int(scale_index),
            num_bands=3,
            num_heads=int(wavelet_num_heads),
            ffn_ratio=float(wavelet_ffn_ratio),
            num_layers=int(wavelet_num_layers),
            dropout=float(dropout),
            sidechain_features=int(wavelet_sidechain_features),
            _norm_eps=float(wavelet_norm_eps),
            _band_gate_floor=float(wavelet_band_gate_floor),
            _resample_mode=wavelet_resample_mode,
        )
        self.mutual_blocks = nn.ModuleList(
            [
                DualDomainMutualAttentionBlock(
                    hidden_dim=int(time_encoder_kwargs["hidden_dim"]),
                    num_heads=int(mutual_num_heads),
                    ffn_ratio=float(mutual_ffn_ratio),
                    dropout=float(dropout),
                    _norm_eps=float(mutual_norm_eps),
                    _gate_floor=float(mutual_gate_floor),
                )
                for _ in range(int(mutual_num_layers))
            ]
        )

    def forward(
        self,
        *,
        time_x: torch.Tensor,
        x_state: torch.Tensor,
        x_pos: torch.Tensor,
        wavelet_coeffs: tuple[torch.Tensor, ...],
        sidechain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_feature_tokens = self.time_encoder(time_x, x_state, x_pos)
        _, time_tokens = self.time_star(time_feature_tokens)
        wavelet_tokens = self.wavelet_encoder(
            wavelet_coeffs,
            x_state,
            x_pos,
            sidechain=sidechain,
        )
        for block in self.mutual_blocks:
            time_tokens, wavelet_tokens = block(time_tokens, wavelet_tokens)
        return time_tokens, wavelet_tokens, time_feature_tokens
