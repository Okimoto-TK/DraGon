"""Wavelet-domain support-aware bottom-up cross-scale fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_support(
    support: torch.Tensor,
    *,
    expected_tokens: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if support.ndim != 2 or support.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [N, 2], got shape={tuple(support.shape)}."
        )
    if support.shape[0] != expected_tokens:
        raise ValueError(
            f"{name} token count mismatch: expected {expected_tokens}, got {support.shape[0]}."
        )
    if torch.any(support[:, 1] < support[:, 0]):
        raise ValueError(f"{name} contains invalid intervals where end < start.")
    return support.to(device=device, dtype=dtype)


def _support_matrix(
    target_support: torch.Tensor,
    source_support: torch.Tensor,
) -> torch.Tensor:
    target_start = target_support[:, 0].unsqueeze(1)
    target_end = target_support[:, 1].unsqueeze(1)
    source_start = source_support[:, 0].unsqueeze(0)
    source_end = source_support[:, 1].unsqueeze(0)

    overlap = torch.minimum(target_end, source_end) - torch.maximum(target_start, source_start)
    overlap = overlap.clamp_min(0.0)
    weight = overlap / overlap.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    zero_rows = overlap.sum(dim=-1) <= 0
    if zero_rows.any():
        target_center = (target_support[:, 0] + target_support[:, 1]) * 0.5
        source_center = (source_support[:, 0] + source_support[:, 1]) * 0.5
        nearest = torch.argmin(
            (target_center.unsqueeze(1) - source_center.unsqueeze(0)).abs(),
            dim=-1,
        )
        fallback = torch.zeros_like(weight)
        fallback[torch.arange(weight.shape[0], device=weight.device), nearest] = 1.0
        weight = torch.where(zero_rows.unsqueeze(1), fallback, weight)
    return weight


class _SupportWriteBlock(nn.Module):
    """Support-aware residual write from source tokens into target tokens."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_ratio: float,
        dropout: float,
        *,
        _norm_eps: float,
        _gate_floor: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
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

        ffn_dim = int(self.hidden_dim * float(ffn_ratio))
        self.fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.write_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
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
        weight: torch.Tensor,
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
        if target_tokens.shape[1] != self.hidden_dim or source_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                "Both target_tokens and source_tokens must use the configured hidden_dim."
            )
        if weight.shape != (target_tokens.shape[2], source_tokens.shape[2]):
            raise ValueError(
                "support weight shape mismatch: expected "
                f"({target_tokens.shape[2]}, {source_tokens.shape[2]}), got {tuple(weight.shape)}."
            )

        aggregated = torch.einsum("bdn,mn->bdm", source_tokens, weight)
        target_seq = self.target_norm(target_tokens.transpose(1, 2))
        source_seq = self.source_norm(aggregated.transpose(1, 2))
        delta = self.fuse(torch.cat([target_seq, source_seq], dim=-1))
        gate = torch.sigmoid(self.write_gate(source_seq.mean(dim=1)))
        gate = self._gate_floor + (1.0 - self._gate_floor) * gate
        out = target_tokens.transpose(1, 2) + gate.unsqueeze(1) * delta
        out = out + self.ffn(self.ffn_norm(out))
        return out.transpose(1, 2)


class WaveletBottomUpSupportFusion(nn.Module):
    """Wavelet-domain micro -> mezzo -> macro -> mezzo fusion chain."""

    def __init__(
        self,
        hidden_dim: int = 128,
        ffn_ratio: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
        _gate_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.micro_to_mezzo = nn.ModuleList(
            [
                _SupportWriteBlock(
                    hidden_dim=hidden_dim,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.mezzo_to_macro = nn.ModuleList(
            [
                _SupportWriteBlock(
                    hidden_dim=hidden_dim,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.macro_to_mezzo = nn.ModuleList(
            [
                _SupportWriteBlock(
                    hidden_dim=hidden_dim,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    _norm_eps=_norm_eps,
                    _gate_floor=_gate_floor,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        macro_tokens: torch.Tensor,
        mezzo_tokens: torch.Tensor,
        micro_tokens: torch.Tensor,
        *,
        macro_support: torch.Tensor,
        mezzo_support: torch.Tensor,
        micro_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_support = _validate_support(
            macro_support,
            expected_tokens=macro_tokens.shape[2],
            name="macro_support",
            device=macro_tokens.device,
            dtype=macro_tokens.dtype,
        )
        mezzo_support = _validate_support(
            mezzo_support,
            expected_tokens=mezzo_tokens.shape[2],
            name="mezzo_support",
            device=mezzo_tokens.device,
            dtype=mezzo_tokens.dtype,
        )
        micro_support = _validate_support(
            micro_support,
            expected_tokens=micro_tokens.shape[2],
            name="micro_support",
            device=micro_tokens.device,
            dtype=micro_tokens.dtype,
        )
        mezzo_from_micro = _support_matrix(
            mezzo_support,
            micro_support,
        )
        macro_from_mezzo = _support_matrix(
            macro_support,
            mezzo_support,
        )
        mezzo_from_macro = _support_matrix(
            mezzo_support,
            macro_support,
        )

        for block in self.micro_to_mezzo:
            mezzo_tokens = block(mezzo_tokens, micro_tokens, mezzo_from_micro)
        for block in self.mezzo_to_macro:
            macro_tokens = block(macro_tokens, mezzo_tokens, macro_from_mezzo)
        for block in self.macro_to_mezzo:
            mezzo_tokens = block(mezzo_tokens, macro_tokens, mezzo_from_macro)
        return macro_tokens, mezzo_tokens, micro_tokens
