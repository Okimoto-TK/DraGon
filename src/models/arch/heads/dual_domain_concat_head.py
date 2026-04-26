"""Prediction head over concatenated mezzo-level time and wavelet tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.arch.layers import LayerNorm1dCF
from src.task_labels import (
    canonical_task_label,
    canonical_training_task,
    field_domain,
    is_quantile_task,
    quantile_level,
)

from .task_query_tower import TaskQueryTower


class DualDomainConcatHead(nn.Module):
    """Fuse mezzo time/wavelet tokens and emit either mu or sigma predictions."""

    def __init__(
        self,
        *,
        mode: str,
        field: str,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        _norm_eps: float = 1e-6,
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

        self.mode = canonical_training_task(mode)
        self.field = canonical_task_label(field)
        self.domain = field_domain(self.field)
        self.hidden_dim = int(hidden_dim)
        self._tau = quantile_level(self.field) if is_quantile_task(self.field) else None
        self.concat_norm = LayerNorm1dCF(2 * self.hidden_dim, eps=float(_norm_eps))
        self.concat_proj = nn.Conv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.global_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim, eps=float(_norm_eps)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.readout = TaskQueryTower(
            hidden_dim=self.hidden_dim,
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            _norm_eps=float(_norm_eps),
        )
        if self.mode == "mu":
            self.value_head = self._make_head(float(dropout), float(_norm_eps))
        else:
            feature_dim = 6 if self.domain == "q" else 5 if self.domain == "ret" else 4
            self.mu_encoder = nn.Sequential(
                nn.LayerNorm(feature_dim, eps=float(_norm_eps)),
                nn.Linear(feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(float(dropout)),
            )
            self.field_embed = nn.Parameter(torch.zeros(1, self.hidden_dim))
            self.q_norm = nn.LayerNorm(self.hidden_dim, eps=float(_norm_eps))
            self.kv_norm = nn.LayerNorm(self.hidden_dim, eps=float(_norm_eps))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=int(num_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            ffn_dim = int(self.hidden_dim * float(ffn_ratio))
            self.conf_ffn_norm = nn.LayerNorm(self.hidden_dim, eps=float(_norm_eps))
            self.conf_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(ffn_dim, self.hidden_dim),
                nn.Dropout(float(dropout)),
            )
            self.sigma_head = nn.Sequential(
                nn.LayerNorm(4 * self.hidden_dim, eps=float(_norm_eps)),
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

    def _make_head(self, dropout: float, norm_eps: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=norm_eps),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        mezzo_time_tokens: torch.Tensor,
        mezzo_wavelet_tokens: torch.Tensor,
        *,
        mu_input: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        for name, value in (
            ("mezzo_time_tokens", mezzo_time_tokens),
            ("mezzo_wavelet_tokens", mezzo_wavelet_tokens),
        ):
            if value.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [B, hidden_dim, N], got shape={tuple(value.shape)}."
                )
            if value.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{name} hidden_dim mismatch: expected {self.hidden_dim}, got {value.shape[1]}."
                )
        if mezzo_time_tokens.shape != mezzo_wavelet_tokens.shape:
            raise ValueError(
                "mezzo_time_tokens and mezzo_wavelet_tokens must have identical shapes, "
                f"got {tuple(mezzo_time_tokens.shape)} and {tuple(mezzo_wavelet_tokens.shape)}."
            )

        head_tokens = torch.cat([mezzo_time_tokens, mezzo_wavelet_tokens], dim=1)
        head_tokens = self.concat_proj(self.concat_norm(head_tokens))
        head_context = self.global_fuse(
            torch.cat(
                [
                    mezzo_time_tokens.mean(dim=-1),
                    mezzo_wavelet_tokens.mean(dim=-1),
                ],
                dim=-1,
            )
        )

        out: dict[str, torch.Tensor] = {
            "head_context": head_context,
            "head_tokens": head_tokens,
        }
        if self.mode == "mu":
            if return_debug:
                task_repr, readout_debug = self.readout(
                    head_tokens,
                    head_context,
                    return_debug=True,
                )
            else:
                task_repr = self.readout(head_tokens, head_context)
            mu_raw = self.value_head(task_repr)
            out.update(
                {
                    "mu_raw": mu_raw,
                    "task_repr": task_repr,
                }
            )
            if return_debug:
                out["_debug"] = readout_debug
            return out

        if mu_input is None:
            raise ValueError("mu_input must be provided when mode='sigma'.")
        if mu_input.ndim != 2 or mu_input.shape[1] != 1:
            raise ValueError(
                f"mu_input must have shape [B, 1], got shape={tuple(mu_input.shape)}."
            )

        query_features = self._build_mu_features(mu_input.float())
        q_base = self.mu_encoder(query_features) + self.field_embed
        q0 = q_base.unsqueeze(1)
        latents = head_tokens.transpose(1, 2)
        q = self.q_norm(q0)
        kv = self.kv_norm(latents)
        context_delta, attn_weights = self.cross_attn(
            q,
            kv,
            kv,
            need_weights=True,
            average_attn_weights=False,
        )
        conf = q0 + context_delta
        conf = conf + self.conf_ffn(self.conf_ffn_norm(conf))
        confidence_query = q0.squeeze(1)
        confidence_repr = conf.squeeze(1)
        fused = torch.cat(
            [
                confidence_query,
                confidence_repr,
                confidence_query * confidence_repr,
                (confidence_query - confidence_repr).abs(),
            ],
            dim=-1,
        )
        sigma_raw = self.sigma_head(fused)
        probs = attn_weights.squeeze(2).clamp_min(1e-12)
        conf_attn_entropy_mean = -(probs * probs.log()).sum(dim=-1).mean()
        conf_attn_max_weight_mean = probs.max(dim=-1).values.mean()
        out.update(
            {
                "sigma_raw": sigma_raw,
                "confidence_query": confidence_query,
                "confidence_repr": confidence_repr,
                "conf_attn_entropy_mean": conf_attn_entropy_mean,
                "conf_attn_max_weight_mean": conf_attn_max_weight_mean,
            }
        )
        if return_debug:
            out["_debug"] = {
                "conf_attn_entropy_mean": float(conf_attn_entropy_mean.detach().cpu()),
                "conf_attn_max_weight_mean": float(conf_attn_max_weight_mean.detach().cpu()),
            }
        return out

    def _build_mu_features(self, mu_input: torch.Tensor) -> torch.Tensor:
        if self.domain in {"ret", "q"}:
            features = [
                mu_input,
                mu_input.abs(),
                torch.sign(mu_input),
                torch.log1p(mu_input.abs()),
                mu_input.square(),
            ]
            if self._tau is not None:
                features.append(mu_input.new_full(mu_input.shape, float(self._tau)))
        else:
            mu_safe = mu_input.clamp_min(1e-6)
            features = [
                mu_safe,
                torch.log(mu_safe),
                torch.sqrt(mu_safe),
                mu_safe.square(),
            ]
        return torch.cat(features, dim=1)
