"""TensorBoard logging for staged mu/sigma training."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.task_labels import canonical_task_label, canonical_training_task, field_domain

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TensorBoardLogger:
    """Log epoch-only training summaries with GPU-resident accumulation."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        task: str,
        field: str = "ret",
        enabled: bool = True,
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.mode = canonical_training_task(task)
        self.field = canonical_task_label(field)
        self.domain = field_domain(self.field)
        self.enabled = bool(enabled)
        self.flush_secs = int(flush_secs)
        self._ret_heatmap_lo = -0.3
        self._ret_heatmap_hi = 0.3
        self._q_heatmap_lo = -0.3
        self._q_heatmap_hi = 0.3
        self._rv_heatmap_lo = 0.0
        self._rv_heatmap_hi = 0.15
        self._phase_state: dict[str, dict[str, object]] = {}
        self.writer: SummaryWriter | None = None
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=self.flush_secs,
            )

    @staticmethod
    def _tag(group: str, name: str) -> str:
        return f"{group}/{name}"

    def _mu_state(self, phase: str) -> dict[str, object]:
        state = self._phase_state.get(phase)
        if state is None:
            state = {
                "predictions": [],
                "targets": [],
            }
            self._phase_state[phase] = state
        return state

    def _sigma_state(self, phase: str, device: torch.device) -> dict[str, object]:
        state = self._phase_state.get(phase)
        if state is None:
            state = {
                "residuals": [],
                "sigmas": [],
                "attn_entropy_sum": torch.zeros((), device=device, dtype=torch.float32),
                "attn_max_sum": torch.zeros((), device=device, dtype=torch.float32),
                "sample_count": torch.zeros((), device=device, dtype=torch.float32),
            }
            self._phase_state[phase] = state
        return state

    def update_mu_state(
        self,
        *,
        phase: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if self.writer is None or predictions.numel() == 0 or targets.numel() == 0:
            return
        state = self._mu_state(phase)
        # Clone on device so we do not keep compiled graph storage alive across steps.
        state["predictions"].append(predictions.detach().reshape(-1).float().clone())
        state["targets"].append(targets.detach().reshape(-1).float().clone())

    def update_sigma_state(
        self,
        *,
        phase: str,
        mu_input: torch.Tensor,
        targets: torch.Tensor,
        sigmas: torch.Tensor,
        attn_entropy: torch.Tensor,
        attn_max_weight: torch.Tensor,
    ) -> None:
        if self.writer is None or targets.numel() == 0 or sigmas.numel() == 0:
            return
        sigma_flat = sigmas.detach().reshape(-1).float().clone()
        residual_flat = (
            targets.detach().reshape(-1).float() - mu_input.detach().reshape(-1).float()
        ).abs().clone()
        state = self._sigma_state(phase, sigma_flat.device)
        state["residuals"].append(residual_flat)
        state["sigmas"].append(sigma_flat)
        sample_count = sigma_flat.new_tensor(float(sigma_flat.numel()))
        state["attn_entropy_sum"] = state["attn_entropy_sum"] + attn_entropy.detach().float() * sample_count
        state["attn_max_sum"] = state["attn_max_sum"] + attn_max_weight.detach().float() * sample_count
        state["sample_count"] = state["sample_count"] + sample_count

    def log_epoch_metrics(
        self,
        *,
        phase: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if self.writer is None:
            return
        epoch_step = int(epoch) + 1
        if self.mode == "mu":
            self.writer.add_scalar(
                self._tag(f"epoch_{phase}", "loss_mu"),
                float(metrics.get("loss_mu", metrics["loss_total"])),
                epoch_step,
            )
            return

        self.writer.add_scalar(
            self._tag(f"epoch_{phase}", "loss_nll"),
            float(metrics.get("loss_nll", metrics["loss_total"])),
            epoch_step,
        )
        state = self._phase_state.get(phase)
        if state is None:
            return
        sample_count = state.get("sample_count")
        attn_entropy_sum = state.get("attn_entropy_sum")
        attn_max_sum = state.get("attn_max_sum")
        if (
            isinstance(sample_count, torch.Tensor)
            and isinstance(attn_entropy_sum, torch.Tensor)
            and isinstance(attn_max_sum, torch.Tensor)
            and float(sample_count.detach().cpu()) > 0.0
        ):
            denom = sample_count.detach().float().cpu()
            self.writer.add_scalar(
                self._tag(f"epoch_{phase}", "conf_attn_entropy_mean"),
                float((attn_entropy_sum.detach().cpu() / denom).item()),
                epoch_step,
            )
            self.writer.add_scalar(
                self._tag(f"epoch_{phase}", "conf_attn_max_weight_mean"),
                float((attn_max_sum.detach().cpu() / denom).item()),
                epoch_step,
            )

    def log_epoch_plots(
        self,
        *,
        phase: str,
        epoch: int,
    ) -> None:
        if self.writer is None:
            return
        epoch_step = int(epoch) + 1
        state = self._phase_state.get(phase)
        if state is None:
            return

        if self.mode == "mu":
            predictions = state.get("predictions")
            targets = state.get("targets")
            if not isinstance(predictions, list) or not isinstance(targets, list) or not predictions or not targets:
                self._phase_state.pop(phase, None)
                return
            pred = torch.cat(predictions, dim=0)
            trg = torch.cat(targets, dim=0)
            finite_mask = torch.isfinite(pred) & torch.isfinite(trg)
            if torch.any(finite_mask):
                fig = self._build_mu_prediction_figure(
                    phase=phase,
                    predictions=pred[finite_mask],
                    targets=trg[finite_mask],
                )
                self.writer.add_figure(
                    self._tag(f"epoch_{phase}", "mu_pred_vs_target_hexbin"),
                    fig,
                    epoch_step,
                )
                plt.close(fig)
            self._phase_state.pop(phase, None)
            return

        residuals = state.get("residuals")
        sigmas = state.get("sigmas")
        if not isinstance(residuals, list) or not isinstance(sigmas, list) or not residuals or not sigmas:
            self._phase_state.pop(phase, None)
            return
        residual = torch.cat(residuals, dim=0)
        sigma = torch.cat(sigmas, dim=0)
        finite_mask = torch.isfinite(residual) & torch.isfinite(sigma) & (sigma > 0.0)
        if torch.any(finite_mask):
            fig = self._build_sigma_density_figure(
                phase=phase,
                residuals=residual[finite_mask],
                sigmas=sigma[finite_mask],
            )
            self.writer.add_figure(
                self._tag(f"epoch_{phase}", "residual_vs_sigma_hexbin"),
                fig,
                epoch_step,
            )
            plt.close(fig)
        self._phase_state.pop(phase, None)

    def _mu_bounds(self) -> tuple[float, float]:
        if self.domain == "ret":
            return float(self._ret_heatmap_lo), float(self._ret_heatmap_hi)
        if self.domain == "rv":
            return float(self._rv_heatmap_lo), float(self._rv_heatmap_hi)
        return float(self._q_heatmap_lo), float(self._q_heatmap_hi)

    def _build_mu_prediction_figure(
        self,
        *,
        phase: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        lo, hi = self._mu_bounds()
        pred_np = predictions.float().clamp(min=lo, max=hi).cpu().numpy()
        trg_np = targets.float().clamp(min=lo, max=hi).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        im = ax.hexbin(
            trg_np,
            pred_np,
            gridsize=48,
            extent=(lo, hi, lo, hi),
            cmap="viridis",
            bins="log",
            mincnt=1,
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0)
        ax.set_xlabel("target")
        ax.set_ylabel("mu_pred")
        ax.set_title(f"{phase} mu pred vs target hexbin density ({self.field})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        return fig

    def _build_sigma_density_figure(
        self,
        *,
        phase: str,
        residuals: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        residual_np = residuals.float().cpu().numpy()
        sigma_np = sigmas.float().cpu().numpy()
        x_hi = _robust_upper(sigma_np)
        y_hi = _robust_upper(residual_np)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        im = ax.hexbin(
            sigma_np,
            residual_np,
            gridsize=48,
            extent=(0.0, x_hi, 0.0, y_hi),
            cmap="viridis",
            bins="log",
            mincnt=1,
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_xlabel("sigma_pred")
        ax.set_ylabel("|target - mu_hat|")
        ax.set_title(f"{phase} residual vs sigma hexbin density ({self.field})")
        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(0.0, y_hi)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        return fig

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        self._phase_state.clear()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def _robust_upper(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    upper = float(np.quantile(values, 0.995))
    max_value = float(values.max())
    return max(upper, max_value * 0.25, 1e-6)
