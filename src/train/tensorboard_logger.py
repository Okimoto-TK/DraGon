"""TensorBoard logging for the training stack."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import torch
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TensorBoardLogger:
    """Log epoch metrics and prediction heatmaps."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        task: str,
        enabled: bool = True,
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.task = task
        self.enabled = bool(enabled)
        self.flush_secs = int(flush_secs)
        self._pred_heatmap_bins = 128
        self._ret_heatmap_lo = -0.3
        self._ret_heatmap_hi = 0.3
        self._q_heatmap_lo = -0.3
        self._q_heatmap_hi = 0.3
        self._rv_heatmap_lo = 0.0
        self._rv_heatmap_hi = 0.15
        self._uncertainty_hist_bins = 128
        self._pred_heatmap_state: dict[str, dict[str, list[torch.Tensor]]] = {}
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

    def log_epoch_metrics(
        self,
        *,
        phase: str,
        global_step: int,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if self.writer is None:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(self._tag(f"epoch_{phase}", name), value, global_step)
        self.writer.add_scalar(self._tag(f"epoch_{phase}", "epoch_index"), epoch, global_step)

    def log_prediction_plot(
        self,
        *,
        phase: str,
        global_step: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor | None = None,
    ) -> None:
        if self.writer is None:
            return
        self.reset_prediction_state(phase=phase)
        self.update_prediction_state(
            phase=phase,
            predictions=predictions,
            targets=targets,
            uncertainties=uncertainties,
        )
        self.log_epoch_prediction_plot(
            phase=phase,
            global_step=global_step,
        )

    def reset_prediction_state(
        self,
        *,
        phase: str,
    ) -> None:
        self._pred_heatmap_state.pop(phase, None)

    def update_prediction_state(
        self,
        *,
        phase: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor | None = None,
    ) -> None:
        if self.writer is None or predictions.numel() == 0 or targets.numel() == 0:
            return
        # Clone on-device so epoch-long logging state does not keep compiled forward
        # output/storage alive across steps.
        pred = predictions.detach().reshape(-1).clone()
        trg = targets.detach().reshape(-1).clone()
        if uncertainties is None:
            unc = torch.ones_like(pred)
        else:
            unc = uncertainties.detach().reshape(-1).clone()
            if unc.shape != pred.shape:
                raise ValueError(
                    "uncertainties shape mismatch: expected flattened shape "
                    f"{tuple(pred.shape)}, got {tuple(unc.shape)}."
                )

        state = self._pred_heatmap_state.get(phase)
        if state is None:
            state = {
                "predictions": [],
                "targets": [],
                "uncertainties": [],
            }
            self._pred_heatmap_state[phase] = state
        state["predictions"].append(pred)
        state["targets"].append(trg)
        state["uncertainties"].append(unc)

    def log_epoch_prediction_plot(
        self,
        *,
        phase: str,
        global_step: int,
    ) -> None:
        if self.writer is None:
            return
        state = self._pred_heatmap_state.get(phase)
        if state is None:
            return
        if not state["predictions"] or not state["targets"] or not state["uncertainties"]:
            return
        pred = torch.cat(state["predictions"], dim=0).float()
        trg = torch.cat(state["targets"], dim=0).float()
        unc = torch.cat(state["uncertainties"], dim=0).float()
        finite_mask = torch.isfinite(pred) & torch.isfinite(trg) & torch.isfinite(unc)
        if not torch.any(finite_mask):
            self.reset_prediction_state(phase=phase)
            return
        pred = pred[finite_mask]
        trg = trg[finite_mask]
        unc = unc[finite_mask].clamp_min(1e-12)

        lo, hi = self._prediction_bounds(pred, trg)
        threshold_lo, threshold_hi, method = self._uncertainty_thresholds(unc)

        self.writer.add_scalar(
            self._tag(f"epoch_{phase}", "uncertainty_confident_upper"),
            float(threshold_lo),
            global_step,
        )
        self.writer.add_scalar(
            self._tag(f"epoch_{phase}", "uncertainty_unconfident_lower"),
            float(threshold_hi),
            global_step,
        )

        bucket_specs = [
            ("confident", "confident", unc <= threshold_lo),
            (
                "moderately_confident",
                "moderately confident",
                (unc > threshold_lo) & (unc <= threshold_hi),
            ),
            ("unconfident", "unconfident", unc > threshold_hi),
        ]
        for bucket_name, bucket_label, bucket_mask in bucket_specs:
            self.writer.add_scalar(
                self._tag(f"epoch_{phase}", f"{bucket_name}_samples"),
                float(bucket_mask.sum().item()),
                global_step,
            )
            fig = self._build_prediction_heatmap_figure(
                pred=pred[bucket_mask],
                trg=trg[bucket_mask],
                lo=lo,
                hi=hi,
                phase=phase,
                bucket_label=bucket_label,
                bucket_name=bucket_name,
                threshold_lo=float(threshold_lo),
                threshold_hi=float(threshold_hi),
                method=method,
            )
            self.writer.add_figure(
                self._tag(f"epoch_{phase}", f"pred_vs_target_heatmap_{bucket_name}"),
                fig,
                global_step,
            )
            plt.close(fig)
        self.reset_prediction_state(phase=phase)

    def _prediction_bounds(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
    ) -> tuple[float, float]:
        if self.task == "ret":
            lo = float(self._ret_heatmap_lo)
            hi = float(self._ret_heatmap_hi)
        elif self.task == "rv":
            lo = float(self._rv_heatmap_lo)
            hi = float(self._rv_heatmap_hi)
        elif self.task == "q":
            lo = float(self._q_heatmap_lo)
            hi = float(self._q_heatmap_hi)
        else:
            lo = float(torch.minimum(pred.min(), trg.min()).item())
            hi = float(torch.maximum(pred.max(), trg.max()).item())
            if abs(hi - lo) <= 1e-12:
                lo -= 1.0
                hi += 1.0
        return lo, hi

    def _uncertainty_thresholds(
        self,
        unc: torch.Tensor,
    ) -> tuple[float, float, str]:
        log_unc = unc.float().clamp_min(1e-12).log()
        if log_unc.numel() < 8:
            return self._mad_thresholds(log_unc)
        if float(log_unc.std(unbiased=False).item()) < 1e-4:
            return self._mad_thresholds(log_unc)

        hist = torch.histc(
            log_unc,
            bins=self._uncertainty_hist_bins,
            min=float(log_unc.min().item()),
            max=float(log_unc.max().item()),
        )
        if torch.count_nonzero(hist).item() < 3:
            return self._mad_thresholds(log_unc)

        total = hist.sum()
        prob = hist / total.clamp_min(1e-12)
        bin_ids = torch.arange(self._uncertainty_hist_bins, dtype=torch.float32)
        omega = torch.cumsum(prob, dim=0)
        mu = torch.cumsum(prob * bin_ids, dim=0)
        mu_total = mu[-1]

        best_score = None
        best_pair: tuple[int, int] | None = None
        for i in range(self._uncertainty_hist_bins - 2):
            w0 = float(omega[i].item())
            if w0 <= 0.0:
                continue
            mu0 = float(mu[i].item()) / w0
            for j in range(i + 1, self._uncertainty_hist_bins - 1):
                w1 = float((omega[j] - omega[i]).item())
                w2 = float((1.0 - omega[j]).item())
                if w1 <= 0.0 or w2 <= 0.0:
                    continue
                mu1 = float((mu[j] - mu[i]).item()) / w1
                mu2 = float((mu_total - mu[j]).item()) / w2
                score = (
                    w0 * (mu0 - float(mu_total.item())) ** 2
                    + w1 * (mu1 - float(mu_total.item())) ** 2
                    + w2 * (mu2 - float(mu_total.item())) ** 2
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            return self._mad_thresholds(log_unc)

        log_min = float(log_unc.min().item())
        log_max = float(log_unc.max().item())
        bin_width = (log_max - log_min) / float(self._uncertainty_hist_bins)
        if bin_width <= 0.0:
            return self._mad_thresholds(log_unc)
        t1 = log_min + bin_width * float(best_pair[0] + 1)
        t2 = log_min + bin_width * float(best_pair[1] + 1)
        if not t1 < t2:
            return self._mad_thresholds(log_unc)
        return float(torch.exp(torch.tensor(t1)).item()), float(torch.exp(torch.tensor(t2)).item()), "multi_otsu_log_sigma"

    def _mad_thresholds(
        self,
        log_unc: torch.Tensor,
    ) -> tuple[float, float, str]:
        median = log_unc.median()
        mad = (log_unc - median).abs().median()
        robust_sigma = (1.4826 * mad).clamp_min(1e-4)
        t1 = median - 0.75 * robust_sigma
        t2 = median + 0.75 * robust_sigma
        if not float(t1.item()) < float(t2.item()):
            t1 = median - robust_sigma
            t2 = median + robust_sigma
        return float(torch.exp(t1).item()), float(torch.exp(t2).item()), "median_mad_log_sigma"

    def _build_prediction_heatmap_figure(
        self,
        *,
        pred: torch.Tensor,
        trg: torch.Tensor,
        lo: float,
        hi: float,
        phase: str,
        bucket_label: str,
        bucket_name: str,
        threshold_lo: float,
        threshold_hi: float,
        method: str,
    ):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        if pred.numel() == 0 or trg.numel() == 0:
            ax.text(
                0.5,
                0.5,
                f"{bucket_label}\\nno samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            fig.tight_layout()
            return fig

        pred_np = pred.float().clamp(min=lo, max=hi).cpu().numpy()
        trg_np = trg.float().clamp(min=lo, max=hi).cpu().numpy()
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
        ax.set_ylabel("prediction")
        title = f"{phase} pred vs target hexbin density ({self.task})"
        subtitle = self._bucket_subtitle(
            bucket_name=bucket_name,
            bucket_label=bucket_label,
            threshold_lo=threshold_lo,
            threshold_hi=threshold_hi,
            method=method,
            sample_count=int(pred.numel()),
        )
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        return fig

    def _bucket_subtitle(
        self,
        *,
        bucket_name: str,
        bucket_label: str,
        threshold_lo: float,
        threshold_hi: float,
        method: str,
        sample_count: int,
    ) -> str:
        if bucket_name == "confident":
            rule = f"sigma <= {threshold_lo:.4g}"
        elif bucket_name == "moderately_confident":
            rule = f"{threshold_lo:.4g} < sigma <= {threshold_hi:.4g}"
        else:
            rule = f"sigma > {threshold_hi:.4g}"
        return f"{bucket_label} | {rule} | n={sample_count} | {method}"

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
