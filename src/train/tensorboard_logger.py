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
        self._rv_heatmap_lo = 0.0
        self._rv_heatmap_hi = 1.0
        self._pred_heatmap_state: dict[str, dict[str, torch.Tensor]] = {}
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

    def log_step(
        self,
        *,
        phase: str,
        global_step: int,
        losses: dict[str, float],
        lr: float | None = None,
        grad_norm: float | None = None,
        param_norm: float | None = None,
        data_time_ms: float | None = None,
        forward_time_ms: float | None = None,
        backward_time_ms: float | None = None,
        optimizer_time_ms: float | None = None,
        step_time_ms: float | None = None,
        samples_per_sec: float | None = None,
        gpu_mem_alloc_mb: float | None = None,
    ) -> None:
        if self.writer is None:
            return

        for name, value in losses.items():
            self.writer.add_scalar(self._tag(f"step_{phase}", name), value, global_step)

        scalar_fields = {
            self._tag(f"system_{phase}", "lr"): lr,
            self._tag(f"system_{phase}", "grad_norm"): grad_norm,
            self._tag(f"system_{phase}", "param_norm"): param_norm,
            self._tag(f"system_{phase}", "data_time_ms"): data_time_ms,
            self._tag(f"system_{phase}", "forward_time_ms"): forward_time_ms,
            self._tag(f"system_{phase}", "backward_time_ms"): backward_time_ms,
            self._tag(f"system_{phase}", "optimizer_time_ms"): optimizer_time_ms,
            self._tag(f"system_{phase}", "step_time_ms"): step_time_ms,
            self._tag(f"system_{phase}", "samples_per_sec"): samples_per_sec,
            self._tag(f"system_{phase}", "gpu_mem_alloc_mb"): gpu_mem_alloc_mb,
        }
        for tag, value in scalar_fields.items():
            if value is not None:
                self.writer.add_scalar(tag, value, global_step)

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
    ) -> None:
        if self.writer is None:
            return
        self.reset_prediction_state(phase=phase)
        self.update_prediction_state(
            phase=phase,
            predictions=predictions,
            targets=targets,
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
    ) -> None:
        if self.writer is None or predictions.numel() == 0 or targets.numel() == 0:
            return
        pred = predictions.detach().float().reshape(-1)
        trg = targets.detach().float().reshape(-1)
        finite_mask = torch.isfinite(pred) & torch.isfinite(trg)
        if not torch.any(finite_mask):
            return
        pred = pred[finite_mask]
        trg = trg[finite_mask]

        state = self._pred_heatmap_state.get(phase)
        if state is None:
            if self.task == "rv":
                lo = pred.new_tensor(self._rv_heatmap_lo)
                hi = pred.new_tensor(self._rv_heatmap_hi)
            else:
                lo = torch.minimum(pred.min(), trg.min())
                hi = torch.maximum(pred.max(), trg.max())
                if torch.isclose(lo, hi):
                    lo = lo - 1.0
                    hi = hi + 1.0
            state = {
                "counts": torch.zeros(
                    (self._pred_heatmap_bins, self._pred_heatmap_bins),
                    device=pred.device,
                    dtype=torch.float32,
                ),
                "lo": lo,
                "hi": hi,
            }
            self._pred_heatmap_state[phase] = state

        lo = state["lo"]
        hi = state["hi"]
        scale = torch.clamp(hi - lo, min=1e-12)
        pred = pred.clamp(min=lo, max=hi)
        trg = trg.clamp(min=lo, max=hi)

        pred_idx = ((pred - lo) / scale * self._pred_heatmap_bins).to(torch.int64)
        trg_idx = ((trg - lo) / scale * self._pred_heatmap_bins).to(torch.int64)
        pred_idx = pred_idx.clamp_(0, self._pred_heatmap_bins - 1)
        trg_idx = trg_idx.clamp_(0, self._pred_heatmap_bins - 1)
        flat_idx = trg_idx * self._pred_heatmap_bins + pred_idx

        counts = state["counts"].reshape(-1)
        ones = torch.ones_like(flat_idx, dtype=counts.dtype)
        counts.scatter_add_(0, flat_idx, ones)

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
        counts = state["counts"]
        if not torch.any(counts > 0):
            return

        lo = float(state["lo"].detach().cpu())
        hi = float(state["hi"].detach().cpu())
        counts_np = torch.log1p(counts).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        im = ax.imshow(
            counts_np.T,
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            extent=(lo, hi, lo, hi),
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"{phase} pred vs target heatmap ({self.task})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        self.writer.add_figure(
            self._tag(f"epoch_{phase}", "pred_vs_target_heatmap"),
            fig,
            global_step,
        )
        plt.close(fig)
        self.reset_prediction_state(phase=phase)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
