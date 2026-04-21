"""TensorBoard logging for the training stack."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TensorBoardLogger:
    """Log training, debug snapshots, and epoch-level prediction plots."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        task: str,
        enabled: bool = True,
        debug_every: int = 200,
        flush_secs: int = 30,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.task = task
        self.enabled = bool(enabled)
        self.debug_every = int(debug_every)
        self.flush_secs = int(flush_secs)
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

    def should_log_debug(
        self,
        *,
        phase: str,
        global_step: int,
        step: int,
    ) -> bool:
        if not self.enabled:
            return False
        if phase == "train":
            return self.debug_every > 0 and global_step > 0 and global_step % self.debug_every == 0
        return step == 1

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

    def log_debug_snapshot(
        self,
        *,
        phase: str,
        global_step: int,
        output: dict[str, torch.Tensor],
    ) -> None:
        if self.writer is None:
            return

        debug = output.get("_debug", {})
        if isinstance(debug, dict):
            for key, value in debug.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(
                        self._tag(f"debug_{phase}", key),
                        float(value),
                        global_step,
                    )
        self._log_feature_activation_histograms(
            phase=phase,
            global_step=global_step,
            output=output,
        )
        self._log_heads(phase=phase, global_step=global_step, output=output)
        self._log_loss_params(phase=phase, global_step=global_step, output=output)

    def log_prediction_plot(
        self,
        *,
        phase: str,
        global_step: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if self.writer is None or predictions.numel() == 0 or targets.numel() == 0:
            return

        preds = predictions.detach().float().cpu().reshape(-1).numpy()
        trgs = targets.detach().float().cpu().reshape(-1).numpy()
        finite_mask = np.isfinite(preds) & np.isfinite(trgs)
        preds = preds[finite_mask]
        trgs = trgs[finite_mask]
        if preds.size == 0:
            return

        lo = float(min(preds.min(), trgs.min()))
        hi = float(max(preds.max(), trgs.max()))
        if math.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        ax.hexbin(trgs, preds, gridsize=80, bins="log", mincnt=1, cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"{phase} pred vs target ({self.task})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        self.writer.add_figure(self._tag(f"epoch_{phase}", "pred_vs_target"), fig, global_step)
        plt.close(fig)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _log_heads(
        self,
        *,
        phase: str,
        global_step: int,
        output: dict[str, torch.Tensor],
    ) -> None:
        for name in ("pred_primary", "pred_aux_raw"):
            value = output.get(name)
            if isinstance(value, torch.Tensor):
                self._add_basic_stats(
                    self._tag(f"debug_{phase}", f"head_{name}"),
                    value,
                    global_step,
                )
                self.writer.add_histogram(
                    self._tag(f"debug_{phase}", f"head_{name}_hist"),
                    value.detach().float().cpu().reshape(-1),
                    global_step,
                )

        task_specific_name = {
            "ret": ("pred_mu_ret", "pred_scale_ret_raw"),
            "rv": ("pred_mean_rv_raw", "pred_shape_rv_raw"),
            "q": ("pred_mu_q", "pred_scale_q_raw"),
        }[self.task]
        for name in task_specific_name:
            value = output.get(name)
            if isinstance(value, torch.Tensor):
                self._add_basic_stats(
                    self._tag(f"debug_{phase}", f"head_{name}"),
                    value,
                    global_step,
                )
                self.writer.add_histogram(
                    self._tag(f"debug_{phase}", f"head_{name}_hist"),
                    value.detach().float().cpu().reshape(-1),
                    global_step,
                )

    def _log_loss_params(
        self,
        *,
        phase: str,
        global_step: int,
        output: dict[str, torch.Tensor],
    ) -> None:
        sigma = output.get("sigma_pred")
        if isinstance(sigma, torch.Tensor):
            self._add_basic_stats(
                self._tag(f"debug_{phase}", "loss_sigma_pred"),
                sigma,
                global_step,
            )
            self.writer.add_histogram(
                self._tag(f"debug_{phase}", "loss_sigma_pred_hist"),
                sigma.detach().float().cpu().reshape(-1),
                global_step,
            )

        if self.task == "ret":
            nu_ret = output.get("nu_ret")
            if isinstance(nu_ret, torch.Tensor):
                self._add_basic_stats(
                    self._tag(f"debug_{phase}", "loss_nu_ret"),
                    nu_ret,
                    global_step,
                )
        elif self.task == "rv":
            shape_rv = output.get("shape_rv")
            if isinstance(shape_rv, torch.Tensor):
                self._add_basic_stats(
                    self._tag(f"debug_{phase}", "loss_shape_rv"),
                    shape_rv,
                    global_step,
                )

    def _log_feature_activation_histograms(
        self,
        *,
        phase: str,
        global_step: int,
        output: dict[str, torch.Tensor],
    ) -> None:
        feature_sets = (
            ("macro", output.get("feature_rms_macro_pre"), output.get("feature_rms_macro_post")),
            ("mezzo", output.get("feature_rms_mezzo_pre"), output.get("feature_rms_mezzo_post")),
            ("micro", output.get("feature_rms_micro_pre"), output.get("feature_rms_micro_post")),
        )
        for scale, pre, post in feature_sets:
            if not isinstance(pre, torch.Tensor) or not isinstance(post, torch.Tensor):
                continue
            pre_np = pre.detach().float().cpu().reshape(-1).numpy()
            post_np = post.detach().float().cpu().reshape(-1).numpy()
            channels = np.arange(pre_np.size, dtype=np.int64)
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=140)
            width = 0.42
            ax.bar(channels - width / 2.0, pre_np, width=width, label="pre", alpha=0.75)
            ax.bar(channels + width / 2.0, post_np, width=width, label="post", alpha=0.75)
            ax.set_xlabel("feature_channel")
            ax.set_ylabel("activation_rms")
            ax.set_title(f"{phase} {scale} feature activation")
            ax.set_xticks(channels)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper right")
            fig.tight_layout()
            self.writer.add_figure(
                self._tag(f"debug_{phase}", f"feature_{scale}_activation_hist"),
                fig,
                global_step,
            )
            plt.close(fig)

    def _add_basic_stats(
        self,
        tag_prefix: str,
        value: torch.Tensor,
        global_step: int,
    ) -> None:
        tensor = value.detach().float()
        self.writer.add_scalar(f"{tag_prefix}_mean", float(tensor.mean().cpu()), global_step)
        self.writer.add_scalar(f"{tag_prefix}_std", float(tensor.std(unbiased=False).cpu()), global_step)
        self.writer.add_scalar(
            f"{tag_prefix}_abs_mean",
            float(tensor.abs().mean().cpu()),
            global_step,
        )
