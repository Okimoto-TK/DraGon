"""MLflow-based training diagnostics for multiscale fusion models."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from config.config import trend_ema_alpha as DEFAULT_EMA_ALPHA
from torch import Tensor, nn

from src.train.utils import grad_norm

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]

_TASKS = ("S", "M", "MDD", "RV")
_HOOK_TARGETS = {
    "E1": "macro_encoder",
    "E2": "mezzo_encoder",
    "E3": "micro_encoder",
    "E4": "side_encoder",
    "T12": "pairwise_lmf_12",
    "T23": "pairwise_lmf_23",
    "H12": "jointnet_12",
    "H23": "jointnet_23",
}
_REALTIME_KEYS = (
    "loss_total",
    "loss_lse",
    "loss_avg",
    "freeze/fixed_scale_active",
    "freeze/scale_unfrozen",
    "freeze/ema_mae/S",
    "freeze/ema_mae/M",
    "freeze/ema_mae/MDD",
    "freeze/ema_mae/RV",
    "mae/S",
    "mae/M",
    "mae/MDD",
    "mae/RV",
    "loss_task/S",
    "loss_task/M",
    "loss_task/MDD",
    "loss_task/RV",
    "scale_mean/S",
    "scale_mean/M",
    "scale_mean/MDD",
    "scale_mean/RV",
    "scale_floor_hit_rate/S",
    "scale_floor_hit_rate/M",
    "scale_floor_hit_rate/MDD",
    "scale_floor_hit_rate/RV",
    "token/M1_norm",
    "token/M2_norm",
    "token/S_norm",
    "fusion/Z0_norm",
    "fusion/Z1_norm",
    "fusion/drift_delta",
    "fusion/diffusion_delta",
    "summary/z_d_norm",
    "summary/z_v_norm",
    "summary/cos_zd_zv",
    "tfn/feat_norm",
    "decoder/head_out_std",
)
_MODULE_GROUPS = (
    "macro_encoder",
    "mezzo_encoder",
    "micro_encoder",
    "side_encoder",
    "jointnet_12",
    "jointnet_23",
    "side_resampler",
    "drift_fusion",
    "diffusion_fusion",
    "drift_summary_head",
    "diffusion_summary_head",
    "decoder_head",
)


def _to_float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().item())
    return float(value)


def _channelwise_l2_mean(x: Tensor, *, channel_dim: int) -> float:
    moved = torch.movedim(x.detach(), channel_dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    return float(flat.norm(dim=-1).mean().item())


def _tensor_std(x: Tensor) -> float:
    return float(x.detach().float().std(unbiased=False).item())


def _mean_abs_delta(x: Tensor, y: Tensor) -> float:
    return float(torch.mean(torch.abs(x.detach() - y.detach())).item())


def _pooled_cosine(x: Tensor, y: Tensor) -> float:
    x_vec = x.detach().mean(dim=1)
    y_vec = y.detach().mean(dim=1)
    return float(F.cosine_similarity(x_vec, y_vec, dim=-1).mean().item())


def _vector_cosine(x: Tensor, y: Tensor) -> float:
    return float(F.cosine_similarity(x.detach(), y.detach(), dim=-1).mean().item())


def _quantile(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.detach().float(), q).item())


def _parameter_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.detach()
        total += float(torch.sum(value * value).item())
    return math.sqrt(total) if total > 0.0 else 0.0


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _tracking_uri(repo: str | Path) -> str:
    path = Path(repo).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _flatten_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


@dataclass
class DiagnosticsAccumulator:
    """Weighted average accumulator for scalar diagnostics."""

    sums: dict[str, float]
    count: int = 0

    def __init__(self) -> None:
        self.sums = defaultdict(float)
        self.count = 0

    def update(self, metrics: Mapping[str, float], *, weight: int) -> None:
        self.count += int(weight)
        for key, value in metrics.items():
            self.sums[key] += float(value) * weight

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: total / self.count for key, total in self.sums.items()}


class EpochStageBuffer:
    """Epoch-level sampled diagnostics and one structural snapshot."""

    def __init__(self) -> None:
        self.pred = {task: [] for task in _TASKS}
        self.raw_scale = {task: [] for task in _TASKS}
        self.scale = {task: [] for task in _TASKS}
        self.label = {task: [] for task in _TASKS}
        self.z_cos: list[np.ndarray] = []
        self.z_d_norm: list[np.ndarray] = []
        self.z_v_norm: list[np.ndarray] = []
        self.tfn_sum = 0.0
        self.tfn_sq_sum = 0.0
        self.tfn_count = 0
        self.decoder_norm_sums: dict[str, float] = defaultdict(float)
        self.decoder_norm_count = 0
        self.joint_heat_sum: dict[str, np.ndarray] = {}
        self.joint_heat_count: dict[str, int] = defaultdict(int)
        self.snapshot: dict[str, Any] | None = None


class MLflowVisualizer:
    """Collect and track meaningful diagnostics for training and validation."""

    def __init__(
        self,
        *,
        run: Any | None = None,
        experiment: str | None = None,
        repo: str | Path | None = None,
        run_name: str | None = None,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._owns_run = False
        if run is None and experiment is not None:
            if mlflow is None:
                raise ImportError(
                    "mlflow is not installed. Add the 'mlflow' package before creating an MLflow run."
                )
            mlflow.set_tracking_uri(_tracking_uri(repo or "mlruns"))
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_name=run_name)
            self._owns_run = True

        self.run = run
        self.ema_alpha = float(ema_alpha)
        self._ema: dict[str, float] = {}
        self._features: dict[str, Tensor] = {}
        self._handles: list[Any] = []
        self._epoch_buffers: dict[str, EpochStageBuffer] = {}

    def attach(self, model: nn.Module) -> None:
        """Attach forward hooks to collect intermediate features."""
        self.detach()

        def make_hook(name: str):
            def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
                if isinstance(output, Tensor):
                    self._features[name] = output.detach()

            return hook

        for name, path in _HOOK_TARGETS.items():
            module = _module_by_path(model, path)
            self._handles.append(module.register_forward_hook(make_hook(name)))

        if hasattr(model, "diffusion_fusion") and hasattr(model.diffusion_fusion, "side_pool"):
            self._handles.append(model.diffusion_fusion.side_pool.register_forward_hook(make_hook("S_global")))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._features.clear()

    def close(self) -> None:
        if self._owns_run and mlflow is not None:
            try:
                mlflow.end_run()
            finally:
                self._owns_run = False

    def make_accumulator(self) -> DiagnosticsAccumulator:
        return DiagnosticsAccumulator()

    def start_epoch_stage(self, stage: str) -> None:
        self._epoch_buffers[stage] = EpochStageBuffer()

    def update_epoch_buffer(
        self,
        stage: str,
        model: nn.Module,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> None:
        buffer = self._epoch_buffers.setdefault(stage, EpochStageBuffer())
        batch_size = int(outputs["pred_S"].shape[0])

        for task in _TASKS:
            buffer.pred[task].append(outputs[f"pred_{task}"].detach().float().cpu().numpy())
            buffer.raw_scale[task].append(outputs[f"raw_scale_{task}"].detach().float().cpu().numpy())
            buffer.scale[task].append(outputs[f"scale_{task}"].detach().float().cpu().numpy())
            buffer.label[task].append(batch[f"label_{task}"].detach().float().cpu().numpy())

        z_d = outputs["z_d"].detach()
        z_v = outputs["z_v"].detach()
        buffer.z_cos.append(F.cosine_similarity(z_d, z_v, dim=-1).float().cpu().numpy())
        buffer.z_d_norm.append(z_d.norm(dim=-1).float().cpu().numpy())
        buffer.z_v_norm.append(z_v.norm(dim=-1).float().cpu().numpy())

        tfn_feat = outputs["tfn_feat"].detach().float()
        buffer.tfn_sum += float(tfn_feat.sum().item())
        buffer.tfn_sq_sum += float(torch.square(tfn_feat).sum().item())
        buffer.tfn_count += int(tfn_feat.numel())

        decoder_debug = getattr(model.decoder_head, "get_last_debug", None)
        if callable(decoder_debug):
            decoder_tensors = decoder_debug()
            for name, tensor in decoder_tensors.items():
                if tensor is None:
                    continue
                buffer.decoder_norm_sums[name] += float(tensor.norm(dim=-1).mean().item()) * batch_size
            buffer.decoder_norm_count += batch_size

        for name in ("H12", "H23"):
            feature = self._features.get(name)
            if feature is None:
                continue
            heat = feature.detach().float().mean(dim=(0, 1)).cpu().numpy()
            if name in buffer.joint_heat_sum:
                buffer.joint_heat_sum[name] += heat
            else:
                buffer.joint_heat_sum[name] = heat
            buffer.joint_heat_count[name] += 1

    def capture_epoch_snapshot(
        self,
        stage: str,
        model: nn.Module,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> None:
        del batch
        buffer = self._epoch_buffers.setdefault(stage, EpochStageBuffer())

        snapshot: dict[str, Any] = {
            "features": {
                name: _to_cpu_tree(self._features.get(name))
                for name in ("T12", "T23", "H12", "H23")
            },
            "outputs": {
                name: _to_cpu_tree(outputs[name])
                for name in ("M1", "M2", "S", "Z0", "Z1", "z_d", "z_v", "tfn_feat", "head_out")
            },
            "drift_fusion": _to_cpu_tree(model.drift_fusion.get_last_debug())
            if hasattr(model.drift_fusion, "get_last_debug")
            else None,
            "diffusion_fusion": _to_cpu_tree(model.diffusion_fusion.get_last_debug())
            if hasattr(model.diffusion_fusion, "get_last_debug")
            else None,
            "side_resampler": _to_cpu_tree(model.side_resampler.get_last_debug())
            if hasattr(model.side_resampler, "get_last_debug")
            else None,
            "drift_summary": _to_cpu_tree(model.drift_summary_head.get_last_debug())
            if hasattr(model.drift_summary_head, "get_last_debug")
            else None,
            "diffusion_summary": _to_cpu_tree(model.diffusion_summary_head.get_last_debug())
            if hasattr(model.diffusion_summary_head, "get_last_debug")
            else None,
            "decoder": _to_cpu_tree(model.decoder_head.get_last_debug())
            if hasattr(model.decoder_head, "get_last_debug")
            else None,
        }
        buffer.snapshot = snapshot

    def _log_figure(self, stage: str, epoch: int, name: str, fig: Any) -> None:
        if fig is None:
            return
        try:
            if self.run is not None and mlflow is not None and plt is not None:
                mlflow.log_figure(fig, f"{stage}/epoch_{epoch:03d}/{name}.png")
        finally:
            if plt is not None:
                plt.close(fig)

    def _plot_drift_fusion(self, snapshot: Mapping[str, Any]) -> Any:
        if plt is None:
            return None
        debug = snapshot.get("drift_fusion")
        if not isinstance(debug, dict):
            return None
        x_to_y = [tensor for tensor in debug.get("x_to_y_attn", []) if isinstance(tensor, Tensor)]
        y_to_x = [tensor for tensor in debug.get("y_to_x_attn", []) if isinstance(tensor, Tensor)]
        if not x_to_y or not y_to_x:
            return None

        x_heat = x_to_y[-1].mean(dim=(0, 1)).numpy()
        y_heat = y_to_x[-1].mean(dim=(0, 1)).numpy()
        x_entropy = []
        y_entropy = []
        for weights in x_to_y:
            probs = weights.clamp_min(1e-9)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            x_entropy.append(float(entropy.item()))
        for weights in y_to_x:
            probs = weights.clamp_min(1e-9)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            y_entropy.append(float(entropy.item()))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(x_heat, aspect="auto", cmap="viridis")
        axes[0].set_title("X attends to Y")
        axes[1].imshow(y_heat, aspect="auto", cmap="viridis")
        axes[1].set_title("Y attends to X")
        axes[2].plot(range(1, len(x_entropy) + 1), x_entropy, marker="o", label="X->Y")
        axes[2].plot(range(1, len(y_entropy) + 1), y_entropy, marker="o", label="Y->X")
        axes[2].set_title("Attention Entropy")
        axes[2].set_xlabel("Layer")
        axes[2].legend()
        fig.tight_layout()
        return fig

    def _plot_diffusion_fusion(self, snapshot: Mapping[str, Any]) -> Any:
        if plt is None:
            return None
        debug = snapshot.get("diffusion_fusion")
        if not isinstance(debug, dict):
            return None
        term_norms = debug.get("term_norms", [])
        gate_activations = [tensor.reshape(-1).numpy() for tensor in debug.get("gate_activations", []) if isinstance(tensor, Tensor)]
        if not term_norms:
            return None

        layers = range(1, len(term_norms) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for key in ("x", "summary", "product", "difference"):
            axes[0].plot(layers, [float(layer.get(key, 0.0)) for layer in term_norms], marker="o", label=key)
        axes[0].set_title("Interaction Term Norms")
        axes[0].set_xlabel("Layer")
        axes[0].legend()
        if gate_activations:
            gate_values = np.concatenate(gate_activations, axis=0)
            axes[1].hist(gate_values, bins=50, color="tab:orange", alpha=0.85)
        axes[1].set_title("Gate Activation Histogram")
        fig.tight_layout()
        return fig

    def _plot_side_resampler(self, snapshot: Mapping[str, Any]) -> Any:
        if plt is None:
            return None
        debug = snapshot.get("side_resampler")
        if not isinstance(debug, dict):
            return None
        attn = debug.get("cross_attn")
        if not isinstance(attn, Tensor):
            return None

        heat = attn.mean(dim=(0, 1)).numpy()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(heat, aspect="auto", cmap="viridis")
        ax.set_title("Side Resampler Attention")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Latent Query")
        fig.tight_layout()
        return fig

    def _plot_joint_maps(self, stage_buffer: EpochStageBuffer) -> Any:
        if plt is None:
            return None
        if "H12" not in stage_buffer.joint_heat_sum or "H23" not in stage_buffer.joint_heat_sum:
            return None

        h12 = stage_buffer.joint_heat_sum["H12"] / max(stage_buffer.joint_heat_count["H12"], 1)
        h23 = stage_buffer.joint_heat_sum["H23"] / max(stage_buffer.joint_heat_count["H23"], 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(h12, aspect="auto", cmap="magma")
        axes[0].set_title("H12 Epoch Mean Activation")
        axes[1].imshow(h23, aspect="auto", cmap="magma")
        axes[1].set_title("H23 Epoch Mean Activation")
        fig.tight_layout()
        return fig

    def _plot_summary(self, snapshot: Mapping[str, Any], stage_buffer: EpochStageBuffer) -> Any:
        if plt is None:
            return None
        drift = snapshot.get("drift_summary")
        diffusion = snapshot.get("diffusion_summary")
        if not isinstance(drift, dict) or not isinstance(diffusion, dict):
            return None

        drift_layers = [tensor for tensor in drift.get("layer_attn", []) if isinstance(tensor, Tensor)]
        diffusion_layers = [tensor for tensor in diffusion.get("layer_attn", []) if isinstance(tensor, Tensor)]
        if not drift_layers or not diffusion_layers:
            return None

        drift_num_summary = int(drift.get("num_summary_tokens", 2))
        diffusion_num_summary = int(diffusion.get("num_summary_tokens", 2))
        drift_heat = drift_layers[-1][:, :, :drift_num_summary, drift_num_summary:].mean(dim=(0, 1)).numpy()
        diffusion_heat = diffusion_layers[-1][:, :, :diffusion_num_summary, diffusion_num_summary:].mean(dim=(0, 1)).numpy()
        cosine = np.sort(_flatten_chunks(stage_buffer.z_cos))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(drift_heat, aspect="auto", cmap="viridis")
        axes[0].set_title("Drift Summary Attn")
        axes[1].imshow(diffusion_heat, aspect="auto", cmap="viridis")
        axes[1].set_title("Diffusion Summary Attn")
        axes[2].plot(cosine)
        axes[2].set_title("z_d vs z_v Cosine")
        axes[2].set_xlabel("Sorted Sample")
        fig.tight_layout()
        return fig

    def _plot_tfn_decoder(self, stage_buffer: EpochStageBuffer, snapshot: Mapping[str, Any]) -> Any:
        if plt is None:
            return None
        if stage_buffer.tfn_count <= 0:
            return None
        decoder = snapshot.get("decoder")
        decoder_norms = {
            name: stage_buffer.decoder_norm_sums[name] / max(stage_buffer.decoder_norm_count, 1)
            for name in ("hidden1", "hidden2", "out")
            if name in stage_buffer.decoder_norm_sums
        }
        tfn_mean = stage_buffer.tfn_sum / stage_buffer.tfn_count
        tfn_var = max(stage_buffer.tfn_sq_sum / stage_buffer.tfn_count - tfn_mean * tfn_mean, 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(list(decoder_norms.keys()), list(decoder_norms.values()))
        axes[0].set_title("Decoder Layer Norms")
        axes[1].bar(["mean", "var"], [tfn_mean, tfn_var])
        axes[1].set_title("TFN Feature Stats")
        fig.tight_layout()
        return fig

    def _plot_scale_calibration(self, stage_buffer: EpochStageBuffer) -> Any:
        if plt is None:
            return None
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for col, task in enumerate(_TASKS):
            pred = _flatten_chunks(stage_buffer.pred[task])
            label = _flatten_chunks(stage_buffer.label[task])
            scale = _flatten_chunks(stage_buffer.scale[task])
            if pred.size == 0 or label.size == 0 or scale.size == 0:
                continue
            error = np.abs(pred - label)
            sample_count = min(2000, error.size)
            sample_idx = np.linspace(0, error.size - 1, num=sample_count, dtype=np.int32)
            axes[0, col].scatter(scale[sample_idx], error[sample_idx], s=6, alpha=0.25)
            axes[0, col].set_title(f"{task}: error vs scale")
            quantiles = np.quantile(scale, np.linspace(0.0, 1.0, 6))
            bucket_scale = []
            bucket_error = []
            for left, right in zip(quantiles[:-1], quantiles[1:]):
                mask = (scale >= left) & (scale <= right if right == quantiles[-1] else scale < right)
                if not np.any(mask):
                    continue
                bucket_scale.append(float(scale[mask].mean()))
                bucket_error.append(float(error[mask].mean()))
            axes[1, col].plot(bucket_scale, bucket_error, marker="o")
            axes[1, col].set_title(f"{task}: calibration")
        fig.tight_layout()
        return fig

    def log_epoch_diagnostics(self, stage: str, epoch: int) -> None:
        buffer = self._epoch_buffers.get(stage)
        if buffer is None:
            return

        metrics: dict[str, float] = {}
        for task in _TASKS:
            pred = _flatten_chunks(buffer.pred[task])
            raw_scale = _flatten_chunks(buffer.raw_scale[task])
            scale = _flatten_chunks(buffer.scale[task])
            label = _flatten_chunks(buffer.label[task])
            error = np.abs(pred - label)
            if scale.size > 0:
                metrics[f"scale_epoch/mean/{task}"] = float(scale.mean())
                metrics[f"scale_epoch/p10/{task}"] = float(np.quantile(scale, 0.10))
                metrics[f"scale_epoch/p50/{task}"] = float(np.quantile(scale, 0.50))
                metrics[f"scale_epoch/p90/{task}"] = float(np.quantile(scale, 0.90))
                metrics[f"corr/error_scale/{task}"] = _safe_corr(error, scale)
            metrics[f"corr/pred_raw_scale/{task}"] = _safe_corr(pred, raw_scale)

        z_cos = _flatten_chunks(buffer.z_cos)
        z_d_norm = _flatten_chunks(buffer.z_d_norm)
        z_v_norm = _flatten_chunks(buffer.z_v_norm)
        if z_d_norm.size > 0:
            metrics["summary_epoch/z_d_norm_mean"] = float(z_d_norm.mean())
        if z_v_norm.size > 0:
            metrics["summary_epoch/z_v_norm_mean"] = float(z_v_norm.mean())
        if z_cos.size > 0:
            metrics["summary_epoch/cos_zd_zv_mean"] = float(z_cos.mean())

        if buffer.tfn_count > 0:
            tfn_mean = buffer.tfn_sum / buffer.tfn_count
            tfn_var = max(buffer.tfn_sq_sum / buffer.tfn_count - tfn_mean * tfn_mean, 0.0)
            metrics["tfn_epoch/outer_mean"] = float(tfn_mean)
            metrics["tfn_epoch/outer_var"] = float(tfn_var)

        if buffer.decoder_norm_count > 0:
            for name, total in buffer.decoder_norm_sums.items():
                metrics[f"decoder_epoch/{name}_norm"] = float(total / buffer.decoder_norm_count)

        self.track(stage, metrics, epoch=epoch, subset="epoch_diag")

        snapshot = buffer.snapshot
        if snapshot is None:
            return
        self._log_figure(stage, epoch, "drift_cross_attention", self._plot_drift_fusion(snapshot))
        self._log_figure(stage, epoch, "diffusion_semantic_gate", self._plot_diffusion_fusion(snapshot))
        self._log_figure(stage, epoch, "side_resampler_attention", self._plot_side_resampler(snapshot))
        self._log_figure(stage, epoch, "jointnet_activation_maps", self._plot_joint_maps(buffer))
        self._log_figure(stage, epoch, "summary_attention", self._plot_summary(snapshot, buffer))
        self._log_figure(stage, epoch, "tfn_decoder_stats", self._plot_tfn_decoder(buffer, snapshot))
        self._log_figure(stage, epoch, "scale_calibration", self._plot_scale_calibration(buffer))

    def collect_batch_metrics(
        self,
        model: nn.Module,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        loss_metrics: Mapping[str, Tensor | float],
    ) -> dict[str, float]:
        """Collect a rich, selective set of batch diagnostics."""
        metrics: dict[str, float] = {}

        metrics["loss_total"] = _to_float(loss_metrics["loss_total"])
        metrics["loss_lse"] = _to_float(loss_metrics["loss_lse"])
        metrics["loss_avg"] = _to_float(loss_metrics["loss_avg"])
        for key in (
            "freeze/fixed_scale_active",
            "freeze/scale_unfrozen",
            "freeze/train_step_count",
            "freeze/good_steps",
            "freeze/min_freeze_steps",
            "freeze/patience_steps",
            "freeze/ema_mae/S",
            "freeze/ema_mae/M",
            "freeze/ema_mae/MDD",
            "freeze/ema_mae/RV",
            "freeze/s0/S",
            "freeze/s0/M",
            "freeze/s0/MDD",
            "freeze/s0/RV",
        ):
            if key in loss_metrics:
                metrics[key] = _to_float(loss_metrics[key])

        for task in _TASKS:
            pred = outputs[f"pred_{task}"]
            target = batch[f"label_{task}"]
            scale = outputs[f"scale_{task}"]
            min_scale = float(getattr(model, f"min_scale_{task}"))

            metrics[f"mae/{task}"] = _to_float(torch.mean(torch.abs(pred - target)))
            metrics[f"loss_task/{task}"] = _to_float(loss_metrics[f"loss_{task}"])
            metrics[f"pred_mean/{task}"] = _to_float(pred.mean())
            metrics[f"label_mean/{task}"] = _to_float(target.mean())
            metrics[f"scale_mean/{task}"] = _to_float(scale.mean())
            metrics[f"scale_p10/{task}"] = _quantile(scale, 0.10)
            metrics[f"scale_p90/{task}"] = _quantile(scale, 0.90)
            floor_hits = (scale.detach() <= min_scale * 1.2).float().mean()
            metrics[f"scale_floor_hit_rate/{task}"] = _to_float(floor_hits)

        feature_defs = {
            "feat/E1_norm": ("E1", 1),
            "feat/E2_norm": ("E2", 1),
            "feat/E3_norm": ("E3", 1),
            "feat/E4_norm": ("E4", 1),
            "map/T12_norm": ("T12", 1),
            "map/T23_norm": ("T23", 1),
            "map/H12_norm": ("H12", 1),
            "map/H23_norm": ("H23", 1),
        }
        for metric_name, (feature_name, channel_dim) in feature_defs.items():
            feature = self._features.get(feature_name)
            if feature is not None:
                metrics[metric_name] = _channelwise_l2_mean(feature, channel_dim=channel_dim)
                metrics[metric_name.replace("_norm", "_std")] = _tensor_std(feature)

        if "H12" in self._features and "T12" in self._features:
            metrics["map/H12_delta"] = _mean_abs_delta(self._features["H12"], self._features["T12"])
        for name in ("M1", "M2", "S", "Z0", "Z1"):
            tensor = outputs[name]
            metrics[f"token/{name}_norm" if name in {"M1", "M2", "S"} else f"fusion/{name}_norm"] = (
                _channelwise_l2_mean(tensor, channel_dim=2)
            )
            metrics[f"token/{name}_std" if name in {"M1", "M2", "S"} else f"fusion/{name}_std"] = _tensor_std(
                tensor
            )

        metrics["token/cos_M1_S"] = _pooled_cosine(outputs["M1"], outputs["S"])
        metrics["token/cos_M2_S"] = _pooled_cosine(outputs["M2"], outputs["S"])
        metrics["fusion/drift_delta"] = _mean_abs_delta(outputs["Z0"], outputs["M1"])
        metrics["fusion/diffusion_delta"] = _mean_abs_delta(outputs["Z1"], outputs["M2"])

        s_global = self._features.get("S_global")
        if s_global is None and hasattr(model, "diffusion_fusion") and hasattr(model.diffusion_fusion, "side_pool"):
            with torch.no_grad():
                s_global = model.diffusion_fusion.side_pool(outputs["S"])
        if s_global is not None:
            metrics["fusion/cos_M2_Sglobal"] = _vector_cosine(outputs["M2"].detach().mean(dim=1), s_global)

        metrics["summary/z_d_norm"] = _channelwise_l2_mean(outputs["z_d"], channel_dim=1)
        metrics["summary/z_v_norm"] = _channelwise_l2_mean(outputs["z_v"], channel_dim=1)
        metrics["summary/z_d_std"] = _tensor_std(outputs["z_d"])
        metrics["summary/z_v_std"] = _tensor_std(outputs["z_v"])
        metrics["summary/cos_zd_zv"] = _vector_cosine(outputs["z_d"], outputs["z_v"])

        metrics["tfn/feat_norm"] = _channelwise_l2_mean(outputs["tfn_feat"], channel_dim=1)
        metrics["tfn/feat_std"] = _tensor_std(outputs["tfn_feat"])
        metrics["decoder/head_out_norm"] = _channelwise_l2_mean(outputs["head_out"], channel_dim=1)
        metrics["decoder/head_out_std"] = _tensor_std(outputs["head_out"])
        metrics["decoder/pred_block_norm"] = _channelwise_l2_mean(outputs["head_out"][:, :4], channel_dim=1)
        metrics["decoder/scale_block_norm"] = _channelwise_l2_mean(outputs["head_out"][:, 4:], channel_dim=1)

        return metrics

    def collect_low_frequency_metrics(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Collect module gradient / parameter norms and global optimizer state."""
        metrics: dict[str, float] = {}

        metrics["grad_global_norm"] = grad_norm(model.parameters())
        metrics["param_global_norm"] = _parameter_norm(model.parameters())
        if optimizer is not None and optimizer.param_groups:
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        for name in _MODULE_GROUPS:
            module = getattr(model, name, None)
            if module is None:
                continue
            metrics[f"grad/{name}"] = grad_norm(module.parameters())
            metrics[f"param/{name}"] = _parameter_norm(module.parameters())

        return metrics

    def realtime_metrics(
        self,
        stage: str,
        metrics: Mapping[str, float],
        *,
        lr: float | None = None,
        grad_global_norm_value: float | None = None,
        param_global_norm_value: float | None = None,
    ) -> dict[str, float]:
        """Filter and EMA-smooth the most useful step-level metrics."""
        values = dict(metrics)
        if lr is not None:
            values["lr"] = float(lr)
        if grad_global_norm_value is not None:
            values["grad_global_norm"] = float(grad_global_norm_value)
        if param_global_norm_value is not None:
            values["param_global_norm"] = float(param_global_norm_value)

        ema_metrics: dict[str, float] = {}
        for key in _REALTIME_KEYS + ("lr", "grad_global_norm", "param_global_norm"):
            if key not in values:
                continue
            ema_key = f"{stage}:{key}"
            current = float(values[key])
            previous = self._ema.get(ema_key, current)
            updated = self.ema_alpha * current + (1.0 - self.ema_alpha) * previous
            self._ema[ema_key] = updated
            ema_metrics[key] = updated
        return ema_metrics

    def track(
        self,
        stage: str,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        epoch: int | None = None,
        subset: str = "summary",
    ) -> None:
        """Track metrics into MLflow when a run is configured."""
        if self.run is None or mlflow is None:
            return

        metric_step = int(step if step is not None else (epoch if epoch is not None else 0))
        payload = {
            f"{stage}/{subset}/{key}": float(value)
            for key, value in metrics.items()
        }
        mlflow.log_metrics(payload, step=metric_step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.run is None or mlflow is None:
            return
        mlflow.log_params({key: str(value) for key, value in params.items()})


__all__ = ["MLflowVisualizer", "DiagnosticsAccumulator"]
