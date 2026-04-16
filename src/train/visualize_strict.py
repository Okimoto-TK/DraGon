"""Strict GPU-first MLflow diagnostics for training and validation."""
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

from src.task_labels import detect_task_from_outputs
from src.task_labels import task_target_column
from src.train.utils import grad_norm

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

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
    "loss_task",
    "loss_mu",
    "loss_unc",
    "loss_task/ret",
    "loss_task/rv",
    "loss_task/p90",
    "mae/ret",
    "mae/rv",
    "mae/p90",
    "unc_mean/ret",
    "unc_mean/rv",
    "unc_mean/p90",
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
_DYNAMIC_REALTIME_PREFIXES = (
    "loss_task/",
    "mae/",
    "unc_mean/",
    "encoder/",
    "map/",
    "fusion/drift_layer",
    "fusion/diffusion_block",
    "summary/drift_block",
    "summary/diffusion_block",
)
_DYNAMIC_REALTIME_EXACT = {
    "fusion/side_global_norm",
    "fusion/drift_fused_tokens_norm",
    "summary/drift_vec_norm",
    "summary/diffusion_vec_norm",
}
_ARTIFACT_SAMPLE_LIMIT = 2048


def _tracking_uri(repo: str | Path) -> str:
    path = Path(repo).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


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


def _has_tensor_tree(value: Any) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_has_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_tensor_tree(item) for item in value)
    return False


def _as_scalar_tensor(value: Tensor | float | int, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.detach()
        if tensor.ndim != 0:
            tensor = tensor.mean()
        return tensor.to(device=device, dtype=torch.float32)
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _channelwise_l2_mean(x: Tensor, *, channel_dim: int) -> Tensor:
    moved = torch.movedim(x.detach(), channel_dim, -1)
    flat = moved.reshape(-1, moved.shape[-1]).float()
    return flat.norm(dim=-1).mean()


def _tensor_std(x: Tensor) -> Tensor:
    return x.detach().float().std(unbiased=False)


def _mean_abs_delta(x: Tensor, y: Tensor) -> Tensor:
    return torch.mean(torch.abs(x.detach() - y.detach()).float())


def _pooled_cosine(x: Tensor, y: Tensor) -> Tensor:
    x_vec = x.detach().mean(dim=1)
    y_vec = y.detach().mean(dim=1)
    return F.cosine_similarity(x_vec, y_vec, dim=-1).mean()


def _vector_cosine(x: Tensor, y: Tensor) -> Tensor:
    return F.cosine_similarity(x.detach(), y.detach(), dim=-1).mean()


def _parameter_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.detach()
        total += float(torch.sum(value * value).item())
    return math.sqrt(total) if total > 0.0 else 0.0


def _sync_scalar_metrics(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics.keys())
    packed = torch.stack([metrics[key].reshape(()) for key in keys], dim=0)
    values = packed.detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _mlflow_metric_key(stage: str, subset: str, key: str) -> str:
    """Flatten metric names so MLflow file-store never mixes files and directories."""
    return ".".join((stage, subset, key)).replace("/", ".")


def _extract_layer_series(metrics: Mapping[str, float], prefix: str, suffix: str) -> tuple[list[int], list[float]]:
    layers: list[int] = []
    values: list[float] = []
    layer_idx = 1
    while True:
        key = f"{prefix}{layer_idx}_{suffix}"
        if key not in metrics:
            break
        layers.append(layer_idx)
        values.append(float(metrics[key]))
        layer_idx += 1
    return layers, values


def _sample_arrays(values: Mapping[str, list[float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(series, dtype=np.float32) for key, series in values.items()}


def _quantile_curve(x: np.ndarray, y: np.ndarray, *, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty
    quantiles = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    if np.allclose(quantiles[0], quantiles[-1]):
        return np.asarray([float(x.mean())], dtype=np.float32), np.asarray([float(y.mean())], dtype=np.float32)
    xs: list[float] = []
    ys: list[float] = []
    for left, right in zip(quantiles[:-1], quantiles[1:]):
        if right <= left:
            continue
        mask = (x >= left) & (x <= right if right == quantiles[-1] else x < right)
        if not np.any(mask):
            continue
        xs.append(float(x[mask].mean()))
        ys.append(float(y[mask].mean()))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _reliability_curve(pred: np.ndarray, label: np.ndarray, *, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if pred.size == 0 or label.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    xs: list[float] = []
    ys: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (pred >= left) & (pred <= right if right == edges[-1] else pred < right)
        if not np.any(mask):
            continue
        xs.append(float(pred[mask].mean()))
        ys.append(float(label[mask].mean()))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


@dataclass
class DiagnosticsAccumulator:
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
    def __init__(self, *, sample_limit: int = _ARTIFACT_SAMPLE_LIMIT) -> None:
        self.metrics_sum: dict[str, Tensor] = {}
        self.weight: Tensor | None = None
        self.heatmap_sum: dict[str, Tensor] = {}
        self.heatmap_count: dict[str, Tensor] = {}
        self.snapshot: dict[str, Any] | None = None
        self.sample_limit = max(1, int(sample_limit))
        self.sample_task: str | None = None
        self.sample_seen = 0
        self.sample_rng = np.random.default_rng(0)
        self.sample_values: dict[str, list[float]] = {
            "pred": [],
            "label": [],
            "unc": [],
        }

    def update(self, metrics: Mapping[str, Tensor], *, weight: int) -> None:
        if not metrics:
            return
        device = next(iter(metrics.values())).device
        if self.weight is None:
            self.weight = torch.zeros((), device=device, dtype=torch.int64)
        weight_i64 = torch.tensor(int(weight), device=device, dtype=torch.int64)
        weight_fp = weight_i64.to(dtype=torch.float32)
        self.weight.add_(weight_i64)
        for key, value in metrics.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = torch.zeros((), device=device, dtype=torch.float32)
            self.metrics_sum[key].add_(value.detach().float() * weight_fp)

    def add_task_samples(self, task: str, pred: Tensor, label: Tensor, unc: Tensor) -> None:
        if self.sample_task is None:
            self.sample_task = task
        if self.sample_task != task:
            return

        pred_np = pred.detach().float().reshape(-1).cpu().numpy()
        label_np = label.detach().float().reshape(-1).cpu().numpy()
        unc_np = unc.detach().float().reshape(-1).cpu().numpy()

        for pred_value, label_value, unc_value in zip(pred_np, label_np, unc_np):
            self.sample_seen += 1
            if len(self.sample_values["pred"]) < self.sample_limit:
                self.sample_values["pred"].append(float(pred_value))
                self.sample_values["label"].append(float(label_value))
                self.sample_values["unc"].append(float(unc_value))
                continue
            slot = int(self.sample_rng.integers(0, self.sample_seen))
            if slot >= self.sample_limit:
                continue
            self.sample_values["pred"][slot] = float(pred_value)
            self.sample_values["label"][slot] = float(label_value)
            self.sample_values["unc"][slot] = float(unc_value)

    def sample_arrays(self) -> dict[str, np.ndarray]:
        return _sample_arrays(self.sample_values)

    def compute(self) -> dict[str, float]:
        if self.weight is None or bool((self.weight <= 0).item()):
            return {}
        denom = self.weight.to(dtype=torch.float32).clamp_min(1.0)
        means = {key: total / denom for key, total in self.metrics_sum.items()}
        return _sync_scalar_metrics(means)


class MLflowVisualizer:
    def __init__(
        self,
        *,
        run: Any | None = None,
        experiment: str | None = None,
        repo: str | Path | None = None,
        run_name: str | None = None,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        self._owns_run = False
        if run is None and experiment is not None:
            if mlflow is None:
                raise ImportError("mlflow is not installed.")
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
        self.detach()

        def make_hook(name: str):
            def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
                if isinstance(output, Tensor):
                    self._features[name] = output.detach()

            return hook

        for name, path in _HOOK_TARGETS.items():
            module = _module_by_path(model, path)
            self._handles.append(module.register_forward_hook(make_hook(name)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._features.clear()

    def close(self) -> None:
        if self._owns_run and mlflow is not None:
            mlflow.end_run()
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
        metrics: Mapping[str, Tensor] | None = None,
    ) -> None:
        buffer = self._epoch_buffers.setdefault(stage, EpochStageBuffer())
        lightweight = dict(metrics) if metrics is not None else self.collect_batch_metrics(model, outputs, batch, {})
        buffer.update(lightweight, weight=int(batch["macro"].shape[0]))

        task = detect_task_from_outputs(outputs)
        target = batch[task_target_column(task)]
        buffer.add_task_samples(task, outputs[f"pred_{task}"], target, outputs[f"unc_{task}"])

        for name in ("H12", "H23"):
            feature = outputs.get(name)
            if not isinstance(feature, Tensor):
                feature = self._features.get(name)
            if feature is None:
                continue
            heat = feature.detach().float().mean(dim=(0, 1))
            if name not in buffer.heatmap_sum:
                buffer.heatmap_sum[name] = torch.zeros_like(heat)
                buffer.heatmap_count[name] = torch.zeros((), device=heat.device, dtype=torch.int64)
            buffer.heatmap_sum[name].add_(heat)
            buffer.heatmap_count[name].add_(1)

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
                name: _to_cpu_tree(outputs[name]) if name in outputs else _to_cpu_tree(self._features.get(name))
                for name in ("T12", "T23", "H12", "H23")
            },
            "outputs": {
                name: _to_cpu_tree(outputs[name])
                for name in ("M1", "M2", "S", "Z0", "Z1", "z_d", "z_v", "tfn_feat", "head_out")
                if name in outputs
            },
        }
        if stage != "train":
            for attr, key in (
                ("drift_fusion", "drift_fusion"),
                ("diffusion_fusion", "diffusion_fusion"),
                ("side_resampler", "side_resampler"),
                ("drift_summary_head", "drift_summary"),
                ("diffusion_summary_head", "diffusion_summary"),
                ("decoder_head", "decoder"),
            ):
                module = getattr(model, attr, None)
                getter = getattr(module, "get_last_debug", None)
                if callable(getter):
                    debug = _to_cpu_tree(getter())
                    if _has_tensor_tree(debug):
                        snapshot[key] = debug
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

    def _plot_joint_maps(self, stage_buffer: EpochStageBuffer) -> Any:
        if plt is None or not stage_buffer.heatmap_sum:
            return None

        names = [name for name in ("H12", "H23") if name in stage_buffer.heatmap_sum]
        if not names:
            return None

        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
        if len(names) == 1:
            axes = [axes]
        for ax, name in zip(axes, names):
            count = stage_buffer.heatmap_count[name].clamp_min(1).to(dtype=torch.float32)
            mean_map = (stage_buffer.heatmap_sum[name] / count).detach().float().cpu().numpy()
            image = ax.imshow(mean_map, aspect="auto", cmap="magma")
            ax.set_title(f"{name} Epoch Mean Activation")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _plot_feature_norm_flow(self, metrics: Mapping[str, float]) -> Any:
        if plt is None or not metrics:
            return None

        groups = [
            ("Encoders", ("encoder/E1_norm", "encoder/E2_norm", "encoder/E3_norm", "encoder/E4_norm")),
            ("Maps", ("map/T12_norm", "map/H12_norm", "map/T23_norm", "map/H23_norm")),
            ("Tokens/Fusion", ("token/M1_norm", "token/M2_norm", "token/S_norm", "fusion/Z0_norm", "fusion/Z1_norm")),
            ("Summary/Head", ("summary/z_d_norm", "summary/z_v_norm", "tfn/feat_norm", "decoder/head_out_std")),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.reshape(-1)
        rendered = 0
        for ax, (title, keys) in zip(axes, groups):
            available = [(key.split("/")[-1], float(metrics[key])) for key in keys if key in metrics]
            if not available:
                ax.axis("off")
                continue
            labels, values = zip(*available)
            ax.bar(range(len(values)), values, color="tab:blue", alpha=0.85)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title(title)
            rendered += 1
        if rendered == 0:
            plt.close(fig)
            return None
        fig.tight_layout()
        return fig

    def _plot_fusion_internal_diagnostics(self, metrics: Mapping[str, float]) -> Any:
        if plt is None or not metrics:
            return None

        term_suffixes = ("x_norm", "summary_norm", "product_norm", "difference_norm")
        residual_suffixes = ("residual_out_norm", "out_norm", "gate_norm")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        rendered = 0
        layers, _ = _extract_layer_series(metrics, "fusion/diffusion_block", "x_norm")
        if layers:
            for suffix in term_suffixes:
                _, values = _extract_layer_series(metrics, "fusion/diffusion_block", suffix)
                axes[0].plot(layers, values, marker="o", label=suffix.removesuffix("_norm"))
            axes[0].set_title("Diffusion Term Norms")
            axes[0].set_xlabel("Block")
            axes[0].legend()
            rendered += 1
        else:
            axes[0].axis("off")

        if layers:
            for suffix in residual_suffixes:
                _, values = _extract_layer_series(metrics, "fusion/diffusion_block", suffix)
                axes[1].plot(layers, values, marker="o", label=suffix.removesuffix("_norm"))
            axes[1].set_title("Diffusion Residual/Gate")
            axes[1].set_xlabel("Block")
            axes[1].legend()
            rendered += 1
        else:
            axes[1].axis("off")

        drift_layers, drift_x = _extract_layer_series(metrics, "fusion/drift_layer", "x_norm")
        _, drift_y = _extract_layer_series(metrics, "fusion/drift_layer", "y_norm")
        if drift_layers:
            axes[2].plot(drift_layers, drift_x, marker="o", label="x")
            axes[2].plot(drift_layers, drift_y, marker="o", label="y")
            if "fusion/drift_fused_tokens_norm" in metrics:
                axes[2].axhline(
                    float(metrics["fusion/drift_fused_tokens_norm"]),
                    color="tab:green",
                    linestyle="--",
                    label="fused",
                )
            axes[2].set_title("Drift Fusion Flow")
            axes[2].set_xlabel("Layer")
            axes[2].legend()
            rendered += 1
        else:
            axes[2].axis("off")

        if rendered == 0:
            plt.close(fig)
            return None
        fig.tight_layout()
        return fig

    def _plot_summary_head_diagnostics(self, metrics: Mapping[str, float]) -> Any:
        if plt is None or not metrics:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        rendered = 0

        drift_layers, drift_tokens = _extract_layer_series(metrics, "summary/drift_block", "tokens_norm")
        diff_layers, diff_tokens = _extract_layer_series(metrics, "summary/diffusion_block", "tokens_norm")
        if drift_layers or diff_layers:
            if drift_layers:
                axes[0].plot(drift_layers, drift_tokens, marker="o", label="drift")
            if diff_layers:
                axes[0].plot(diff_layers, diff_tokens, marker="o", label="diffusion")
            axes[0].set_title("Summary Token Norms")
            axes[0].set_xlabel("Layer")
            axes[0].legend()
            rendered += 1
        else:
            axes[0].axis("off")

        drift_summary_layers, drift_summary = _extract_layer_series(metrics, "summary/drift_block", "summary_tokens_norm")
        diff_summary_layers, diff_summary = _extract_layer_series(metrics, "summary/diffusion_block", "summary_tokens_norm")
        if drift_summary_layers or diff_summary_layers:
            if drift_summary_layers:
                axes[1].plot(drift_summary_layers, drift_summary, marker="o", label="drift")
            if diff_summary_layers:
                axes[1].plot(diff_summary_layers, diff_summary, marker="o", label="diffusion")
            axes[1].set_title("Summary-Token Focus Norms")
            axes[1].set_xlabel("Layer")
            axes[1].legend()
            rendered += 1
        else:
            axes[1].axis("off")

        final_items = [
            ("drift_vec", metrics.get("summary/drift_vec_norm")),
            ("diffusion_vec", metrics.get("summary/diffusion_vec_norm")),
        ]
        final_items = [(name, float(value)) for name, value in final_items if value is not None]
        cosine = metrics.get("summary/cos_zd_zv")
        if final_items or cosine is not None:
            if final_items:
                labels, values = zip(*final_items)
                axes[2].bar(range(len(values)), values, color="tab:purple", alpha=0.85)
                axes[2].set_xticks(range(len(values)))
                axes[2].set_xticklabels(labels, rotation=15, ha="right")
            if cosine is not None:
                axes[2].set_title(f"Final Summary Vectors (cos={float(cosine):.3f})")
            else:
                axes[2].set_title("Final Summary Vectors")
            rendered += 1
        else:
            axes[2].axis("off")

        if rendered == 0:
            plt.close(fig)
            return None
        fig.tight_layout()
        return fig

    def _plot_task_diagnostics(self, stage_buffer: EpochStageBuffer) -> Any:
        if plt is None or stage_buffer.sample_task is None:
            return None

        task = stage_buffer.sample_task
        samples = stage_buffer.sample_arrays()
        pred = samples["pred"]
        label = samples["label"]
        unc = samples["unc"]
        if pred.size == 0 or label.size == 0 or unc.size == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        error = np.abs(pred - label)
        if pred.size >= 128:
            axes[0, 0].hexbin(label, pred, gridsize=30, cmap="viridis", mincnt=1)
        else:
            axes[0, 0].scatter(label, pred, s=10, alpha=0.35)
        limits = [float(min(label.min(), pred.min())), float(max(label.max(), pred.max()))]
        axes[0, 0].plot(limits, limits, linestyle="--", color="tab:gray")
        axes[0, 0].set_title(f"{task} Prediction vs Label")
        axes[0, 0].set_xlabel("Label")
        axes[0, 0].set_ylabel("Prediction")

        axes[0, 1].hist(label, bins=30, alpha=0.55, label="label")
        axes[0, 1].hist(pred, bins=30, alpha=0.55, label="pred")
        axes[0, 1].set_title(f"{task} Distribution")
        axes[0, 1].legend()

        axes[1, 0].hist(unc, bins=30, alpha=0.85, color="tab:orange")
        axes[1, 0].set_title(f"{task} Uncertainty")
        axes[1, 0].set_xlabel("Predicted Uncertainty")

        cal_x, cal_y = _quantile_curve(unc, error)
        if cal_x.size > 0:
            axes[1, 1].plot(cal_x, cal_y, marker="o")
        axes[1, 1].set_title("Uncertainty vs Abs Error")
        axes[1, 1].set_xlabel("Mean Predicted Uncertainty")
        axes[1, 1].set_ylabel("Mean Absolute Error")
        fig.tight_layout()
        return fig

    def _plot_drift_cross_attention(self, snapshot: Mapping[str, Any]) -> Any:
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
            x_entropy.append(float((-(probs * probs.log()).sum(dim=-1).mean()).item()))
        for weights in y_to_x:
            probs = weights.clamp_min(1e-9)
            y_entropy.append(float((-(probs * probs.log()).sum(dim=-1).mean()).item()))

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

    def _plot_side_resampler_attention(self, snapshot: Mapping[str, Any]) -> Any:
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

    def _plot_summary_attention(self, snapshot: Mapping[str, Any]) -> Any:
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

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(drift_heat, aspect="auto", cmap="viridis")
        axes[0].set_title("Drift Summary Attn")
        axes[1].imshow(diffusion_heat, aspect="auto", cmap="viridis")
        axes[1].set_title("Diffusion Summary Attn")
        fig.tight_layout()
        return fig

    def log_epoch_diagnostics(self, stage: str, epoch: int) -> None:
        buffer = self._epoch_buffers.get(stage)
        if buffer is None:
            return

        metrics = buffer.compute()
        if buffer.heatmap_sum:
            heatmap_metrics: dict[str, Tensor] = {}
            for name, heat_sum in buffer.heatmap_sum.items():
                count = buffer.heatmap_count[name].clamp_min(1).to(dtype=torch.float32)
                mean_map = heat_sum / count
                heatmap_metrics[f"heatmap_epoch/{name}_mean"] = mean_map.mean()
                heatmap_metrics[f"heatmap_epoch/{name}_std"] = mean_map.std(unbiased=False)
            metrics.update(_sync_scalar_metrics(heatmap_metrics))
        if metrics:
            self.track(stage, metrics, epoch=epoch, subset="epoch_diag")

        self._log_figure(stage, epoch, "jointnet_activation_maps", self._plot_joint_maps(buffer))
        self._log_figure(stage, epoch, "feature_norm_flow", self._plot_feature_norm_flow(metrics))
        self._log_figure(stage, epoch, "fusion_internal_diagnostics", self._plot_fusion_internal_diagnostics(metrics))
        self._log_figure(stage, epoch, "summary_head_diagnostics", self._plot_summary_head_diagnostics(metrics))
        self._log_figure(stage, epoch, "task_diagnostics", self._plot_task_diagnostics(buffer))

        snapshot = buffer.snapshot
        if snapshot is None:
            return
        self._log_figure(stage, epoch, "drift_cross_attention", self._plot_drift_cross_attention(snapshot))
        self._log_figure(stage, epoch, "side_resampler_attention", self._plot_side_resampler_attention(snapshot))
        self._log_figure(stage, epoch, "summary_attention", self._plot_summary_attention(snapshot))

    def collect_batch_metrics(
        self,
        model: nn.Module,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        loss_metrics: Mapping[str, Tensor | float],
    ) -> dict[str, Tensor]:
        del model
        task = detect_task_from_outputs(outputs)
        device = next(value.device for value in outputs.values() if isinstance(value, Tensor))
        metrics: dict[str, Tensor] = {}
        for key, value in outputs.items():
            if not isinstance(value, Tensor) or not key.startswith("diag/"):
                continue
            metrics[key.removeprefix("diag/")] = _as_scalar_tensor(value, device)
        for key in ("loss_total", "loss_task", "loss_mu", "loss_unc"):
            if key in loss_metrics:
                metrics[key] = _as_scalar_tensor(loss_metrics[key], device)
        pred = outputs[f"pred_{task}"]
        target = batch[task_target_column(task)]
        metrics.setdefault(f"mae/{task}", torch.mean(torch.abs(pred - target)).detach().float())
        metrics.setdefault(f"unc_mean/{task}", outputs[f"unc_{task}"].detach().float().mean())
        loss_key = f"loss_{task}"
        if loss_key in loss_metrics:
            metrics[f"loss_task/{task}"] = _as_scalar_tensor(loss_metrics[loss_key], device)
        metrics.setdefault("token/M1_norm", _channelwise_l2_mean(outputs["M1"], channel_dim=2))
        metrics.setdefault("token/M2_norm", _channelwise_l2_mean(outputs["M2"], channel_dim=2))
        metrics.setdefault("token/S_norm", _channelwise_l2_mean(outputs["S"], channel_dim=2))
        metrics.setdefault("fusion/Z0_norm", _channelwise_l2_mean(outputs["Z0"], channel_dim=2))
        metrics.setdefault("fusion/Z1_norm", _channelwise_l2_mean(outputs["Z1"], channel_dim=2))
        metrics.setdefault("fusion/drift_delta", _mean_abs_delta(outputs["Z0"], outputs["M1"]))
        metrics.setdefault("fusion/diffusion_delta", _mean_abs_delta(outputs["Z1"], outputs["M2"]))
        metrics.setdefault("summary/z_d_norm", _channelwise_l2_mean(outputs["z_d"], channel_dim=1))
        metrics.setdefault("summary/z_v_norm", _channelwise_l2_mean(outputs["z_v"], channel_dim=1))
        metrics.setdefault("summary/cos_zd_zv", _vector_cosine(outputs["z_d"], outputs["z_v"]))
        metrics.setdefault("tfn/feat_norm", _channelwise_l2_mean(outputs["tfn_feat"], channel_dim=1))
        metrics.setdefault("decoder/head_out_std", _tensor_std(outputs["head_out"]))
        metrics.setdefault("token/cos_M1_S", _pooled_cosine(outputs["M1"], outputs["S"]))
        metrics.setdefault("token/cos_M2_S", _pooled_cosine(outputs["M2"], outputs["S"]))
        return {key: value.detach() for key, value in metrics.items()}

    def collect_low_frequency_metrics(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        metrics["grad_global_norm"] = grad_norm(model.parameters())
        metrics["param_global_norm"] = _parameter_norm(model.parameters())
        if optimizer is not None and optimizer.param_groups:
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])
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
        values = dict(metrics)
        if lr is not None:
            values["lr"] = float(lr)
        if grad_global_norm_value is not None:
            values["grad_global_norm"] = float(grad_global_norm_value)
        if param_global_norm_value is not None:
            values["param_global_norm"] = float(param_global_norm_value)
        ema_metrics: dict[str, float] = {}
        tracked_keys = list(_REALTIME_KEYS) + ["lr", "grad_global_norm", "param_global_norm"]
        for key in values:
            if key in _DYNAMIC_REALTIME_EXACT or key.startswith(_DYNAMIC_REALTIME_PREFIXES):
                tracked_keys.append(key)
        for key in dict.fromkeys(tracked_keys):
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
        if self.run is None or mlflow is None or not metrics:
            return
        metric_step = int(step if step is not None else (epoch if epoch is not None else 0))
        payload = {
            _mlflow_metric_key(stage, subset, key): float(value)
            for key, value in metrics.items()
        }
        mlflow.log_metrics(payload, step=metric_step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.run is None or mlflow is None:
            return
        mlflow.log_params({key: str(value) for key, value in params.items()})


__all__ = ["MLflowVisualizer", "DiagnosticsAccumulator"]
