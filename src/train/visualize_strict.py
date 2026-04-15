"""Strict GPU-first MLflow diagnostics for training and validation."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from config.config import trend_ema_alpha as DEFAULT_EMA_ALPHA
from torch import Tensor, nn

from src.task_labels import detect_task_from_outputs
from src.train.utils import grad_norm

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]

try:
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
    "loss_task/Edge",
    "loss_task/Persist",
    "loss_task/DownRisk",
    "mae/Edge",
    "mae/Persist",
    "mae/DownRisk",
    "brier/Persist",
    "prob_mean/Persist",
    "unc_mean/Edge",
    "unc_mean/Persist",
    "unc_mean/DownRisk",
    "nu/Edge",
    "nu/DownRisk",
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
    "encoder/",
    "map/",
    "fusion/diffusion_block",
)
_DYNAMIC_REALTIME_EXACT = {
    "fusion/side_global_norm",
}


def _tracking_uri(repo: str | Path) -> str:
    path = Path(repo).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


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
    def __init__(self) -> None:
        self.metrics_sum: dict[str, Tensor] = {}
        self.weight: Tensor | None = None
        self.heatmap_sum: dict[str, Tensor] = {}
        self.heatmap_count: dict[str, Tensor] = {}

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
        del stage, model, outputs, batch

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
        for key in ("loss_total", "loss_task"):
            if key in loss_metrics:
                metrics[key] = _as_scalar_tensor(loss_metrics[key], device)
        if task == "Persist":
            pred = outputs["pred_Persist"]
            target = batch["label_Persist"]
            metrics.setdefault("mae/Persist", torch.mean(torch.abs(pred - target)).detach().float())
            metrics.setdefault("brier/Persist", torch.mean((pred - target).float() ** 2).detach())
            metrics.setdefault("prob_mean/Persist", pred.detach().float().mean())
            metrics.setdefault("unc_mean/Persist", outputs["Persist_unc"].detach().float().mean())
        else:
            pred = outputs[f"pred_{task}"]
            target = batch[f"label_{task}"]
            metrics.setdefault(f"mae/{task}", torch.mean(torch.abs(pred - target)).detach().float())
            metrics.setdefault(f"unc_mean/{task}", outputs[f"unc_{task}"].detach().float().mean())
        loss_key = f"loss_{task}"
        if loss_key in loss_metrics:
            metrics[f"loss_task/{task}"] = _as_scalar_tensor(loss_metrics[loss_key], device)
        nu_key = f"nu_{task}"
        if nu_key in loss_metrics:
            metrics[f"nu/{task}"] = _as_scalar_tensor(loss_metrics[nu_key], device)
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
