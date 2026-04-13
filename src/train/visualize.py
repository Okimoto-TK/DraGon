"""MLflow-based training diagnostics for multiscale fusion models."""
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

from src.train.utils import grad_norm

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
    "T23_proj": "jointnet_23_in_proj",
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
        if "H23" in self._features and "T23_proj" in self._features:
            metrics["map/H23_delta"] = _mean_abs_delta(self._features["H23"], self._features["T23_proj"])

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
