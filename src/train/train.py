"""Training loop for one epoch."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch.amp import GradScaler, autocast
from config.config import diagnostics_every_steps as DEFAULT_DIAGNOSTICS_EVERY_STEPS
from config.config import cuda_graph_warmup_steps as DEFAULT_CUDA_GRAPH_WARMUP_STEPS
from config.config import log_every as DEFAULT_LOG_EVERY
from config.config import nan_debug_enabled as DEFAULT_NAN_DEBUG_ENABLED
from config.config import nan_debug_max_param_reports as DEFAULT_NAN_DEBUG_MAX_PARAM_REPORTS
from config.config import use_cuda_graph as DEFAULT_USE_CUDA_GRAPH
from torch import nn
from torch.utils.data import DataLoader

from src.train.visualize_strict import MLflowVisualizer
from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    move_batch_to_device,
)


def _tensor_nonfinite_summary(name: str, tensor: torch.Tensor) -> str | None:
    det = tensor.detach()
    if det.numel() == 0:
        return None
    nan_count = int(torch.isnan(det).sum().item())
    inf_count = int(torch.isinf(det).sum().item())
    if nan_count == 0 and inf_count == 0:
        return None
    finite = torch.isfinite(det)
    finite_count = int(finite.sum().item())
    total = int(det.numel())
    finite_ratio = float(finite_count / max(total, 1))
    if finite_count > 0:
        finite_values = det[finite]
        f_min = float(finite_values.min().item())
        f_max = float(finite_values.max().item())
        f_mean = float(finite_values.mean().item())
    else:
        f_min = float("nan")
        f_max = float("nan")
        f_mean = float("nan")
    return (
        f"{name}: shape={tuple(det.shape)} dtype={det.dtype} "
        f"finite_ratio={finite_ratio:.6f} nan={nan_count} inf={inf_count} "
        f"finite_min={f_min:.6e} finite_max={f_max:.6e} finite_mean={f_mean:.6e}"
    )


def _collect_nonfinite_lines(
    mapping: Mapping[str, torch.Tensor],
    prefix: str,
) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        line = _tensor_nonfinite_summary(f"{prefix}.{key}", value)
        if line is not None:
            lines.append(line)
    return lines


def _collect_nonfinite_grad_lines(
    model: nn.Module,
    *,
    max_reports: int,
) -> list[str]:
    lines: list[str] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        line = _tensor_nonfinite_summary(f"grad.{name}", grad)
        if line is None:
            continue
        lines.append(line)
        if len(lines) >= max_reports:
            break
    return lines


def _raise_nonfinite(
    *,
    phase: str,
    step: int,
    lines: list[str],
) -> None:
    if not lines:
        return
    message = "\n".join([f"[NaNDebug] phase={phase} step={step}", *lines])
    print(message)
    raise RuntimeError(message)


def _make_static_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.empty_like(value) for key, value in batch.items()}


def _copy_batch_to_static(
    static_batch: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    for key, value in batch.items():
        static_batch[key].copy_(value, non_blocking=True)


def _run_eager_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    *,
    grad_clip: float | None,
    step_callback: Callable[[dict[str, object]], None] | None,
    visualizer: MLflowVisualizer | None,
    epoch: int | None,
    global_step_offset: int,
    log_every: int,
    scaler: GradScaler | None,
    amp_enabled: bool,
    nan_debug_enabled: bool,
    nan_debug_max_param_reports: int,
) -> dict[str, float]:
    autocast_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    tracker = MetricTracker(device=device)
    window_tracker = MetricTracker(device=device)
    total_steps = max(len(dataloader), 1)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
            loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        diag_metrics: dict[str, torch.Tensor | float] = {}
        if visualizer is not None:
            diag_metrics = visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            visualizer.update_epoch_buffer("train", model, outputs, batch, metrics=diag_metrics)
        if visualizer is not None and step_idx == total_steps:
            visualizer.capture_epoch_snapshot("train", model, outputs, batch)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if nan_debug_enabled:
            debug_lines = _collect_nonfinite_lines(batch, "batch")
            debug_lines.extend(_collect_nonfinite_lines(outputs, "outputs"))
            debug_lines.extend(_collect_nonfinite_lines(loss_metrics, "loss_metrics"))
            loss_line = _tensor_nonfinite_summary("loss", loss)
            if loss_line is not None:
                debug_lines.append(loss_line)
            debug_lines.extend(
                _collect_nonfinite_grad_lines(
                    model,
                    max_reports=nan_debug_max_param_reports,
                )
            )
            _raise_nonfinite(phase="eager", step=step_idx, lines=debug_lines)

        if grad_clip is not None:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        current_grad_norm = None

        if amp_enabled and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_size = int(batch["macro"].shape[0])
        step_metrics = {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}
        tracker.update(step_metrics, weight=batch_size)
        window_tracker.update(step_metrics, weight=batch_size)

        should_sync = (
            log_every <= 0
            or step_idx % log_every == 0
            or step_idx == total_steps
        )
        if should_sync:
            synced_metrics = window_tracker.compute_and_reset(reset=True)
            if visualizer is not None:
                lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
                realtime_metrics = visualizer.realtime_metrics(
                    "train",
                    synced_metrics,
                    lr=lr,
                    grad_global_norm_value=current_grad_norm,
                )
                visualizer.track(
                    "train",
                    realtime_metrics,
                    step=global_step_offset + step_idx,
                    epoch=epoch,
                    subset="realtime",
                )
            if step_callback is not None:
                step_callback(
                    {
                        "step": float(step_idx),
                        "total_steps": float(total_steps),
                        "metrics": synced_metrics,
                    }
                )

    return tracker.compute()


def _run_cuda_graph_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    *,
    grad_clip: float | None,
    step_callback: Callable[[dict[str, object]], None] | None,
    visualizer: MLflowVisualizer | None,
    epoch: int | None,
    global_step_offset: int,
    log_every: int,
    amp_enabled: bool,
    cuda_graph_warmup_steps: int,
    nan_debug_enabled: bool,
    nan_debug_max_param_reports: int,
) -> dict[str, float]:
    autocast_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    hooks_suspended = visualizer is not None
    if hooks_suspended:
        visualizer.detach()

    tracker = MetricTracker(device=device)
    window_tracker = MetricTracker(device=device)
    total_steps = max(len(dataloader), 1)
    data_iter = iter(dataloader)

    step_idx = 0

    def _sync_window_if_needed(current_step: int) -> None:
        should_sync = (
            log_every <= 0
            or current_step % log_every == 0
            or current_step == total_steps
        )
        if not should_sync:
            return
        synced_metrics = window_tracker.compute_and_reset(reset=True)
        if visualizer is not None:
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
            realtime_metrics = visualizer.realtime_metrics(
                "train",
                synced_metrics,
                lr=lr,
                grad_global_norm_value=None,
            )
            visualizer.track(
                "train",
                realtime_metrics,
                step=global_step_offset + current_step,
                epoch=epoch,
                subset="realtime",
            )
        if step_callback is not None:
            step_callback(
                {
                    "step": float(current_step),
                    "total_steps": float(total_steps),
                    "metrics": synced_metrics,
                }
            )

    def _run_eager_step(batch: dict[str, torch.Tensor], current_step: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )
            loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        diag_metrics: dict[str, torch.Tensor | float] = {}
        if visualizer is not None:
            diag_metrics = visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            visualizer.update_epoch_buffer("train", model, outputs, batch, metrics=diag_metrics)
        if visualizer is not None and current_step == total_steps:
            visualizer.capture_epoch_snapshot("train", model, outputs, batch)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if nan_debug_enabled:
            debug_lines = _collect_nonfinite_lines(batch, "batch")
            debug_lines.extend(_collect_nonfinite_lines(outputs, "outputs"))
            debug_lines.extend(_collect_nonfinite_lines(loss_metrics, "loss_metrics"))
            loss_line = _tensor_nonfinite_summary("loss", loss)
            if loss_line is not None:
                debug_lines.append(loss_line)
            debug_lines.extend(
                _collect_nonfinite_grad_lines(
                    model,
                    max_reports=nan_debug_max_param_reports,
                )
            )
            _raise_nonfinite(phase="cuda_graph_warmup", step=current_step, lines=debug_lines)

        batch_size = int(batch["macro"].shape[0])
        step_metrics = {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}
        tracker.update(step_metrics, weight=batch_size)
        window_tracker.update(step_metrics, weight=batch_size)
        _sync_window_if_needed(current_step)

    try:
        warmup_steps = max(int(cuda_graph_warmup_steps), 0)
        for _ in range(warmup_steps):
            host_batch = next(data_iter, None)
            if host_batch is None:
                return tracker.compute()
            step_idx += 1
            _run_eager_step(move_batch_to_device(host_batch, device), step_idx)

        capture_batch_host = next(data_iter, None)
        if capture_batch_host is None:
            return tracker.compute()
        capture_batch = move_batch_to_device(capture_batch_host, device)
        static_batch = _make_static_batch(capture_batch)
        _copy_batch_to_static(static_batch, capture_batch)

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
                    static_outputs = model(
                        static_batch["macro"],
                        static_batch["mezzo"],
                        static_batch["micro"],
                        static_batch["sidechain"],
                    )
                    static_loss, _ = criterion(
                        static_outputs,
                        static_batch,
                        return_metrics=False,
                        update_state=False,
                    )
                static_loss.backward()
        except Exception as exc:
            raise RuntimeError(
                "CUDA Graph capture failed. Non-CUDA-Graph fallback is disabled by policy."
            ) from exc

        def _iter_captured_batches():
            yield capture_batch
            for host_batch in data_iter:
                yield move_batch_to_device(host_batch, device)

        for batch in _iter_captured_batches():
            step_idx += 1
            _copy_batch_to_static(static_batch, batch)
            graph.replay()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            with torch.no_grad():
                _, static_loss_metrics = criterion(
                    static_outputs,
                    static_batch,
                    return_metrics=True,
                    update_state=True,
                )
            static_mean_metrics = batch_prediction_metrics(static_outputs, static_batch)

            if nan_debug_enabled:
                debug_lines = _collect_nonfinite_lines(static_batch, "batch")
                debug_lines.extend(_collect_nonfinite_lines(static_outputs, "outputs"))
                debug_lines.extend(_collect_nonfinite_lines(static_loss_metrics, "loss_metrics"))
                loss_line = _tensor_nonfinite_summary("loss", static_loss)
                if loss_line is not None:
                    debug_lines.append(loss_line)
                debug_lines.extend(
                    _collect_nonfinite_grad_lines(
                        model,
                        max_reports=nan_debug_max_param_reports,
                    )
                )
                _raise_nonfinite(phase="cuda_graph_replay", step=step_idx, lines=debug_lines)

            diag_metrics: dict[str, torch.Tensor | float] = {}
            if visualizer is not None:
                diag_metrics = visualizer.collect_batch_metrics(model, static_outputs, static_batch, static_loss_metrics)
                visualizer.update_epoch_buffer("train", model, static_outputs, static_batch, metrics=diag_metrics)
            if visualizer is not None and step_idx == total_steps:
                visualizer.capture_epoch_snapshot("train", model, static_outputs, static_batch)

            batch_size = int(static_batch["macro"].shape[0])
            step_metrics = {
                "loss": static_loss.detach(),
                **static_loss_metrics,
                **static_mean_metrics,
                **diag_metrics,
            }
            tracker.update(step_metrics, weight=batch_size)
            window_tracker.update(step_metrics, weight=batch_size)
            _sync_window_if_needed(step_idx)

        return tracker.compute()
    finally:
        if hooks_suspended:
            visualizer.attach(model)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    *,
    grad_clip: float | None = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
    visualizer: MLflowVisualizer | None = None,
    epoch: int | None = None,
    global_step_offset: int = 0,
    diagnostics_every_steps: int = DEFAULT_DIAGNOSTICS_EVERY_STEPS,
    log_every: int = DEFAULT_LOG_EVERY,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    use_cuda_graph: bool = DEFAULT_USE_CUDA_GRAPH,
    cuda_graph_warmup_steps: int = DEFAULT_CUDA_GRAPH_WARMUP_STEPS,
    nan_debug_enabled: bool = DEFAULT_NAN_DEBUG_ENABLED,
    nan_debug_max_param_reports: int = DEFAULT_NAN_DEBUG_MAX_PARAM_REPORTS,
) -> dict[str, float]:
    """Run one training epoch and return averaged metrics."""
    del diagnostics_every_steps
    model.train()
    criterion.train()
    if not bool(use_cuda_graph):
        raise RuntimeError("Non-CUDA-Graph training is disabled. Set use_cuda_graph=True.")
    can_use_cuda_graph = (
        bool(use_cuda_graph)
        and device.startswith("cuda")
        and torch.cuda.is_available()
        and scaler is None
    )
    if can_use_cuda_graph:
        return _run_cuda_graph_epoch(
            model,
            criterion,
            optimizer,
            dataloader,
            device,
            grad_clip=grad_clip,
            step_callback=step_callback,
            visualizer=visualizer,
            epoch=epoch,
            global_step_offset=global_step_offset,
            log_every=log_every,
            amp_enabled=amp_enabled,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            nan_debug_enabled=nan_debug_enabled,
            nan_debug_max_param_reports=nan_debug_max_param_reports,
        )
    raise RuntimeError(
        "CUDA Graph prerequisites not met. "
        "Require CUDA device, use_cuda_graph=True, and scaler=None."
    )


__all__ = ["train_one_epoch"]
