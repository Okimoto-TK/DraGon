"""Training loop for one epoch."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from config.config import diagnostics_every_steps as DEFAULT_DIAGNOSTICS_EVERY_STEPS
from config.config import cuda_graph_warmup_steps as DEFAULT_CUDA_GRAPH_WARMUP_STEPS
from config.config import log_every as DEFAULT_LOG_EVERY
from config.config import use_cuda_graph as DEFAULT_USE_CUDA_GRAPH
from torch import nn
from torch.utils.data import DataLoader

from src.train.visualize_strict import MLflowVisualizer
from src.train.utils import (
    MetricTracker,
    batch_prediction_metrics,
    move_batch_to_device,
)


@dataclass
class CudaGraphState:
    capture_device: torch.device
    capture_device_str: str
    graph: torch.cuda.CUDAGraph
    static_batch: dict[str, torch.Tensor]
    static_outputs: dict[str, torch.Tensor]
    static_loss: torch.Tensor


def _make_static_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.empty_like(value) for key, value in batch.items()}


def _copy_batch_to_static(
    static_batch: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    for key, value in batch.items():
        static_batch[key].copy_(value, non_blocking=True)


def _cuda_amp_autocast_kwargs(amp_enabled: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "device_type": "cuda",
        "enabled": amp_enabled,
        # Keep AMP behavior consistent across train/val and avoid extra cache
        # bookkeeping inside CUDA Graph capture.
        "cache_enabled": False,
    }
    if amp_enabled:
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def _zero_grads_for_cuda_graph(optimizer: torch.optim.Optimizer) -> None:
    # CUDA Graph replay does not re-run Python-side `set_to_none=True` logic.
    # Keep grad buffers alive and zero them in-place outside capture.
    optimizer.zero_grad(set_to_none=False)


def _initialize_cuda_graph_state(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_iter,
    *,
    amp_enabled: bool,
    cuda_graph_warmup_steps: int,
) -> tuple[CudaGraphState | None, int]:
    capture_device = next(model.parameters()).device
    if capture_device.type != "cuda":
        raise RuntimeError(f"CUDA Graph requires model parameters on CUDA, got {capture_device}")
    torch.cuda.set_device(capture_device)
    capture_device_str = str(capture_device)

    warmup_steps = max(int(cuda_graph_warmup_steps), 0)
    consumed_steps = 0
    for _ in range(warmup_steps):
        host_batch = next(data_iter, None)
        if host_batch is None:
            return None, consumed_steps
        batch = move_batch_to_device(host_batch, capture_device_str)
        _zero_grads_for_cuda_graph(optimizer)
        with autocast(**_cuda_amp_autocast_kwargs(amp_enabled)):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )
            loss, _ = criterion(outputs, batch, return_metrics=False, update_state=False)
        loss.backward()
        optimizer.step()
        consumed_steps += 1
        del outputs
        del loss

    capture_batch_host = next(data_iter, None)
    if capture_batch_host is None:
        return None, consumed_steps
    capture_batch = move_batch_to_device(capture_batch_host, capture_device_str)
    static_batch = _make_static_batch(capture_batch)
    _copy_batch_to_static(static_batch, capture_batch)
    _zero_grads_for_cuda_graph(optimizer)

    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        static_warmup_iters = max(1, min(max(int(cuda_graph_warmup_steps), 0), 3))
        for _ in range(static_warmup_iters):
            _zero_grads_for_cuda_graph(optimizer)
            with autocast(**_cuda_amp_autocast_kwargs(amp_enabled)):
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
            del static_outputs
            del static_loss
        _zero_grads_for_cuda_graph(optimizer)

        capture_stage = "graph_init"
        try:
            graph = torch.cuda.CUDAGraph()
            capture_stage = "graph_context"
            with torch.cuda.graph(graph):
                capture_stage = "autocast_context"
                with autocast(**_cuda_amp_autocast_kwargs(amp_enabled)):
                    capture_stage = "forward"
                    static_outputs = model(
                        static_batch["macro"],
                        static_batch["mezzo"],
                        static_batch["micro"],
                        static_batch["sidechain"],
                    )
                    capture_stage = "loss"
                    static_loss, _ = criterion(
                        static_outputs,
                        static_batch,
                        return_metrics=False,
                        update_state=False,
                    )
                capture_stage = "backward"
                static_loss.backward()
        except Exception as exc:
            raise RuntimeError(
                "CUDA Graph capture failed. "
                f"stage={capture_stage}. "
                "Non-CUDA-Graph fallback is disabled by policy."
            ) from exc
    finally:
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    state = CudaGraphState(
        capture_device=capture_device,
        capture_device_str=capture_device_str,
        graph=graph,
        static_batch=static_batch,
        static_outputs=static_outputs,
        static_loss=static_loss,
    )
    return state, consumed_steps


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
) -> tuple[dict[str, float], None]:
    tracker = MetricTracker(device=device)
    window_tracker = MetricTracker(device=device)
    total_steps = max(len(dataloader), 1)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(
                batch["macro"],
                batch["mezzo"],
                batch["micro"],
                batch["sidechain"],
            )

        with autocast(device_type="cuda", enabled=amp_enabled):
            loss, loss_metrics = criterion(outputs, batch)
        mean_metrics = batch_prediction_metrics(outputs, batch)
        diag_metrics: dict[str, torch.Tensor | float] = {}
        if visualizer is not None:
            diag_metrics = visualizer.collect_batch_metrics(model, outputs, batch, loss_metrics)
            visualizer.update_epoch_buffer("train", model, outputs, batch, metrics=diag_metrics)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

    return tracker.compute(), None


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
    cuda_graph_state: CudaGraphState | None,
) -> tuple[dict[str, float], CudaGraphState | None]:
    capture_device = next(model.parameters()).device
    if capture_device.type != "cuda":
        raise RuntimeError(f"CUDA Graph requires model parameters on CUDA, got {capture_device}")
    torch.cuda.set_device(capture_device)

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
        _zero_grads_for_cuda_graph(optimizer)
        with autocast(**_cuda_amp_autocast_kwargs(amp_enabled)):
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

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(batch["macro"].shape[0])
        step_metrics = {"loss": loss.detach(), **loss_metrics, **mean_metrics, **diag_metrics}
        tracker.update(step_metrics, weight=batch_size)
        window_tracker.update(step_metrics, weight=batch_size)
        _sync_window_if_needed(current_step)
        del outputs
        del loss
        del loss_metrics
        del mean_metrics
        del diag_metrics
        del step_metrics

    try:
        graph_initialized_this_epoch = False
        if cuda_graph_state is None:
            cuda_graph_state, warmup_consumed = _initialize_cuda_graph_state(
                model,
                criterion,
                optimizer,
                data_iter,
                amp_enabled=amp_enabled,
                cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            )
            step_idx += warmup_consumed
            if cuda_graph_state is None:
                return tracker.compute(), None
            graph_initialized_this_epoch = True

        assert cuda_graph_state is not None

        if graph_initialized_this_epoch:
            step_idx += 1
            _zero_grads_for_cuda_graph(optimizer)
            cuda_graph_state.graph.replay()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            with torch.no_grad():
                _, static_loss_metrics = criterion(
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    return_metrics=True,
                    update_state=True,
                )
            static_mean_metrics = batch_prediction_metrics(cuda_graph_state.static_outputs, cuda_graph_state.static_batch)
            diag_metrics: dict[str, torch.Tensor | float] = {}
            if visualizer is not None:
                diag_metrics = visualizer.collect_batch_metrics(
                    model,
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    static_loss_metrics,
                )
                visualizer.update_epoch_buffer(
                    "train",
                    model,
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    metrics=diag_metrics,
                )
            batch_size = int(cuda_graph_state.static_batch["macro"].shape[0])
            step_metrics = {
                "loss": cuda_graph_state.static_loss.detach(),
                **static_loss_metrics,
                **static_mean_metrics,
                **diag_metrics,
            }
            tracker.update(step_metrics, weight=batch_size)
            window_tracker.update(step_metrics, weight=batch_size)
            _sync_window_if_needed(step_idx)

        for host_batch in data_iter:
            batch = move_batch_to_device(host_batch, cuda_graph_state.capture_device_str)
            step_idx += 1
            _zero_grads_for_cuda_graph(optimizer)
            _copy_batch_to_static(cuda_graph_state.static_batch, batch)
            cuda_graph_state.graph.replay()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            with torch.no_grad():
                _, static_loss_metrics = criterion(
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    return_metrics=True,
                    update_state=True,
                )
            static_mean_metrics = batch_prediction_metrics(cuda_graph_state.static_outputs, cuda_graph_state.static_batch)

            diag_metrics: dict[str, torch.Tensor | float] = {}
            if visualizer is not None:
                diag_metrics = visualizer.collect_batch_metrics(
                    model,
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    static_loss_metrics,
                )
                visualizer.update_epoch_buffer(
                    "train",
                    model,
                    cuda_graph_state.static_outputs,
                    cuda_graph_state.static_batch,
                    metrics=diag_metrics,
                )

            batch_size = int(cuda_graph_state.static_batch["macro"].shape[0])
            step_metrics = {
                "loss": cuda_graph_state.static_loss.detach(),
                **static_loss_metrics,
                **static_mean_metrics,
                **diag_metrics,
            }
            tracker.update(step_metrics, weight=batch_size)
            window_tracker.update(step_metrics, weight=batch_size)
            _sync_window_if_needed(step_idx)

        return tracker.compute(), cuda_graph_state
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
    cuda_graph_state: CudaGraphState | None = None,
) -> tuple[dict[str, float], CudaGraphState | None]:
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
    if can_use_cuda_graph and amp_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA Graph AMP requires bfloat16 autocast support on the active CUDA device.")
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
            cuda_graph_state=cuda_graph_state,
        )
    raise RuntimeError(
        "CUDA Graph prerequisites not met. "
        "Require CUDA device, use_cuda_graph=True, and scaler=None."
    )


__all__ = ["train_one_epoch"]
