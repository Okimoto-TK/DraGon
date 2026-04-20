"""Lightweight torch.profiler helpers for short training captures."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
import torch


@dataclass(frozen=True)
class TrainingProfilerConfig:
    """Short-run profiler settings for the training loop."""

    warmup_steps: int = 2
    active_steps: int = 20
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False


class _TraceExporter:
    """Persist one chrome trace and one text summary per capture window."""

    def __init__(self, output_dir: Path, *, sort_by: str) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sort_by = sort_by

    def __call__(self, prof: torch.profiler.profile) -> None:
        step_num = int(getattr(prof, "step_num", 0))
        stem = f"trace_step_{step_num:05d}"
        trace_path = self.output_dir / f"{stem}.json"
        summary_path = self.output_dir / f"{stem}_summary.txt"
        prof.export_chrome_trace(str(trace_path))
        summary_path.write_text(
            prof.key_averages().table(
                sort_by=self.sort_by,
                row_limit=80,
            ),
            encoding="utf-8",
        )


class TrainingProfiler:
    """Small wrapper around torch.profiler with a stable close/step API."""

    def __init__(self, profiler: torch.profiler.profile, output_dir: Path) -> None:
        self._profiler = profiler
        self.output_dir = output_dir
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def step(self) -> None:
        self._profiler.step()

    def add_cleanup(self, callback) -> None:
        self._cleanup_callbacks.append(callback)

    def close(self) -> None:
        if self._profiler is None:
            return
        for callback in reversed(self._cleanup_callbacks):
            callback()
        self._cleanup_callbacks.clear()
        self._profiler.stop()
        self._profiler = None


class _ModuleRangeHooks:
    """Attach module-level forward/backward ranges for profiler attribution."""

    def __init__(self, module_names: Iterable[str], model: nn.Module) -> None:
        self._contexts: dict[tuple[int, str], list[object]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing when gradients are computed with respect to module outputs.*",
            category=UserWarning,
        )
        named_children = dict(model.named_children())
        for name in module_names:
            module = named_children.get(name)
            if module is None:
                continue
            self._handles.extend(
                [
                    module.register_forward_pre_hook(self._make_forward_pre_hook(name)),
                    module.register_forward_hook(self._make_forward_post_hook()),
                    module.register_full_backward_pre_hook(self._make_backward_pre_hook(name)),
                    module.register_full_backward_hook(self._make_backward_post_hook()),
                ]
            )

    def _enter_range(self, module: nn.Module, phase: str, name: str) -> None:
        context = torch.profiler.record_function(f"module_{phase}:{name}")
        context.__enter__()
        self._contexts[(id(module), phase)].append(context)

    def _exit_range(self, module: nn.Module, phase: str) -> None:
        key = (id(module), phase)
        if key not in self._contexts or not self._contexts[key]:
            return
        context = self._contexts[key].pop()
        context.__exit__(None, None, None)

    def _make_forward_pre_hook(self, name: str):
        def _hook(module: nn.Module, _inputs) -> None:
            self._enter_range(module, "forward", name)

        return _hook

    def _make_forward_post_hook(self):
        def _hook(module: nn.Module, _inputs, _output) -> None:
            self._exit_range(module, "forward")

        return _hook

    def _make_backward_pre_hook(self, name: str):
        def _hook(module: nn.Module, _grad_output) -> None:
            self._enter_range(module, "backward", name)

        return _hook

    def _make_backward_post_hook(self):
        def _hook(module: nn.Module, _grad_input, _grad_output) -> None:
            self._exit_range(module, "backward")

        return _hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for contexts in self._contexts.values():
            while contexts:
                contexts.pop().__exit__(None, None, None)
        self._contexts.clear()


def build_training_profiler(
    *,
    device: torch.device,
    output_dir: Path,
    active_steps: int,
    warmup_steps: int = 2,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
) -> TrainingProfiler | None:
    """Create and start a short-window torch.profiler session."""

    if active_steps <= 0:
        return None

    warmup_steps = max(0, int(warmup_steps))
    active_steps = max(1, int(active_steps))
    output_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"

    exporter = _TraceExporter(output_dir, sort_by=sort_by)
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1),
        on_trace_ready=exporter,
        acc_events=True,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )
    profiler.start()
    return TrainingProfiler(profiler=profiler, output_dir=output_dir)


def attach_module_range_profiler(
    *,
    model: nn.Module,
    profiler: TrainingProfiler | None,
    module_names: Iterable[str] | None = None,
) -> None:
    """Attach top-level module ranges to an active training profiler."""

    if profiler is None:
        return

    selected = list(module_names) if module_names is not None else list(dict(model.named_children()))
    if not selected:
        return
    hooks = _ModuleRangeHooks(selected, model)
    profiler.add_cleanup(hooks.close)
