"""torch.profiler helpers for targeted training profiling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TorchProfilerSession:
    """Container for one optional epoch profiler session."""

    profiler: Any | None
    output_dir: Path | None

    @property
    def enabled(self) -> bool:
        return self.profiler is not None and self.output_dir is not None


def create_epoch_profiler(
    *,
    enabled: bool,
    epoch: int,
    target_epoch: int,
    run_name: str,
    output_root: str | Path,
    device: str,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    with_flops: bool,
) -> TorchProfilerSession:
    """Create a torch.profiler session for one epoch if requested."""
    if not enabled or epoch != target_epoch:
        return TorchProfilerSession(profiler=None, output_dir=None)

    output_dir = Path(output_root) / run_name / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=max(0, int(wait)),
        warmup=max(0, int(warmup)),
        active=max(1, int(active)),
        repeat=max(1, int(repeat)),
    )
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=bool(record_shapes),
        profile_memory=bool(profile_memory),
        with_stack=bool(with_stack),
        with_flops=bool(with_flops),
    )
    return TorchProfilerSession(profiler=profiler, output_dir=output_dir)


def write_profiler_reports(
    session: TorchProfilerSession,
    *,
    row_limit: int = 60,
) -> None:
    """Persist human-readable top-op tables next to profiler traces."""
    if not session.enabled:
        return

    assert session.profiler is not None
    assert session.output_dir is not None
    key_avg = session.profiler.key_averages()

    cpu_table = key_avg.table(sort_by="self_cpu_time_total", row_limit=row_limit)
    (session.output_dir / "top_ops_cpu.txt").write_text(cpu_table, encoding="utf-8")

    # key_averages internals differ across versions; try CUDA table best-effort.
    try:
        cuda_table = key_avg.table(sort_by="self_cuda_time_total", row_limit=row_limit)
    except Exception:
        cuda_table = ""
    if cuda_table.strip():
        (session.output_dir / "top_ops_cuda.txt").write_text(cuda_table, encoding="utf-8")


__all__ = [
    "TorchProfilerSession",
    "create_epoch_profiler",
    "write_profiler_reports",
]
