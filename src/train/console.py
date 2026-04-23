"""Rich console logging for training and validation epochs."""

from __future__ import annotations

import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class EpochConsoleLogger:
    """Single-task Rich progress logger reused across train and val phases."""

    def __init__(
        self,
        *,
        log_every: int,
        task: str = "ret",
        enabled: bool = True,
        console: Console | None = None,
    ) -> None:
        self.log_every = int(log_every)
        self.task = task
        self._task_display = "q10" if task == "q" else task
        self.enabled = bool(enabled)
        self.console = console or Console()
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[metrics]}"),
            TextColumn("step={task.fields[step_ms]}"),
            TextColumn("eta={task.fields[eta]}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            disable=not self.enabled,
        )
        self.task_id: TaskID | None = None
        self._phase = ""
        self._phase_start_time: float | None = None
        self._started = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.progress.start()
        self._started = True

    def start_phase(self, *, epoch: int, phase: str, total_steps: int) -> None:
        """Reset the single progress task for a new phase."""

        self._ensure_started()
        display_epoch = epoch + 1
        description = f"{phase} epoch={display_epoch}"
        if self.task_id is None:
            self.task_id = self.progress.add_task(
                description,
                total=total_steps,
                metrics="loss=--",
                step_ms="--ms",
                eta="--:--",
            )
        else:
            self.progress.reset(
                self.task_id,
                total=total_steps,
                completed=0,
                description=description,
                metrics="loss=--",
                step_ms="--ms",
                eta="--:--",
            )
        self._phase = phase
        self._phase_start_time = time.perf_counter()

    def _estimate_timing(
        self,
        *,
        step: int,
        total_steps: int,
    ) -> tuple[float | None, float | None]:
        if self._phase_start_time is None or step <= 0:
            return None, None
        elapsed = max(time.perf_counter() - self._phase_start_time, 0.0)
        avg_step_seconds = elapsed / step
        eta_seconds = avg_step_seconds * max(total_steps - step, 0)
        return avg_step_seconds, eta_seconds

    def advance(self, step: int) -> None:
        """Advance the current phase progress."""

        if self.task_id is None:
            raise RuntimeError("Progress task has not been initialized.")
        task = self.progress.tasks[self.task_id]
        total_steps = int(task.total) if task.total is not None else step
        avg_step_seconds, eta_seconds = self._estimate_timing(
            step=step,
            total_steps=total_steps,
        )
        step_ms_text = "--ms"
        if avg_step_seconds is not None:
            step_ms_text = f"{avg_step_seconds * 1000.0:.1f}ms"
        self.progress.update(
            self.task_id,
            completed=step,
            step_ms=step_ms_text,
            eta=_format_seconds(eta_seconds),
        )

    def log_metrics(
        self,
        *,
        epoch: int,
        phase: str,
        step: int,
        total_steps: int,
        losses: dict[str, float],
        lr: float,
    ) -> None:
        """Log metrics and update ETA using the fixed logged-step formula."""

        if self.task_id is None:
            raise RuntimeError("Progress task has not been initialized.")

        display_epoch = epoch + 1
        avg_step_seconds, eta_seconds = self._estimate_timing(
            step=step,
            total_steps=total_steps,
        )
        step_ms_text = "--ms"
        if avg_step_seconds is not None:
            step_ms_text = f"{avg_step_seconds * 1000.0:.1f}ms"

        metric_text = (
            f"loss={losses['loss_total']:.4f} "
            f"{self._task_display}={losses['loss_task']:.4f} "
            f"lr={lr:.2e}"
        )
        self.progress.update(
            self.task_id,
            metrics=metric_text,
            step_ms=step_ms_text,
            eta=_format_seconds(eta_seconds),
        )
        if self.enabled:
            self.console.log(
                f"phase={phase} epoch={display_epoch} step={step}/{total_steps} "
                f"loss_total={losses['loss_total']:.6f} "
                f"loss_{self._task_display}={losses['loss_task']:.6f} "
                f"lr={lr:.6e} step_ms={step_ms_text} eta={_format_seconds(eta_seconds)}"
            )

    def close(self) -> None:
        """Stop the underlying Rich progress display."""

        if self._started:
            self.progress.stop()
