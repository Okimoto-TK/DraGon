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
        self.enabled = bool(enabled)
        self.console = console or Console()
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[metrics]}"),
            TextColumn("eta={task.fields[eta]}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            disable=not self.enabled,
        )
        self.task_id: TaskID | None = None
        self._phase = ""
        self._last_log_time: float | None = None
        self._last_log_step = 0
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
                eta="--:--",
            )
        else:
            self.progress.reset(
                self.task_id,
                total=total_steps,
                completed=0,
                description=description,
                metrics="loss=--",
                eta="--:--",
            )
        self._phase = phase
        self._last_log_time = time.perf_counter()
        self._last_log_step = 0

    def advance(self, step: int) -> None:
        """Advance the current phase progress."""

        if self.task_id is None:
            raise RuntimeError("Progress task has not been initialized.")
        self.progress.update(self.task_id, completed=step)

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

        now = time.perf_counter()
        display_epoch = epoch + 1
        eta_seconds: float | None = None
        if self._last_log_time is not None:
            logged_steps = step - self._last_log_step
            if logged_steps > 0:
                avg_step_time = (now - self._last_log_time) / logged_steps
                eta_seconds = avg_step_time * max(total_steps - step, 0)
        self._last_log_time = now
        self._last_log_step = step

        metric_text = (
            f"loss={losses['loss_total']:.4f} "
            f"{self.task}={losses['loss_task']:.4f} "
            f"lr={lr:.2e}"
        )
        self.progress.update(
            self.task_id,
            metrics=metric_text,
            eta=_format_seconds(eta_seconds),
        )
        if self.enabled:
            self.console.log(
                f"phase={phase} epoch={display_epoch} step={step}/{total_steps} "
                f"loss_total={losses['loss_total']:.6f} "
                f"loss_{self.task}={losses['loss_task']:.6f} "
                f"lr={lr:.6e} eta={_format_seconds(eta_seconds)}"
            )

    def close(self) -> None:
        """Stop the underlying Rich progress display."""

        if self._started:
            self.progress.stop()
