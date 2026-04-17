"""Rich-based training dashboard with in-place refresh."""
from __future__ import annotations

import math
import time
from collections.abc import Mapping

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
)
from rich.table import Table

from config.config import trend_ema_alpha as DEFAULT_TREND_EMA_ALPHA

_DISPLAY_KEYS = [
    "loss_total",
    "loss_task",
    "loss_mu",
    "loss_unc",
    "loss_task/ret",
    "loss_task/rv",
    "loss_task/p90",
    "corr/ret",
    "mae/ret",
    "mae/rv",
    "mae/p90",
    "unc_mean/ret",
    "unc_mean/rv",
    "unc_mean/p90",
]


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value) or math.isinf(value):
        return str(value)
    return f"{value:.6f}"


def _trend_symbol(symbol: str | None) -> str:
    return symbol or "→"


def _metrics_table(
    title: str,
    metrics: Mapping[str, float],
    trends: Mapping[str, str],
) -> Panel:
    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        expand=True,
        pad_edge=False,
    )
    table.add_column(justify="left", no_wrap=True, overflow="ellipsis")

    visible_keys = [key for key in _DISPLAY_KEYS if key in metrics]
    if not visible_keys:
        visible_keys = sorted(metrics.keys())

    if visible_keys:
        for key in visible_keys:
            table.add_row(f"{key}: {_format_metric(metrics[key])} {_trend_symbol(trends.get(key))}")
    else:
        table.add_row("status: -")

    return Panel(table, title=title, border_style="cyan")


class TrainingUI:
    """Manage a compact live dashboard for training progress."""

    def __init__(self, *, ema_alpha: float = DEFAULT_TREND_EMA_ALPHA) -> None:
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.fields[step_time]}"),
            expand=True,
        )
        self._ema_alpha = float(ema_alpha)
        self._task_id: TaskID | None = None
        self._train_metrics: dict[str, float] = {}
        self._val_metrics: dict[str, float] = {}
        self._train_ema: dict[str, float] = {}
        self._val_ema: dict[str, float] = {}
        self._train_trends: dict[str, str] = {}
        self._val_trends: dict[str, str] = {}
        self._status = "idle"
        self._last_step_time: float | None = None
        self._live = Live(
            self._render(),
            refresh_per_second=8,
            transient=False,
        )

    def _format_step_time(self, elapsed_seconds: float | None) -> str:
        if elapsed_seconds is None or elapsed_seconds <= 0.0:
            return "step --"
        if elapsed_seconds < 1.0:
            return f"step {elapsed_seconds * 1000.0:.0f}ms"
        return f"step {elapsed_seconds:.2f}s"

    def _render(self) -> Group:
        progress_panel = Panel(
            self._progress,
            title=self._status,
            border_style="green",
        )
        return Group(
            progress_panel,
            Columns(
                [
                    _metrics_table("Train Metrics", self._train_metrics, self._train_trends),
                    _metrics_table("Val Metrics", self._val_metrics, self._val_trends),
                ],
                expand=True,
                equal=True,
            ),
        )

    def start(self) -> None:
        self._live.start()
        self.refresh()

    def stop(self) -> None:
        self._live.stop()

    def refresh(self) -> None:
        self._live.update(self._render(), refresh=True)

    def start_epoch(self, epoch: int, num_epochs: int, total_steps: int) -> None:
        self._status = f"Epoch {epoch}/{num_epochs}"
        self._last_step_time = None
        self._train_metrics = {}
        self._train_ema = {}
        self._train_trends = {}
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
            self._task_id = None
        self._task_id = self._progress.add_task("train", total=max(total_steps, 1), step_time="step --")
        self.refresh()

    def _update_trends(
        self,
        *,
        metrics: Mapping[str, float],
        ema_store: dict[str, float],
        trend_store: dict[str, str],
    ) -> None:
        for key, value in metrics.items():
            previous = ema_store.get(key)
            if previous is None:
                ema_store[key] = float(value)
                trend_store[key] = "→"
                continue

            current = self._ema_alpha * float(value) + (1.0 - self._ema_alpha) * previous
            delta = current - previous
            scale = max(abs(previous), 1e-2)
            weak_threshold = 5e-4 * scale
            strong_threshold = 2e-3 * scale
            if delta > strong_threshold:
                trend_store[key] = "↑"
            elif delta > weak_threshold:
                trend_store[key] = "↗"
            elif delta < -strong_threshold:
                trend_store[key] = "↓"
            elif delta < -weak_threshold:
                trend_store[key] = "↘"
            else:
                trend_store[key] = "→"
            ema_store[key] = current

    def update_train_step(
        self,
        *,
        step: int,
        total_steps: int,
        metrics: Mapping[str, float],
        step_time_seconds: float | None = None,
    ) -> None:
        if self._task_id is None:
            self.start_epoch(1, 1, total_steps)
        self._train_metrics = dict(metrics)
        self._update_trends(
            metrics=self._train_metrics,
            ema_store=self._train_ema,
            trend_store=self._train_trends,
        )
        self._progress.update(
            self._task_id,
            completed=min(step, max(total_steps, 1)),
            total=max(total_steps, 1),
            step_time=self._format_step_time(step_time_seconds),
        )
        self.refresh()

    def set_val_metrics(self, metrics: Mapping[str, float]) -> None:
        self._val_metrics = dict(metrics)
        self._update_trends(
            metrics=self._val_metrics,
            ema_store=self._val_ema,
            trend_store=self._val_trends,
        )
        self.refresh()

    def set_status(self, status: str) -> None:
        self._status = status
        self.refresh()
