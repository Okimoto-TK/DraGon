"""Minimal web monitor for up to three MLflow runs.

Usage:
    python tools/monitor.py <mlflow_run_url> [<mlflow_run_url> ...]

The script accepts one to three MLflow UI run URLs such as:
    http://host:5000/#/experiments/123/runs/abcde12345

It polls the selected runs in parallel with a ProcessPoolExecutor, keeps a
small in-memory snapshot cache, and serves a lightweight web UI with
matplotlib-rendered charts.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

from mlflow.tracking import MlflowClient

from src.task_labels import TASK_LABELS, canonical_task_label, is_quantile_task

DEFAULT_RUN_URLS: list[str] = [
    # "http://127.0.0.1:5000/#/experiments/1/runs/<run_id>",
]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_POLL_SECONDS = 12.0
DEFAULT_FETCH_TIMEOUT = 20.0
MAX_RUNS = 3
GROUPS_PER_RUN = 5
_RUN_URL_RE = re.compile(r"/?experiments/(?P<experiment_id>[^/]+)/runs/(?P<run_id>[^/?#]+)")

_COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "mu": "#111827",
    "unc": "#0ea5e9",
    "prob": "#14b8a6",
    "quality": "#dc2626",
    "health_a": "#0f766e",
    "health_b": "#7c3aed",
    "health_c": "#ea580c",
    "health_d": "#2563eb",
}


def _get_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    return plt, MaxNLocator


def _base_tracking_uri(parsed_url: Any) -> str:
    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path.rstrip("/"),
            "",
            "",
            "",
        )
    )


def _parse_run_url(run_url: str) -> tuple[str, str]:
    parsed = urlparse(run_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Expected a full MLflow URL, got {run_url!r}")
    for candidate in (parsed.fragment, parsed.path):
        match = _RUN_URL_RE.search(candidate)
        if match is not None:
            return _base_tracking_uri(parsed), str(match.group("run_id"))
    raise ValueError(
        "Could not parse MLflow run ID from URL. Expected something like "
        "'http://host:5000/#/experiments/<exp_id>/runs/<run_id>'."
    )


def _metric_history(client: MlflowClient, run_id: str, key: str) -> list[tuple[int, float]]:
    history = client.get_metric_history(run_id, key)
    if not history:
        return []
    return sorted(
        ((int(point.step), float(point.value)) for point in history),
        key=lambda item: item[0],
    )


def _infer_task(metric_keys: set[str], params: dict[str, str]) -> str:
    explicit = params.get("label")
    if explicit:
        return canonical_task_label(explicit)
    joined = " ".join(sorted(metric_keys))
    for task in TASK_LABELS:
        if f".{task}" in joined or f"_{task}" in joined or task in joined:
            return task
    return "ret"


def _task_metric_keys(task: str) -> dict[str, tuple[str, str] | tuple[str, str, str]]:
    if is_quantile_task(task):
        return {
            "loss_total": ("train.realtime.loss_total", "val.epoch_diag.loss_total"),
            "loss_mu": ("train.realtime.loss_mu", "val.epoch_diag.loss_mu"),
            "loss_unc": ("train.realtime.loss_unc", "val.epoch_diag.loss_unc"),
            "loss_quantile": (f"train.realtime.loss_{task}", f"val.epoch_diag.loss_{task}"),
            "mae": (f"train.realtime.mae.{task}", f"val.epoch_diag.mae.{task}"),
            "unc_mean": (f"train.realtime.unc_mean.{task}", f"val.epoch_diag.unc_mean.{task}"),
        }
    if task == "rv":
        return {
            "loss_total": ("train.realtime.loss_total", "val.epoch_diag.loss_total"),
            "loss_mu": ("train.realtime.loss_mu", "val.epoch_diag.loss_mu"),
            "loss_unc": ("train.realtime.loss_unc", "val.epoch_diag.loss_unc"),
            "loss_rv": ("train.realtime.loss_rv", "val.epoch_diag.loss_rv"),
            "mae": ("train.realtime.mae.rv", "val.epoch_diag.mae.rv"),
            "unc_mean": ("train.realtime.unc_mean.rv", "val.epoch_diag.unc_mean.rv"),
        }
    return {
        "loss_total": ("train.realtime.loss_total", "val.epoch_diag.loss_total"),
        "loss_mu": ("train.realtime.loss_mu", "val.epoch_diag.loss_mu"),
        "loss_unc": ("train.realtime.loss_unc", "val.epoch_diag.loss_unc"),
        "mae": (f"train.realtime.mae.{task}", f"val.epoch_diag.mae.{task}"),
        "unc_mean": (f"train.realtime.unc_mean.{task}", f"val.epoch_diag.unc_mean.{task}"),
    }


def _health_metric_keys() -> dict[str, str]:
    return {
        "lr": "train.realtime.lr",
        "mezzo_joint_norm": "train.realtime.token.mezzo_joint_norm",
        "macro_conditioned_mezzo_norm": "train.realtime.fusion.macro_conditioned_mezzo_norm",
        "micro_refined_mezzo_norm": "train.realtime.fusion.micro_refined_mezzo_norm",
        "macro_to_mezzo_delta": "train.realtime.fusion.macro_to_mezzo_delta",
        "micro_to_mezzo_delta": "train.realtime.fusion.micro_to_mezzo_delta",
        "joint_side_interaction_norm": "train.realtime.fusion.joint_side_interaction_norm",
    }


def _latest_point(points: list[tuple[int, float]]) -> dict[str, float | int] | None:
    if not points:
        return None
    step, value = points[-1]
    return {"step": step, "value": value}


def _summary_entry(points: list[tuple[int, float]]) -> dict[str, float | int] | None:
    if not points:
        return None
    latest = points[-1]
    previous = points[-2] if len(points) > 1 else None
    out: dict[str, float | int] = {
        "step": int(latest[0]),
        "value": float(latest[1]),
    }
    if previous is not None:
        out["previous"] = float(previous[1])
    return out


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return str(numeric)
    abs_value = abs(numeric)
    if abs_value >= 1_000.0:
        return f"{numeric:,.1f}"
    if 0.0 < abs_value < 1e-3:
        return f"{numeric:.2e}"
    return f"{numeric:.4f}"


def _trend_symbol(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "→"
    delta = current - previous
    scale = max(abs(previous), 1e-6)
    if delta > 2e-3 * scale:
        return "↑"
    if delta < -2e-3 * scale:
        return "↓"
    return "→"


def _metric_groups_for_task(task: str) -> list[list[str]]:
    keys = _task_metric_keys(task)
    health = _health_metric_keys()
    if is_quantile_task(task):
        return [
            [keys["loss_total"][0], keys["loss_total"][1]],
            [keys["loss_mu"][0], keys["loss_mu"][1], keys["loss_unc"][0], keys["loss_unc"][1]],
            [
                keys["loss_quantile"][0],
                keys["loss_quantile"][1],
                keys["mae"][0],
                keys["mae"][1],
                keys["unc_mean"][0],
                keys["unc_mean"][1],
            ],
            list(health.values()),
        ]
    return [
        [keys["loss_total"][0], keys["loss_total"][1]],
        [keys["loss_mu"][0], keys["loss_mu"][1], keys["loss_unc"][0], keys["loss_unc"][1]],
        [keys["mae"][0], keys["mae"][1]],
        [keys["unc_mean"][0], keys["unc_mean"][1]],
        list(health.values()),
    ]


def _fetch_run_meta(run_url: str) -> dict[str, Any]:
    tracking_uri, run_id = _parse_run_url(run_url)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    metric_keys = set(run.data.metrics.keys())
    task = _infer_task(metric_keys, run.data.params)
    return {
        "source_url": run_url,
        "tracking_uri": tracking_uri,
        "run_id": run_id,
        "run_name": run.data.tags.get("mlflow.runName", run_id[:8]),
        "task": task,
        "status": run.info.status,
        "groups": _metric_groups_for_task(task),
    }


def _fetch_metric_group(
    tracking_uri: str,
    run_id: str,
    metric_keys: list[str],
) -> dict[str, list[tuple[int, float]]]:
    client = MlflowClient(tracking_uri=tracking_uri)
    metrics: dict[str, list[tuple[int, float]]] = {}
    for key in metric_keys:
        try:
            points = _metric_history(client, run_id, key)
        except Exception:
            continue
        if points:
            metrics[key] = points
    return metrics


def _build_summary(task: str, metrics: dict[str, list[tuple[int, float]]]) -> dict[str, dict[str, float | int] | None]:
    task_keys = _task_metric_keys(task)
    summary: dict[str, dict[str, float | int] | None] = {}
    for label, pair in task_keys.items():
        train_key, val_key = pair
        summary[f"train_{label}"] = _summary_entry(metrics.get(train_key, []))
        summary[f"val_{label}"] = _summary_entry(metrics.get(val_key, []))
    for label, key in _health_metric_keys().items():
        summary[label] = _summary_entry(metrics.get(key, []))
    return summary


@dataclass
class RunState:
    source_url: str
    last_snapshot: dict[str, Any] | None = None
    last_error: str | None = None
    refreshed_at: float | None = None
    version: int = 0


class MonitorStore:
    def __init__(self, run_urls: list[str], poll_seconds: float) -> None:
        self.poll_seconds = poll_seconds
        self._runs = [RunState(source_url=url) for url in run_urls]
        self._lock = threading.Lock()

    def update(self, index: int, snapshot: dict[str, Any] | None, error: str | None) -> None:
        with self._lock:
            run = self._runs[index]
            if snapshot is not None:
                run.last_snapshot = snapshot
                run.last_error = None
                run.refreshed_at = time.time()
                run.version += 1
            else:
                run.last_error = error
                run.refreshed_at = time.time()

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            runs: list[dict[str, Any]] = []
            for index, run in enumerate(self._runs):
                payload: dict[str, Any] = {
                    "index": index,
                    "source_url": run.source_url,
                    "error": run.last_error,
                    "refreshed_at": run.refreshed_at,
                }
                if run.last_snapshot is not None:
                    payload.update(
                        {
                            "run_name": run.last_snapshot["run_name"],
                            "task": run.last_snapshot["task"],
                            "status": run.last_snapshot["status"],
                            "updated_at": run.last_snapshot["updated_at"],
                            "summary": run.last_snapshot["summary"],
                            "version": run.version,
                        }
                    )
                runs.append(payload)
        return {
            "generated_at": time.time(),
            "poll_seconds": self.poll_seconds,
            "runs": runs,
        }

    def get_snapshot(self, index: int) -> dict[str, Any] | None:
        with self._lock:
            if index < 0 or index >= len(self._runs):
                return None
            return self._runs[index].last_snapshot


class Poller(threading.Thread):
    def __init__(
        self,
        *,
        store: MonitorStore,
        run_urls: list[str],
        poll_seconds: float,
        fetch_timeout: float,
    ) -> None:
        super().__init__(daemon=True)
        self.store = store
        self.run_urls = run_urls
        self.poll_seconds = poll_seconds
        self.fetch_timeout = fetch_timeout
        self.stop_event = threading.Event()
        self.executor = ProcessPoolExecutor(max_workers=max(1, len(run_urls) * GROUPS_PER_RUN))

    def shutdown(self) -> None:
        self.stop_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self) -> None:
        while not self.stop_event.is_set():
            loop_started = time.time()
            meta_futures = {
                self.executor.submit(_fetch_run_meta, run_url): index
                for index, run_url in enumerate(self.run_urls)
            }
            meta_results: dict[int, dict[str, Any]] = {}
            for future, index in meta_futures.items():
                try:
                    meta_results[index] = future.result(timeout=self.fetch_timeout)
                except TimeoutError:
                    self.store.update(index, None, "Fetch timeout")
                except Exception as exc:  # pragma: no cover - runtime integration
                    self.store.update(index, None, str(exc))

            group_futures: dict[Any, tuple[int, dict[str, Any]]] = {}
            for index, meta in meta_results.items():
                for metric_keys in meta["groups"]:
                    future = self.executor.submit(
                        _fetch_metric_group,
                        meta["tracking_uri"],
                        meta["run_id"],
                        metric_keys,
                    )
                    group_futures[future] = (index, meta)

            grouped_metrics: dict[int, dict[str, list[tuple[int, float]]]] = {
                index: {} for index in meta_results
            }
            group_errors: dict[int, str] = {}
            for future, (index, meta) in group_futures.items():
                try:
                    payload = future.result(timeout=self.fetch_timeout)
                    grouped_metrics[index].update(payload)
                except TimeoutError:
                    group_errors[index] = "Fetch timeout"
                except Exception as exc:  # pragma: no cover - runtime integration
                    group_errors[index] = str(exc)

            for index, meta in meta_results.items():
                if index in group_errors and not grouped_metrics.get(index):
                    self.store.update(index, None, group_errors[index])
                    continue
                snapshot = {
                    "source_url": meta["source_url"],
                    "tracking_uri": meta["tracking_uri"],
                    "run_id": meta["run_id"],
                    "run_name": meta["run_name"],
                    "task": meta["task"],
                    "status": meta["status"],
                    "summary": _build_summary(meta["task"], grouped_metrics[index]),
                    "metrics": grouped_metrics[index],
                    "updated_at": time.strftime("%H:%M:%S"),
                }
                self.store.update(index, snapshot, group_errors.get(index))

            elapsed = time.time() - loop_started
            remaining = max(0.0, self.poll_seconds - elapsed)
            self.stop_event.wait(remaining)


def _series(snapshot: dict[str, Any], key: str) -> list[tuple[int, float]]:
    return list(snapshot.get("metrics", {}).get(key, []))


def _plot_style() -> None:
    plt, _ = _get_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": "#d1d5db",
            "axes.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#94a3b8",
            "figure.facecolor": "#f8fafc",
            "savefig.facecolor": "#f8fafc",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _empty_png(title: str, message: str) -> bytes:
    _plot_style()
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.axis("off")
    ax.text(0.02, 0.70, title, fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.38, message, fontsize=10, color="#475569", transform=ax.transAxes)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_lines(ax: Any, series_map: list[tuple[str, list[tuple[int, float]], str, str]]) -> None:
    _, MaxNLocator = _get_matplotlib()
    for label, points, color, style in series_map:
        if not points:
            continue
        xs = [step for step, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.8, linestyle=style)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.tick_params(colors="#334155")


def _loss_plot(snapshot: dict[str, Any]) -> bytes:
    task = str(snapshot["task"])
    keys = _task_metric_keys(task)
    _plot_style()
    plt, _ = _get_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)

    total_train = _series(snapshot, keys["loss_total"][0])
    total_val = _series(snapshot, keys["loss_total"][1])
    _plot_lines(
        axes[0],
        [
            ("train total", total_train, _COLORS["train"], "-"),
            ("val total", total_val, _COLORS["val"], "--"),
        ],
    )
    axes[0].set_title("Total Loss")
    axes[0].legend(frameon=False, loc="best")

    components: list[tuple[str, list[tuple[int, float]], str, str]] = [
        ("train mu", _series(snapshot, keys["loss_mu"][0]), _COLORS["mu"], "-"),
        ("train unc", _series(snapshot, keys["loss_unc"][0]), _COLORS["unc"], "-"),
    ]
    if is_quantile_task(task):
        components.append(
            ("train quantile", _series(snapshot, keys["loss_quantile"][0]), _COLORS["prob"], "-")
        )
    elif task == "rv":
        components.append(
            ("train rv", _series(snapshot, keys["loss_rv"][0]), _COLORS["prob"], "-")
        )
    _plot_lines(axes[1], components)
    axes[1].set_title("Train Components")
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle(f"{snapshot['run_name']} · {task} · Loss", x=0.03, ha="left", fontsize=12, fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _task_plot(snapshot: dict[str, Any]) -> bytes:
    task = str(snapshot["task"])
    keys = _task_metric_keys(task)
    _plot_style()
    plt, _ = _get_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)

    _plot_lines(
        axes[0],
        [
            ("train mae", _series(snapshot, keys["mae"][0]), _COLORS["quality"], "-"),
            ("val mae", _series(snapshot, keys["mae"][1]), _COLORS["val"], "--"),
        ],
    )
    axes[0].set_title("Prediction Error")
    uncertainty_series: list[tuple[str, list[tuple[int, float]], str, str]] = [
        ("train unc", _series(snapshot, keys["unc_mean"][0]), _COLORS["unc"], "-"),
        ("val unc", _series(snapshot, keys["unc_mean"][1]), _COLORS["val"], "--"),
    ]
    if is_quantile_task(task):
        uncertainty_series.append(
            ("train quantile", _series(snapshot, keys["loss_quantile"][0]), _COLORS["prob"], "-.")
        )
        axes[1].set_title("Quantile / Uncertainty")
    elif task == "rv":
        uncertainty_series.append(
            ("train rv", _series(snapshot, keys["loss_rv"][0]), _COLORS["prob"], "-.")
        )
        axes[1].set_title("RV / Uncertainty")
    else:
        axes[1].set_title("Uncertainty")
    _plot_lines(axes[1], uncertainty_series)

    fig.suptitle(f"{snapshot['run_name']} · {task} · Task", x=0.03, ha="left", fontsize=12, fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _health_plot(snapshot: dict[str, Any]) -> bytes:
    health = _health_metric_keys()
    _plot_style()
    plt, _ = _get_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)

    _plot_lines(
        axes[0],
        [
            ("mezzo joint", _series(snapshot, health["mezzo_joint_norm"]), _COLORS["health_a"], "-"),
            ("macro->mezzo", _series(snapshot, health["macro_conditioned_mezzo_norm"]), _COLORS["health_b"], "-"),
            ("micro->mezzo", _series(snapshot, health["micro_refined_mezzo_norm"]), _COLORS["health_c"], "-"),
            ("joint×side", _series(snapshot, health["joint_side_interaction_norm"]), _COLORS["health_d"], "-"),
        ],
    )
    axes[0].set_title("Norm Flow")

    _plot_lines(
        axes[1],
        [
            ("macro->mezzo Δ", _series(snapshot, health["macro_to_mezzo_delta"]), _COLORS["health_b"], "-"),
            ("micro->mezzo Δ", _series(snapshot, health["micro_to_mezzo_delta"]), _COLORS["health_c"], "-"),
            ("lr", _series(snapshot, health["lr"]), _COLORS["mu"], "--"),
        ],
    )
    axes[1].set_title("Fusion Health")

    fig.suptitle(f"{snapshot['run_name']} · Health", x=0.03, ha="left", fontsize=12, fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_plot(snapshot: dict[str, Any] | None, kind: str) -> bytes:
    if snapshot is None:
        return _empty_png("Run unavailable", "No cached data yet. Wait for the first polling cycle.")
    if kind == "loss":
        return _loss_plot(snapshot)
    if kind == "task":
        return _task_plot(snapshot)
    if kind == "health":
        return _health_plot(snapshot)
    return _empty_png("Unknown plot", f"Unsupported plot kind: {kind}")


def _html_page(title: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: rgba(255,255,255,0.92);
      --line: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --danger: #dc2626;
      --shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.09), transparent 34%),
        radial-gradient(circle at top right, rgba(20,184,166,0.08), transparent 30%),
        var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px 22px 48px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .title {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
    }}
    .subtitle {{
      color: var(--muted);
      margin-top: 6px;
      font-size: 14px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    .runs {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(226,232,240,0.9);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-head {{
      padding: 18px 18px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .run-line {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .run-name {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chip {{
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
    }}
    .chip.error {{
      background: #fef2f2;
      color: #b91c1c;
      border-color: #fecaca;
    }}
    .source {{
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .summary {{
      padding: 14px 18px 2px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.88);
    }}
    .stat-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat-value {{
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .stat-sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .plots {{
      padding: 16px 18px 18px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .plot {{
      background: rgba(255,255,255,0.85);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    .plot img {{
      display: block;
      width: 100%;
      height: auto;
      background: #f8fafc;
      transition: opacity 0.18s ease;
      opacity: 1;
    }}
    .plot img.loading {{ opacity: 0.96; }}
    @media (max-width: 900px) {{
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .meta {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1 class="title">Dragon Monitor</h1>
        <div class="subtitle">Minimal MLflow live board · key metrics only · auto refresh</div>
      </div>
      <div class="meta" id="meta">Loading…</div>
    </div>
    <div class="runs" id="runs"></div>
  </div>
  <script>
    const runCards = new Map();
    let refreshTimer = null;
    let refreshIntervalMs = 12000;

    function fmt(value) {{
      if (value === null || value === undefined) return "-";
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return String(value);
      const absValue = Math.abs(numeric);
      if (absValue >= 1000) return numeric.toLocaleString(undefined, {{ maximumFractionDigits: 1 }});
      if (absValue > 0 && absValue < 1e-3) return numeric.toExponential(2);
      return numeric.toFixed(4);
    }}

    function trend(entry) {{
      if (!entry || entry.previous === undefined) return "→";
      const current = Number(entry.value);
      const previous = Number(entry.previous);
      const scale = Math.max(Math.abs(previous), 1e-6);
      if (current - previous > 2e-3 * scale) return "↑";
      if (current - previous < -2e-3 * scale) return "↓";
      return "→";
    }}

    function statCard(label, entry) {{
      if (!entry) {{
        return `<div class="stat"><div class="stat-label">${{label}}</div><div class="stat-value">-</div><div class="stat-sub">no data</div></div>`;
      }}
      const step = entry.step !== undefined ? `step ${entry.step}` : "latest";
      return `
        <div class="stat">
          <div class="stat-label">${{label}}</div>
          <div class="stat-value">${{fmt(entry.value)}} <span style="color:#64748b;font-size:14px;">${{trend(entry)}}</span></div>
          <div class="stat-sub">${{step}}</div>
        </div>
      `;
    }}

    function plotImg(kind, runIndex, version) {{
      return `/plot/${{kind}}.png?run=${{runIndex}}&v=${{version}}`;
    }}

    function setImageWhenReady(img, src, version) {{
      if (String(version) === img.dataset.version) return;
      img.classList.add("loading");
      const preload = new Image();
      preload.onload = function() {{
        img.src = src;
        img.dataset.version = String(version);
        img.classList.remove("loading");
      }};
      preload.onerror = function() {{
        img.classList.remove("loading");
      }};
      preload.src = src;
    }}

    function ensureCard(run) {{
      let card = runCards.get(run.index);
      if (card) return card;

      card = document.createElement("section");
      card.className = "card";
      card.dataset.runIndex = String(run.index);
      card.innerHTML = `
        <div class="card-head">
          <div class="run-line">
            <div class="run-name"></div>
            <div class="chips"></div>
          </div>
          <div class="source"></div>
        </div>
        <div class="summary"></div>
        <div class="plots">
          <div class="plot"><img alt="loss plot" data-kind="loss"></div>
          <div class="plot"><img alt="task plot" data-kind="task"></div>
          <div class="plot"><img alt="health plot" data-kind="health"></div>
        </div>
      `;
      runCards.set(run.index, card);
      return card;
    }}

    function updateCard(card, run) {{
      const summary = run.summary || {};
      const version = Number(run.version || 0);
      const chips = [
        run.task ? `<span class="chip">${run.task}</span>` : "",
        run.status ? `<span class="chip">${run.status}</span>` : "",
        run.updated_at ? `<span class="chip">MLflow ${run.updated_at}</span>` : "",
        run.error ? `<span class="chip error">${run.error}</span>` : "",
      ].join("");

      card.querySelector(".run-name").textContent = run.run_name || "Loading…";
      card.querySelector(".chips").innerHTML = chips;
      card.querySelector(".source").textContent = run.source_url;
      card.querySelector(".summary").innerHTML = [
        statCard("Train Total", summary.train_loss_total),
        statCard("Val Total", summary.val_loss_total),
        statCard("Train Mu", summary.train_loss_mu),
        statCard("Train Unc", summary.train_loss_unc),
        statCard("Quality", summary.train_loss_quantile || summary.train_loss_rv || summary.train_mae),
        statCard("LR", summary.lr),
      ].join("");

      for (const img of card.querySelectorAll("img[data-kind]")) {{
        const kind = img.dataset.kind;
        setImageWhenReady(img, plotImg(kind, run.index, version), version);
      }}
    }}

    function renderRuns(payload) {{
      const meta = document.getElementById("meta");
      meta.textContent = `Refresh ${payload.poll_seconds}s · ${new Date(payload.generated_at * 1000).toLocaleTimeString()}`;

      const root = document.getElementById("runs");
      const seen = new Set();

      for (const run of payload.runs) {{
        const card = ensureCard(run);
        updateCard(card, run);
        root.appendChild(card);
        seen.add(run.index);
      }}

      for (const [index, card] of runCards.entries()) {{
        if (!seen.has(index)) {{
          card.remove();
          runCards.delete(index);
        }}
      }}
    }}

    function scheduleRefresh(delayMs) {{
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(refresh, delayMs);
    }}

    async function refresh() {{
      try {{
        const response = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
        const payload = await response.json();
        renderRuns(payload);
        refreshIntervalMs = Math.max(Math.round((payload.poll_seconds || 12) * 1000), 10000);
      }} catch (error) {{
        document.getElementById("meta").textContent = `Monitor fetch failed: ${error}`;
      }} finally {{
        scheduleRefresh(refreshIntervalMs);
      }}
    }}

    refresh();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", title).replace("{{", "{").replace("}}", "}")


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "DragonMonitor/1.0"

    @property
    def store(self) -> MonitorStore:
        return self.server.store  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_html_page("Dragon Monitor"))
            return
        if parsed.path == "/api/state":
            self._send_json(self.store.state_payload())
            return
        if parsed.path.startswith("/plot/") and parsed.path.endswith(".png"):
            kind = parsed.path.split("/")[-1].removesuffix(".png")
            params = parse_qs(parsed.query)
            run_index = int(params.get("run", ["0"])[0])
            snapshot = self.store.get_snapshot(run_index)
            png = _render_plot(snapshot, kind)
            self._send_png(png)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_html(self, payload: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, payload: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live web monitor for up to three MLflow runs.")
    parser.add_argument(
        "run_urls",
        metavar="URL",
        nargs="*",
        help="One to three MLflow run URLs.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the local web server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the local web server.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=DEFAULT_FETCH_TIMEOUT,
        help="Per-run fetch timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_urls = list(args.run_urls or DEFAULT_RUN_URLS)
    if not run_urls:
        raise SystemExit("Provide one to three MLflow run URLs.")
    if len(run_urls) > MAX_RUNS:
        raise SystemExit(f"Expected at most {MAX_RUNS} run URLs, got {len(run_urls)}.")

    for run_url in run_urls:
        _parse_run_url(run_url)

    store = MonitorStore(run_urls=run_urls, poll_seconds=float(args.poll_seconds))
    poller = Poller(
        store=store,
        run_urls=run_urls,
        poll_seconds=float(args.poll_seconds),
        fetch_timeout=float(args.fetch_timeout),
    )
    try:
        server = ReusableThreadingHTTPServer((args.host, int(args.port)), MonitorHandler)
        server.store = store  # type: ignore[attr-defined]
        poller.start()
        print(f"Dragon monitor running at http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        poller.shutdown()
        raise SystemExit(
            f"Failed to bind http://{args.host}:{args.port}: {exc}. "
            "Pick another port or stop the process already listening there."
        ) from exc
    finally:
        poller.shutdown()
        if "server" in locals():
            server.server_close()


if __name__ == "__main__":
    main()
