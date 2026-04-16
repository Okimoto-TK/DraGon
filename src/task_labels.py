"""Shared task-label definitions for single-target training."""
from __future__ import annotations

from collections.abc import Mapping


def _quantile_labels() -> tuple[str, ...]:
    return tuple(f"p{i:02d}" for i in range(1, 100))


TASK_LABELS = ("ret", "rv", *_quantile_labels())
LABEL_COLUMNS = ("label_ret", "label_rv")


def is_quantile_task(label: str) -> bool:
    return len(label) == 3 and label.startswith("p") and label[1:].isdigit() and 1 <= int(label[1:]) <= 99


def task_target_column(label: str) -> str:
    if label == "ret":
        return "label_ret"
    if label == "rv":
        return "label_rv"
    if is_quantile_task(label):
        return "label_ret"
    raise ValueError(f"Unsupported task label: {label!r}")


def quantile_level(label: str) -> float:
    if not is_quantile_task(label):
        raise ValueError(f"Task {label!r} is not a quantile task.")
    return float(int(label[1:])) / 100.0


def canonical_task_label(label: str) -> str:
    """Validate and return a supported task label."""
    if label not in TASK_LABELS:
        raise ValueError(f"Unsupported task label: {label}. Expected one of {TASK_LABELS[:5]}...{TASK_LABELS[-3:]}.")
    return label


def detect_task_from_outputs(outputs: Mapping[str, object]) -> str:
    """Infer the active task from model output keys."""
    for task in TASK_LABELS:
        if (
            f"pred_{task}" in outputs
            or f"log_var_{task}" in outputs
            or f"log_b_{task}" in outputs
            or f"unc_{task}" in outputs
        ):
            return task
    raise KeyError(f"Could not infer task from outputs keys: {sorted(outputs.keys())}")


__all__ = [
    "LABEL_COLUMNS",
    "TASK_LABELS",
    "canonical_task_label",
    "detect_task_from_outputs",
    "is_quantile_task",
    "quantile_level",
    "task_target_column",
]
