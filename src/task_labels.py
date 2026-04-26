"""Shared field-label definitions for single-target training."""
from __future__ import annotations

from collections.abc import Mapping


def _quantile_labels() -> tuple[str, ...]:
    return tuple(f"p{i:02d}" for i in range(1, 100))


TASK_LABELS = ("ret", "rv", *_quantile_labels())
TRAINING_TASKS = ("mu", "sigma")
LABEL_COLUMNS = ("label_ret", "label_rv")


def is_quantile_task(label: str) -> bool:
    return len(label) == 3 and label.startswith("p") and label[1:].isdigit() and 1 <= int(label[1:]) <= 99


def canonical_task_label(label: str) -> str:
    """Validate and return a supported field label."""
    if label not in TASK_LABELS:
        raise ValueError(
            f"Unsupported task label: {label}. Expected one of {TASK_LABELS[:5]}...{TASK_LABELS[-3:]}."
        )
    return label


def canonical_training_task(task: str) -> str:
    """Validate and return a supported optimization stage."""
    if task not in TRAINING_TASKS:
        raise ValueError(f"Unsupported training task: {task!r}. Expected one of {TRAINING_TASKS}.")
    return task


def field_domain(label: str) -> str:
    label = canonical_task_label(label)
    if label == "ret":
        return "ret"
    if label == "rv":
        return "rv"
    return "q"


def task_target_column(label: str) -> str:
    label = canonical_task_label(label)
    if label == "ret":
        return "label_ret"
    if label == "rv":
        return "label_rv"
    return "label_ret"


def field_target_key(label: str) -> str:
    label = canonical_task_label(label)
    if label == "ret":
        return "target_ret"
    if label == "rv":
        return "target_rv"
    return "target_q"


def quantile_level(label: str) -> float:
    if not is_quantile_task(label):
        raise ValueError(f"Task {label!r} is not a quantile task.")
    return float(int(label[1:])) / 100.0


def detect_task_from_outputs(outputs: Mapping[str, object]) -> str:
    """Infer the active field from model output keys."""
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
    "TRAINING_TASKS",
    "canonical_task_label",
    "canonical_training_task",
    "detect_task_from_outputs",
    "field_domain",
    "field_target_key",
    "is_quantile_task",
    "quantile_level",
    "task_target_column",
]
