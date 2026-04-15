"""Shared task-label definitions."""
from __future__ import annotations

from collections.abc import Mapping

TASK_LABELS = ("Edge", "Persist", "DownRisk")
LABEL_COLUMNS = tuple(f"label_{task}" for task in TASK_LABELS)


def canonical_task_label(label: str) -> str:
    """Validate and return a supported task label."""
    if label not in TASK_LABELS:
        raise ValueError(f"Unsupported task label: {label}. Expected one of {TASK_LABELS}.")
    return label


def detect_task_from_outputs(outputs: Mapping[str, object]) -> str:
    """Infer the active task from model output keys."""
    for task in TASK_LABELS:
        if (
            f"pred_{task}" in outputs
            or f"logit_{task}" in outputs
            or f"pred_log_{task}" in outputs
            or task in outputs
        ):
            return task
    raise KeyError(f"Could not infer task from outputs keys: {sorted(outputs.keys())}")


__all__ = ["LABEL_COLUMNS", "TASK_LABELS", "canonical_task_label", "detect_task_from_outputs"]
