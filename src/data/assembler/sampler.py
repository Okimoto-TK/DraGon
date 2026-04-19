"""Sample builders for turning assembled per-code arrays into model windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config.data import assembled_dir
from src.data.assembler.assemble import (
    LABEL_COLS,
    MACRO_FEATURES,
    MEZZO_FEATURES,
    MICRO_FEATURES,
    SIDECHAIN_FEATURES,
    MEZZO_BARS_PER_DAY,
    MICRO_BARS_PER_DAY,
)
from src.data.registry.dataset import (
    MACRO_LOOKBACK,
    MEZZO_LOOKBACK,
    MICRO_LOOKBACK,
    WARMUP_BARS,
)
from src.data.storage.npy_io import read_npy

DATE_IDX = 0
IS_VALID_IDX = 1
LABEL_SLICE = slice(2, 2 + len(LABEL_COLS))
MACRO_SLICE = slice(LABEL_SLICE.stop, LABEL_SLICE.stop + len(MACRO_FEATURES))
SIDECHAIN_SLICE = slice(MACRO_SLICE.stop, MACRO_SLICE.stop + len(SIDECHAIN_FEATURES))
MEZZO_SLICE = slice(SIDECHAIN_SLICE.stop, SIDECHAIN_SLICE.stop + len(MEZZO_FEATURES) * MEZZO_BARS_PER_DAY)
MICRO_USED_FEATURES = MICRO_FEATURES
FULL_MICRO_SLICE = slice(MEZZO_SLICE.stop, MEZZO_SLICE.stop + len(MICRO_FEATURES) * MICRO_BARS_PER_DAY)
MICRO_SLICE = slice(MEZZO_SLICE.stop, MEZZO_SLICE.stop + len(MICRO_USED_FEATURES) * MICRO_BARS_PER_DAY)

# Daily window sizes (lookback + warmup, expressed in days)
MACRO_WINDOW_DAYS = MACRO_LOOKBACK + WARMUP_BARS
MEZZO_WINDOW_DAYS = MEZZO_LOOKBACK // MEZZO_BARS_PER_DAY + (WARMUP_BARS + MEZZO_BARS_PER_DAY - 1) // MEZZO_BARS_PER_DAY
MICRO_WINDOW_DAYS = MICRO_LOOKBACK // MICRO_BARS_PER_DAY + (WARMUP_BARS + MICRO_BARS_PER_DAY - 1) // MICRO_BARS_PER_DAY

# Total bar counts per scale
MACRO_TOTAL_BARS = MACRO_WINDOW_DAYS
MEZZO_TOTAL_BARS = MEZZO_WINDOW_DAYS * MEZZO_BARS_PER_DAY
MICRO_TOTAL_BARS = MICRO_WINDOW_DAYS * MICRO_BARS_PER_DAY


def _resolve_path(code: str) -> Path:
    path = assembled_dir / f"{code}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Assembled file not found for {code}: {path}")
    return path


def _validate_layout(data: np.ndarray) -> None:
    expected_width = FULL_MICRO_SLICE.stop
    if data.ndim != 2:
        raise ValueError(f"Expected assembled array to be 2D, got shape {data.shape}")
    if data.shape[1] != expected_width:
        raise ValueError(
            f"Unexpected assembled width {data.shape[1]}, expected {expected_width}"
        )


def _window_spec() -> tuple[int, int, int]:
    start_idx = max(MACRO_WINDOW_DAYS, MEZZO_WINDOW_DAYS, MICRO_WINDOW_DAYS) - 1
    return MEZZO_WINDOW_DAYS, MICRO_WINDOW_DAYS, start_idx


def _compute_sample_valid(step_valid: np.ndarray) -> np.ndarray:
    macro_valid = sliding_window_view(step_valid, MACRO_WINDOW_DAYS).all(axis=-1)
    mezzo_valid = sliding_window_view(step_valid, MEZZO_WINDOW_DAYS).all(axis=-1)
    micro_valid = sliding_window_view(step_valid, MICRO_WINDOW_DAYS).all(axis=-1)
    start_idx = max(MACRO_WINDOW_DAYS, MEZZO_WINDOW_DAYS, MICRO_WINDOW_DAYS) - 1
    return (
        macro_valid[start_idx - MACRO_WINDOW_DAYS + 1 :]
        & mezzo_valid[start_idx - MEZZO_WINDOW_DAYS + 1 :]
        & micro_valid[start_idx - MICRO_WINDOW_DAYS + 1 :]
    )


def build_sample_index_for_code(code: str) -> dict[str, np.ndarray]:
    """Build a lightweight per-code sample index without materializing windows."""
    data = read_npy(_resolve_path(code))
    _validate_layout(data)

    _, _, start_idx = _window_spec()
    n_rows = data.shape[0]
    if n_rows <= start_idx:
        return {
            "t": np.empty((0,), dtype=np.int32),
            "sample_idx": np.empty((0,), dtype=np.int32),
            "date": np.empty((0,), dtype=np.float32),
            "is_valid": np.empty((0,), dtype=bool),
        }

    sample_count = n_rows - start_idx
    t = np.arange(start_idx, n_rows, dtype=np.int32)
    sample_idx = np.arange(sample_count, dtype=np.int32)
    date = np.asarray(data[start_idx:, DATE_IDX], dtype=np.float32)
    step_valid = np.asarray(data[:, IS_VALID_IDX] > 0.5, dtype=bool)
    is_valid = _compute_sample_valid(step_valid)

    return {
        "t": t,
        "sample_idx": sample_idx,
        "date": date,
        "is_valid": is_valid,
    }


def _extract_sample_from_array(data: np.ndarray, t: int) -> dict[str, np.ndarray]:
    return {
        "date": np.asarray(data[t, DATE_IDX], dtype=np.float32),
        "label": np.asarray(data[t, LABEL_SLICE], dtype=np.float32),
        "macro": np.asarray(data[t - MACRO_WINDOW_DAYS + 1 : t + 1, MACRO_SLICE], dtype=np.float32),
        "sidechain": np.asarray(
            data[t - MACRO_WINDOW_DAYS + 1 : t + 1, SIDECHAIN_SLICE],
            dtype=np.float32,
        ),
        "mezzo": np.asarray(data[t - MEZZO_WINDOW_DAYS + 1 : t + 1, MEZZO_SLICE], dtype=np.float32),
        "micro": np.asarray(data[t - MICRO_WINDOW_DAYS + 1 : t + 1, MICRO_SLICE], dtype=np.float32),
    }


def get_sample_at(code: str, t: int) -> dict[str, np.ndarray]:
    """Load one rolling sample at absolute row index ``t`` from ``code.npy``."""
    data = read_npy(_resolve_path(code))
    _validate_layout(data)
    if t < 0 or t >= data.shape[0]:
        raise IndexError(f"Sample row {t} out of bounds for {code} with {data.shape[0]} rows")
    return _extract_sample_from_array(data, t)


def get_samples(code: str) -> dict[str, np.ndarray]:
    """Load ``code.npy`` and assemble rolling samples for model consumption.

    Sampling spec:
    - macro window: last ``MACRO_WINDOW_DAYS`` rows of macro features (MACRO_TOTAL_BARS bars)
    - sidechain window: last ``MACRO_WINDOW_DAYS`` rows of sidechain features
    - mezzo window: last ``MEZZO_WINDOW_DAYS`` rows of per-day mezzo payload (MEZZO_TOTAL_BARS bars)
    - micro window: last ``MICRO_WINDOW_DAYS`` rows of per-day micro payload (MICRO_TOTAL_BARS bars)
    - sample validity: AND over ``is_valid_step`` across the daily window
    - label: current row label
    """
    data = read_npy(_resolve_path(code))
    _validate_layout(data)

    mezzo_days, micro_days, start_idx = _window_spec()

    if data.shape[0] <= start_idx:
        macro_width = MACRO_SLICE.stop - MACRO_SLICE.start
        sidechain_width = SIDECHAIN_SLICE.stop - SIDECHAIN_SLICE.start
        mezzo_width = MEZZO_SLICE.stop - MEZZO_SLICE.start
        micro_width = MICRO_SLICE.stop - MICRO_SLICE.start
        return {
            "date": np.empty((0,), dtype=np.float32),
            "is_valid": np.empty((0,), dtype=bool),
            "label": np.empty((0, len(LABEL_COLS)), dtype=np.float32),
            "macro": np.empty((0, MACRO_TOTAL_BARS, macro_width), dtype=np.float32),
            "sidechain": np.empty((0, MACRO_TOTAL_BARS, sidechain_width), dtype=np.float32),
            "mezzo": np.empty((0, mezzo_days, mezzo_width), dtype=np.float32),
            "micro": np.empty((0, micro_days, micro_width), dtype=np.float32),
        }

    sample_index = build_sample_index_for_code(code)
    dates = []
    labels = []
    macro_samples = []
    sidechain_samples = []
    mezzo_samples = []
    micro_samples = []

    for t in sample_index["t"]:
        sample = _extract_sample_from_array(data, int(t))
        dates.append(sample["date"])
        labels.append(sample["label"])
        macro_samples.append(sample["macro"])
        sidechain_samples.append(sample["sidechain"])
        mezzo_samples.append(sample["mezzo"])
        micro_samples.append(sample["micro"])

    return {
        "date": np.asarray(dates, dtype=np.float32),
        "is_valid": np.asarray(sample_index["is_valid"], dtype=bool),
        "label": np.stack(labels).astype(np.float32, copy=False),
        "macro": np.stack(macro_samples).astype(np.float32, copy=False),
        "sidechain": np.stack(sidechain_samples).astype(np.float32, copy=False),
        "mezzo": np.stack(mezzo_samples).astype(np.float32, copy=False),
        "micro": np.stack(micro_samples).astype(np.float32, copy=False),
    }
