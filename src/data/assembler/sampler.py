"""Sample builders for turning assembled per-code arrays into model windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config.config import assembled_dir
from src.data.assembler.assemble import (
    LABEL_COLS,
    MACRO_FEATURES,
    MEZZO_FEATURES,
    MICRO_FEATURES,
    SIDECHAIN_FEATURES,
)
from src.data.registry.dataset import (
    MACRO_LOOKBACK,
    MEZZO_LOOKBACK,
    MICRO_LOOKBACK,
)
from src.data.storage.npy_io import read_npy

DATE_IDX = 0
IS_VALID_IDX = 1
LABEL_SLICE = slice(2, 2 + len(LABEL_COLS))
MACRO_SLICE = slice(LABEL_SLICE.stop, LABEL_SLICE.stop + len(MACRO_FEATURES))
SIDECHAIN_SLICE = slice(MACRO_SLICE.stop, MACRO_SLICE.stop + len(SIDECHAIN_FEATURES))
MEZZO_SLICE = slice(SIDECHAIN_SLICE.stop, SIDECHAIN_SLICE.stop + len(MEZZO_FEATURES) * 8)
MICRO_SLICE = slice(MEZZO_SLICE.stop, MEZZO_SLICE.stop + len(MICRO_FEATURES) * 48)


def _resolve_path(code: str) -> Path:
    path = assembled_dir / f"{code}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Assembled file not found for {code}: {path}")
    return path


def _validate_layout(data: np.ndarray) -> None:
    expected_width = MICRO_SLICE.stop
    if data.ndim != 2:
        raise ValueError(f"Expected assembled array to be 2D, got shape {data.shape}")
    if data.shape[1] != expected_width:
        raise ValueError(
            f"Unexpected assembled width {data.shape[1]}, expected {expected_width}"
        )


def get_samples(code: str) -> dict[str, np.ndarray]:
    """Load ``code.npy`` and assemble rolling samples for model consumption.

    Sampling spec:
    - macro window: last ``MACRO_LOOKBACK`` rows of macro features
    - sidechain window: last ``MACRO_LOOKBACK`` rows of sidechain features
    - mezzo window: last ``MEZZO_LOOKBACK / 8`` rows of per-day mezzo payload
    - micro window: last ``MICRO_LOOKBACK / 48`` rows of per-day micro payload
    - sample validity: AND over ``is_valid_step`` across the daily window
    - label: current row label
    """
    data = read_npy(_resolve_path(code))
    _validate_layout(data)

    mezzo_days = MEZZO_LOOKBACK // 8
    micro_days = MICRO_LOOKBACK // 48
    start_idx = max(MACRO_LOOKBACK, mezzo_days, micro_days) - 1

    if data.shape[0] <= start_idx:
        macro_width = MACRO_SLICE.stop - MACRO_SLICE.start
        sidechain_width = SIDECHAIN_SLICE.stop - SIDECHAIN_SLICE.start
        mezzo_width = MEZZO_SLICE.stop - MEZZO_SLICE.start
        micro_width = MICRO_SLICE.stop - MICRO_SLICE.start
        return {
            "date": np.empty((0,), dtype=np.float32),
            "is_valid": np.empty((0,), dtype=bool),
            "label": np.empty((0, len(LABEL_COLS)), dtype=np.float32),
            "macro": np.empty((0, MACRO_LOOKBACK, macro_width), dtype=np.float32),
            "sidechain": np.empty((0, MACRO_LOOKBACK, sidechain_width), dtype=np.float32),
            "mezzo": np.empty((0, mezzo_days, mezzo_width), dtype=np.float32),
            "micro": np.empty((0, micro_days, micro_width), dtype=np.float32),
        }

    dates = []
    is_valid = []
    labels = []
    macro_samples = []
    sidechain_samples = []
    mezzo_samples = []
    micro_samples = []

    for t in range(start_idx, data.shape[0]):
        macro_window = np.asarray(
            data[t - MACRO_LOOKBACK + 1 : t + 1, MACRO_SLICE], dtype=np.float32
        )
        sidechain_window = np.asarray(
            data[t - MACRO_LOOKBACK + 1 : t + 1, SIDECHAIN_SLICE], dtype=np.float32
        )
        mezzo_window = np.asarray(
            data[t - mezzo_days + 1 : t + 1, MEZZO_SLICE], dtype=np.float32
        )
        micro_window = np.asarray(
            data[t - micro_days + 1 : t + 1, MICRO_SLICE], dtype=np.float32
        )

        dates.append(np.float32(data[t, DATE_IDX]))
        is_valid.append(bool(np.all(data[t - MACRO_LOOKBACK + 1 : t + 1, IS_VALID_IDX] > 0.5)))
        labels.append(np.asarray(data[t, LABEL_SLICE], dtype=np.float32))
        macro_samples.append(macro_window)
        sidechain_samples.append(sidechain_window)
        mezzo_samples.append(mezzo_window)
        micro_samples.append(micro_window)

    return {
        "date": np.asarray(dates, dtype=np.float32),
        "is_valid": np.asarray(is_valid, dtype=bool),
        "label": np.stack(labels).astype(np.float32, copy=False),
        "macro": np.stack(macro_samples).astype(np.float32, copy=False),
        "sidechain": np.stack(sidechain_samples).astype(np.float32, copy=False),
        "mezzo": np.stack(mezzo_samples).astype(np.float32, copy=False),
        "micro": np.stack(micro_samples).astype(np.float32, copy=False),
    }
