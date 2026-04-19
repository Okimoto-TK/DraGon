"""Assembled NPZ dataset and network-batch adapters."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

NETWORK_SAMPLE_SHAPES: dict[str, tuple[int, ...]] = {
    "macro_float_long": (9, 112),
    "macro_i8_long": (2, 112),
    "mezzo_float_long": (9, 144),
    "mezzo_i8_long": (2, 144),
    "micro_float_long": (9, 192),
    "micro_i8_long": (2, 192),
    "sidechain_cond": (13, 64),
    "target_ret": (1,),
    "target_rv": (1,),
    "target_q": (1,),
}

FLOAT_BATCH_KEYS = (
    "macro_float_long",
    "mezzo_float_long",
    "micro_float_long",
    "sidechain_cond",
    "target_ret",
    "target_rv",
    "target_q",
)
INT_BATCH_KEYS = ("macro_i8_long", "mezzo_i8_long", "micro_i8_long")
REQUIRED_RAW_KEYS = (
    "label",
    "macro",
    "macro_i8",
    "mezzo",
    "mezzo_i8",
    "micro",
    "micro_i8",
    "sidechain",
)


def _ensure_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} shape mismatch: expected {expected}, got {tuple(value.shape)}."
        )


def _validate_network_sample(sample: dict[str, np.ndarray]) -> None:
    missing = [key for key in NETWORK_SAMPLE_SHAPES if key not in sample]
    if missing:
        raise ValueError(f"Missing network sample keys: {missing}.")

    for key, expected in NETWORK_SAMPLE_SHAPES.items():
        _ensure_shape(key, sample[key], expected)


def _extract_sample_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    local_index: int,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"Missing assembled key {key!r} in NPZ file.")
    return np.asarray(archive[key][local_index])


def _adapt_scale_float(
    float_data: np.ndarray,
    int_data: np.ndarray,
    *,
    float_key: str,
    int_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    expected_float_len = NETWORK_SAMPLE_SHAPES[float_key][1]
    expected_int_shape = NETWORK_SAMPLE_SHAPES[int_key]
    _ensure_shape(int_key, int_data, expected_int_shape)

    if float_data.ndim != 2:
        raise ValueError(
            f"{float_key} source must be rank-2, got shape {tuple(float_data.shape)}."
        )
    if float_data.shape[1] != expected_float_len:
        raise ValueError(
            f"{float_key} source length mismatch: expected {expected_float_len}, "
            f"got {float_data.shape[1]}."
        )
    if float_data.shape[0] != NETWORK_SAMPLE_SHAPES[float_key][0]:
        raise ValueError(
            f"{float_key} source channel mismatch: expected {NETWORK_SAMPLE_SHAPES[float_key][0]}, "
            f"got {float_data.shape[0]}."
        )
    adapted_float = np.asarray(float_data, dtype=np.float32)
    _ensure_shape(float_key, adapted_float, NETWORK_SAMPLE_SHAPES[float_key])
    return adapted_float, np.asarray(int_data, dtype=np.int8)


def _adapt_macro(raw_sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _adapt_scale_float(
        raw_sample["macro"],
        raw_sample["macro_i8"],
        float_key="macro_float_long",
        int_key="macro_i8_long",
    )


def _adapt_mezzo(raw_sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _adapt_scale_float(
        raw_sample["mezzo"],
        raw_sample["mezzo_i8"],
        float_key="mezzo_float_long",
        int_key="mezzo_i8_long",
    )


def _adapt_micro(raw_sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _adapt_scale_float(
        raw_sample["micro"],
        raw_sample["micro_i8"],
        float_key="micro_float_long",
        int_key="micro_i8_long",
    )


def _adapt_sidechain(raw_sample: dict[str, np.ndarray]) -> np.ndarray:
    sidechain = np.asarray(raw_sample["sidechain"], dtype=np.float32)
    if sidechain.ndim != 2:
        raise ValueError(
            f"sidechain source must be rank-2, got shape {tuple(sidechain.shape)}."
        )
    if sidechain.shape[1] < 64:
        raise ValueError(
            f"sidechain source length mismatch: expected at least 64, got {sidechain.shape[1]}."
        )
    if sidechain.shape[0] != NETWORK_SAMPLE_SHAPES["sidechain_cond"][0]:
        raise ValueError(
            "sidechain source channel mismatch: "
            f"expected {NETWORK_SAMPLE_SHAPES['sidechain_cond'][0]}, got {sidechain.shape[0]}."
        )
    adapted = sidechain[:, -64:]
    _ensure_shape("sidechain_cond", adapted, NETWORK_SAMPLE_SHAPES["sidechain_cond"])
    return adapted


def _adapt_targets(raw_sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(raw_sample["label"], dtype=np.float32).reshape(-1)
    if labels.shape[0] < 2:
        raise ValueError(
            f"label source width mismatch: expected at least 2, got {labels.shape[0]}."
        )
    target_ret = labels[0:1].astype(np.float32, copy=False)
    target_rv = labels[1:2].astype(np.float32, copy=False)
    target_q = target_ret
    return target_ret, target_rv, target_q


class AssembledNPZDataset(torch.utils.data.Dataset):
    """Read training-ready assembled NPZ files and adapt them to network samples."""

    def __init__(
        self,
        file_paths: list[str],
        mmap_mode: str | None = None,
        validate_shapes: bool = True,
        max_open_archives: int = 16,
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths must not be empty.")
        if max_open_archives < 1:
            raise ValueError("max_open_archives must be at least 1.")

        self.file_paths = [Path(path) for path in file_paths]
        self.mmap_mode = mmap_mode
        self.validate_shapes = bool(validate_shapes)
        self.max_open_archives = int(max_open_archives)
        self._archive_cache: OrderedDict[int, np.lib.npyio.NpzFile] = OrderedDict()
        self._sample_index, self._file_sample_ranges = self._build_sample_index()

    def _build_sample_index(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        index: list[tuple[int, int]] = []
        file_sample_ranges: list[tuple[int, int]] = []
        for file_id, path in enumerate(self.file_paths):
            with np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False) as archive:
                for key in REQUIRED_RAW_KEYS:
                    if key not in archive:
                        raise ValueError(f"Missing assembled key {key!r} in {path}.")
                sample_count = int(np.asarray(archive["label"]).shape[0])
            start = len(index)
            index.extend((file_id, local_index) for local_index in range(sample_count))
            file_sample_ranges.append((start, len(index)))
        if not index:
            raise ValueError("AssembledNPZDataset found zero samples.")
        return index, file_sample_ranges

    def _get_archive(self, file_id: int) -> np.lib.npyio.NpzFile:
        archive = self._archive_cache.get(file_id)
        if archive is not None:
            self._archive_cache.move_to_end(file_id)
            return archive

        archive = np.load(
            self.file_paths[file_id],
            mmap_mode=self.mmap_mode,
            allow_pickle=False,
        )
        self._archive_cache[file_id] = archive
        if len(self._archive_cache) > self.max_open_archives:
            _, evicted = self._archive_cache.popitem(last=False)
            evicted.close()
        return archive

    @property
    def file_sample_ranges(self) -> list[tuple[int, int]]:
        return list(self._file_sample_ranges)

    def close(self) -> None:
        while self._archive_cache:
            _, archive = self._archive_cache.popitem(last=False)
            archive.close()

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        file_id, local_index = self._sample_index[index]
        archive = self._get_archive(file_id)
        raw_sample = {
            key: _extract_sample_array(archive, key, local_index) for key in REQUIRED_RAW_KEYS
        }

        macro_float_long, macro_i8_long = _adapt_macro(raw_sample)
        mezzo_float_long, mezzo_i8_long = _adapt_mezzo(raw_sample)
        micro_float_long, micro_i8_long = _adapt_micro(raw_sample)
        sidechain_cond = _adapt_sidechain(raw_sample)
        target_ret, target_rv, target_q = _adapt_targets(raw_sample)

        sample = {
            "macro_float_long": macro_float_long,
            "macro_i8_long": macro_i8_long,
            "mezzo_float_long": mezzo_float_long,
            "mezzo_i8_long": mezzo_i8_long,
            "micro_float_long": micro_float_long,
            "micro_i8_long": micro_i8_long,
            "sidechain_cond": sidechain_cond,
            "target_ret": target_ret,
            "target_rv": target_rv,
            "target_q": target_q,
        }
        if self.validate_shapes:
            _validate_network_sample(sample)
        return sample

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
