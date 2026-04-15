"""Packed training datasets backed directly by assembled ``.npz`` tensors."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config.config import assembled_dir
from config.config import lazy_cache_codes as DEFAULT_LAZY_CACHE_CODES
from config.config import val_ratio as DEFAULT_VAL_RATIO
from src.data.assembler.assemble import LABEL_COLS

MemoryMode = Literal["lazy_packed", "auto"]
_LABEL_INDEX = {name: idx for idx, name in enumerate(LABEL_COLS)}
_MANIFEST_NAME = "_packed_manifest.json"


@dataclass(frozen=True)
class SampleIndexTable:
    """Compact per-sample index over packed per-code tensors."""

    codebook: tuple[str, ...]
    code_ids: np.ndarray
    sample_idx: np.ndarray
    date: np.ndarray

    def __len__(self) -> int:
        return int(self.sample_idx.shape[0])

    def subset(self, mask: np.ndarray) -> "SampleIndexTable":
        return SampleIndexTable(
            codebook=self.codebook,
            code_ids=self.code_ids[mask],
            sample_idx=self.sample_idx[mask],
            date=self.date[mask],
        )

    def code_at(self, index: int) -> str:
        return self.codebook[int(self.code_ids[index])]


def discover_codes(root: Path = assembled_dir) -> list[str]:
    """Return codes that already have packed training tensors."""
    manifest = _get_packed_manifest(root)
    return sorted(code for code, meta in manifest.items() if int(meta.get("count", 0)) > 0)


def _packed_path(code: str) -> Path:
    return assembled_dir / f"{code}.npz"


def _manifest_path(root: Path) -> Path:
    return root / _MANIFEST_NAME


def _scan_packed_file(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as packed:
        count = int(np.asarray(packed["date"]).shape[0])
    stat = path.stat()
    return {
        "count": count,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_packed_manifest(root: Path) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for path in sorted(root.glob("*.npz")):
        manifest[path.stem] = _scan_packed_file(path)
    try:
        _manifest_path(root).write_text(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        pass
    return manifest


def _get_packed_manifest(root: Path) -> dict[str, dict[str, int]]:
    manifest_path = _manifest_path(root)
    current_files = {path.stem: path for path in root.glob("*.npz")}

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = None
        else:
            if isinstance(manifest, dict):
                same_keys = set(manifest.keys()) == set(current_files.keys())
                if same_keys:
                    up_to_date = True
                    for code, path in current_files.items():
                        meta = manifest.get(code, {})
                        stat = path.stat()
                        if (
                            int(meta.get("size", -1)) != int(stat.st_size)
                            or int(meta.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
                        ):
                            up_to_date = False
                            break
                    if up_to_date:
                        return {
                            code: {
                                "count": int(meta.get("count", 0)),
                                "size": int(meta.get("size", 0)),
                                "mtime_ns": int(meta.get("mtime_ns", 0)),
                            }
                            for code, meta in manifest.items()
                        }

    return _build_packed_manifest(root)


def build_packed_sample_index(codes: list[str]) -> SampleIndexTable:
    """Build a lightweight index by scanning packed ``.npz`` metadata only."""
    codebook = tuple(codes)
    manifest = _get_packed_manifest(assembled_dir)
    code_id_parts: list[np.ndarray] = []
    sample_idx_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []

    for code_id, code in enumerate(codebook):
        path = _packed_path(code)
        if not path.exists():
            continue
        if int(manifest.get(code, {}).get("count", 0)) <= 0:
            continue

        with np.load(path, allow_pickle=False) as packed:
            date = np.asarray(packed["date"], dtype=np.float32)

        if date.size == 0:
            continue

        count = int(date.shape[0])
        code_id_parts.append(np.full(count, code_id, dtype=np.int32))
        sample_idx_parts.append(np.arange(count, dtype=np.int32))
        date_parts.append(date)

    if not code_id_parts:
        empty_i32 = np.empty((0,), dtype=np.int32)
        empty_f32 = np.empty((0,), dtype=np.float32)
        return SampleIndexTable(
            codebook=codebook,
            code_ids=empty_i32,
            sample_idx=empty_i32,
            date=empty_f32,
        )

    return SampleIndexTable(
        codebook=codebook,
        code_ids=np.concatenate(code_id_parts),
        sample_idx=np.concatenate(sample_idx_parts),
        date=np.concatenate(date_parts),
    )


def split_index_by_date(
    sample_index: SampleIndexTable,
    *,
    split_date: float | int | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> tuple[SampleIndexTable, SampleIndexTable, float | None]:
    """Split samples chronologically using one global date boundary."""
    if len(sample_index) == 0:
        empty_mask = np.zeros((0,), dtype=bool)
        empty = sample_index.subset(empty_mask)
        return empty, empty, split_date

    if split_date is not None:
        resolved_split_date = float(split_date)
    else:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        unique_dates = np.unique(sample_index.date)
        if val_ratio == 0.0 or unique_dates.size <= 1:
            resolved_split_date = None
        else:
            split_pos = int(np.floor(unique_dates.size * (1.0 - val_ratio)))
            split_pos = min(max(split_pos, 1), unique_dates.size - 1)
            resolved_split_date = float(unique_dates[split_pos])

    if resolved_split_date is None:
        train_mask = np.ones(len(sample_index), dtype=bool)
        val_mask = np.zeros(len(sample_index), dtype=bool)
    else:
        train_mask = sample_index.date < resolved_split_date
        val_mask = sample_index.date >= resolved_split_date

    return sample_index.subset(train_mask), sample_index.subset(val_mask), resolved_split_date


class PackedTensorDataset(Dataset[dict[str, Tensor]]):
    """Training-ready dataset backed by per-code packed ``.npz`` tensors."""

    def __init__(
        self,
        sample_index: SampleIndexTable,
        *,
        max_cached_codes: int = DEFAULT_LAZY_CACHE_CODES,
    ) -> None:
        self.sample_index = sample_index
        self.max_cached_codes = max(1, int(max_cached_codes))
        self._cache: OrderedDict[str, dict[str, Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_index)

    def _load_payload(self, code: str) -> dict[str, Tensor]:
        cached = self._cache.pop(code, None)
        if cached is None:
            with np.load(_packed_path(code), allow_pickle=False) as packed:
                cached = {
                    "date": torch.from_numpy(np.ascontiguousarray(packed["date"], dtype=np.float32)),
                    "label": torch.from_numpy(np.ascontiguousarray(packed["label"], dtype=np.float32)),
                    "macro": torch.from_numpy(np.ascontiguousarray(packed["macro"], dtype=np.float32)),
                    "mezzo": torch.from_numpy(np.ascontiguousarray(packed["mezzo"], dtype=np.float32)),
                    "micro": torch.from_numpy(np.ascontiguousarray(packed["micro"], dtype=np.float32)),
                    "sidechain": torch.from_numpy(np.ascontiguousarray(packed["sidechain"], dtype=np.float32)),
                }

        self._cache[code] = cached
        while len(self._cache) > self.max_cached_codes:
            self._cache.popitem(last=False)
        return cached

    def clear_cache(self) -> None:
        self._cache.clear()

    def _item_from_payload(self, payload: dict[str, Tensor], sample_idx: int) -> dict[str, Tensor]:
        label = payload["label"][sample_idx]
        item = {
            "date": payload["date"][sample_idx],
            "macro": payload["macro"][sample_idx],
            "mezzo": payload["mezzo"][sample_idx],
            "micro": payload["micro"][sample_idx],
            "sidechain": payload["sidechain"][sample_idx],
            "label": label,
        }
        for name, idx in _LABEL_INDEX.items():
            item[name] = label[idx]
        return item

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        code = self.sample_index.code_at(index)
        payload = self._load_payload(code)
        sample_idx = int(self.sample_index.sample_idx[index])
        return self._item_from_payload(payload, sample_idx)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Tensor]]:
        items: list[dict[str, Tensor]] = []
        current_code: str | None = None
        current_payload: dict[str, Tensor] | None = None

        for index in indices:
            code = self.sample_index.code_at(index)
            if code != current_code or current_payload is None:
                current_code = code
                current_payload = self._load_payload(code)
            sample_idx = int(self.sample_index.sample_idx[index])
            items.append(self._item_from_payload(current_payload, sample_idx))

        return items


LazyPackedDataset = PackedTensorDataset


def create_train_val_datasets(
    *,
    codes: list[str] | None = None,
    split_date: float | int | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    max_codes: int | None = None,
    filter_invalid: bool = True,
    memory_mode: MemoryMode = "lazy_packed",
) -> tuple[Dataset[dict[str, Tensor]], Dataset[dict[str, Tensor]]]:
    """Create train/validation datasets from packed per-code ``.npz`` tensors."""
    del filter_invalid

    if memory_mode not in {"lazy_packed", "auto"}:
        raise ValueError(
            f"Unsupported memory_mode: {memory_mode}. Packed training only supports "
            "'lazy_packed' or 'auto'."
        )

    selected_codes = discover_codes() if codes is None else list(codes)
    if max_codes is not None:
        selected_codes = selected_codes[:max_codes]

    sample_index = build_packed_sample_index(selected_codes)
    if len(sample_index) == 0 and selected_codes:
        raise FileNotFoundError(
            "Packed training tensors were not found. Run the assembler to generate "
            f"'{assembled_dir}'."
        )

    train_index, val_index, _ = split_index_by_date(
        sample_index,
        split_date=split_date,
        val_ratio=val_ratio,
    )
    return PackedTensorDataset(train_index), PackedTensorDataset(val_index)


__all__ = [
    "LazyPackedDataset",
    "PackedTensorDataset",
    "SampleIndexTable",
    "build_packed_sample_index",
    "create_train_val_datasets",
    "discover_codes",
    "split_index_by_date",
]
