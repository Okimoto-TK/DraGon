"""Packed training datasets backed directly by assembled ``.npz`` tensors."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
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
from src.data.assembler.assemble import PACKED_LABEL_SCHEMA_VERSION

MemoryMode = Literal["lazy_packed", "auto"]
_LABEL_INDEX = {name: idx for idx, name in enumerate(LABEL_COLS)}
_MANIFEST_NAME = "_packed_manifest.json"


def _resolve_scan_workers(
    requested_workers: int | None,
    *,
    task_count: int,
) -> int:
    if task_count <= 1:
        return 1
    cpu_limit = max(1, (os.cpu_count() or 1) // 2)
    if requested_workers is None:
        return min(cpu_limit, task_count)
    return max(1, min(int(requested_workers), cpu_limit, task_count))


def _validate_packed_schema(packed: np.lib.npyio.NpzFile, *, path: Path) -> None:
    if "label_schema_version" not in packed or "label_names" not in packed:
        raise ValueError(
            f"Packed tensor file {path} is missing label schema metadata. "
            "Rebuild assembled data with `dragon prepare assembled`."
        )

    schema_version = int(np.asarray(packed["label_schema_version"]).reshape(()).item())
    label_names_raw = np.asarray(packed["label_names"])
    label_names = [str(name) for name in label_names_raw.reshape(-1).tolist()]
    if schema_version != int(PACKED_LABEL_SCHEMA_VERSION) or label_names != list(LABEL_COLS):
        raise ValueError(
            f"Packed tensor file {path} has label schema {schema_version}:{label_names}, "
            f"expected {PACKED_LABEL_SCHEMA_VERSION}:{LABEL_COLS}. "
            "Rebuild processed labels and assembled data."
        )

    if "label" not in packed:
        raise ValueError(f"Packed tensor file {path} is missing `label` payload.")
    label = np.asarray(packed["label"])
    if label.ndim != 2 or int(label.shape[1]) != len(LABEL_COLS):
        raise ValueError(
            f"Packed tensor file {path} has label shape {tuple(label.shape)}, "
            f"expected (*, {len(LABEL_COLS)}). Rebuild assembled data."
        )


@dataclass(frozen=True)
class SampleIndexTable:
    """Compact per-sample index over packed payload shards."""

    payloadbook: tuple[str, ...]
    payload_ids: np.ndarray
    sample_idx: np.ndarray
    date: np.ndarray

    def __len__(self) -> int:
        return int(self.sample_idx.shape[0])

    def subset(self, mask: np.ndarray) -> "SampleIndexTable":
        return SampleIndexTable(
            payloadbook=self.payloadbook,
            payload_ids=self.payload_ids[mask],
            sample_idx=self.sample_idx[mask],
            date=self.date[mask],
        )

    def payload_at(self, index: int) -> str:
        return self.payloadbook[int(self.payload_ids[index])]


def _payload_code(stem: str) -> str:
    return stem.split("__", 1)[0]


def discover_codes(root: Path = assembled_dir, *, scan_workers: int | None = None) -> list[str]:
    """Return logical codes that already have packed training tensors."""
    manifest = _get_packed_manifest(root, scan_workers=scan_workers)
    totals: dict[str, int] = {}
    for stem, meta in manifest.items():
        code = str(meta.get("code", _payload_code(stem)))
        totals[code] = totals.get(code, 0) + int(meta.get("count", 0))
    return sorted(code for code, total in totals.items() if total > 0)


def _packed_path(payload_name: str) -> Path:
    return assembled_dir / f"{payload_name}.npz"


def _manifest_path(root: Path) -> Path:
    return root / _MANIFEST_NAME


def _scan_packed_file(path: Path) -> dict[str, int | str]:
    with np.load(path, allow_pickle=False) as packed:
        _validate_packed_schema(packed, path=path)
        count = int(np.asarray(packed["date"]).shape[0])
    stat = path.stat()
    return {
        "code": _payload_code(path.stem),
        "count": count,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "schema_version": int(PACKED_LABEL_SCHEMA_VERSION),
        "label_names": list(LABEL_COLS),
    }


def _load_packed_dates(path: Path) -> tuple[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as packed:
        _validate_packed_schema(packed, path=path)
        date = np.asarray(packed["date"], dtype=np.float32)
    return path.stem, date


def _build_packed_manifest(
    root: Path,
    *,
    scan_workers: int | None = None,
) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    paths = sorted(root.glob("*.npz"))
    max_workers = _resolve_scan_workers(scan_workers, task_count=len(paths))
    if max_workers <= 1:
        for path in paths:
            manifest[path.stem] = _scan_packed_file(path)
    else:
        ctx = mp.get_context("spawn")
        chunksize = max(1, min(64, len(paths) // (max_workers * 4) or 1))
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for path, meta in zip(
                paths,
                executor.map(_scan_packed_file, paths, chunksize=chunksize),
                strict=True,
            ):
                manifest[path.stem] = meta
    try:
        _manifest_path(root).write_text(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        pass
    return manifest


def _get_packed_manifest(
    root: Path,
    *,
    scan_workers: int | None = None,
) -> dict[str, dict[str, int | str]]:
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
                    for payload_name, path in current_files.items():
                        meta = manifest.get(payload_name, {})
                        stat = path.stat()
                        if (
                            str(meta.get("code", "")) != _payload_code(payload_name)
                            or
                            int(meta.get("size", -1)) != int(stat.st_size)
                            or int(meta.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
                            or int(meta.get("schema_version", -1)) != int(PACKED_LABEL_SCHEMA_VERSION)
                            or list(meta.get("label_names", [])) != list(LABEL_COLS)
                        ):
                            up_to_date = False
                            break
                    if up_to_date:
                        return {
                            payload_name: {
                                "code": str(meta.get("code", _payload_code(payload_name))),
                                "count": int(meta.get("count", 0)),
                                "size": int(meta.get("size", 0)),
                                "mtime_ns": int(meta.get("mtime_ns", 0)),
                                "schema_version": int(meta.get("schema_version", PACKED_LABEL_SCHEMA_VERSION)),
                                "label_names": list(meta.get("label_names", LABEL_COLS)),
                            }
                            for payload_name, meta in manifest.items()
                        }

    return _build_packed_manifest(root, scan_workers=scan_workers)


def build_packed_sample_index(
    codes: list[str],
    *,
    scan_workers: int | None = None,
) -> SampleIndexTable:
    """Build a lightweight index by scanning packed ``.npz`` metadata only."""
    selected_codes = set(codes)
    manifest = _get_packed_manifest(assembled_dir, scan_workers=scan_workers)
    payloadbook: list[str] = []
    payload_id_parts: list[np.ndarray] = []
    sample_idx_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []

    selected_payload_names: list[str] = []
    for payload_name in sorted(manifest.keys()):
        meta = manifest.get(payload_name, {})
        code = str(meta.get("code", _payload_code(payload_name)))
        if code not in selected_codes:
            continue
        path = _packed_path(payload_name)
        if not path.exists():
            continue
        if int(meta.get("count", 0)) <= 0:
            continue
        selected_payload_names.append(payload_name)

    paths = [_packed_path(payload_name) for payload_name in selected_payload_names]
    max_workers = _resolve_scan_workers(scan_workers, task_count=len(paths))
    if max_workers <= 1:
        loaded_dates = [_load_packed_dates(path) for path in paths]
    else:
        ctx = mp.get_context("spawn")
        chunksize = max(1, min(64, len(paths) // (max_workers * 4) or 1))
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            loaded_dates = list(executor.map(_load_packed_dates, paths, chunksize=chunksize))

    for payload_name, date in loaded_dates:
        payload_id = len(payloadbook)
        payloadbook.append(payload_name)

        if date.size == 0:
            continue

        count = int(date.shape[0])
        payload_id_parts.append(np.full(count, payload_id, dtype=np.int32))
        sample_idx_parts.append(np.arange(count, dtype=np.int32))
        date_parts.append(date)

    if not payload_id_parts:
        empty_i32 = np.empty((0,), dtype=np.int32)
        empty_f32 = np.empty((0,), dtype=np.float32)
        return SampleIndexTable(
            payloadbook=tuple(payloadbook),
            payload_ids=empty_i32,
            sample_idx=empty_i32,
            date=empty_f32,
        )

    return SampleIndexTable(
        payloadbook=tuple(payloadbook),
        payload_ids=np.concatenate(payload_id_parts),
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
    """Training-ready dataset backed by packed payload shard ``.npz`` tensors."""

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

    def _load_payload(self, payload_name: str) -> dict[str, Tensor]:
        cached = self._cache.pop(payload_name, None)
        if cached is None:
            with np.load(_packed_path(payload_name), allow_pickle=False) as packed:
                _validate_packed_schema(packed, path=_packed_path(payload_name))
                cached = {
                    "date": torch.from_numpy(np.ascontiguousarray(packed["date"], dtype=np.float32)),
                    "label": torch.from_numpy(np.ascontiguousarray(packed["label"], dtype=np.float32)),
                    "macro": torch.from_numpy(np.ascontiguousarray(packed["macro"], dtype=np.float32)),
                    "mezzo": torch.from_numpy(np.ascontiguousarray(packed["mezzo"], dtype=np.float32)),
                    "micro": torch.from_numpy(np.ascontiguousarray(packed["micro"], dtype=np.float32)),
                    "sidechain": torch.from_numpy(np.ascontiguousarray(packed["sidechain"], dtype=np.float32)),
                }

        self._cache[payload_name] = cached
        while len(self._cache) > self.max_cached_codes:
            self._cache.popitem(last=False)
        return cached

    def clear_cache(self) -> None:
        self._cache.clear()

    def _preload_payloads(self, payload_names: list[str]) -> None:
        seen: set[str] = set()
        for payload_name in payload_names:
            if payload_name in seen:
                continue
            seen.add(payload_name)
            self._load_payload(payload_name)

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
        payload_name = self.sample_index.payload_at(index)
        payload = self._load_payload(payload_name)
        sample_idx = int(self.sample_index.sample_idx[index])
        return self._item_from_payload(payload, sample_idx)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Tensor]]:
        payload_names = [self.sample_index.payload_at(index) for index in indices]
        self._preload_payloads(payload_names)

        items: list[dict[str, Tensor]] = []
        current_payload_name: str | None = None
        current_payload: dict[str, Tensor] | None = None

        for index, payload_name in zip(indices, payload_names, strict=True):
            if payload_name != current_payload_name or current_payload is None:
                current_payload_name = payload_name
                current_payload = self._load_payload(payload_name)
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
    scan_workers: int | None = None,
) -> tuple[Dataset[dict[str, Tensor]], Dataset[dict[str, Tensor]]]:
    """Create train/validation datasets from packed per-code ``.npz`` tensors."""
    del filter_invalid

    if memory_mode not in {"lazy_packed", "auto"}:
        raise ValueError(
            f"Unsupported memory_mode: {memory_mode}. Packed training only supports "
            "'lazy_packed' or 'auto'."
        )

    selected_codes = discover_codes(scan_workers=scan_workers) if codes is None else list(codes)
    if max_codes is not None:
        selected_codes = selected_codes[:max_codes]

    sample_index = build_packed_sample_index(selected_codes, scan_workers=scan_workers)
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
