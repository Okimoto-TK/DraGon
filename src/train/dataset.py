"""Dataset adapters built on top of assembled ``.npy`` samples."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config.config import assembled_dir
from config.config import lazy_cache_codes as DEFAULT_LAZY_CACHE_CODES
from config.config import val_ratio as DEFAULT_VAL_RATIO
from src.data.assembler.assemble import LABEL_COLS, MEZZO_FEATURES, MICRO_FEATURES
from src.data.assembler.sampler import (
    DATE_IDX,
    LABEL_SLICE,
    MACRO_SLICE,
    MEZZO_SLICE,
    MICRO_SLICE,
    MICRO_USED_FEATURES,
    SIDECHAIN_SLICE,
    build_sample_index_for_code,
    get_samples,
)
from src.data.registry.dataset import MACRO_LOOKBACK, MEZZO_LOOKBACK, MICRO_LOOKBACK

MemoryMode = Literal["eager", "lazy_memmap"]

_MEZZO_STEPS_PER_DAY: Final[int] = 8
_MICRO_STEPS_PER_DAY: Final[int] = 48
_MEZZO_DAYS: Final[int] = MEZZO_LOOKBACK // _MEZZO_STEPS_PER_DAY
_MICRO_DAYS: Final[int] = MICRO_LOOKBACK // _MICRO_STEPS_PER_DAY
_USED_MICRO_FEATURES: Final[list[str]] = MICRO_FEATURES[:-2]


@dataclass(frozen=True)
class SampleIndexTable:
    """Compact sample index used by lazy/eager datasets."""

    codebook: tuple[str, ...]
    code_ids: np.ndarray
    t: np.ndarray
    sample_idx: np.ndarray
    date: np.ndarray
    is_valid: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def subset(self, mask: np.ndarray) -> "SampleIndexTable":
        return SampleIndexTable(
            codebook=self.codebook,
            code_ids=self.code_ids[mask],
            t=self.t[mask],
            sample_idx=self.sample_idx[mask],
            date=self.date[mask],
            is_valid=self.is_valid[mask],
        )

    def code_at(self, index: int) -> str:
        return self.codebook[int(self.code_ids[index])]


def discover_codes(root: Path = assembled_dir) -> list[str]:
    """Return all assembled codes available under ``root``."""
    return sorted(path.stem for path in root.glob("*.npy"))


def _reshape_mezzo(samples: np.ndarray) -> np.ndarray:
    """Convert [N, 8, 72] day-major mezzo windows into [N, 64, 9]."""
    if samples.size == 0:
        return np.empty((0, _MEZZO_DAYS * _MEZZO_STEPS_PER_DAY, len(MEZZO_FEATURES)), dtype=np.float32)

    n_samples, n_days, flat_dim = samples.shape
    expected_flat_dim = len(MEZZO_FEATURES) * _MEZZO_STEPS_PER_DAY
    if flat_dim != expected_flat_dim:
        raise ValueError(f"Unexpected mezzo flat dim {flat_dim}, expected {expected_flat_dim}")

    reshaped = samples.reshape(n_samples, n_days, len(MEZZO_FEATURES), _MEZZO_STEPS_PER_DAY)
    return reshaped.transpose(0, 1, 3, 2).reshape(
        n_samples,
        n_days * _MEZZO_STEPS_PER_DAY,
        len(MEZZO_FEATURES),
    )


def _reshape_micro(samples: np.ndarray) -> np.ndarray:
    """Convert [N, 1, 336] day-major micro windows into [N, 48, 7]."""
    if samples.size == 0:
        return np.empty((0, _MICRO_DAYS * _MICRO_STEPS_PER_DAY, len(_USED_MICRO_FEATURES)), dtype=np.float32)

    n_samples, n_days, flat_dim = samples.shape
    expected_flat_dim = len(_USED_MICRO_FEATURES) * _MICRO_STEPS_PER_DAY
    if flat_dim != expected_flat_dim:
        raise ValueError(f"Unexpected micro flat dim {flat_dim}, expected {expected_flat_dim}")

    reshaped = samples.reshape(
        n_samples,
        n_days,
        len(_USED_MICRO_FEATURES),
        _MICRO_STEPS_PER_DAY,
    )
    return reshaped.transpose(0, 1, 3, 2).reshape(
        n_samples,
        n_days * _MICRO_STEPS_PER_DAY,
        len(_USED_MICRO_FEATURES),
    )


def _reshape_mezzo_sample(sample: np.ndarray) -> np.ndarray:
    """Convert one mezzo sample from [8, 72] into [64, 9]."""
    if sample.shape != (_MEZZO_DAYS, len(MEZZO_FEATURES) * _MEZZO_STEPS_PER_DAY):
        raise ValueError(f"Unexpected mezzo sample shape {sample.shape}")
    reshaped = sample.reshape(_MEZZO_DAYS, len(MEZZO_FEATURES), _MEZZO_STEPS_PER_DAY)
    return reshaped.transpose(0, 2, 1).reshape(
        _MEZZO_DAYS * _MEZZO_STEPS_PER_DAY,
        len(MEZZO_FEATURES),
    )


def _reshape_micro_sample(sample: np.ndarray) -> np.ndarray:
    """Convert one micro sample from [1, 336] into [48, 7]."""
    if sample.shape != (_MICRO_DAYS, len(_USED_MICRO_FEATURES) * _MICRO_STEPS_PER_DAY):
        raise ValueError(f"Unexpected micro sample shape {sample.shape}")
    reshaped = sample.reshape(_MICRO_DAYS, len(_USED_MICRO_FEATURES), _MICRO_STEPS_PER_DAY)
    return reshaped.transpose(0, 2, 1).reshape(
        _MICRO_DAYS * _MICRO_STEPS_PER_DAY,
        len(_USED_MICRO_FEATURES),
    )


def _to_bcl(samples: np.ndarray) -> Tensor:
    """Convert [N, L, C] arrays into float32 [N, C, L] tensors."""
    transposed = np.transpose(samples, (0, 2, 1))
    return torch.from_numpy(np.ascontiguousarray(transposed)).to(dtype=torch.float32)


def _to_cl(sample: np.ndarray) -> Tensor:
    """Convert one [L, C] sample into float32 [C, L]."""
    transposed = np.transpose(sample, (1, 0))
    return torch.from_numpy(np.array(transposed, dtype=np.float32, copy=True))


def _build_item(
    *,
    date: np.ndarray | float,
    macro: np.ndarray,
    mezzo: np.ndarray,
    micro: np.ndarray,
    sidechain: np.ndarray,
    label: np.ndarray,
) -> dict[str, Tensor]:
    if not (
        np.isfinite(macro).all()
        and np.isfinite(mezzo).all()
        and np.isfinite(micro).all()
        and np.isfinite(sidechain).all()
        and np.isfinite(label).all()
    ):
        raise ValueError(
            "Non-finite values remain after is_valid filtering. "
            "This indicates the upstream is_valid_step/assembled data needs to be fixed."
        )

    label_tensor = torch.from_numpy(np.array(label, dtype=np.float32, copy=True))
    date_tensor = torch.tensor(float(date), dtype=torch.float32)
    label_index = {name: idx for idx, name in enumerate(LABEL_COLS)}
    return {
        "date": date_tensor,
        "macro": _to_cl(macro),
        "mezzo": _to_cl(mezzo),
        "micro": _to_cl(micro),
        "sidechain": _to_cl(sidechain),
        "label": label_tensor,
        "label_S": label_tensor[label_index["label_S"]],
        "label_M": label_tensor[label_index["label_M"]],
        "label_MDD": label_tensor[label_index["label_MDD"]],
        "label_RV": label_tensor[label_index["label_RV"]],
    }


def build_sample_index(
    codes: list[str],
    *,
    filter_invalid: bool = True,
) -> SampleIndexTable:
    """Build a lightweight cross-code sample index."""
    codebook = tuple(codes)
    code_id_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    sample_idx_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    is_valid_parts: list[np.ndarray] = []

    for code_id, code in enumerate(codebook):
        per_code = build_sample_index_for_code(code)
        if per_code["t"].size == 0:
            continue

        keep_mask = per_code["is_valid"]
        if not filter_invalid:
            keep_mask = np.ones_like(keep_mask, dtype=bool)
        if not np.any(keep_mask):
            continue

        keep_count = int(keep_mask.sum())
        code_id_parts.append(np.full(keep_count, code_id, dtype=np.int32))
        t_parts.append(np.asarray(per_code["t"][keep_mask], dtype=np.int32))
        sample_idx_parts.append(np.asarray(per_code["sample_idx"][keep_mask], dtype=np.int32))
        date_parts.append(np.asarray(per_code["date"][keep_mask], dtype=np.float32))
        is_valid_parts.append(np.asarray(per_code["is_valid"][keep_mask], dtype=bool))

    if not code_id_parts:
        empty_i32 = np.empty((0,), dtype=np.int32)
        empty_f32 = np.empty((0,), dtype=np.float32)
        empty_bool = np.empty((0,), dtype=bool)
        return SampleIndexTable(
            codebook=codebook,
            code_ids=empty_i32,
            t=empty_i32,
            sample_idx=empty_i32,
            date=empty_f32,
            is_valid=empty_bool,
        )

    return SampleIndexTable(
        codebook=codebook,
        code_ids=np.concatenate(code_id_parts),
        t=np.concatenate(t_parts),
        sample_idx=np.concatenate(sample_idx_parts),
        date=np.concatenate(date_parts),
        is_valid=np.concatenate(is_valid_parts),
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

    resolved_split_date: float | None
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

    return (
        sample_index.subset(train_mask),
        sample_index.subset(val_mask),
        resolved_split_date,
    )


class SamplerDataset(Dataset[dict[str, Tensor]]):
    """Eager dataset that materializes selected samples in memory."""

    def __init__(self, sample_index: SampleIndexTable) -> None:
        self.sample_index = sample_index
        self.label_index = {name: idx for idx, name in enumerate(LABEL_COLS)}

        macro_parts: list[Tensor] = []
        mezzo_parts: list[Tensor] = []
        micro_parts: list[Tensor] = []
        sidechain_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        date_parts: list[Tensor] = []

        for code_id, code in enumerate(self.sample_index.codebook):
            code_mask = self.sample_index.code_ids == code_id
            if not np.any(code_mask):
                continue

            sample_positions = self.sample_index.sample_idx[code_mask]
            samples = get_samples(code)
            if sample_positions.size == 0:
                continue

            macro = samples["macro"][sample_positions]
            sidechain = samples["sidechain"][sample_positions]
            mezzo = _reshape_mezzo(samples["mezzo"][sample_positions])
            micro = _reshape_micro(samples["micro"][sample_positions])
            label = samples["label"][sample_positions]
            date = samples["date"][sample_positions]

            macro_parts.append(_to_bcl(macro))
            sidechain_parts.append(_to_bcl(sidechain))
            mezzo_parts.append(_to_bcl(mezzo))
            micro_parts.append(_to_bcl(micro))
            label_parts.append(torch.from_numpy(np.ascontiguousarray(label)).to(dtype=torch.float32))
            date_parts.append(torch.from_numpy(np.ascontiguousarray(date)).to(dtype=torch.float32))

        if macro_parts:
            self.macro = torch.cat(macro_parts, dim=0)
            self.sidechain = torch.cat(sidechain_parts, dim=0)
            self.mezzo = torch.cat(mezzo_parts, dim=0)
            self.micro = torch.cat(micro_parts, dim=0)
            self.label = torch.cat(label_parts, dim=0)
            self.date = torch.cat(date_parts, dim=0)
        else:
            self.macro = torch.empty((0, 9, 64), dtype=torch.float32)
            self.sidechain = torch.empty((0, 8, 64), dtype=torch.float32)
            self.mezzo = torch.empty((0, 9, 64), dtype=torch.float32)
            self.micro = torch.empty((0, 7, 48), dtype=torch.float32)
            self.label = torch.empty((0, len(LABEL_COLS)), dtype=torch.float32)
            self.date = torch.empty((0,), dtype=torch.float32)

        if len(self) > 0:
            tensors_to_check = {
                "macro": self.macro,
                "mezzo": self.mezzo,
                "micro": self.micro,
                "sidechain": self.sidechain,
                "label": self.label,
            }
            for name, tensor in tensors_to_check.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Non-finite values remain in {name} after is_valid filtering. "
                        "This indicates the upstream is_valid_step/assembled data needs to be fixed."
                    )

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def clear_cache(self) -> None:
        """Eager datasets do not hold external file caches."""
        return None

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        labels = self.label[index]
        return {
            "date": self.date[index],
            "macro": self.macro[index],
            "mezzo": self.mezzo[index],
            "micro": self.micro[index],
            "sidechain": self.sidechain[index],
            "label": labels,
            "label_S": labels[self.label_index["label_S"]],
            "label_M": labels[self.label_index["label_M"]],
            "label_MDD": labels[self.label_index["label_MDD"]],
            "label_RV": labels[self.label_index["label_RV"]],
        }


class LazySamplerDataset(Dataset[dict[str, Tensor]]):
    """Low-memory dataset backed by memmapped assembled ``.npy`` files."""

    def __init__(
        self,
        sample_index: SampleIndexTable,
        *,
        max_cached_codes: int = DEFAULT_LAZY_CACHE_CODES,
    ) -> None:
        self.sample_index = sample_index
        self.max_cached_codes = max(1, int(max_cached_codes))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_index)

    def _get_array(self, code: str) -> np.ndarray:
        cached = self._cache.pop(code, None)
        if cached is None:
            cached = np.load(assembled_dir / f"{code}.npy", mmap_mode="r")
        self._cache[code] = cached

        while len(self._cache) > self.max_cached_codes:
            _, evicted = self._cache.popitem(last=False)
            mmap_obj = getattr(evicted, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()

        return cached

    def clear_cache(self) -> None:
        while self._cache:
            _, cached = self._cache.popitem(last=False)
            mmap_obj = getattr(cached, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        code = self.sample_index.code_at(index)
        data = self._get_array(code)
        t = int(self.sample_index.t[index])

        macro = np.asarray(
            data[t - MACRO_LOOKBACK + 1 : t + 1, MACRO_SLICE],
            dtype=np.float32,
        )
        sidechain = np.asarray(
            data[t - MACRO_LOOKBACK + 1 : t + 1, SIDECHAIN_SLICE],
            dtype=np.float32,
        )
        mezzo_raw = np.asarray(
            data[t - _MEZZO_DAYS + 1 : t + 1, MEZZO_SLICE],
            dtype=np.float32,
        )
        micro_raw = np.asarray(
            data[t - _MICRO_DAYS + 1 : t + 1, MICRO_SLICE],
            dtype=np.float32,
        )
        label = np.asarray(data[t, LABEL_SLICE], dtype=np.float32)
        date = np.asarray(data[t, DATE_IDX], dtype=np.float32)

        return _build_item(
            date=date,
            macro=macro,
            mezzo=_reshape_mezzo_sample(mezzo_raw),
            micro=_reshape_micro_sample(micro_raw),
            sidechain=sidechain,
            label=label,
        )


def create_train_val_datasets(
    *,
    codes: list[str] | None = None,
    split_date: float | int | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    max_codes: int | None = None,
    filter_invalid: bool = True,
    memory_mode: MemoryMode = "lazy_memmap",
) -> tuple[Dataset[dict[str, Tensor]], Dataset[dict[str, Tensor]]]:
    """Create train/validation datasets using chronological splitting."""
    selected_codes = discover_codes() if codes is None else list(codes)
    if max_codes is not None:
        selected_codes = selected_codes[:max_codes]

    sample_index = build_sample_index(selected_codes, filter_invalid=filter_invalid)
    train_index, val_index, _ = split_index_by_date(
        sample_index,
        split_date=split_date,
        val_ratio=val_ratio,
    )

    if memory_mode == "eager":
        return SamplerDataset(train_index), SamplerDataset(val_index)
    if memory_mode == "lazy_memmap":
        return LazySamplerDataset(train_index), LazySamplerDataset(val_index)
    raise ValueError(f"Unsupported memory_mode: {memory_mode}")


__all__ = [
    "LazySamplerDataset",
    "SamplerDataset",
    "SampleIndexTable",
    "build_sample_index",
    "create_train_val_datasets",
    "discover_codes",
    "split_index_by_date",
]
