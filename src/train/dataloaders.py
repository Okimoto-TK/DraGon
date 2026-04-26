"""Training and validation dataloader builders."""

from __future__ import annotations

from functools import partial
import random

from torch.utils.data import BatchSampler, DataLoader, Dataset

from config.train import training
from .dataset import AssembledNPZDataset

from .collate import collate_network_batch, resolve_collate_chunk_size


def _resolve_loader_parallelism(
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> tuple[int, int]:
    """Resolve safe dataloader worker and prefetch settings."""
    _ = batch_size

    if num_workers <= 0:
        return 0, prefetch_factor
    return num_workers, max(1, prefetch_factor)


class FileLocalityBatchSampler(BatchSampler):
    """Build mostly sequential batches while allowing file boundaries to fill leftovers."""

    def __init__(
        self,
        dataset: AssembledNPZDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._file_ranges = dataset.file_sample_ranges
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

    def __iter__(self):
        file_order = list(range(len(self._file_ranges)))
        if self.shuffle:
            random.shuffle(file_order)

        batch: list[int] = []
        for file_id in file_order:
            start, end = self._file_ranges[file_id]
            file_indices = list(range(start, end))
            if self.shuffle:
                chunk_starts = list(range(0, len(file_indices), self.batch_size))
                random.shuffle(chunk_starts)
                shuffled_indices: list[int] = []
                for chunk_start in chunk_starts:
                    chunk_end = min(chunk_start + self.batch_size, len(file_indices))
                    shuffled_indices.extend(file_indices[chunk_start:chunk_end])
                file_indices = shuffled_indices

            batch.extend(file_indices)
            while len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int,
    in_order: bool | None = None,
) -> DataLoader:
    resolved_num_workers, resolved_prefetch_factor = _resolve_loader_parallelism(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "num_workers": resolved_num_workers,
        "pin_memory": training.pin_memory,
        "persistent_workers": training.persistent_workers and resolved_num_workers > 0,
        "in_order": training.dataloader_in_order if in_order is None else bool(in_order),
        "collate_fn": partial(
            collate_network_batch,
            chunk_size=resolve_collate_chunk_size(batch_size=batch_size),
            validate_shapes=training.validate_shapes,
        ),
    }
    if isinstance(dataset, AssembledNPZDataset):
        loader_kwargs["batch_sampler"] = FileLocalityBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["drop_last"] = drop_last
    if resolved_num_workers > 0:
        loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
    return DataLoader(**loader_kwargs)


def build_train_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = training.batch_size,
    num_workers: int = training.num_workers,
) -> DataLoader:
    """Build the train dataloader with fixed-shape batch settings."""

    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=training.prefetch_factor_train,
    )


def build_val_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = training.val_batch_size,
    num_workers: int = training.val_num_workers,
    in_order: bool | None = None,
) -> DataLoader:
    """Build the validation dataloader."""

    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=training.prefetch_factor_val,
        in_order=in_order,
    )
