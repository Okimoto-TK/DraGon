"""Training and validation dataloader builders."""

from __future__ import annotations

import random

from torch.utils.data import BatchSampler, DataLoader, Dataset

from config.train import training
from .dataset import AssembledNPZDataset

from .collate import collate_network_batch


class FileLocalityBatchSampler(BatchSampler):
    """Shuffle by file, then by sample within file, to keep batches I/O-local."""

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
                random.shuffle(file_indices)
            for sample_index in file_indices:
                batch.append(sample_index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

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
) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": training.pin_memory,
        "persistent_workers": training.persistent_workers and num_workers > 0,
        "collate_fn": collate_network_batch,
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
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
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
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=training.prefetch_factor_train,
    )


def build_val_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = training.val_batch_size,
    num_workers: int = training.num_workers,
) -> DataLoader:
    """Build the validation dataloader."""

    return _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=training.prefetch_factor_val,
    )
