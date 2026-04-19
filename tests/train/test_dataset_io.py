from __future__ import annotations

from pathlib import Path

import numpy as np

from src.train.dataloaders import FileLocalityBatchSampler
from src.train.dataset import AssembledNPZDataset, REQUIRED_RAW_KEYS


class _FakeArchive:
    def __init__(self, name: str, sample_count: int) -> None:
        self.name = name
        self.sample_count = sample_count
        self.closed = False

    def __contains__(self, key: str) -> bool:
        return key in REQUIRED_RAW_KEYS

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "label":
            return np.zeros((self.sample_count, 2), dtype=np.float32)
        return np.zeros((self.sample_count, 1), dtype=np.float32)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _FakeArchive:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def test_dataset_lru_cache_closes_evicted_archives(monkeypatch) -> None:
    sample_counts = {
        "a.npz": 2,
        "b.npz": 3,
        "c.npz": 4,
    }
    opened: list[_FakeArchive] = []

    def fake_load(path, mmap_mode=None, allow_pickle=False):
        archive = _FakeArchive(Path(path).name, sample_counts[Path(path).name])
        opened.append(archive)
        return archive

    monkeypatch.setattr("src.train.dataset.np.load", fake_load)

    dataset = AssembledNPZDataset(
        ["a.npz", "b.npz", "c.npz"],
        validate_shapes=False,
        max_open_archives=2,
    )

    archive_a = dataset._get_archive(0)
    archive_b = dataset._get_archive(1)
    archive_c = dataset._get_archive(2)

    assert archive_a.closed is True
    assert archive_b.closed is False
    assert archive_c.closed is False
    assert list(dataset._archive_cache) == [1, 2]

    dataset.close()
    assert archive_b.closed is True
    assert archive_c.closed is True


def test_file_locality_batch_sampler_groups_indices_by_file_ranges() -> None:
    dataset = type(
        "DummyDataset",
        (),
        {
            "file_sample_ranges": [(0, 3), (3, 5), (5, 9)],
            "__len__": lambda self: 9,
        },
    )()

    sampler = FileLocalityBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    assert list(sampler) == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
