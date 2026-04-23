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


def test_dataset_reuses_loaded_file_payload(monkeypatch) -> None:
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
        max_open_archives=16,
    )

    payload_a = dataset._load_file_payload(0)
    payload_a_again = dataset._load_file_payload(0)
    payload_b = dataset._load_file_payload(1)

    assert payload_a["label"].shape == (2, 2)
    assert payload_a_again is payload_a
    assert payload_b["label"].shape == (3, 2)
    assert dataset._loaded_file_id == 1
    assert dataset._loaded_payload is payload_b

    dataset.close()
    assert dataset._loaded_file_id is None
    assert dataset._loaded_payload is None


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


def test_file_locality_batch_sampler_preserves_file_order_for_train_shards() -> None:
    dataset = type(
        "DummyDataset",
        (),
        {
            "file_sample_ranges": [(0, 2), (2, 5), (5, 7)],
            "__len__": lambda self: 7,
        },
    )()

    sampler = FileLocalityBatchSampler(
        dataset,
        batch_size=3,
        shuffle=False,
        drop_last=True,
    )

    assert list(sampler) == [[0, 1, 2], [3, 4, 5]]
