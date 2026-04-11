"""NPY file I/O operations for reading and writing numpy arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def write_npy(target_path: str | Path, data: np.ndarray, dtype: str = "float32") -> None:
    """Write numpy array to .npy file using memory mapping.

    Args:
        target_path: Path to save the .npy file.
        data: Numpy array to write.
        dtype: Data type for the output file (default: "float32").
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    mmap_file = open_memmap(target_path, mode="w+", dtype=dtype, shape=data.shape)
    mmap_file[:] = data
    mmap_file.flush()
    del mmap_file


def read_npy(file_path: str | Path) -> np.ndarray:
    """Load numpy array from .npy file.

    Args:
        file_path: Path to the .npy file.

    Returns:
        Loaded numpy array.
    """
    file_path = Path(file_path)
    return np.load(file_path, mmap_mode="r")
