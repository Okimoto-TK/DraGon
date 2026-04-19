from __future__ import annotations

import torch

from src.train.runtime import configure_training_backends, resolve_amp_dtype


def test_resolve_amp_dtype_supports_tf32() -> None:
    assert resolve_amp_dtype("tf32") == torch.float32


def test_configure_training_backends_enables_tf32_and_benchmark_for_cuda() -> None:
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_precision = torch.get_float32_matmul_precision()

    try:
        configure_training_backends(
            device=torch.device("cuda"),
            allow_tf32=True,
            cudnn_benchmark=True,
        )

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.set_float32_matmul_precision(old_precision)
