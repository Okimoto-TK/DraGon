from __future__ import annotations

import torch

from src.train.runtime import (
    configure_training_backends,
    move_batch_to_device,
    resolve_amp_dtype,
)


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


def test_move_batch_to_device_explicitly_casts_floats_to_bf16() -> None:
    batch = {
        "macro_float_long": torch.randn(2, 9, 112, dtype=torch.float32),
        "target_ret": torch.randn(2, 1, dtype=torch.float32),
        "macro_i8_long": torch.zeros(2, 2, 112, dtype=torch.int64),
    }

    moved = move_batch_to_device(
        batch,
        torch.device("cpu"),
        float_dtype=torch.bfloat16,
    )

    assert moved["macro_float_long"].dtype == torch.bfloat16
    assert moved["target_ret"].dtype == torch.bfloat16
    assert moved["macro_i8_long"].dtype == torch.int64
