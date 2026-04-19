"""Runtime builders for model, optimizer, scheduler, and batch device moves."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from config.models import multi_scale_forecast_network
from config.train import training
from src.models.arch.networks import MultiScaleForecastNetwork
from src.models.config.hparams import TRAIN_RUNTIME_HPARAMS


def configure_training_backends(
    *,
    device: torch.device,
    allow_tf32: bool = training.allow_tf32,
    cudnn_benchmark: bool = training.cudnn_benchmark,
) -> None:
    """Configure global torch backend flags for the training runtime."""

    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def build_model() -> torch.nn.Module:
    """Instantiate the full forecast network with configured open parameters."""

    return MultiScaleForecastNetwork(
        hidden_dim=multi_scale_forecast_network.hidden_dim,
        cond_dim=multi_scale_forecast_network.cond_dim,
        num_latents=multi_scale_forecast_network.num_latents,
    )


def maybe_compile_model(
    model: torch.nn.Module,
    enabled: bool,
    mode: str = TRAIN_RUNTIME_HPARAMS._compile_mode,
) -> torch.nn.Module:
    """Compile the full model body when enabled."""

    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this environment.")
    return torch.compile(model, mode=mode)


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float = training.lr,
    weight_decay: float = training.weight_decay,
) -> torch.optim.Optimizer:
    """Build the default AdamW optimizer."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a minimal epoch-stepped scheduler."""

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    setattr(scheduler, "_step_per_batch", False)
    return scheduler


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a tensor batch to the target device with non-blocking copies."""

    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    """Convert config amp dtype names to torch dtypes."""

    mapping: Mapping[str, torch.dtype] = {
        "tf32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if amp_dtype not in mapping:
        raise ValueError(
            f"Unsupported amp_dtype {amp_dtype!r}. Expected one of {sorted(mapping)}."
        )
    return mapping[amp_dtype]
