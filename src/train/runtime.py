"""Runtime builders for model, optimizer, scheduler, and batch device moves."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch

from config.models import multi_scale_forecast_network
from config.train import training
from src.models.arch.networks import MultiScaleForecastNetwork


def configure_training_backends(
    *,
    device: torch.device,
    allow_tf32: bool = training.allow_tf32,
    cudnn_benchmark: bool = training.cudnn_benchmark,
    compile_mode: str = training.compile_mode,
    dynamo_recompile_limit: int = training.dynamo_recompile_limit,
    dynamo_accumulated_recompile_limit: int = training.dynamo_accumulated_recompile_limit,
) -> None:
    """Configure global torch backend flags for the training runtime."""

    _configure_dynamo_backends(
        recompile_limit=dynamo_recompile_limit,
        accumulated_recompile_limit=dynamo_accumulated_recompile_limit,
    )

    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    _configure_inductor_backends(compile_mode=compile_mode)


def _configure_dynamo_backends(
    *,
    recompile_limit: int,
    accumulated_recompile_limit: int,
) -> None:
    """Raise Dynamo guard-specialization ceilings for hard-to-trace subgraphs."""

    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        return

    limit = int(recompile_limit)
    accumulated_limit = int(accumulated_recompile_limit)
    if hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = limit
    if hasattr(dynamo_config, "cache_size_limit"):
        dynamo_config.cache_size_limit = limit
    if hasattr(dynamo_config, "accumulated_recompile_limit"):
        dynamo_config.accumulated_recompile_limit = accumulated_limit
    if hasattr(dynamo_config, "accumulated_cache_size_limit"):
        dynamo_config.accumulated_cache_size_limit = accumulated_limit


def _configure_inductor_backends(*, compile_mode: str) -> None:
    """Enable the inductor settings that help static-shape training stay saturated."""

    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return

    if hasattr(inductor_config, "fx_graph_cache"):
        inductor_config.fx_graph_cache = True
    if hasattr(inductor_config, "shape_padding"):
        inductor_config.shape_padding = True
    if compile_mode.startswith("max-autotune") and hasattr(
        inductor_config, "coordinate_descent_tuning"
    ):
        inductor_config.coordinate_descent_tuning = True

    triton_config = getattr(inductor_config, "triton", None)
    if triton_config is not None and hasattr(triton_config, "cudagraphs"):
        triton_config.cudagraphs = compile_mode == "max-autotune"


def _compile_target(target, *, mode: str):
    """Compile with mode-only kwargs; inductor tuning is configured separately."""

    _configure_inductor_backends(compile_mode=mode)
    return torch.compile(
        target,
        mode=mode,
        fullgraph=False,
        dynamic=False,
    )


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
    mode: str = training.compile_mode,
) -> torch.nn.Module:
    """Compile the full model body when enabled."""

    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this environment.")
    return _compile_target(model, mode=mode)


def maybe_compile_loss_fn(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = training.compile_mode,
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Compile the actual forward+loss callable used by the trainer."""

    forward_loss = partial(model.forward_loss, return_aux=False)
    if not enabled:
        return forward_loss
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this environment.")
    return _compile_target(forward_loss, mode=mode)


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
            moved[key] = (
                value
                if value.device == device
                else value.to(device=device, non_blocking=True)
            )
        else:
            moved[key] = value
    return moved


class DevicePrefetchLoader:
    """Overlap host-to-device copies with the current training step on CUDA."""

    def __init__(self, loader, *, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        if self.device.type != "cuda":
            yield from iterator
            return

        stream = torch.cuda.Stream(device=self.device)
        next_batch: dict[str, Any] | None = None

        def _preload() -> None:
            nonlocal next_batch
            try:
                batch = next(iterator)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(stream):
                next_batch = move_batch_to_device(batch, self.device)

        _preload()
        current_stream = torch.cuda.current_stream(device=self.device)
        while next_batch is not None:
            current_stream.wait_stream(stream)
            batch = next_batch
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(current_stream)
            _preload()
            yield batch


def maybe_prefetch_loader(loader, *, device: torch.device, enabled: bool = True):
    """Wrap a loader with CUDA prefetch when the target device can benefit."""

    if not enabled or device.type != "cuda":
        return loader
    return DevicePrefetchLoader(loader, device=device)


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
