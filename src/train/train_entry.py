"""Top-level training entrypoint for assembled NPZ datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from config.data import assembled_dir, checkpoint_dir as DEFAULT_CHECKPOINT_ROOT
from config.train import training

from .checkpoint import load_checkpoint
from .console import EpochConsoleLogger
from .dataloaders import build_train_dataloader, build_val_dataloader
from .dataset import AssembledNPZDataset
from .runtime import (
    build_model,
    build_optimizer,
    build_scheduler,
    configure_training_backends,
    maybe_compile_loss_fn,
    maybe_prefetch_loader,
    resolve_amp_dtype,
)
from .tensorboard_logger import TensorBoardLogger
from .trainer import Trainer


def _default_file_split() -> tuple[list[str], list[str]]:
    file_paths = sorted(str(path) for path in Path(assembled_dir).glob("*.npz"))
    if not file_paths:
        raise ValueError(f"No assembled NPZ files found under {assembled_dir}.")
    if len(file_paths) == 1:
        return file_paths, file_paths

    split_index = max(1, int(round(len(file_paths) * 0.9)))
    split_index = min(split_index, len(file_paths) - 1)
    return file_paths[:split_index], file_paths[split_index:]


def _default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _resolve_resume_target(path: Path) -> Path:
    return path / "latest.pt" if path.is_dir() else path


def _resolve_checkpoint_runtime(
    *,
    checkpoint_root: Path,
    name: str | None,
    load_name: str | None,
    checkpoint: str | Path | None,
) -> tuple[str, Path, Path | None]:
    selected = sum(value is not None for value in (name, load_name, checkpoint))
    if selected > 1:
        raise ValueError("Only one of name, load_name, or checkpoint may be set.")

    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        resume_from = _resolve_resume_target(checkpoint_path)
        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {resume_from}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir.name, checkpoint_dir, resume_from

    if load_name is not None:
        checkpoint_dir = checkpoint_root / load_name
        resume_from = checkpoint_dir / "latest.pt"
        if not resume_from.exists():
            raise FileNotFoundError(f"Named checkpoint latest not found: {resume_from}")
        return load_name, checkpoint_dir, resume_from

    run_name = name or _default_run_name()
    checkpoint_dir = checkpoint_root / run_name
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            f"Checkpoint directory already exists and is not empty: {checkpoint_dir}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_name, checkpoint_dir, None


def run_training(
    train_files: Sequence[str] | None = None,
    val_files: Sequence[str] | None = None,
    *,
    name: str | None = None,
    load_name: str | None = None,
    checkpoint: str | Path | None = None,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    batch_size: int = training.batch_size,
    val_batch_size: int = training.val_batch_size,
    num_workers: int = training.num_workers,
    lr: float = training.lr,
    weight_decay: float = training.weight_decay,
    max_grad_norm: float = training.max_grad_norm,
    task: str = training.task,
    log_every: int = training.log_every,
    enable_tensorboard: bool = training.enable_tensorboard,
    tensorboard_root: str | Path = training.tensorboard_root,
    tensorboard_debug_every: int = training.tensorboard_debug_every,
    tensorboard_flush_secs: int = training.tensorboard_flush_secs,
    num_epochs: int = training.num_epochs,
    save_every: int = training.save_every,
    compile_model: bool = training.compile_model,
    compile_mode: str = training.compile_mode,
    amp_dtype: str = training.amp_dtype,
    device: str | torch.device | None = None,
    use_amp: bool = True,
    mmap_mode: str | None = training.mmap_mode,
    validate_shapes: bool = True,
    enable_console: bool = True,
) -> dict[str, Any]:
    """Run end-to-end training over assembled NPZ inputs."""

    selected_task = task or training.task

    if train_files is None or val_files is None:
        default_train_files, default_val_files = _default_file_split()
        train_files = list(train_files or default_train_files)
        val_files = list(val_files or default_val_files)

    resolved_run_name, resolved_checkpoint_dir, resume_from = _resolve_checkpoint_runtime(
        checkpoint_root=Path(checkpoint_root),
        name=name,
        load_name=load_name,
        checkpoint=checkpoint,
    )

    train_dataset = AssembledNPZDataset(
        list(train_files),
        mmap_mode=mmap_mode,
        validate_shapes=validate_shapes,
        max_open_archives=training.max_open_archives,
    )
    val_dataset = AssembledNPZDataset(
        list(val_files),
        mmap_mode=mmap_mode,
        validate_shapes=validate_shapes,
        max_open_archives=training.max_open_archives,
    )

    train_loader = build_train_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = build_val_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    configure_training_backends(
        device=resolved_device,
        compile_mode=compile_mode,
    )
    model_dtype = resolve_amp_dtype(amp_dtype)
    # CUDA graph capture in torch.compile can be invalidated by our explicit
    # prefetch stream issuing overlapping H2D copies on another stream.
    prefetch_enabled = not compile_model
    train_loader = maybe_prefetch_loader(
        train_loader,
        device=resolved_device,
        enabled=prefetch_enabled,
        float_dtype=model_dtype if model_dtype in {torch.bfloat16, torch.float16} else None,
    )
    val_loader = maybe_prefetch_loader(
        val_loader,
        device=resolved_device,
        enabled=prefetch_enabled,
        float_dtype=model_dtype if model_dtype in {torch.bfloat16, torch.float16} else None,
    )
    model = build_model(task=selected_task)
    if model_dtype in {torch.bfloat16, torch.float16}:
        model = model.to(device=resolved_device, dtype=model_dtype)
    else:
        model = model.to(device=resolved_device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer)
    console_logger = EpochConsoleLogger(
        log_every=log_every,
        enabled=enable_console,
        task=selected_task,
    )
    tensorboard_log_dir = Path(tensorboard_root) / resolved_run_name
    tensorboard_logger = TensorBoardLogger(
        log_dir=tensorboard_log_dir,
        task=selected_task,
        enabled=enable_tensorboard,
        debug_every=tensorboard_debug_every,
        flush_secs=tensorboard_flush_secs,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=resolved_device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        max_grad_norm=max_grad_norm,
        task=selected_task,
        log_every=log_every,
        save_every=save_every,
        console_logger=console_logger,
        tensorboard_logger=tensorboard_logger,
    )
    trainer.checkpoint_dir = resolved_checkpoint_dir
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        state = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            map_location=resolved_device,
            load_training_state=True,
        )
        trainer.start_epoch = int(state["epoch"]) + 1
        trainer.global_step = int(state["global_step"])

    trainer.forward_loss = maybe_compile_loss_fn(
        trainer.model,
        enabled=compile_model,
        mode=compile_mode,
    )
    trainer.fit(num_epochs=num_epochs)
    best_val_loss = min(
        (row["val"]["loss_total"] for row in trainer.history),
        default=float("inf"),
    )
    return {
        "run_name": resolved_run_name,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "best_val_loss": best_val_loss,
        "global_step": trainer.global_step,
        "epochs_completed": len(trainer.history),
        "device": str(resolved_device),
        "task": selected_task,
        "tensorboard_log_dir": str(tensorboard_log_dir) if enable_tensorboard else None,
        "tensorboard_command": (
            f"tensorboard --logdir {Path(tensorboard_root)} "
            f"--host {training.tensorboard_host} --port {training.tensorboard_port}"
        )
        if enable_tensorboard
        else None,
    }
