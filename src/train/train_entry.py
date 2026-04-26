"""Top-level training entrypoint for assembled NPZ datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from contextlib import nullcontext

import torch

from config.data import assembled_dir, checkpoint_dir as DEFAULT_CHECKPOINT_ROOT
from config.train import training
from src.task_labels import canonical_task_label, canonical_training_task

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
    move_batch_to_device,
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


def _autocast_context(
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype_name: str,
    amp_dtype: torch.dtype,
):
    if not use_amp:
        return nullcontext()
    if amp_dtype_name == "tf32":
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _infer_mu_cache(
    *,
    dataset: AssembledNPZDataset,
    mu_model_path: str | Path,
    field: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype_name: str,
) -> torch.Tensor:
    loader = build_val_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        in_order=True,
    )
    amp_dtype = resolve_amp_dtype(amp_dtype_name)
    float_dtype = amp_dtype if amp_dtype in {torch.bfloat16, torch.float16} else None
    model = build_model(mode="mu", field=field)
    if float_dtype is not None:
        model = model.to(device=device, dtype=float_dtype)
    else:
        model = model.to(device=device)
    load_checkpoint(
        mu_model_path,
        model=model,
        map_location=device,
        load_training_state=False,
    )
    model.eval()

    predictions = torch.empty((len(dataset), 1), dtype=torch.float32)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(
                batch,
                device,
                float_dtype=float_dtype,
            )
            with _autocast_context(
                device=device,
                use_amp=use_amp,
                amp_dtype_name=amp_dtype_name,
                amp_dtype=amp_dtype,
            ):
                output = model.forward_loss(batch, return_aux=False)
            mu_pred = output["mu_pred"].detach().float().cpu()
            next_cursor = cursor + mu_pred.shape[0]
            predictions[cursor:next_cursor] = mu_pred
            cursor = next_cursor
    if cursor != len(dataset):
        raise RuntimeError(
            f"mu cache size mismatch: expected {len(dataset)} predictions, got {cursor}."
        )
    return predictions


def _resolve_resume_target(path: Path) -> Path:
    return path / "latest.pt" if path.is_dir() else path


def _resolve_checkpoint_runtime(
    *,
    checkpoint_root: Path,
    name: str | None,
    load_name: str | None,
    checkpoint: str | Path | None,
) -> tuple[str, Path, Path | None]:
    if load_name is not None and checkpoint is not None:
        raise ValueError("Only one of load_name or checkpoint may be set.")

    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if name is not None:
        run_name = name
        checkpoint_dir = checkpoint_root / run_name
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"Checkpoint directory already exists and is not empty: {checkpoint_dir}"
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            resume_from = _resolve_resume_target(checkpoint_path)
            if not resume_from.exists():
                raise FileNotFoundError(f"Checkpoint path does not exist: {resume_from}")
            return run_name, checkpoint_dir, resume_from

        if load_name is not None:
            source_checkpoint_dir = checkpoint_root / load_name
            resume_from = source_checkpoint_dir / "latest.pt"
            if not resume_from.exists():
                raise FileNotFoundError(f"Named checkpoint latest not found: {resume_from}")
            return run_name, checkpoint_dir, resume_from

        return run_name, checkpoint_dir, None

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
    num_workers: int | None = training.num_workers,
    val_num_workers: int | None = training.val_num_workers,
    lr: float = training.lr,
    weight_decay: float = training.weight_decay,
    max_grad_norm: float = training.max_grad_norm,
    task: str = training.task,
    field: str = training.field,
    log_every: int = training.log_every,
    enable_tensorboard: bool = training.enable_tensorboard,
    tensorboard_root: str | Path = training.tensorboard_root,
    tensorboard_flush_secs: int = training.tensorboard_flush_secs,
    mu_model: str | Path | None = training.mu_model,
    num_epochs: int = training.num_epochs,
    save_every: int = training.save_every,
    compile_model: bool = training.compile_model,
    compile_mode: str = training.compile_mode,
    amp_dtype: str = training.amp_dtype,
    device: str | torch.device | None = None,
    use_amp: bool = True,
    mmap_mode: str | None = training.mmap_mode,
    validate_shapes: bool = training.validate_shapes,
    device_prefetch: bool = training.device_prefetch,
    device_prefetch_batches: int = training.device_prefetch_batches,
    enable_console: bool = True,
) -> dict[str, Any]:
    """Run end-to-end training over assembled NPZ inputs."""

    selected_task = canonical_training_task(task or training.task)
    selected_field = canonical_task_label(field or training.field)
    resolved_train_workers = training.num_workers if num_workers is None else int(num_workers)
    resolved_val_workers = (
        training.val_num_workers if val_num_workers is None else int(val_num_workers)
    )

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
    if selected_task == "sigma":
        if mu_model is None:
            raise ValueError("--mu-model is required when --task=sigma.")
        resolved_device_for_cache = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_dataset.attach_mu_cache(
            _infer_mu_cache(
                dataset=train_dataset,
                mu_model_path=mu_model,
                field=selected_field,
                batch_size=val_batch_size,
                num_workers=resolved_val_workers,
                device=resolved_device_for_cache,
                use_amp=use_amp,
                amp_dtype_name=amp_dtype,
            ).numpy()
        )
        val_dataset.attach_mu_cache(
            _infer_mu_cache(
                dataset=val_dataset,
                mu_model_path=mu_model,
                field=selected_field,
                batch_size=val_batch_size,
                num_workers=resolved_val_workers,
                device=resolved_device_for_cache,
                use_amp=use_amp,
                amp_dtype_name=amp_dtype,
            ).numpy()
        )
    elif mu_model is not None:
        raise ValueError("--mu-model is only supported when --task=sigma.")

    train_loader = build_train_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=resolved_train_workers,
    )
    val_loader = build_val_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=resolved_val_workers,
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
    train_loader = maybe_prefetch_loader(
        train_loader,
        device=resolved_device,
        enabled=device_prefetch,
        float_dtype=model_dtype if model_dtype in {torch.bfloat16, torch.float16} else None,
        num_prefetch_batches=device_prefetch_batches,
    )
    val_loader = maybe_prefetch_loader(
        val_loader,
        device=resolved_device,
        enabled=device_prefetch,
        float_dtype=model_dtype if model_dtype in {torch.bfloat16, torch.float16} else None,
        num_prefetch_batches=device_prefetch_batches,
    )
    model = build_model(mode=selected_task, field=selected_field)
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
        field=selected_field,
    )
    tensorboard_log_dir = Path(tensorboard_root) / resolved_run_name
    tensorboard_logger = TensorBoardLogger(
        log_dir=tensorboard_log_dir,
        task=selected_task,
        field=selected_field,
        enabled=enable_tensorboard,
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
        field=selected_field,
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
        "field": selected_field,
        "mu_model": str(mu_model) if mu_model is not None else None,
        "tensorboard_log_dir": str(tensorboard_log_dir) if enable_tensorboard else None,
        "tensorboard_command": (
            f"tensorboard --logdir {Path(tensorboard_root)} "
            f"--host {training.tensorboard_host} --port {training.tensorboard_port}"
        )
        if enable_tensorboard
        else None,
    }
