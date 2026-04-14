"""End-to-end training entrypoints."""
from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from config.config import amp_enabled as DEFAULT_AMP_ENABLED
from config.config import batch_size as DEFAULT_BATCH_SIZE
from config.config import checkpoint_dir as DEFAULT_CHECKPOINT_DIR
from config.config import early_stopping_patience as DEFAULT_EARLY_STOPPING_PATIENCE
from config.config import mlflow_dir as DEFAULT_MLFLOW_DIR
from config.config import mlflow_enabled as DEFAULT_MLFLOW_ENABLED
from config.config import grad_clip as DEFAULT_GRAD_CLIP
from config.config import learning_rate as DEFAULT_LEARNING_RATE
from config.config import log_every_steps as DEFAULT_LOG_EVERY_STEPS
from config.config import memory_mode as DEFAULT_MEMORY_MODE
from config.config import num_epochs as DEFAULT_NUM_EPOCHS
from config.config import num_workers as DEFAULT_NUM_WORKERS
from config.config import prefetch_factor as DEFAULT_PREFETCH_FACTOR
from config.config import profile_active_steps as DEFAULT_PROFILE_ACTIVE_STEPS
from config.config import profile_dir as DEFAULT_PROFILE_DIR
from config.config import profile_enabled as DEFAULT_PROFILE_ENABLED
from config.config import profile_epoch as DEFAULT_PROFILE_EPOCH
from config.config import profile_memory as DEFAULT_PROFILE_MEMORY
from config.config import profile_record_shapes as DEFAULT_PROFILE_RECORD_SHAPES
from config.config import profile_repeat as DEFAULT_PROFILE_REPEAT
from config.config import profile_row_limit as DEFAULT_PROFILE_ROW_LIMIT
from config.config import profile_wait_steps as DEFAULT_PROFILE_WAIT_STEPS
from config.config import profile_warmup_steps as DEFAULT_PROFILE_WARMUP_STEPS
from config.config import profile_with_flops as DEFAULT_PROFILE_WITH_FLOPS
from config.config import profile_with_stack as DEFAULT_PROFILE_WITH_STACK
from config.config import scheduler_factor as DEFAULT_SCHEDULER_FACTOR
from config.config import scheduler_min_lr as DEFAULT_SCHEDULER_MIN_LR
from config.config import scheduler_name as DEFAULT_SCHEDULER_NAME
from config.config import scheduler_patience as DEFAULT_SCHEDULER_PATIENCE
from config.config import save_every as DEFAULT_SAVE_EVERY
from config.config import train_seed as DEFAULT_TRAIN_SEED
from config.config import train_samples_per_epoch as DEFAULT_TRAIN_SAMPLES_PER_EPOCH
from config.config import train_val_split_date as DEFAULT_TRAIN_VAL_SPLIT_DATE
from config.config import val_ratio as DEFAULT_VAL_RATIO
from config.config import val_samples_per_epoch as DEFAULT_VAL_SAMPLES_PER_EPOCH
from config.config import weight_decay as DEFAULT_WEIGHT_DECAY
from src.models import MultiScaleFusionNet
from src.models.losses import LaplaceLSELoss
from src.train.dataset import create_train_val_datasets
from src.train.train import train_one_epoch
from src.train.profiler import create_epoch_profiler, write_profiler_reports
from src.train.ui import TrainingUI
from src.train.validate import validate
from src.train.visualize import MLflowVisualizer


def _step_scheduler(scheduler: Any, monitored_value: float) -> None:
    if scheduler is None:
        return
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(monitored_value)
    else:
        scheduler.step()


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str | None,
    scheduler_factor: float,
    scheduler_patience: int,
    scheduler_min_lr: float,
) -> Any:
    """Build the training scheduler from config-level knobs."""
    if scheduler_name is None:
        return None

    normalized = scheduler_name.strip().lower()
    if normalized in {"", "none", "off"}:
        return None

    if normalized == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")


def _resolve_dataset_cache_owner(dataset: Any) -> Any:
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current


def _clear_dataset_cache(dataset: Any) -> None:
    owner = _resolve_dataset_cache_owner(dataset)
    clear_fn = getattr(owner, "clear_cache", None)
    if callable(clear_fn):
        clear_fn()


def _run_checkpoint_dir(root: str | Path, run_name: str) -> Path:
    return Path(root) / run_name


def _checkpoint_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "latest": run_dir / "latest.pt",
        "best": run_dir / "best.pt",
    }


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: dict[str, list[dict[str, float]]],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_epoch: int,
    best_val_loss: float,
    global_train_step: int,
    run_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "global_train_step": global_train_step,
        },
        path,
    )


def _resolve_checkpoint_path(load_checkpoint: str | Path) -> Path:
    checkpoint_path = Path(load_checkpoint)
    if checkpoint_path.is_dir():
        latest = checkpoint_path / "latest.pt"
        if latest.exists():
            return latest
        raise FileNotFoundError(f"No latest.pt found under checkpoint directory: {checkpoint_path}")
    return checkpoint_path


def _load_checkpoint_state(
    load_checkpoint: str | Path,
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(load_checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if state.get("criterion_state_dict") is not None:
        criterion.load_state_dict(state["criterion_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def _grouped_epoch_indices(
    dataset: Any,
    *,
    samples_per_epoch: int | None,
    seed: int,
    epoch: int,
) -> list[int] | None:
    sample_index = getattr(dataset, "sample_index", None)
    if sample_index is None:
        return None

    total = len(sample_index)
    if total == 0:
        return []

    rng = np.random.default_rng(seed + epoch)
    if samples_per_epoch is None or samples_per_epoch <= 0 or total <= samples_per_epoch:
        selected = np.arange(total, dtype=np.int32)
    else:
        selected = np.asarray(
            rng.choice(total, size=samples_per_epoch, replace=False),
            dtype=np.int32,
        )

    selected_code_ids = sample_index.code_ids[selected]
    unique_code_ids = np.unique(selected_code_ids)
    shuffled_code_ids = np.asarray(rng.permutation(unique_code_ids), dtype=selected_code_ids.dtype)
    code_rank = np.empty(int(sample_index.code_ids.max()) + 1, dtype=np.int32)
    code_rank.fill(-1)
    code_rank[shuffled_code_ids] = np.arange(shuffled_code_ids.size, dtype=np.int32)

    order = np.lexsort((selected, code_rank[selected_code_ids]))
    return selected[order].tolist()


def _build_epoch_loader(
    base_loader: DataLoader,
    *,
    samples_per_epoch: int | None,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> DataLoader:
    """Create one epoch loader with optional code-grouped sampling."""
    if samples_per_epoch is None and not shuffle:
        return base_loader

    dataset = base_loader.dataset
    grouped_indices = _grouped_epoch_indices(
        dataset,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        epoch=epoch,
    )
    if grouped_indices is None:
        if samples_per_epoch is None or samples_per_epoch <= 0 or len(dataset) <= samples_per_epoch:
            return base_loader
        rng = np.random.default_rng(seed + epoch)
        grouped_indices = rng.choice(len(dataset), size=samples_per_epoch, replace=False).tolist()

    if isinstance(grouped_indices, list) and len(grouped_indices) == len(dataset):
        # Still rebuild the loader so order is grouped by code rather than global random shuffle.
        epoch_dataset = Subset(dataset, grouped_indices)
    else:
        epoch_dataset = Subset(dataset, grouped_indices)

    return DataLoader(
        epoch_dataset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=base_loader.pin_memory,
        drop_last=base_loader.drop_last,
        persistent_workers=base_loader.persistent_workers,
        prefetch_factor=base_loader.prefetch_factor if base_loader.num_workers > 0 else None,
    )


def fit(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    *,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    scheduler: Any = None,
    grad_clip: float | None = DEFAULT_GRAD_CLIP,
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    run_name: str = "default",
    load_checkpoint: str | Path | None = None,
    save_every: int = DEFAULT_SAVE_EVERY,
    patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
    use_rich_ui: bool = True,
    use_mlflow: bool = DEFAULT_MLFLOW_ENABLED,
    mlflow_repo: str | Path = DEFAULT_MLFLOW_DIR,
    train_samples_per_epoch: int | None = DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
    val_samples_per_epoch: int | None = DEFAULT_VAL_SAMPLES_PER_EPOCH,
    seed: int = DEFAULT_TRAIN_SEED,
    amp_enabled: bool = DEFAULT_AMP_ENABLED,
    log_every_steps: int = DEFAULT_LOG_EVERY_STEPS,
    profile_enabled: bool = DEFAULT_PROFILE_ENABLED,
    profile_epoch: int = DEFAULT_PROFILE_EPOCH,
    profile_wait_steps: int = DEFAULT_PROFILE_WAIT_STEPS,
    profile_warmup_steps: int = DEFAULT_PROFILE_WARMUP_STEPS,
    profile_active_steps: int = DEFAULT_PROFILE_ACTIVE_STEPS,
    profile_repeat: int = DEFAULT_PROFILE_REPEAT,
    profile_record_shapes: bool = DEFAULT_PROFILE_RECORD_SHAPES,
    profile_memory: bool = DEFAULT_PROFILE_MEMORY,
    profile_with_stack: bool = DEFAULT_PROFILE_WITH_STACK,
    profile_with_flops: bool = DEFAULT_PROFILE_WITH_FLOPS,
    profile_row_limit: int = DEFAULT_PROFILE_ROW_LIMIT,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> dict[str, Any]:
    """Run the full training loop."""
    model.to(device)
    criterion.to(device)

    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 0
    global_train_step = 0
    run_dir = _run_checkpoint_dir(checkpoint_dir, run_name)
    checkpoint_paths = _checkpoint_paths(run_dir)
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else checkpoint_paths["latest"]
    ui = TrainingUI() if use_rich_ui else None
    visualizer = (
        MLflowVisualizer(
            experiment="dragon",
            repo=str(mlflow_repo),
            run_name=run_name,
        )
        if use_mlflow
        else None
    )
    resolved_amp_enabled = bool(amp_enabled and device.startswith("cuda") and torch.cuda.is_available())
    scaler = GradScaler("cuda", enabled=resolved_amp_enabled)

    try:
        if visualizer is not None:
            visualizer.attach(model)
            visualizer.log_params(
                {
                    "run_name": run_name,
                    "checkpoint_dir": str(run_dir),
                    "batch_size": train_loader.batch_size,
                    "num_epochs": num_epochs,
                    "amp_enabled": resolved_amp_enabled,
                    "train_samples_per_epoch": train_samples_per_epoch,
                    "val_samples_per_epoch": val_samples_per_epoch,
                    "grad_clip": grad_clip,
                    "log_every_steps": log_every_steps,
                    "profile_enabled": profile_enabled,
                    "profile_epoch": profile_epoch,
                    "profile_wait_steps": profile_wait_steps,
                    "profile_warmup_steps": profile_warmup_steps,
                    "profile_active_steps": profile_active_steps,
                    "profile_repeat": profile_repeat,
                    "freeze_min_steps": getattr(criterion, "min_freeze_steps", None),
                    "freeze_patience_steps": getattr(criterion, "patience_steps", None),
                    "freeze_ema_beta": getattr(criterion, "ema_beta", None),
                    "freeze_s0_S": float(getattr(criterion, "s0_S").item()) if hasattr(criterion, "s0_S") else None,
                    "freeze_s0_M": float(getattr(criterion, "s0_M").item()) if hasattr(criterion, "s0_M") else None,
                    "freeze_s0_MDD": float(getattr(criterion, "s0_MDD").item()) if hasattr(criterion, "s0_MDD") else None,
                    "freeze_s0_RV": float(getattr(criterion, "s0_RV").item()) if hasattr(criterion, "s0_RV") else None,
                }
            )

        if load_checkpoint is not None:
            resume_state = _load_checkpoint_state(
                load_checkpoint,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            start_epoch = int(resume_state.get("epoch", -1)) + 1
            history = resume_state.get("history", history)
            best_epoch = int(resume_state.get("best_epoch", best_epoch))
            best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
            global_train_step = int(resume_state.get("global_train_step", 0))

        if ui is not None:
            ui.start()

        for epoch in range(start_epoch, num_epochs):
            epoch_train_loader = _build_epoch_loader(
                train_loader,
                samples_per_epoch=train_samples_per_epoch,
                seed=seed,
                epoch=epoch * 2,
                shuffle=False,
            )
            epoch_val_loader = _build_epoch_loader(
                val_loader,
                samples_per_epoch=val_samples_per_epoch,
                seed=seed,
                epoch=epoch * 2 + 1,
                shuffle=False,
            )
            if ui is not None:
                ui.start_epoch(epoch + 1, num_epochs, len(epoch_train_loader))
            if visualizer is not None:
                visualizer.start_epoch_stage("train")

            def _step_callback(payload: dict[str, Any]) -> None:
                if ui is None:
                    return
                ui.update_train_step(
                    step=int(payload["step"]),
                    total_steps=int(payload["total_steps"]),
                    metrics=payload["metrics"],
                )

            profiler_session = create_epoch_profiler(
                enabled=profile_enabled,
                epoch=epoch + 1,
                target_epoch=profile_epoch,
                run_name=run_name,
                output_root=profile_dir,
                device=device,
                wait=profile_wait_steps,
                warmup=profile_warmup_steps,
                active=profile_active_steps,
                repeat=profile_repeat,
                record_shapes=profile_record_shapes,
                profile_memory=profile_memory,
                with_stack=profile_with_stack,
                with_flops=profile_with_flops,
            )
            train_context = profiler_session.profiler if profiler_session.enabled else nullcontext()
            with train_context:
                train_metrics = train_one_epoch(
                    model,
                    criterion,
                    optimizer,
                    epoch_train_loader,
                    device,
                    grad_clip=grad_clip,
                    step_callback=_step_callback if ui is not None else None,
                    visualizer=visualizer,
                    epoch=epoch + 1,
                    global_step_offset=global_train_step,
                    scaler=scaler,
                    amp_enabled=resolved_amp_enabled,
                    profiler=profiler_session.profiler,
                    log_every_steps=log_every_steps,
                )
            if profiler_session.enabled:
                write_profiler_reports(profiler_session, row_limit=profile_row_limit)
            global_train_step += len(epoch_train_loader)
            if visualizer is not None:
                visualizer.log_epoch_diagnostics("train", epoch + 1)
            if ui is not None:
                ui.set_status(f"Epoch {epoch + 1}/{num_epochs} validating")
            if visualizer is not None:
                visualizer.start_epoch_stage("val")
            val_metrics = validate(
                model,
                criterion,
                epoch_val_loader,
                device,
                visualizer=visualizer,
                amp_enabled=resolved_amp_enabled,
            )
            if visualizer is not None:
                visualizer.log_epoch_diagnostics("val", epoch + 1)
            if ui is not None:
                ui.set_val_metrics(val_metrics)

            epoch_low_metrics = (
                visualizer.collect_low_frequency_metrics(model, optimizer)
                if visualizer is not None
                else {}
            )
            train_summary_metrics = {**train_metrics, **epoch_low_metrics}
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            if visualizer is not None:
                visualizer.track("train", train_summary_metrics, epoch=epoch + 1, subset="epoch")
                visualizer.track("val", val_metrics, epoch=epoch + 1, subset="epoch")

            current_val_loss = float(val_metrics.get("loss_total", val_metrics.get("loss", 0.0)))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                stale_epochs = 0

                _save_checkpoint(
                    checkpoint_paths["best"],
                    epoch=epoch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    global_train_step=global_train_step,
                    run_name=run_name,
                )
                if checkpoint_file != checkpoint_paths["latest"] and checkpoint_file != checkpoint_paths["best"]:
                    _save_checkpoint(
                        checkpoint_file,
                        epoch=epoch,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        history=history,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        global_train_step=global_train_step,
                        run_name=run_name,
                    )
            else:
                stale_epochs += 1

            _save_checkpoint(
                checkpoint_paths["latest"],
                epoch=epoch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                global_train_step=global_train_step,
                run_name=run_name,
            )

            if save_every > 0 and (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    run_dir / f"epoch_{epoch + 1:03d}.pt",
                    epoch=epoch,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    global_train_step=global_train_step,
                    run_name=run_name,
                )

            _step_scheduler(scheduler, current_val_loss)

            if ui is not None:
                ui.set_status(f"Epoch {epoch + 1}/{num_epochs} complete")

            _clear_dataset_cache(epoch_train_loader.dataset)
            _clear_dataset_cache(epoch_val_loader.dataset)
            del epoch_train_loader
            del epoch_val_loader
            gc.collect()

            if patience is not None and stale_epochs >= patience:
                break
    finally:
        if visualizer is not None:
            visualizer.detach()
            visualizer.close()
        if ui is not None:
            ui.stop()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def run_training(
    *,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    split_date: float | int | None = DEFAULT_TRAIN_VAL_SPLIT_DATE,
    grad_clip: float | None = DEFAULT_GRAD_CLIP,
    val_ratio: float = DEFAULT_VAL_RATIO,
    memory_mode: str = DEFAULT_MEMORY_MODE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_TRAIN_SEED,
    max_codes: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    run_name: str = "default",
    load_checkpoint: str | Path | None = None,
    save_every: int = DEFAULT_SAVE_EVERY,
    patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
    scheduler_name: str | None = DEFAULT_SCHEDULER_NAME,
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR,
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE,
    scheduler_min_lr: float = DEFAULT_SCHEDULER_MIN_LR,
    use_rich_ui: bool = True,
    use_mlflow: bool = DEFAULT_MLFLOW_ENABLED,
    mlflow_repo: str | Path = DEFAULT_MLFLOW_DIR,
    train_samples_per_epoch: int | None = DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
    val_samples_per_epoch: int | None = DEFAULT_VAL_SAMPLES_PER_EPOCH,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    amp_enabled: bool = DEFAULT_AMP_ENABLED,
    log_every_steps: int = DEFAULT_LOG_EVERY_STEPS,
    profile_enabled: bool = DEFAULT_PROFILE_ENABLED,
    profile_epoch: int = DEFAULT_PROFILE_EPOCH,
    profile_wait_steps: int = DEFAULT_PROFILE_WAIT_STEPS,
    profile_warmup_steps: int = DEFAULT_PROFILE_WARMUP_STEPS,
    profile_active_steps: int = DEFAULT_PROFILE_ACTIVE_STEPS,
    profile_repeat: int = DEFAULT_PROFILE_REPEAT,
    profile_record_shapes: bool = DEFAULT_PROFILE_RECORD_SHAPES,
    profile_memory: bool = DEFAULT_PROFILE_MEMORY,
    profile_with_stack: bool = DEFAULT_PROFILE_WITH_STACK,
    profile_with_flops: bool = DEFAULT_PROFILE_WITH_FLOPS,
    profile_row_limit: int = DEFAULT_PROFILE_ROW_LIMIT,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> dict[str, Any]:
    """Build datasets, model, loss, optimizer, and run training."""
    torch.manual_seed(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader_uses_cuda = resolved_device.startswith("cuda")
    persistent_workers = num_workers > 0
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": loader_uses_cuda,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataset, val_dataset = create_train_val_datasets(
        split_date=split_date,
        val_ratio=val_ratio,
        max_codes=max_codes,
        memory_mode=memory_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = MultiScaleFusionNet()
    criterion = LaplaceLSELoss().to(resolved_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
    )

    return fit(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        resolved_device,
        num_epochs=num_epochs,
        scheduler=scheduler,
        grad_clip=grad_clip,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        load_checkpoint=load_checkpoint,
        save_every=save_every,
        patience=patience,
        use_rich_ui=use_rich_ui,
        use_mlflow=use_mlflow,
        mlflow_repo=mlflow_repo,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=val_samples_per_epoch,
        seed=seed,
        amp_enabled=amp_enabled,
        log_every_steps=log_every_steps,
        profile_enabled=profile_enabled,
        profile_epoch=profile_epoch,
        profile_wait_steps=profile_wait_steps,
        profile_warmup_steps=profile_warmup_steps,
        profile_active_steps=profile_active_steps,
        profile_repeat=profile_repeat,
        profile_record_shapes=profile_record_shapes,
        profile_memory=profile_memory,
        profile_with_stack=profile_with_stack,
        profile_with_flops=profile_with_flops,
        profile_row_limit=profile_row_limit,
        profile_dir=profile_dir,
    )


def smoke_test(
    *,
    device: str | None = None,
    max_codes: int = 1,
    memory_mode: str = DEFAULT_MEMORY_MODE,
) -> dict[str, Any]:
    """Run one batch through model and loss for shape verification."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, _ = create_train_val_datasets(
        val_ratio=0.0,
        max_codes=max_codes,
        memory_mode=memory_mode,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No samples available for smoke test")

    batch = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=min(4, len(train_dataset)),
                shuffle=False,
                num_workers=0,
            )
        )
    )
    batch = {key: value.to(resolved_device) for key, value in batch.items()}

    model = MultiScaleFusionNet().to(resolved_device)
    criterion = LaplaceLSELoss().to(resolved_device)
    outputs = model(
        batch["macro"],
        batch["mezzo"],
        batch["micro"],
        batch["sidechain"],
    )
    loss, metrics = criterion(outputs, batch)

    return {
        "macro_shape": tuple(batch["macro"].shape),
        "mezzo_shape": tuple(batch["mezzo"].shape),
        "micro_shape": tuple(batch["micro"].shape),
        "sidechain_shape": tuple(batch["sidechain"].shape),
        "M1_shape": tuple(outputs["M1"].shape),
        "M2_shape": tuple(outputs["M2"].shape),
        "S_shape": tuple(outputs["S"].shape),
        "z_d_shape": tuple(outputs["z_d"].shape),
        "z_v_shape": tuple(outputs["z_v"].shape),
        "pred_S_shape": tuple(outputs["pred_S"].shape),
        "pred_M_shape": tuple(outputs["pred_M"].shape),
        "pred_MDD_shape": tuple(outputs["pred_MDD"].shape),
        "pred_RV_shape": tuple(outputs["pred_RV"].shape),
        "loss": float(loss.detach().item()),
        "metrics": {key: float(value.detach().item()) for key, value in metrics.items()},
    }


def main() -> None:
    print(run_training())


if __name__ == "__main__":
    main()
