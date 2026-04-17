"""End-to-end training entrypoints."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler, Subset

from config.config import amp_enabled as DEFAULT_AMP_ENABLED
from config.config import batch_size as DEFAULT_BATCH_SIZE
from config.config import checkpoint_dir as DEFAULT_CHECKPOINT_DIR
from config.config import early_stopping_patience as DEFAULT_EARLY_STOPPING_PATIENCE
from config.config import mlflow_dir as DEFAULT_MLFLOW_DIR
from config.config import mlflow_enabled as DEFAULT_MLFLOW_ENABLED
from config.config import grad_clip as DEFAULT_GRAD_CLIP
from config.config import learning_rate as DEFAULT_LEARNING_RATE
from config.config import log_every as DEFAULT_LOG_EVERY
from config.config import memory_mode as DEFAULT_MEMORY_MODE
from config.config import num_epochs as DEFAULT_NUM_EPOCHS
from config.config import num_workers as DEFAULT_NUM_WORKERS
from config.config import prefetch_factor as DEFAULT_PREFETCH_FACTOR
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
from config.config import preload_index_into_memory as DEFAULT_PRELOAD_INDEX
from config.config import compile_mode as DEFAULT_COMPILE_MODE
from src.models import MultiScaleFusionNet
from src.models.losses import SingleTaskLoss
from src.task_labels import TASK_LABELS, canonical_task_label
from src.train.dataset import create_train_val_datasets
from src.train.train import train_one_epoch
from src.train.ui import TrainingUI
from src.train.validate import validate
from src.train.visualize_strict import MLflowVisualizer


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


def _preload_sample_index(dataset: Any, num_workers: int) -> None:
    """Load sample_index into RAM permanently to avoid repeated disk reads each epoch."""
    owner = _resolve_dataset_cache_owner(dataset)
    si = getattr(owner, "sample_index", None)
    if si is None:
        return
    # Check if already preloaded (has _index_cached attribute)
    if getattr(owner, "_index_preloaded", False):
        return
    # Read once into numpy arrays
    payload_ids = si.payload_ids
    sample_idx = si.sample_idx
    # Store as numpy arrays for fast access
    owner._index_payload_ids = np.asarray(payload_ids, dtype=np.int32)
    owner._index_sample_idx = np.asarray(sample_idx, dtype=np.int32)
    owner._index_payload_ids_max = int(payload_ids.max())
    owner._index_preloaded = True
    return None  # Explicitly return None to indicate preload done


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
    label: str,
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
            "label": label,
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
    expected_label: str,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(load_checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    checkpoint_label = state.get("label")
    if checkpoint_label != expected_label:
        raise ValueError(
            "Checkpoint label mismatch. "
            f"Expected {expected_label!r}, found {checkpoint_label!r} in {checkpoint_path}."
        )
    model.load_state_dict(state["model_state_dict"])
    if state.get("criterion_state_dict") is not None:
        criterion.load_state_dict(state["criterion_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def _run_name_from_checkpoint(load_checkpoint: str | Path) -> str:
    checkpoint_path = Path(load_checkpoint)
    if checkpoint_path.is_dir():
        return checkpoint_path.name
    return checkpoint_path.parent.name if checkpoint_path.parent.name else checkpoint_path.stem


def _grouped_epoch_indices(
    dataset: Any,
    *,
    samples_per_epoch: int | None,
    seed: int,
    epoch: int,
    num_workers: int = 1,
) -> list[int] | None:
    """Compute epoch-specific sample indices with optional multiprocessing."""
    owner = _resolve_dataset_cache_owner(dataset)
    si = getattr(owner, "sample_index", None)
    if si is None:
        return None

    # Use preloaded numpy arrays if available, otherwise read from sample_index
    if getattr(owner, "_index_preloaded", False):
        payload_ids = owner._index_payload_ids
        sample_idx = owner._index_sample_idx
        payload_ids_max = owner._index_payload_ids_max
    else:
        payload_ids = np.asarray(si.payload_ids, dtype=np.int32)
        sample_idx = np.asarray(si.sample_idx, dtype=np.int32)
        payload_ids_max = int(si.payload_ids.max())

    total = len(payload_ids)
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

    selected_payload_ids = payload_ids[selected]
    unique_payload_ids = np.unique(selected_payload_ids)
    shuffled_payload_ids = np.asarray(rng.permutation(unique_payload_ids), dtype=selected_payload_ids.dtype)
    payload_rank = np.empty(payload_ids_max + 1, dtype=np.int32)
    payload_rank.fill(-1)
    payload_rank[shuffled_payload_ids] = np.arange(shuffled_payload_ids.size, dtype=np.int32)

    order = np.lexsort((selected, sample_idx[selected], payload_rank[selected_payload_ids]))
    return selected[order].tolist()


class MutableIndexSampler(Sampler[int]):
    """Mutable sampler to keep one worker pool alive across epochs."""

    def __init__(self, indices: list[int] | range | np.ndarray) -> None:
        self._indices: list[int] = [int(idx) for idx in indices]

    def set_indices(self, indices: list[int] | np.ndarray) -> None:
        self._indices = [int(idx) for idx in indices]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def _set_epoch_loader_indices(
    loader: DataLoader,
    *,
    samples_per_epoch: int | None,
    seed: int,
    epoch: int,
    num_workers: int,
    shuffle: bool,
) -> None:
    """Update one persistent loader's sampler with epoch-specific indices."""
    del shuffle

    sampler = loader.sampler
    if not isinstance(sampler, MutableIndexSampler):
        raise TypeError("Expected loader.sampler to be MutableIndexSampler")

    dataset = loader.dataset
    grouped_indices = _grouped_epoch_indices(
        dataset,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        epoch=epoch,
        num_workers=num_workers,
    )
    if grouped_indices is None:
        if samples_per_epoch is None or samples_per_epoch <= 0 or len(dataset) <= samples_per_epoch:
            grouped_indices = list(range(len(dataset)))
        else:
            rng = np.random.default_rng(seed + epoch)
            grouped_indices = rng.choice(len(dataset), size=samples_per_epoch, replace=False).tolist()

    sampler.set_indices(grouped_indices)


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
    label: str = "ret",
    run_name: str = "default",
    load_checkpoint: str | Path | None = None,
    save_every: int = DEFAULT_SAVE_EVERY,
    patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
    use_rich_ui: bool = True,
    use_mlflow: bool = DEFAULT_MLFLOW_ENABLED,
    mlflow_repo: str | Path = DEFAULT_MLFLOW_DIR,
    train_samples_per_epoch: int | None = DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
    val_samples_per_epoch: int | None = DEFAULT_VAL_SAMPLES_PER_EPOCH,
    log_every: int = DEFAULT_LOG_EVERY,
    seed: int = DEFAULT_TRAIN_SEED,
    amp_enabled: bool = DEFAULT_AMP_ENABLED,
    compile_mode: str = DEFAULT_COMPILE_MODE,
    preload_index: bool = DEFAULT_PRELOAD_INDEX,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, Any]:
    """Run the full training loop."""
    resolved_label = canonical_task_label(label)
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError("Training requires a CUDA device.")

    model.to(device)
    criterion.to(device)

    # Preload sample indices into RAM once to avoid repeated disk reads per epoch.
    if preload_index:
        _preload_sample_index(train_loader.dataset, num_workers)
        _preload_sample_index(val_loader.dataset, num_workers)

    # Favor throughput on fixed-shape CUDA training.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
    scaler = None

    try:
        if visualizer is not None:
            visualizer.attach(model)
            visualizer.prepare_reference_ret_samples(val_loader.dataset)
            visualizer.log_params(
                {
                    "label": resolved_label,
                    "run_name": run_name,
                    "checkpoint_dir": str(run_dir),
                    "batch_size": train_loader.batch_size,
                    "num_epochs": num_epochs,
                    "amp_enabled": resolved_amp_enabled,
                    "compile_mode": compile_mode,
                    "train_samples_per_epoch": train_samples_per_epoch,
                    "val_samples_per_epoch": val_samples_per_epoch,
                    "log_every": log_every,
                    "grad_clip": grad_clip,
                }
            )

        if load_checkpoint is not None:
            resume_state = _load_checkpoint_state(
                load_checkpoint,
                expected_label=resolved_label,
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
            _set_epoch_loader_indices(
                train_loader,
                samples_per_epoch=train_samples_per_epoch,
                seed=seed,
                epoch=epoch * 2,
                num_workers=num_workers,
                shuffle=False,
            )
            _set_epoch_loader_indices(
                val_loader,
                samples_per_epoch=val_samples_per_epoch,
                seed=seed,
                epoch=epoch * 2 + 1,
                num_workers=num_workers,
                shuffle=False,
            )
            epoch_train_loader = train_loader
            epoch_val_loader = val_loader
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
                    step_time_seconds=float(payload.get("step_time_seconds", 0.0)),
                )

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
                log_every=log_every,
                scaler=scaler,
                amp_enabled=resolved_amp_enabled,
            )
            global_train_step += len(epoch_train_loader)
            if visualizer is not None:
                visualizer.log_epoch_diagnostics("train", epoch + 1, model=model, device=device)
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
                visualizer.log_epoch_diagnostics("val", epoch + 1, model=model, device=device)
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
                    label=resolved_label,
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
                        label=resolved_label,
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
                label=resolved_label,
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
                    label=resolved_label,
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

            _clear_dataset_cache(train_loader.dataset)
            _clear_dataset_cache(val_loader.dataset)
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
    label: str,
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
    run_name: str | None = None,
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
    log_every: int = DEFAULT_LOG_EVERY,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    amp_enabled: bool = DEFAULT_AMP_ENABLED,
    compile_mode: str = DEFAULT_COMPILE_MODE,
    preload_index: bool = DEFAULT_PRELOAD_INDEX,
) -> dict[str, Any]:
    """Build datasets, model, loss, optimizer, and run training."""
    resolved_label = canonical_task_label(label)
    if (run_name is None) == (load_checkpoint is None):
        raise ValueError("Exactly one of run_name or load_checkpoint must be provided.")
    resolved_run_name = str(run_name) if run_name is not None else _run_name_from_checkpoint(load_checkpoint)
    torch.manual_seed(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader_uses_cuda = resolved_device.startswith("cuda")
    if not (loader_uses_cuda and torch.cuda.is_available()):
        raise RuntimeError("Training requires a CUDA device.")
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
        scan_workers=max(1, num_workers),
    )
    train_sampler = MutableIndexSampler(range(len(train_dataset)))
    val_sampler = MutableIndexSampler(range(len(val_dataset)))
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **loader_kwargs,
    )

    model = MultiScaleFusionNet(task_label=resolved_label)

    model = torch.compile(model, mode=compile_mode)

    criterion = SingleTaskLoss(task_label=resolved_label).to(resolved_device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
        capturable=compile_mode == "max-autotune",
    )
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
        label=resolved_label,
        run_name=resolved_run_name,
        load_checkpoint=load_checkpoint,
        save_every=save_every,
        patience=patience,
        use_rich_ui=use_rich_ui,
        use_mlflow=use_mlflow,
        mlflow_repo=mlflow_repo,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=val_samples_per_epoch,
        log_every=log_every,
        seed=seed,
        amp_enabled=amp_enabled,
        compile_mode=compile_mode,
        preload_index=preload_index,
        num_workers=num_workers,
    )


def smoke_test(
    *,
    label: str = "ret",
    device: str | None = None,
    max_codes: int = 1,
    memory_mode: str = DEFAULT_MEMORY_MODE,
) -> dict[str, Any]:
    """Run one batch through model and loss for shape verification."""
    resolved_label = canonical_task_label(label)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, _ = create_train_val_datasets(
        val_ratio=0.0,
        max_codes=max_codes,
        memory_mode=memory_mode,
        scan_workers=1,
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

    model = MultiScaleFusionNet(task_label=resolved_label).to(resolved_device)
    criterion = SingleTaskLoss(task_label=resolved_label).to(resolved_device)
    outputs = model(
        batch["macro"],
        batch["mezzo"],
        batch["micro"],
        batch["sidechain"],
    )
    loss, metrics = criterion(outputs, batch)

    return {
        "label": resolved_label,
        "macro_shape": tuple(batch["macro"].shape),
        "mezzo_shape": tuple(batch["mezzo"].shape),
        "micro_shape": tuple(batch["micro"].shape),
        "sidechain_shape": tuple(batch["sidechain"].shape),
        "macro_joint_tokens_shape": tuple(outputs["macro_joint_tokens"].shape),
        "mezzo_joint_tokens_shape": tuple(outputs["mezzo_joint_tokens"].shape),
        "micro_joint_tokens_shape": tuple(outputs["micro_joint_tokens"].shape),
        "joint_summary_shape": tuple(outputs["joint_summary"].shape),
        "side_summary_shape": tuple(outputs["side_summary"].shape),
        "head_out_shape": tuple(outputs["head_out"].shape),
        "loss": float(loss.detach().item()),
        "metrics": {key: float(value.detach().item()) for key, value in metrics.items()},
    }


def main() -> None:
    print(run_training(label=TASK_LABELS[0], run_name="default"))


if __name__ == "__main__":
    main()
