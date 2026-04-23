from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.collate import collate_network_batch
from src.train.console import EpochConsoleLogger
from src.train.dataloaders import build_train_dataloader, build_val_dataloader
from src.train.dataset import AssembledNPZDataset
from src.train.runtime import build_optimizer, build_scheduler, maybe_compile_model
from src.train.tensorboard_logger import TensorBoardLogger
from src.train.train_entry import _resolve_checkpoint_runtime
from src.train.trainer import Trainer


def _write_minimal_npz(path: Path, samples: int = 4) -> Path:
    macro = np.zeros((samples, 9, 112), dtype=np.float32)
    mezzo = np.zeros((samples, 9, 144), dtype=np.float32)
    micro = np.zeros((samples, 9, 192), dtype=np.float32)
    sidechain = np.zeros((samples, 13, 112), dtype=np.float32)
    label = np.zeros((samples, 2), dtype=np.float32)

    macro_i8 = np.zeros((samples, 2, 112), dtype=np.int8)
    mezzo_i8 = np.zeros((samples, 2, 144), dtype=np.int8)
    micro_i8 = np.zeros((samples, 2, 192), dtype=np.int8)

    for idx in range(samples):
        macro[idx] = idx + np.linspace(0.0, 1.0, num=9 * 112, dtype=np.float32).reshape(9, 112)
        mezzo[idx] = idx + np.linspace(0.0, 1.0, num=9 * 144, dtype=np.float32).reshape(9, 144)
        micro[idx] = idx + np.linspace(0.0, 1.0, num=9 * 192, dtype=np.float32).reshape(9, 192)
        sidechain[idx] = idx + np.linspace(
            0.0,
            1.0,
            num=13 * 112,
            dtype=np.float32,
        ).reshape(13, 112)
        label[idx] = np.asarray([0.1 * idx, 0.2 + 0.1 * idx], dtype=np.float32)

        macro_i8[idx, 0] = idx % 16
        macro_i8[idx, 1] = np.arange(112, dtype=np.int16) % 8
        mezzo_i8[idx, 0] = (idx + 1) % 16
        mezzo_i8[idx, 1] = np.arange(144, dtype=np.int16) % 16
        micro_i8[idx, 0] = (idx + 2) % 16
        micro_i8[idx, 1] = np.arange(192, dtype=np.int16) % 64

    np.savez(
        path,
        label_schema_version=np.asarray(8, dtype=np.int32),
        label_names=np.asarray(["label_ret", "label_rv"]),
        date=np.arange(samples, dtype=np.float32),
        label=label,
        macro=macro,
        mezzo=mezzo,
        micro=micro,
        sidechain=sidechain,
        macro_i8=macro_i8,
        mezzo_i8=mezzo_i8,
        micro_i8=micro_i8,
    )
    return path


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task = "ret"
        self.proj = nn.Linear(1, 2).to(dtype=torch.bfloat16)

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = batch["macro_float_long"].mean(dim=(1, 2), keepdim=True)
        pred = self.proj(x.view(-1, 1))
        pred_primary = pred[:, 0:1]
        pred_aux_raw = pred[:, 1:2]
        loss_task = ((pred_primary - batch["target_ret"]) ** 2).mean()
        return {
            "loss_total": loss_task,
            "loss_task": loss_task,
            "pred_primary": pred_primary,
            "pred_aux_raw": pred_aux_raw,
            "fused_latents": batch["micro_float_long"],
        }


class _TrackingConsoleLogger:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.started = False

    def start_phase(self, *, epoch: int, phase: str, total_steps: int) -> None:
        assert getattr(self.model, "first_forward_done", False) is True
        self.started = True

    def advance(self, step: int) -> None:
        return None

    def log_metrics(
        self,
        *,
        epoch: int,
        phase: str,
        step: int,
        total_steps: int,
        losses: dict[str, float],
        lr: float,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class _CompileTrackingModel(_TinyTrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.first_forward_done = False

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        out = super().forward_loss(batch, return_aux=return_aux)
        self.first_forward_done = True
        return out


class _CountingTensorBoardLogger:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, int]] = []
        self.epoch_metric_calls = 0
        self.epoch_plot_calls = 0

    def update_prediction_state(
        self,
        *,
        phase: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor | None = None,
    ) -> None:
        self.update_calls.append((phase, int(predictions.shape[0])))

    def log_epoch_metrics(
        self,
        *,
        phase: str,
        global_step: int,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        self.epoch_metric_calls += 1

    def log_epoch_prediction_plot(
        self,
        *,
        phase: str,
        global_step: int,
    ) -> None:
        self.epoch_plot_calls += 1


def test_dataset_can_read_and_adapt(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=3)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["macro_float_long"].shape == (9, 112)
    assert sample["mezzo_float_long"].shape == (9, 144)
    assert sample["micro_float_long"].shape == (9, 192)
    assert sample["sidechain_cond"].shape == (13, 64)
    assert sample["target_q"].shape == (1,)
    assert np.allclose(sample["target_q"], sample["target_ret"])


def test_collate_network_batch_shapes_and_dtypes(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=2)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)

    batch = collate_network_batch([dataset[0], dataset[1]])

    assert batch["macro_float_long"].shape == (2, 9, 112)
    assert batch["macro_i8_long"].shape == (2, 2, 112)
    assert batch["sidechain_cond"].shape == (2, 13, 64)
    assert batch["macro_float_long"].dtype == torch.bfloat16
    assert batch["macro_i8_long"].dtype == torch.int64


def test_train_loader_uses_drop_last_true(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=5)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)

    assert loader.batch_sampler.drop_last is True


def test_train_step_smoke_runs(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-1, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        console_logger=EpochConsoleLogger(log_every=1, enabled=False),
    )

    before = model.proj.weight.detach().clone()
    metrics = trainer.train_one_epoch(epoch=0)

    assert metrics["loss_total"] >= 0.0
    assert not torch.equal(before, model.proj.weight.detach())


def test_console_phase_starts_only_after_first_forward(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _CompileTrackingModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    console_logger = _TrackingConsoleLogger(model)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        console_logger=console_logger,
    )

    trainer.train_one_epoch(epoch=0)

    assert console_logger.started is True


def test_val_step_smoke_runs_without_backward(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        console_logger=EpochConsoleLogger(log_every=1, enabled=False),
    )

    metrics = trainer.validate_one_epoch(epoch=0)

    assert metrics["loss_total"] >= 0.0
    assert all(param.grad is None for param in model.parameters())


def test_trainer_fit_runs_without_external_logger(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        console_logger=EpochConsoleLogger(log_every=1, enabled=False),
    )

    trainer.fit(num_epochs=1)

    assert len(trainer.history) == 1
    assert trainer.history[0]["train"]["loss_total"] >= 0.0
    assert trainer.history[0]["val"]["loss_total"] >= 0.0


def test_trainer_fit_writes_tensorboard_events(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    logger = TensorBoardLogger(
        log_dir=tmp_path / "tb" / "smoke",
        task="ret",
        enabled=True,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        console_logger=EpochConsoleLogger(log_every=1, enabled=False),
        tensorboard_logger=logger,
    )

    trainer.fit(num_epochs=1)

    assert list((tmp_path / "tb" / "smoke").glob("events.out.tfevents.*"))


def test_prediction_state_updates_every_step(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=6)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    logger = _CountingTensorBoardLogger()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=2,
        console_logger=EpochConsoleLogger(log_every=2, enabled=False),
        tensorboard_logger=logger,
    )

    trainer.train_one_epoch(epoch=0)

    assert len(logger.update_calls) == 3
    assert logger.update_calls[0][0] == "train"
    assert logger.update_calls[1][0] == "train"
    assert logger.update_calls[2][0] == "train"
    assert logger.epoch_metric_calls == 1
    assert logger.epoch_plot_calls == 1


def test_compile_path_uses_reduce_overhead(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 2)
    recorded: dict[str, object] = {}

    def _fake_compile(module: nn.Module, *, mode: str, **kwargs) -> nn.Module:
        recorded["module"] = module
        recorded["mode"] = mode
        recorded["kwargs"] = kwargs
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    compiled = maybe_compile_model(model, enabled=True)

    assert compiled is model
    assert recorded["mode"] == "reduce-overhead"


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)

    reference_weight = model.proj.weight.detach().clone()
    checkpoint_path = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=3,
        global_step=17,
    )

    with torch.no_grad():
        model.proj.weight.add_(1.0)

    state = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        load_training_state=True,
    )

    assert state["epoch"] == 3
    assert state["global_step"] == 17
    assert torch.allclose(model.proj.weight.detach(), reference_weight)


def test_trainer_fit_writes_epoch_latest_and_best_checkpoints(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        log_every=1,
        save_every=1,
        console_logger=EpochConsoleLogger(log_every=1, enabled=False),
    )
    trainer.checkpoint_dir = tmp_path / "models" / "checkpoints" / "smoke"
    trainer.fit(num_epochs=1)

    assert (trainer.checkpoint_dir / "epoch_001.pt").exists()
    assert (trainer.checkpoint_dir / "latest.pt").exists()
    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_resolve_checkpoint_runtime_for_name_and_load(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "models" / "checkpoints"

    run_name, checkpoint_dir, resume_from = _resolve_checkpoint_runtime(
        checkpoint_root=checkpoint_root,
        name="fresh",
        load_name=None,
        checkpoint=None,
    )
    assert run_name == "fresh"
    assert checkpoint_dir == checkpoint_root / "fresh"
    assert resume_from is None

    (checkpoint_root / "resume").mkdir(parents=True)
    (checkpoint_root / "resume" / "latest.pt").write_bytes(b"ckpt")
    run_name, checkpoint_dir, resume_from = _resolve_checkpoint_runtime(
        checkpoint_root=checkpoint_root,
        name=None,
        load_name="resume",
        checkpoint=None,
    )
    assert run_name == "resume"
    assert checkpoint_dir == checkpoint_root / "resume"
    assert resume_from == checkpoint_root / "resume" / "latest.pt"

    run_name, checkpoint_dir, resume_from = _resolve_checkpoint_runtime(
        checkpoint_root=checkpoint_root,
        name="migrated",
        load_name="resume",
        checkpoint=None,
    )
    assert run_name == "migrated"
    assert checkpoint_dir == checkpoint_root / "migrated"
    assert resume_from == checkpoint_root / "resume" / "latest.pt"


def test_resolve_checkpoint_runtime_for_name_and_checkpoint(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "models" / "checkpoints"
    source_dir = checkpoint_root / "source"
    source_dir.mkdir(parents=True)
    source_checkpoint = source_dir / "latest.pt"
    source_checkpoint.write_bytes(b"ckpt")

    run_name, checkpoint_dir, resume_from = _resolve_checkpoint_runtime(
        checkpoint_root=checkpoint_root,
        name="migrated",
        load_name=None,
        checkpoint=source_checkpoint,
    )
    assert run_name == "migrated"
    assert checkpoint_dir == checkpoint_root / "migrated"
    assert resume_from == source_checkpoint


def test_resolve_checkpoint_runtime_rejects_load_and_checkpoint_together(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "models" / "checkpoints"
    checkpoint_path = checkpoint_root / "source" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"ckpt")

    with pytest.raises(ValueError, match="Only one of load_name or checkpoint"):
        _resolve_checkpoint_runtime(
            checkpoint_root=checkpoint_root,
            name="migrated",
            load_name="resume",
            checkpoint=checkpoint_path,
        )
