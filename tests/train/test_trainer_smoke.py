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
from src.train.train_entry import _resolve_checkpoint_runtime
from src.train.trainer import Trainer
from src.train.wandb_logger import _loss_shares


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
        self.proj = nn.Linear(1, 2)

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


class _FakeWandbLogger:
    def __init__(self) -> None:
        self.train_logs = 0
        self.val_logs = 0
        self.val_snapshots = 0
        self.captured = False
        self.fixed_val_batches: dict[str, dict[str, torch.Tensor]] = {}

    def capture_fixed_val_batch(self, loader) -> None:
        self.captured = True
        base_loader = loader.loader if hasattr(loader, "loader") else loader
        sample = base_loader.dataset[0]
        batch = base_loader.collate_fn([sample])
        self.fixed_val_batches = {"fixed_val": batch}

    def get_fixed_val_batch(self) -> dict[str, torch.Tensor] | None:
        return None

    def get_fixed_val_batches(self) -> dict[str, dict[str, torch.Tensor]]:
        return self.fixed_val_batches

    def should_log_histograms(self, global_step: int) -> bool:
        return global_step % 2 == 0

    def should_log_visuals(self, global_step: int) -> bool:
        return global_step % 2 == 0

    def log_train_step(self, **_: object) -> None:
        self.train_logs += 1

    def log_val_epoch(self, **_: object) -> None:
        self.val_logs += 1

    def log_fixed_val_snapshot(self, **_: object) -> None:
        self.val_snapshots += 1

    def finish(self) -> None:
        return None


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
    assert batch["macro_float_long"].dtype == torch.float32
    assert batch["macro_i8_long"].dtype == torch.int64


def test_loss_shares_use_absolute_contributions_for_negative_nlls() -> None:
    shares = _loss_shares(-2.0, -3.0, -5.0)

    assert shares == pytest.approx((0.2, 0.3, 0.5), rel=1e-6, abs=1e-6)


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

    before = model.proj.weight.detach().clone()
    metrics = trainer.train_one_epoch(epoch=0)

    assert metrics["loss_total"] >= 0.0
    assert not torch.equal(before, model.proj.weight.detach())


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


def test_trainer_wandb_debug_fallback_does_not_require_model_support(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    wandb_logger = _FakeWandbLogger()
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
        wandb_logger=wandb_logger,
    )

    trainer.fit(num_epochs=1)

    assert wandb_logger.captured is True
    assert wandb_logger.train_logs >= 1
    assert wandb_logger.val_logs == 1
    assert wandb_logger.val_snapshots == 1


def test_trainer_logs_one_val_snapshot_per_epoch(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=4)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    train_loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)
    val_loader = build_val_dataloader(dataset, batch_size=2, num_workers=0)

    model = _TinyTrainModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)
    wandb_logger = _FakeWandbLogger()
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
        wandb_logger=wandb_logger,
    )

    trainer.fit(num_epochs=2)

    assert wandb_logger.val_logs == 2
    assert wandb_logger.val_snapshots == 2


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
