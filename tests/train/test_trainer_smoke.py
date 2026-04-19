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
        self.proj = nn.Linear(1, 3)

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = batch["macro_float_long"].mean(dim=(1, 2), keepdim=True)
        pred = self.proj(x.view(-1, 1))
        pred_ret = pred[:, 0:1]
        pred_rv = pred[:, 1:2]
        pred_q = pred[:, 2:3]
        loss_ret = ((pred_ret - batch["target_ret"]) ** 2).mean()
        loss_rv = ((pred_rv - batch["target_rv"]) ** 2).mean()
        loss_q = ((pred_q - batch["target_q"]) ** 2).mean()
        return {
            "loss_total": loss_ret + loss_rv + loss_q,
            "loss_ret": loss_ret,
            "loss_rv": loss_rv,
            "loss_q": loss_q,
        }


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


def test_train_loader_uses_drop_last_true(tmp_path: Path) -> None:
    path = _write_minimal_npz(tmp_path / "000001.SZ.npz", samples=5)
    dataset = AssembledNPZDataset([str(path)], mmap_mode=None)
    loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)

    assert loader.drop_last is True


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


def test_compile_path_uses_max_autotune(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 2)
    recorded: dict[str, object] = {}

    def _fake_compile(module: nn.Module, *, mode: str) -> nn.Module:
        recorded["module"] = module
        recorded["mode"] = mode
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    compiled = maybe_compile_model(model, enabled=True)

    assert compiled is model
    assert recorded["mode"] == "max-autotune"


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
