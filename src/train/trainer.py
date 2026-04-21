"""Training loop orchestration for the forecast network."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_checkpoint
from .console import EpochConsoleLogger
from .runtime import move_batch_to_device, resolve_amp_dtype


class Trainer:
    """Single-device trainer with train/val loops, AMP, and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
        device: torch.device,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        max_grad_norm: float = 1.0,
        task: str = "ret",
        log_every: int = 50,
        save_every: int = 1,
        console_logger=None,
        forward_loss: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        wandb_logger=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = bool(use_amp)
        self.amp_dtype_name = amp_dtype
        self.amp_dtype = resolve_amp_dtype(amp_dtype)
        self.max_grad_norm = float(max_grad_norm)
        self.task = getattr(model, "task", task)
        self.log_every = int(log_every)
        if save_every <= 0:
            raise ValueError(f"save_every must be > 0, got {save_every}.")
        self.save_every = int(save_every)
        self.console_logger = console_logger or EpochConsoleLogger(
            log_every=self.log_every,
            task=self.task,
        )
        self.wandb_logger = wandb_logger

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []
        self.checkpoint_dir: Path | None = None
        self.forward_loss = forward_loss or self._default_forward_loss
        self._event_step = 0
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.device.type == "cuda"
            and self.use_amp
            and self.amp_dtype == torch.float16
            else None
        )

    def _default_forward_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.model.forward_loss(batch, return_aux=False)

    def _compute_param_norm(self) -> float:
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                total = total + param.detach().float().square().sum()
        return float(total.sqrt().cpu())

    def _compute_grad_norm(self) -> float:
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is None:
                    continue
                total = total + param.grad.detach().float().square().sum()
        return float(total.sqrt().cpu())

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        if self.amp_dtype_name == "tf32":
            return nullcontext()
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        if self.device.type == "cpu" and self.amp_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def _extract_log_losses(
        self,
        output: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        task_loss = output.get("loss_task")
        if task_loss is None:
            legacy_key = f"loss_{self.task}"
            if legacy_key not in output:
                raise KeyError(
                    f"Expected 'loss_task' or {legacy_key!r} in model output, got keys={tuple(output.keys())}."
                )
            task_loss = output[legacy_key]
        return {
            "loss_total": float(output["loss_total"].detach().float().cpu()),
            "loss_task": float(task_loss.detach().float().cpu()),
        }

    def _run_phase(
        self,
        *,
        epoch: int,
        phase: str,
        loader,
        training: bool,
    ) -> dict[str, float]:
        if loader is None:
            return {
                "loss_total": float("nan"),
                "loss_task": float("nan"),
            }

        total_steps = len(loader)
        self.console_logger.start_phase(epoch=epoch, phase=phase, total_steps=total_steps)

        if training:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        aggregate = {
            "loss_total": torch.zeros((), device=self.device, dtype=torch.float32),
            "loss_task": torch.zeros((), device=self.device, dtype=torch.float32),
        }

        last_batch_end = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            self.console_logger.advance(step)
            self._event_step += 1
            should_log_wandb = (
                training
                and self.wandb_logger is not None
                and self._event_step % self.log_every == 0
            )
            data_time_ms = (time.perf_counter() - last_batch_end) * 1000.0
            step_start = time.perf_counter()
            batch = move_batch_to_device(batch, self.device)
            grad_norm: float | None = None

            if training:
                forward_start = time.perf_counter()
                with self._autocast_context():
                    output = self.forward_loss(batch)
                    loss = output["loss_total"]
                forward_time_ms = (time.perf_counter() - forward_start) * 1000.0

                backward_start = time.perf_counter()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    ).detach().float().cpu())
                else:
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    ).detach().float().cpu())
                backward_time_ms = (time.perf_counter() - backward_start) * 1000.0

                optimizer_start = time.perf_counter()
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                optimizer_time_ms = (time.perf_counter() - optimizer_start) * 1000.0

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and getattr(
                    self.scheduler, "_step_per_batch", False
                ):
                    self.scheduler.step()
                self.global_step += 1

                log_output: dict[str, torch.Tensor] | None = None
                if should_log_wandb:
                    with torch.inference_mode():
                        with self._autocast_context():
                            try:
                                log_output = self.model.forward_loss(
                                    batch,
                                    return_aux=True,
                                    return_debug=True,
                                )
                            except TypeError:
                                log_output = self.model.forward_loss(
                                    batch,
                                    return_aux=True,
                                )
                    for key in aggregate:
                        log_output[key] = output[key].detach()
            else:
                forward_start = time.perf_counter()
                with torch.inference_mode():
                    with self._autocast_context():
                        output = self.forward_loss(batch)
                forward_time_ms = (time.perf_counter() - forward_start) * 1000.0
                backward_time_ms = 0.0
                optimizer_time_ms = 0.0
                log_output = None

            for key in aggregate:
                aggregate[key] = aggregate[key] + output[key].detach().float()

            step_time_ms = (time.perf_counter() - step_start) * 1000.0
            samples_per_sec = (
                float(batch["macro_float_long"].shape[0]) * 1000.0 / max(step_time_ms, 1e-12)
            )

            if self._event_step % self.log_every == 0:
                lr = float(self.optimizer.param_groups[0]["lr"])
                self.console_logger.log_metrics(
                    epoch=epoch,
                    phase=phase,
                    step=step,
                    total_steps=total_steps,
                    losses=self._extract_log_losses(output),
                    lr=lr,
                )
            if should_log_wandb:
                self.wandb_logger.log_train_step(
                    global_step=self.global_step,
                    epoch=epoch,
                    batch=batch,
                    output=log_output or output,
                    model=self.model,
                    grad_norm=grad_norm,
                    param_norm=self._compute_param_norm(),
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    step_time_ms=step_time_ms,
                    data_time_ms=data_time_ms,
                    forward_time_ms=forward_time_ms,
                    backward_time_ms=backward_time_ms,
                    optimizer_time_ms=optimizer_time_ms,
                    samples_per_sec=samples_per_sec,
                )

            last_batch_end = time.perf_counter()

        num_steps = max(total_steps, 1)
        metrics = {
            key: float((value / num_steps).detach().cpu()) for key, value in aggregate.items()
        }

        if total_steps > 0 and self._event_step % self.log_every != 0:
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.console_logger.log_metrics(
                epoch=epoch,
                phase=phase,
                step=total_steps,
                total_steps=total_steps,
                losses=metrics,
                lr=lr,
            )

        return metrics

    def _log_fixed_val_snapshots(self, *, epoch: int) -> None:
        if self.wandb_logger is None or not hasattr(self.wandb_logger, "get_fixed_val_batches"):
            return
        fixed_batches = self.wandb_logger.get_fixed_val_batches()
        if not fixed_batches:
            return

        self.model.eval()
        with torch.inference_mode():
            for bucket_name, batch in fixed_batches.items():
                batch_on_device = move_batch_to_device(batch, self.device)
                with self._autocast_context():
                    try:
                        output = self.model.forward_loss(
                            batch_on_device,
                            return_aux=True,
                            return_debug=True,
                        )
                    except TypeError:
                        output = self.model.forward_loss(
                            batch_on_device,
                            return_aux=True,
                        )
                self.wandb_logger.log_fixed_val_snapshot(
                    global_step=self.global_step,
                    epoch=epoch,
                    batch=batch_on_device,
                    output=output,
                    bucket_name=bucket_name,
                )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        return self._run_phase(
            epoch=epoch,
            phase="train",
            loader=self.train_loader,
            training=True,
        )

    def validate_one_epoch(self, epoch: int) -> dict[str, float]:
        return self._run_phase(
            epoch=epoch,
            phase="val",
            loader=self.val_loader,
            training=False,
        )

    def fit(self, num_epochs: int) -> None:
        try:
            if self.wandb_logger is not None:
                self.wandb_logger.capture_fixed_val_batch(self.val_loader)
            for epoch in range(self.start_epoch, num_epochs):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.validate_one_epoch(epoch)

                if self.scheduler is not None and not getattr(
                    self.scheduler, "_step_per_batch", False
                ):
                    self.scheduler.step()

                history_row = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                self.history.append(history_row)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_val_epoch(
                        global_step=self.global_step,
                        epoch=epoch,
                        metrics=val_metrics,
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                    )
                    self._log_fixed_val_snapshots(epoch=epoch)

                if self.checkpoint_dir is not None:
                    epoch_number = epoch + 1
                    if epoch_number % self.save_every == 0:
                        save_checkpoint(
                            self.checkpoint_dir / f"epoch_{epoch_number:03d}.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.global_step,
                        )
                    latest_path = self.checkpoint_dir / "latest.pt"
                    save_checkpoint(
                        latest_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                    )
                    val_loss = val_metrics["loss_total"]
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            self.checkpoint_dir / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.global_step,
                        )
        finally:
            self.console_logger.close()
            if self.wandb_logger is not None:
                self.wandb_logger.finish()
