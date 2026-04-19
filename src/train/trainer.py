"""Training loop orchestration for the forecast network."""

from __future__ import annotations

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
        log_every: int = 50,
        save_every: int = 1,
        console_logger=None,
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
        self.log_every = int(log_every)
        if save_every <= 0:
            raise ValueError(f"save_every must be > 0, got {save_every}.")
        self.save_every = int(save_every)
        self.console_logger = console_logger or EpochConsoleLogger(
            log_every=self.log_every
        )

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []
        self.checkpoint_dir: Path | None = None
        self._event_step = 0
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.device.type == "cuda"
            and self.use_amp
            and self.amp_dtype == torch.float16
            else None
        )

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
        return {
            "loss_total": float(output["loss_total"].detach().float().cpu()),
            "loss_ret": float(output["loss_ret"].detach().float().cpu()),
            "loss_rv": float(output["loss_rv"].detach().float().cpu()),
            "loss_q": float(output["loss_q"].detach().float().cpu()),
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
                "loss_ret": float("nan"),
                "loss_rv": float("nan"),
                "loss_q": float("nan"),
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
            "loss_ret": torch.zeros((), device=self.device, dtype=torch.float32),
            "loss_rv": torch.zeros((), device=self.device, dtype=torch.float32),
            "loss_q": torch.zeros((), device=self.device, dtype=torch.float32),
        }

        for step, batch in enumerate(loader, start=1):
            self.console_logger.advance(step)
            self._event_step += 1
            batch = move_batch_to_device(batch, self.device)

            if training:
                with self._autocast_context():
                    output = self.model.forward_loss(batch, return_aux=False)
                    loss = output["loss_total"]

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and getattr(
                    self.scheduler, "_step_per_batch", False
                ):
                    self.scheduler.step()
                self.global_step += 1
            else:
                with torch.no_grad():
                    with self._autocast_context():
                        output = self.model.forward_loss(batch, return_aux=False)

            for key in aggregate:
                aggregate[key] = aggregate[key] + output[key].detach().float()

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
