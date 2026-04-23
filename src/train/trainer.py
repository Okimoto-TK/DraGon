"""Training loop orchestration for the forecast network."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config.train import training as training_config
from .checkpoint import save_checkpoint
from .console import EpochConsoleLogger
from .runtime import move_batch_to_device, resolve_amp_dtype
from .tensorboard_logger import TensorBoardLogger


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
        tensorboard_logger: TensorBoardLogger | None = None,
        forward_loss: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
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
        self.float_dtype = (
            self.amp_dtype if self.amp_dtype in {torch.bfloat16, torch.float16} else None
        )
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
        self.tensorboard_logger = tensorboard_logger

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []
        self.checkpoint_dir: Path | None = None
        self.forward_loss = forward_loss or self._default_forward_loss
        self._event_step = 0
        self._phase_prefetched: dict[str, dict[str, torch.Tensor] | None] = {
            "train": None,
            "val": None,
        }
        self._phase_prefetch_future: dict[str, Future | None] = {
            "train": None,
            "val": None,
        }
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="phase-prefetch",
        )
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

    def _target_key(self) -> str:
        return {
            "ret": "target_ret",
            "rv": "target_rv",
            "q": "target_q",
        }[self.task]

    def _prefetch_loader_once(self, loader) -> dict[str, torch.Tensor] | None:
        iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            return None
        return batch

    def _collect_prefetch(self, phase: str, *, wait: bool) -> None:
        future = self._phase_prefetch_future.get(phase)
        if future is None:
            return
        if not wait and not future.done():
            return
        self._phase_prefetch_future[phase] = None
        self._phase_prefetched[phase] = future.result()

    def _start_prefetch_for(self, phase: str) -> None:
        if self._phase_prefetched.get(phase) is not None:
            return
        if self._phase_prefetch_future.get(phase) is not None:
            return
        loader = self.train_loader if phase == "train" else self.val_loader
        if loader is None:
            return
        self._phase_prefetch_future[phase] = self._prefetch_executor.submit(
            self._prefetch_loader_once,
            loader,
        )

    def _extract_log_losses(
        self,
        output: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        loss_metrics = self._extract_all_loss_metrics(output)
        return {
            "loss_total": loss_metrics["loss_total"],
            "loss_task": loss_metrics["loss_task"],
        }

    def _extract_all_loss_metrics(
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

        metrics: dict[str, float] = {
            "loss_total": float(output["loss_total"].detach().float().cpu()),
            "loss_task": float(task_loss.detach().float().cpu()),
        }
        for key, value in output.items():
            if key in metrics or not key.startswith("loss_") or not isinstance(value, torch.Tensor):
                continue
            if value.numel() != 1:
                continue
            metrics[key] = float(value.detach().float().cpu())
        return metrics

    def _prediction_for_logging(
        self,
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred = output["pred_primary"]
        if self.task == "rv":
            # Align visualization semantics with rv target domain (>0):
            # loss uses mean = softplus(raw) + eps.
            return F.softplus(pred.float()) + 1e-6
        return pred

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
        phase_started = False

        if training:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        aggregate: dict[str, torch.Tensor] | None = None
        target_key = self._target_key()

        phase_name = "train" if training else "val"
        next_phase_name = "val" if training else "train"

        self._collect_prefetch(phase_name, wait=True)
        prefetched_batch = self._phase_prefetched.get(phase_name)
        self._phase_prefetched[phase_name] = None
        self._start_prefetch_for(next_phase_name)

        batch_iter = iter(loader)
        if prefetched_batch is not None:
            try:
                next(batch_iter)
            except StopIteration:
                batch_iter = iter(())
            base_iter = batch_iter

            def _iter_prefetched():
                yield prefetched_batch
                for _batch in base_iter:
                    yield _batch

            batch_iter = _iter_prefetched()

        for step, batch in enumerate(batch_iter, start=1):
            self._event_step += 1
            batch = move_batch_to_device(
                batch,
                self.device,
                float_dtype=self.float_dtype,
            )
            if training:
                with self._autocast_context():
                    output = self.forward_loss(batch)
                    loss = output["loss_total"]

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and getattr(
                    self.scheduler, "_step_per_batch", False
                ):
                    self.scheduler.step()
                self.global_step += 1

            else:
                with torch.inference_mode():
                    with self._autocast_context():
                        output = self.forward_loss(batch)

            loss_metric_tensors = {
                key: value.detach().float()
                for key, value in output.items()
                if key.startswith("loss_") and isinstance(value, torch.Tensor) and value.numel() == 1
            }
            if "loss_task" not in loss_metric_tensors:
                legacy_key = f"loss_{self.task}"
                if legacy_key in output and isinstance(output[legacy_key], torch.Tensor):
                    loss_metric_tensors["loss_task"] = output[legacy_key].detach().float()
            if aggregate is None:
                aggregate = {
                    key: torch.zeros((), device=self.device, dtype=torch.float32)
                    for key in loss_metric_tensors
                }
            for key, value in loss_metric_tensors.items():
                if key not in aggregate:
                    aggregate[key] = torch.zeros((), device=self.device, dtype=torch.float32)
                aggregate[key] = aggregate[key] + value
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.update_prediction_state(
                    phase=phase,
                    predictions=self._prediction_for_logging(output),
                    targets=batch[target_key],
                    uncertainties=output.get("sigma_pred"),
                )

            if not phase_started:
                self.console_logger.start_phase(
                    epoch=epoch,
                    phase=phase,
                    total_steps=total_steps,
                )
                phase_started = True
            self.console_logger.advance(step)

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
        aggregate = aggregate or {
            "loss_total": torch.zeros((), device=self.device, dtype=torch.float32),
            "loss_task": torch.zeros((), device=self.device, dtype=torch.float32),
        }
        metrics = {key: float((value / num_steps).detach().cpu()) for key, value in aggregate.items()}

        if phase_started and self._event_step % self.log_every != 0:
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.console_logger.log_metrics(
                epoch=epoch,
                phase=phase,
                step=total_steps,
                total_steps=total_steps,
                losses=metrics,
                lr=lr,
            )

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_epoch_metrics(
                phase=phase,
                global_step=self.global_step,
                epoch=epoch,
                metrics=metrics,
            )
            self.tensorboard_logger.log_epoch_prediction_plot(
                phase=phase,
                global_step=self.global_step,
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
        no_improve_epochs = 0
        early_stop_patience = int(training_config.early_stop_patience)
        try:
            for epoch in range(self.start_epoch, num_epochs):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.validate_one_epoch(epoch)

                if self.scheduler is not None and not getattr(
                    self.scheduler, "_step_per_batch", False
                ):
                    val_loss_for_scheduler = float(val_metrics["loss_total"])
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss_for_scheduler)
                    else:
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
                        no_improve_epochs = 0
                        save_checkpoint(
                            self.checkpoint_dir / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.global_step,
                        )
                    else:
                        no_improve_epochs += 1
                else:
                    val_loss = val_metrics["loss_total"]
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                if no_improve_epochs >= early_stop_patience:
                    break
        finally:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.close()
            self.console_logger.close()
