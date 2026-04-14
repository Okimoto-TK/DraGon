from __future__ import annotations

import torch
from config.config import freeze_ema_beta as DEFAULT_FREEZE_EMA_BETA
from config.config import freeze_min_steps as DEFAULT_FREEZE_MIN_STEPS
from config.config import freeze_patience_steps as DEFAULT_FREEZE_PATIENCE_STEPS
from config.config import freeze_scale_s0_M as DEFAULT_S0_M
from config.config import freeze_scale_s0_MDD as DEFAULT_S0_MDD
from config.config import freeze_scale_s0_RV as DEFAULT_S0_RV
from config.config import freeze_scale_s0_S as DEFAULT_S0_S
from torch import Tensor, nn


class LaplaceLSELoss(nn.Module):
    """Physics-consistent heteroscedastic Laplace loss + smooth LSE aggregation.

    Expected prediction keys:
        pred_S, pred_M, pred_MDD, pred_RV,
        scale_S, scale_M, scale_MDD, scale_RV

    Expected target keys:
        label_S, label_M, label_MDD, label_RV
        (or S, M, MDD, RV)
    """

    TASKS = ("S", "M", "MDD", "RV")

    def __init__(
        self,
        tau: float = 2.0,
        lse_mix: float = 0.25,
        eps: float = 1e-6,
        use_target_normalization: bool = False,
        *,
        s0_S: float = DEFAULT_S0_S,
        s0_M: float = DEFAULT_S0_M,
        s0_MDD: float = DEFAULT_S0_MDD,
        s0_RV: float = DEFAULT_S0_RV,
        min_freeze_steps: int = DEFAULT_FREEZE_MIN_STEPS,
        patience_steps: int = DEFAULT_FREEZE_PATIENCE_STEPS,
        ema_beta: float = DEFAULT_FREEZE_EMA_BETA,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        if not (0.0 <= lse_mix <= 1.0):
            raise ValueError("lse_mix must be in [0, 1]")
        if min_freeze_steps < 0:
            raise ValueError("min_freeze_steps must be non-negative")
        if patience_steps <= 0:
            raise ValueError("patience_steps must be positive")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0, 1)")

        self.tau = float(tau)
        self.lse_mix = float(lse_mix)
        self.eps = float(eps)
        self.use_target_normalization = bool(use_target_normalization)
        self.min_freeze_steps = int(min_freeze_steps)
        self.patience_steps = int(patience_steps)
        self.ema_beta = float(ema_beta)

        # default to identity normalization
        for task in self.TASKS:
            self.register_buffer(f"mean_{task}", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer(f"std_{task}", torch.tensor(1.0, dtype=torch.float32))
            s0_value = float(locals()[f"s0_{task}"])
            self.register_buffer(f"s0_{task}", torch.tensor(s0_value, dtype=torch.float32))
            self.register_buffer(f"ema_mae_{task}", torch.tensor(s0_value, dtype=torch.float32))

        self.register_buffer("train_step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("freeze_good_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("scale_unfrozen", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("min_freeze_steps_tensor", torch.tensor(float(self.min_freeze_steps), dtype=torch.float32))
        self.register_buffer("patience_steps_tensor", torch.tensor(float(self.patience_steps), dtype=torch.float32))

    @torch.no_grad()
    def set_target_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        """stats example:
        {
            "S": (mean_S, std_S),
            "M": (mean_M, std_M),
            "MDD": (mean_MDD, std_MDD),
            "RV": (mean_RV, std_RV),
        }
        """
        for task in self.TASKS:
            if task not in stats:
                raise KeyError(f"Missing stats for task {task}")
            mean, std = stats[task]
            std = max(float(std), self.eps)
            getattr(self, f"mean_{task}").copy_(torch.tensor(mean, dtype=torch.float32))
            getattr(self, f"std_{task}").copy_(torch.tensor(std, dtype=torch.float32))

    def _get_target(self, targets: dict[str, Tensor], task: str) -> Tensor:
        if f"label_{task}" in targets:
            y = targets[f"label_{task}"]
        elif task in targets:
            y = targets[task]
        else:
            raise KeyError(f"Missing target for task {task}")

        if y.ndim == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        return y

    def _laplace_nll(self, pred: Tensor, scale: Tensor, target: Tensor, task: str) -> Tensor:
        if pred.ndim == 2 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        if scale.ndim == 2 and scale.size(-1) == 1:
            scale = scale.squeeze(-1)
        if target.ndim == 2 and target.size(-1) == 1:
            target = target.squeeze(-1)

        if self.use_target_normalization:
            mean = getattr(self, f"mean_{task}").to(device=pred.device, dtype=pred.dtype)
            std = getattr(self, f"std_{task}").to(device=pred.device, dtype=pred.dtype).clamp_min(self.eps)

            pred = (pred - mean) / std
            target = (target - mean) / std
            scale = scale / std

        scale = scale.clamp_min(self.eps)
        return torch.abs(target - pred) / scale + torch.log(scale)

    def _fixed_scale(self, pred: Tensor, task: str) -> Tensor:
        s0 = getattr(self, f"s0_{task}").to(device=pred.device, dtype=pred.dtype)
        return torch.ones_like(pred) * s0

    @torch.no_grad()
    def _update_freeze_state(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> None:
        active_mask = (~self.scale_unfrozen).to(dtype=torch.bool)
        all_good = torch.ones((), device=self.scale_unfrozen.device, dtype=torch.bool)
        for task in self.TASKS:
            pred = preds[f"pred_{task}"].detach()
            target = self._get_target(targets, task).detach().to(device=pred.device, dtype=pred.dtype)
            if pred.ndim == 2 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if target.ndim == 2 and target.size(-1) == 1:
                target = target.squeeze(-1)

            ema_buffer = getattr(self, f"ema_mae_{task}")
            mae = torch.mean(torch.abs(pred - target)).float().to(device=ema_buffer.device)
            ema_buffer.mul_(self.ema_beta).add_(mae * (1.0 - self.ema_beta))

            threshold = (0.9 * getattr(self, f"s0_{task}")).to(device=ema_buffer.device)
            all_good = all_good & (ema_buffer <= threshold)

        self.train_step_count.add_(active_mask.to(dtype=self.train_step_count.dtype))
        meets_min_steps = self.train_step_count >= self.min_freeze_steps
        eligible = active_mask & meets_min_steps & all_good

        next_good_steps = torch.where(
            eligible,
            self.freeze_good_steps + 1,
            torch.zeros_like(self.freeze_good_steps),
        )
        self.freeze_good_steps.copy_(
            torch.where(
                active_mask,
                next_good_steps,
                self.freeze_good_steps,
            )
        )

        should_unfreeze = self.scale_unfrozen | (self.freeze_good_steps >= self.patience_steps)
        self.scale_unfrozen.copy_(should_unfreeze)

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        *,
        return_metrics: bool = True,
        update_state: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses = []
        scale_unfrozen = self.scale_unfrozen.to(dtype=preds["pred_S"].dtype, device=preds["pred_S"].device)
        scale_frozen = 1.0 - scale_unfrozen
        metrics: dict[str, Tensor] = {}

        for task in self.TASKS:
            pred = preds[f"pred_{task}"]
            learned_scale = preds[f"scale_{task}"]
            fixed_scale = self._fixed_scale(pred, task)
            scale = learned_scale * scale_unfrozen + fixed_scale * scale_frozen
            target = self._get_target(targets, task)

            per_sample = self._laplace_nll(pred, scale, target, task)
            task_loss = per_sample.mean()

            losses.append(task_loss)
            if return_metrics:
                metrics[f"loss_{task}"] = task_loss.detach()

        loss_vec = torch.stack(losses, dim=0)  # [4]
        loss_avg = loss_vec.mean()
        loss_lse = torch.logsumexp(self.tau * loss_vec, dim=0) / self.tau
        loss_total = (1.0 - self.lse_mix) * loss_avg + self.lse_mix * loss_lse

        if update_state and self.training:
            self._update_freeze_state(preds, targets)

        if return_metrics:
            metrics["loss_avg"] = loss_avg.detach()
            metrics["loss_lse"] = loss_lse.detach()
            metrics["loss_total"] = loss_total.detach()
            metrics["freeze/fixed_scale_active"] = (1.0 - self.scale_unfrozen.to(dtype=loss_total.dtype)).to(
                device=loss_total.device,
            )
            metrics["freeze/scale_unfrozen"] = self.scale_unfrozen.to(dtype=loss_total.dtype, device=loss_total.device)
            metrics["freeze/train_step_count"] = self.train_step_count.detach().to(
                device=loss_total.device,
                dtype=torch.float32,
            )
            metrics["freeze/good_steps"] = self.freeze_good_steps.detach().to(device=loss_total.device, dtype=torch.float32)
            metrics["freeze/min_freeze_steps"] = self.min_freeze_steps_tensor.detach().to(
                device=loss_total.device,
                dtype=torch.float32,
            )
            metrics["freeze/patience_steps"] = self.patience_steps_tensor.detach().to(
                device=loss_total.device,
                dtype=torch.float32,
            )
            for task in self.TASKS:
                metrics[f"freeze/ema_mae/{task}"] = getattr(self, f"ema_mae_{task}").detach().to(
                    device=loss_total.device,
                    dtype=torch.float32,
                )
                metrics[f"freeze/s0/{task}"] = getattr(self, f"s0_{task}").detach().to(
                    device=loss_total.device,
                    dtype=torch.float32,
                )

        return loss_total, metrics
