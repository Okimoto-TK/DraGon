from __future__ import annotations

import torch
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
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        if not (0.0 <= lse_mix <= 1.0):
            raise ValueError("lse_mix must be in [0, 1]")

        self.tau = float(tau)
        self.lse_mix = float(lse_mix)
        self.eps = float(eps)
        self.use_target_normalization = bool(use_target_normalization)

        # default to identity normalization
        for task in self.TASKS:
            self.register_buffer(f"mean_{task}", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer(f"std_{task}", torch.tensor(1.0, dtype=torch.float32))

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

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses = []
        metrics: dict[str, Tensor] = {}

        for task in self.TASKS:
            pred = preds[f"pred_{task}"]
            scale = preds[f"scale_{task}"]
            target = self._get_target(targets, task)

            per_sample = self._laplace_nll(pred, scale, target, task)
            task_loss = per_sample.mean()

            losses.append(task_loss)
            metrics[f"loss_{task}"] = task_loss.detach()

        loss_vec = torch.stack(losses, dim=0)  # [4]
        loss_avg = loss_vec.mean()
        loss_lse = torch.logsumexp(self.tau * loss_vec, dim=0) / self.tau
        loss_total = (1.0 - self.lse_mix) * loss_avg + self.lse_mix * loss_lse

        metrics["loss_avg"] = loss_avg.detach()
        metrics["loss_lse"] = loss_lse.detach()
        metrics["loss_total"] = loss_total.detach()

        return loss_total, metrics