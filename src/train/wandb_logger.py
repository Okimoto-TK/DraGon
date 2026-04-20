"""W&B logging utilities for structured model visualization."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WandbLoggerConfig:
    """Runtime knobs for W&B logging frequency and connectivity."""

    enabled: bool = False
    log_every: int = 50
    hist_every: int = 500
    viz_every: int = 1000
    project: str | None = None
    group: str | None = None
    base_url: str | None = None


def _detach_float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float()


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def _std(value: torch.Tensor) -> float:
    if value.numel() <= 1:
        return 0.0
    return float(value.detach().float().std(unbiased=False).cpu())


def _section_name(*parts: str) -> str:
    return "_".join(part for part in parts if part)


def _metric_key(*section_parts: str, name: str) -> str:
    return f"{_section_name(*section_parts)}/{name}"


def _loss_shares(
    loss_ret: torch.Tensor | float,
    loss_rv: torch.Tensor | float,
    loss_q: torch.Tensor | float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    ret = abs(_scalar(loss_ret))
    rv = abs(_scalar(loss_rv))
    q = abs(_scalar(loss_q))
    total = ret + rv + q
    if total <= eps:
        return 0.0, 0.0, 0.0
    return ret / total, rv / total, q / total


def _histogram_bins(
    value: torch.Tensor,
    *,
    bins: int = 64,
) -> tuple[np.ndarray, np.ndarray] | None:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return None
    lo = flat.amin()
    hi = flat.amax()
    if torch.equal(lo, hi):
        lo = lo - 0.5
        hi = hi + 0.5
    counts = torch.histc(flat, bins=bins, min=float(lo), max=float(hi))
    edges = torch.linspace(lo, hi, steps=bins + 1, device=flat.device)
    return counts.cpu().numpy(), edges.cpu().numpy()


def _binary_entropy(value: torch.Tensor, eps: float = 1e-6) -> float:
    p = value.detach().float().clamp(min=eps, max=1.0 - eps)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return float(entropy.mean().cpu())


def _distribution_entropy(weights: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> float:
    w = weights.detach().float().clamp_min(eps)
    entropy = -(w * w.log()).sum(dim=dim)
    return float(entropy.mean().cpu())


def _gini(weights: torch.Tensor, eps: float = 1e-6) -> float:
    flat = weights.detach().float().reshape(-1).clamp_min(eps)
    flat = flat / flat.sum()
    sorted_vals, _ = torch.sort(flat)
    n = sorted_vals.numel()
    if n == 0:
        return float("nan")
    index = torch.arange(1, n + 1, device=sorted_vals.device, dtype=sorted_vals.dtype)
    gini = (2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n
    return float(gini.cpu())


def _offdiag_mean_cosine_distance(z_scale: torch.Tensor, eps: float = 1e-6) -> float:
    # z_scale: [B, F, D, N]
    tokens = z_scale.detach().float().permute(0, 3, 1, 2).reshape(-1, z_scale.shape[1], z_scale.shape[2])
    if tokens.numel() == 0 or tokens.shape[1] <= 1:
        return float("nan")
    tokens = F.normalize(tokens, dim=-1, eps=eps)
    sim = torch.matmul(tokens, tokens.transpose(1, 2))
    mask = ~torch.eye(tokens.shape[1], device=sim.device, dtype=torch.bool)
    dist = 1.0 - sim
    return float(dist[:, mask].mean().cpu())


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    return float(F.cosine_similarity(a.detach().float(), b.detach().float(), dim=-1, eps=eps).mean().cpu())


class WandbVisualizationLogger:
    """Structured W&B logger with scalar, histogram, and low-frequency visual logs."""

    def __init__(
        self,
        *,
        config: WandbLoggerConfig,
        run_name: str,
        run_config: Mapping[str, object],
    ) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._wandb = None
        self._fixed_val_batches: dict[str, dict[str, torch.Tensor]] = {}
        if not self.enabled:
            return

        project = config.project or os.getenv("WANDB_PROJECT")
        group = config.group or os.getenv("WANDB_RUN_GROUP")
        base_url = config.base_url or os.getenv("WANDB_BASE_URL")
        if project is None:
            raise ValueError(
                "W&B logging is enabled but no project was provided. "
                "Set WANDB_PROJECT or pass wandb_project."
            )
        if base_url:
            os.environ["WANDB_BASE_URL"] = base_url

        try:
            import wandb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-path only
            raise ImportError(
                "W&B logging requires the 'wandb' package. Install it with `pip install wandb`."
            ) from exc

        self._wandb = wandb
        self._wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=dict(run_config),
        )
        self._wandb.define_metric("trainer/global_step")
        self._wandb.define_metric("*", step_metric="trainer/global_step")

    @property
    def hist_every(self) -> int:
        return max(int(self.config.hist_every), int(self.config.log_every))

    @property
    def viz_every(self) -> int:
        return max(int(self.config.viz_every), int(self.hist_every))

    @staticmethod
    def _ret_bucket_name(ret_pct: float) -> str | None:
        if ret_pct <= -10.0:
            return "ret_le_neg10"
        if -10.0 < ret_pct <= -5.0:
            return "ret_neg10_to_neg5"
        if 5.0 <= ret_pct < 10.0:
            return "ret_5_to_10"
        if ret_pct >= 10.0:
            return "ret_ge_10"
        return None

    @staticmethod
    def _ret_bucket_title(bucket_name: str) -> str:
        titles = {
            "ret_le_neg10": "RET <= -10%",
            "ret_neg10_to_neg5": "-10% < RET <= -5%",
            "ret_5_to_10": "5% <= RET < 10%",
            "ret_ge_10": "RET >= 10%",
        }
        return titles.get(bucket_name, bucket_name)

    @staticmethod
    def _unwrap_loader(loader):
        base = loader
        while hasattr(base, "loader"):
            base = base.loader
        return base

    @staticmethod
    def _clone_batch_to_cpu(batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        cloned: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.detach().cpu().clone()
        return cloned

    def capture_fixed_val_batch(self, loader) -> None:
        self._fixed_val_batches = {}
        if not self.enabled:
            return

        base_loader = self._unwrap_loader(loader)
        dataset = getattr(base_loader, "dataset", None)
        collate_fn = getattr(base_loader, "collate_fn", None)
        batch_size = getattr(base_loader, "batch_size", None)
        if batch_size is None:
            batch_sampler = getattr(base_loader, "batch_sampler", None)
            batch_size = getattr(batch_sampler, "batch_size", None)
        batch_size = int(batch_size or 1)
        if dataset is None or collate_fn is None:
            return

        bucket_samples: dict[str, list[dict[str, np.ndarray]]] = {
            "ret_le_neg10": [],
            "ret_neg10_to_neg5": [],
            "ret_5_to_10": [],
            "ret_ge_10": [],
        }

        for index in range(len(dataset)):
            sample = dataset[index]
            ret_pct = float(np.asarray(sample["target_ret"]).reshape(-1)[0] * 100.0)
            bucket_name = self._ret_bucket_name(ret_pct)
            if bucket_name is None:
                continue
            if len(bucket_samples[bucket_name]) >= batch_size:
                continue
            bucket_samples[bucket_name].append(sample)
            if all(len(samples) >= batch_size for samples in bucket_samples.values()):
                break

        for bucket_name, samples in bucket_samples.items():
            if not samples:
                continue
            collated = collate_fn(samples)
            self._fixed_val_batches[bucket_name] = self._clone_batch_to_cpu(collated)

    def get_fixed_val_batch(self) -> dict[str, torch.Tensor] | None:
        return None

    def get_fixed_val_batches(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            bucket_name: self._clone_batch_to_cpu(batch)
            for bucket_name, batch in self._fixed_val_batches.items()
        }

    def should_log_histograms(self, global_step: int) -> bool:
        return self.enabled and global_step > 0 and global_step % self.hist_every == 0

    def should_log_visuals(self, global_step: int) -> bool:
        return self.enabled and global_step > 0 and global_step % self.viz_every == 0

    def _log(self, payload: Mapping[str, object], *, global_step: int) -> None:
        if not self.enabled or self._wandb is None or not payload:
            return
        data = {"trainer/global_step": global_step, **payload}
        self._wandb.log(data, step=global_step)

    def _figure_to_image(self, fig) -> np.ndarray:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[..., :3].copy()

    def _build_heatmap_image(
        self,
        matrix: np.ndarray,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        cmap: str = "viridis",
        xticks: list[float] | None = None,
        xticklabels: list[str] | None = None,
        yticks: list[float] | None = None,
        yticklabels: list[str] | None = None,
    ) -> np.ndarray:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticks is not None:
            ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
        fig.colorbar(im, ax=ax, shrink=0.9)
        image = self._figure_to_image(fig)
        plt.close(fig)
        return image

    def _build_density_scatter_image(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        *,
        title: str,
        xlabel: str = "target",
        ylabel: str = "pred",
        bins: int = 64,
    ) -> np.ndarray:
        import matplotlib.pyplot as plt

        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        mask = np.isfinite(pred) & np.isfinite(target)
        pred = pred[mask]
        target = target[mask]
        if pred.size == 0:
            pred = np.array([0.0], dtype=np.float32)
            target = np.array([0.0], dtype=np.float32)

        lo = float(min(pred.min(), target.min()))
        hi = float(max(pred.max(), target.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo -= 0.5
            hi += 0.5

        hist, xedges, yedges = np.histogram2d(target, pred, bins=bins, range=[[lo, hi], [lo, hi]])
        fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        im = ax.imshow(
            np.log1p(hist.T),
            origin="lower",
            aspect="equal",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="magma",
            interpolation="nearest",
        )
        ax.plot([lo, hi], [lo, hi], color="cyan", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, shrink=0.9)
        image = self._figure_to_image(fig)
        plt.close(fig)
        return image

    def _add_stats(
        self,
        payload: dict[str, object],
        *,
        prefix: str,
        value: torch.Tensor,
        include_minmax: bool = False,
    ) -> None:
        payload[f"{prefix}/mean"] = _scalar(value)
        payload[f"{prefix}/std"] = _std(value)
        if include_minmax:
            payload[f"{prefix}/min"] = float(value.detach().float().amin().cpu())
            payload[f"{prefix}/max"] = float(value.detach().float().amax().cpu())

    def _log_common_scalars(
        self,
        *,
        split: str,
        payload: dict[str, object],
        batch: dict[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
        grad_norm: float | None,
        param_norm: float,
        lr: float,
        step_time_ms: float,
        data_time_ms: float,
        forward_time_ms: float,
        backward_time_ms: float,
        optimizer_time_ms: float,
        samples_per_sec: float,
        gpu_mem_alloc_mb: float,
    ) -> None:
        payload[_metric_key(split, "loss", name="total")] = _scalar(output["loss_total"])
        payload[_metric_key(split, "loss", name="ret_nll")] = _scalar(output["loss_ret"])
        payload[_metric_key(split, "loss", name="rv_nll")] = _scalar(output["loss_rv"])
        payload[_metric_key(split, "loss", name="q_nll")] = _scalar(output["loss_q"])
        share_ret, share_rv, share_q = _loss_shares(
            output["loss_ret"],
            output["loss_rv"],
            output["loss_q"],
        )
        payload[_metric_key(split, "loss_share", name="ret")] = share_ret
        payload[_metric_key(split, "loss_share", name="rv")] = share_rv
        payload[_metric_key(split, "loss_share", name="q")] = share_q
        payload[_metric_key(split, "optimizer", name="lr")] = lr
        payload[_metric_key(split, "health", name="global_param_norm")] = param_norm
        if grad_norm is not None:
            payload[_metric_key(split, "health", name="global_grad_norm")] = grad_norm
        if "fused_latents" in output:
            fused_latents = output["fused_latents"]
            payload[_metric_key(split, "health", name="global_act_mean")] = _scalar(fused_latents)
            payload[_metric_key(split, "health", name="global_act_std")] = _std(fused_latents)
        payload[_metric_key(split, "system", name="step_time_ms")] = step_time_ms
        payload[_metric_key(split, "system", name="data_time_ms")] = data_time_ms
        payload[_metric_key(split, "system", name="forward_time_ms")] = forward_time_ms
        payload[_metric_key(split, "system", name="backward_time_ms")] = backward_time_ms
        payload[_metric_key(split, "system", name="optimizer_time_ms")] = optimizer_time_ms
        payload[_metric_key(split, "system", name="samples_per_sec")] = samples_per_sec
        payload[_metric_key(split, "system", name="gpu_mem_alloc_mb")] = gpu_mem_alloc_mb

        for scale, key in (
            ("macro", "macro_float_long"),
            ("mezzo", "mezzo_float_long"),
            ("micro", "micro_float_long"),
        ):
            self._add_stats(
                payload,
                prefix=_section_name(split, "input", scale),
                value=batch[key],
                include_minmax=True,
            )
        self._add_stats(
            payload,
            prefix=_section_name(split, "input", "sidechain"),
            value=batch["sidechain_cond"],
            include_minmax=True,
        )

        self._add_stats(payload, prefix=_section_name(split, "loss_params", "ret_sigma"), value=output["sigma_ret_pred"])
        self._add_stats(payload, prefix=_section_name(split, "loss_params", "rv_sigma"), value=output["sigma_rv_pred"])
        self._add_stats(payload, prefix=_section_name(split, "loss_params", "q_sigma"), value=output["sigma_q_pred"])
        payload[_metric_key(split, "loss_params", name="ret_nu")] = _scalar(output["nu_ret"])

        for name in (
            "pred_mu_ret",
            "pred_scale_ret_raw",
            "pred_mean_rv_raw",
            "pred_shape_rv_raw",
            "pred_mu_q",
            "pred_scale_q_raw",
        ):
            if name in output:
                self._add_stats(
                    payload,
                    prefix=_section_name(split, "heads", name),
                    value=output[name],
                )

    def _log_debug_scalars(
        self,
        *,
        split: str,
        payload: dict[str, object],
        output: Mapping[str, torch.Tensor],
    ) -> None:
        debug = output.get("_debug")
        if not isinstance(debug, Mapping):
            return

        wavelet = debug.get("wavelet")
        if isinstance(wavelet, Mapping):
            for scale in ("macro", "mezzo", "micro"):
                raw_tail = wavelet.get(f"{scale}_raw_tail")
                denoised = wavelet.get(f"{scale}_denoised")
                if isinstance(raw_tail, torch.Tensor) and isinstance(denoised, torch.Tensor):
                    raw_energy = raw_tail.detach().float().square().mean()
                    denoised_energy = denoised.detach().float().square().mean()
                    payload[_metric_key(split, "wavelet", scale, name="energy_raw")] = float(raw_energy.cpu())
                    payload[_metric_key(split, "wavelet", scale, name="energy_denoised")] = float(denoised_energy.cpu())
                    payload[_metric_key(split, "wavelet", scale, name="energy_ratio_denoised_over_raw")] = float(
                        (denoised_energy / raw_energy.clamp_min(1e-12)).cpu()
                    )

        encoder = debug.get("encoder")
        if isinstance(encoder, Mapping):
            for scale in ("macro", "mezzo", "micro"):
                z = encoder.get(scale)
                if isinstance(z, torch.Tensor):
                    payload[_metric_key(split, "encoder", scale, name="final_block_act_mean")] = _scalar(z)
                    payload[_metric_key(split, "encoder", scale, name="final_block_act_std")] = _std(z)
                    payload[_metric_key(split, "encoder", scale, name="final_block_act_abs_mean")] = _scalar(z.abs())

        within_scale = debug.get("within_scale")
        if isinstance(within_scale, Mapping):
            for scale in ("macro", "mezzo", "micro"):
                z_pre = within_scale.get(f"{scale}_pre")
                z_post = within_scale.get(f"{scale}_post")
                if isinstance(z_pre, torch.Tensor) and isinstance(z_post, torch.Tensor):
                    pre_div = _offdiag_mean_cosine_distance(z_pre)
                    post_div = _offdiag_mean_cosine_distance(z_post)
                    payload[_metric_key(split, "within_scale", scale, name="feature_cosdist_pre")] = pre_div
                    payload[_metric_key(split, "within_scale", scale, name="feature_cosdist_post")] = post_div
                    payload[_metric_key(split, "within_scale", scale, name="feature_cosdist_ratio")] = post_div / max(pre_div, 1e-12)

        conditioning = debug.get("conditioning")
        if isinstance(conditioning, Mapping):
            cond_seq = conditioning.get("cond_seq")
            cond_global = conditioning.get("cond_global")
            if isinstance(cond_seq, torch.Tensor):
                payload[_metric_key(split, "conditioning", name="cond_seq_mean")] = _scalar(cond_seq)
                payload[_metric_key(split, "conditioning", name="cond_seq_std")] = _std(cond_seq)
                payload[_metric_key(split, "conditioning", name="cond_seq_abs_mean")] = _scalar(cond_seq.abs())
            if isinstance(cond_global, torch.Tensor):
                cond_global_norm = cond_global.detach().float().norm(dim=-1)
                payload[_metric_key(split, "conditioning", name="cond_global_l2_mean")] = _scalar(cond_global_norm)
                payload[_metric_key(split, "conditioning", name="cond_global_l2_std")] = _std(cond_global_norm)

        side_memory = debug.get("side_memory")
        if isinstance(side_memory, Mapping):
            g1 = side_memory.get("g1")
            g2 = side_memory.get("g2")
            g3 = side_memory.get("g3")
            for name in ("s1", "s2", "s3", "g1", "g2", "g3"):
                value = side_memory.get(name)
                if isinstance(value, torch.Tensor):
                    payload[_metric_key(split, "side_memory", name=f"{name}_l2_mean")] = _scalar(
                        value.detach().float().norm(dim=1 if value.ndim == 3 else -1)
                    )
            if all(isinstance(value, torch.Tensor) for value in (g1, g2, g3)):
                payload[_metric_key(split, "side_memory", name="cos_g1_g2")] = _cosine(g1, g2)
                payload[_metric_key(split, "side_memory", name="cos_g2_g3")] = _cosine(g2, g3)
                payload[_metric_key(split, "side_memory", name="cos_g1_g3")] = _cosine(g1, g3)

        bridge = debug.get("bridge")
        if isinstance(bridge, Mapping):
            for scale in ("macro", "mezzo", "micro"):
                bridge_debug = bridge.get(scale)
                if not isinstance(bridge_debug, Mapping):
                    continue
                gate = bridge_debug.get("gate")
                bridge_token = bridge_debug.get("bridge_token")
                fusion_delta = bridge_debug.get("fusion_delta")
                if isinstance(bridge_token, torch.Tensor):
                    payload[_metric_key(split, "bridge", scale, name="bridge_token_l2_mean")] = _scalar(
                        bridge_token.detach().float().norm(dim=-1)
                    )
                if isinstance(gate, torch.Tensor):
                    payload[_metric_key(split, "bridge", scale, name="gate_mean")] = _scalar(gate)
                    payload[_metric_key(split, "bridge", scale, name="gate_std")] = _std(gate)
                    payload[_metric_key(split, "bridge", scale, name="gate_min")] = float(gate.detach().float().amin().cpu())
                    payload[_metric_key(split, "bridge", scale, name="gate_max")] = float(gate.detach().float().amax().cpu())
                    payload[_metric_key(split, "bridge", scale, name="gate_entropy")] = _binary_entropy(gate)
                if isinstance(fusion_delta, torch.Tensor):
                    payload[_metric_key(split, "bridge", scale, name="delta_l2")] = _scalar(
                        fusion_delta.detach().float().norm(dim=-1)
                    )

        cross_scale = debug.get("cross_scale")
        if isinstance(cross_scale, Mapping):
            cross_attn = cross_scale.get("cross_attn_weights")
            latent_norms = cross_scale.get("latent_norms")
            if isinstance(latent_norms, torch.Tensor):
                for idx in range(latent_norms.shape[1]):
                    payload[_metric_key(split, "cross_scale", name=f"latent_{idx:02d}_norm")] = _scalar(
                        latent_norms[:, idx]
                    )
            if isinstance(cross_attn, torch.Tensor):
                attn_mean = cross_attn.detach().float().mean(dim=(0, 1, 2))
                payload[_metric_key(split, "cross_scale", name="latent_attn_to_macro")] = float(attn_mean[:16].sum().cpu())
                payload[_metric_key(split, "cross_scale", name="latent_attn_to_mezzo")] = float(attn_mean[16:40].sum().cpu())
                payload[_metric_key(split, "cross_scale", name="latent_attn_to_micro")] = float(attn_mean[40:].sum().cpu())
                usage = cross_attn.detach().float().mean(dim=(0, 1, 3))
                payload[_metric_key(split, "cross_scale", name="latent_usage_entropy")] = _distribution_entropy(
                    usage / usage.sum().clamp_min(1e-12)
                )
                payload[_metric_key(split, "cross_scale", name="latent_usage_gini")] = _gini(usage)

        heads = debug.get("heads")
        if isinstance(heads, Mapping):
            reprs: dict[str, torch.Tensor] = {}
            for task in ("ret", "rv", "q"):
                task_repr = heads.get(f"{task}_task_repr")
                if isinstance(task_repr, torch.Tensor):
                    reprs[task] = task_repr
                    payload[_metric_key(split, "heads", name=f"task_repr_{task}_l2")] = _scalar(
                        task_repr.detach().float().norm(dim=-1)
                    )
            if len(reprs) == 3:
                payload[_metric_key(split, "heads", name="cos_task_repr_ret_rv")] = _cosine(reprs["ret"], reprs["rv"])
                payload[_metric_key(split, "heads", name="cos_task_repr_ret_q")] = _cosine(reprs["ret"], reprs["q"])
                payload[_metric_key(split, "heads", name="cos_task_repr_rv_q")] = _cosine(reprs["rv"], reprs["q"])

    def log_train_step(
        self,
        *,
        global_step: int,
        epoch: int,
        batch: dict[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
        model: torch.nn.Module,
        grad_norm: float,
        param_norm: float,
        lr: float,
        step_time_ms: float,
        data_time_ms: float,
        forward_time_ms: float,
        backward_time_ms: float,
        optimizer_time_ms: float,
        samples_per_sec: float,
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, object] = {"trainer/epoch": epoch}
        gpu_mem_alloc_mb = (
            torch.cuda.memory_allocated(device=batch["macro_float_long"].device) / (1024.0 * 1024.0)
            if batch["macro_float_long"].device.type == "cuda"
            else 0.0
        )
        self._log_common_scalars(
            split="train",
            payload=payload,
            batch=batch,
            output=output,
            grad_norm=grad_norm,
            param_norm=param_norm,
            lr=lr,
            step_time_ms=step_time_ms,
            data_time_ms=data_time_ms,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            optimizer_time_ms=optimizer_time_ms,
            samples_per_sec=samples_per_sec,
            gpu_mem_alloc_mb=gpu_mem_alloc_mb,
        )
        self._log_debug_scalars(split="train", payload=payload, output=output)

        if self.should_log_histograms(global_step):
            for name in (
                "pred_mu_ret",
                "pred_scale_ret_raw",
                "pred_mean_rv_raw",
                "pred_shape_rv_raw",
                "pred_mu_q",
                "pred_scale_q_raw",
                "sigma_ret_pred",
                "sigma_rv_pred",
                "sigma_q_pred",
            ):
                if name in output:
                    hist = _histogram_bins(output[name])
                    if hist is not None:
                        payload[_metric_key("train", "heads", name, name="hist")] = self._wandb.Histogram(
                            np_histogram=hist
                        )

        self._log(payload, global_step=global_step)

    def log_val_epoch(
        self,
        *,
        global_step: int,
        epoch: int,
        metrics: Mapping[str, float],
        lr: float,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "trainer/epoch": epoch,
            _metric_key("val", "loss", name="total"): float(metrics["loss_total"]),
            _metric_key("val", "loss", name="ret_nll"): float(metrics["loss_ret"]),
            _metric_key("val", "loss", name="rv_nll"): float(metrics["loss_rv"]),
            _metric_key("val", "loss", name="q_nll"): float(metrics["loss_q"]),
            _metric_key("val", "optimizer", name="lr"): lr,
        }
        share_ret, share_rv, share_q = _loss_shares(
            float(metrics["loss_ret"]),
            float(metrics["loss_rv"]),
            float(metrics["loss_q"]),
        )
        payload[_metric_key("val", "loss_share", name="ret")] = share_ret
        payload[_metric_key("val", "loss_share", name="rv")] = share_rv
        payload[_metric_key("val", "loss_share", name="q")] = share_q
        self._log(payload, global_step=global_step)

    def log_fixed_val_snapshot(
        self,
        *,
        global_step: int,
        epoch: int,
        batch: dict[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
        bucket_name: str = "fixed_val",
    ) -> None:
        if not self.enabled or self._wandb is None:
            return

        payload: dict[str, object] = {"trainer/epoch": epoch}
        bucket_title = self._ret_bucket_title(bucket_name)

        def _to_np(value: torch.Tensor) -> np.ndarray:
            return value.detach().float().cpu().numpy()

        sample_idx = 0
        payload[_metric_key("viz", bucket_name, "input", "macro", name="heatmap")] = self._wandb.Image(
            self._build_heatmap_image(
                _to_np(batch["macro_float_long"][sample_idx]),
                title=f"{bucket_title} | Macro Input",
                xlabel="time",
                ylabel="feature",
            )
        )
        payload[_metric_key("viz", bucket_name, "input", "mezzo", name="heatmap")] = self._wandb.Image(
            self._build_heatmap_image(
                _to_np(batch["mezzo_float_long"][sample_idx]),
                title=f"{bucket_title} | Mezzo Input",
                xlabel="time",
                ylabel="feature",
            )
        )
        payload[_metric_key("viz", bucket_name, "input", "micro", name="heatmap")] = self._wandb.Image(
            self._build_heatmap_image(
                _to_np(batch["micro_float_long"][sample_idx]),
                title=f"{bucket_title} | Micro Input",
                xlabel="time",
                ylabel="feature",
            )
        )
        payload[_metric_key("viz", bucket_name, "input", "sidechain", name="heatmap")] = self._wandb.Image(
            self._build_heatmap_image(
                _to_np(batch["sidechain_cond"][sample_idx]),
                title=f"{bucket_title} | Sidechain Input",
                xlabel="time",
                ylabel="feature",
            )
        )

        debug = output.get("_debug")
        if isinstance(debug, Mapping):
            encoder = debug.get("encoder")
            if isinstance(encoder, Mapping):
                for scale in ("macro", "mezzo", "micro"):
                    z = encoder.get(scale)
                    if isinstance(z, torch.Tensor):
                        matrix = _to_np(z[sample_idx].abs().mean(dim=1))
                        payload[_metric_key("viz", bucket_name, "encoder", scale, name="feature_patch_heatmap")] = self._wandb.Image(
                            self._build_heatmap_image(
                                matrix,
                                title=f"{bucket_title} | {scale.title()} Encoder Feature-Patch",
                                xlabel="patch",
                                ylabel="feature",
                            )
                        )

            bridge = debug.get("bridge")
            if isinstance(bridge, Mapping):
                for scale in ("macro", "mezzo", "micro"):
                    bridge_debug = bridge.get(scale)
                    if isinstance(bridge_debug, Mapping):
                        gate = bridge_debug.get("gate")
                        if isinstance(gate, torch.Tensor):
                            matrix = _to_np(gate[sample_idx].unsqueeze(1))
                            payload[_metric_key("viz", bucket_name, "bridge", scale, name="gate_heatmap")] = self._wandb.Image(
                                self._build_heatmap_image(
                                    matrix,
                                    title=f"{bucket_title} | {scale.title()} Bridge Gate",
                                    xlabel="gate",
                                    ylabel="channel",
                                )
                            )

            cross_scale = debug.get("cross_scale")
            if isinstance(cross_scale, Mapping):
                cross_attn = cross_scale.get("cross_attn_weights")
                lengths = cross_scale.get("scale_token_lengths")
                if isinstance(cross_attn, torch.Tensor):
                    attn = _to_np(cross_attn[sample_idx])
                    rows = attn.reshape(attn.shape[0] * attn.shape[1], attn.shape[2])
                    xticks = None
                    xticklabels = None
                    if isinstance(lengths, torch.Tensor):
                        lens = lengths.detach().cpu().tolist()
                        edges = np.cumsum([0, *[int(v) for v in lens]])
                        xticks = [0, edges[1], edges[2], edges[3] - 1]
                        xticklabels = ["macro", "mezzo", "micro", "end"]
                    payload[_metric_key("viz", bucket_name, "cross_scale", name="latent_attention_heatmap")] = self._wandb.Image(
                        self._build_heatmap_image(
                            rows,
                            title=f"{bucket_title} | Cross-Scale Latent Attention",
                            xlabel="source token",
                            ylabel="head x latent",
                            xticks=xticks,
                            xticklabels=xticklabels,
                        )
                    )

            heads = debug.get("heads")
            if isinstance(heads, Mapping):
                task_rows: list[np.ndarray] = []
                row_labels: list[str] = []
                for task in ("ret", "rv", "q"):
                    weights = heads.get(f"{task}_task_attn_weights")
                    if isinstance(weights, torch.Tensor):
                        task_attn = _to_np(weights[sample_idx].squeeze(1))
                        task_rows.append(task_attn)
                        row_labels.extend([f"{task}_h{idx}" for idx in range(task_attn.shape[0])])
                if task_rows:
                    matrix = np.concatenate(task_rows, axis=0)
                    payload[_metric_key("viz", bucket_name, "heads", name="task_attention_heatmap")] = self._wandb.Image(
                        self._build_heatmap_image(
                            matrix,
                            title=f"{bucket_title} | Task Query Attention",
                            xlabel="latent",
                            ylabel="task x head",
                            yticks=list(range(len(row_labels))),
                            yticklabels=row_labels,
                        )
                    )

        ret_pred = _to_np(output["pred_mu_ret"])
        rv_pred = _to_np(F.softplus(output["pred_mean_rv_raw"]))
        q_pred = _to_np(output["pred_mu_q"])
        payload[_metric_key("viz", bucket_name, "pred_ret", name="density_heatmap")] = self._wandb.Image(
            self._build_density_scatter_image(
                ret_pred,
                _to_np(batch["target_ret"]),
                title=f"{bucket_title} | RET Pred vs Target",
            )
        )
        payload[_metric_key("viz", bucket_name, "pred_rv", name="density_heatmap")] = self._wandb.Image(
            self._build_density_scatter_image(
                rv_pred,
                _to_np(batch["target_rv"]),
                title=f"{bucket_title} | RV Pred vs Target",
            )
        )
        payload[_metric_key("viz", bucket_name, "pred_q", name="density_heatmap")] = self._wandb.Image(
            self._build_density_scatter_image(
                q_pred,
                _to_np(batch["target_q"]),
                title=f"{bucket_title} | Q Pred vs Target",
            )
        )

        self._log(payload, global_step=global_step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
