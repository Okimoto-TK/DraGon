"""Export a model architecture diagram image for MultiScaleForecastNetwork."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
import torch.nn as nn

from src.models.arch.networks import MultiScaleForecastNetwork
from src.task_labels import TASK_LABELS, TRAINING_TASKS, canonical_task_label, canonical_training_task
from src.train.checkpoint import load_checkpoint


@dataclass(frozen=True)
class NodeSpec:
    key: str
    title: str
    x: float
    y: float
    color: str
    width: float = 2.95
    height: float = 1.25


class _ShapeRecorder:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, str]] = {}
        self._handles: list[Any] = []

    def add_hook(self, name: str, module: nn.Module) -> None:
        def _hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            self.records[name] = {
                "input": _summarize_value(inputs),
                "output": _summarize_value(output),
            }

        self._handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _TailCropDenoise(nn.Module):
    """Fast deterministic denoise replacement for export-time shape capture."""

    def __init__(self, target_len: int) -> None:
        super().__init__()
        self.target_len = int(target_len)

    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        return x_long[..., -self.target_len :]

    def forward_features(
        self,
        x_long: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        y = self.forward(x_long)
        return y, (y, y, y)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a MultiScaleForecastNetwork architecture image.",
    )
    parser.add_argument(
        "--mode",
        choices=TRAINING_TASKS,
        default="mu",
        help="Training stage to visualize.",
    )
    parser.add_argument(
        "--field",
        choices=TASK_LABELS,
        default="ret",
        help="Prediction field to visualize.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint file or directory to load before export.",
    )
    parser.add_argument(
        "--output",
        default="models/diagrams/model_graph.png",
        help="Output image path. Extension controls the format.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for shape capture. Only 1 is supported.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--fast-dummy-denoise",
        action="store_true",
        help="Replace denoise modules with deterministic tail-crop stubs for faster export.",
    )
    return parser


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda, but CUDA is not available.")
    return torch.device(device_name)


def _make_dummy_batch(
    *,
    batch_size: int,
    mode: str,
    field: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    macro_state = torch.randint(0, 16, (batch_size, 1, 112), generator=generator, dtype=torch.int64)
    macro_pos = torch.randint(0, 8, (batch_size, 1, 112), generator=generator, dtype=torch.int64)
    mezzo_state = torch.randint(0, 16, (batch_size, 1, 144), generator=generator, dtype=torch.int64)
    mezzo_pos = torch.randint(0, 16, (batch_size, 1, 144), generator=generator, dtype=torch.int64)
    micro_state = torch.randint(0, 16, (batch_size, 1, 192), generator=generator, dtype=torch.int64)
    micro_pos = torch.randint(0, 64, (batch_size, 1, 192), generator=generator, dtype=torch.int64)

    batch = {
        "macro_float_long": torch.randn(batch_size, 9, 112, generator=generator),
        "macro_i8_long": torch.cat([macro_state, macro_pos], dim=1),
        "mezzo_float_long": torch.randn(batch_size, 9, 144, generator=generator),
        "mezzo_i8_long": torch.cat([mezzo_state, mezzo_pos], dim=1),
        "micro_float_long": torch.randn(batch_size, 9, 192, generator=generator),
        "micro_i8_long": torch.cat([micro_state, micro_pos], dim=1),
        "sidechain_cond": torch.randn(batch_size, 13, 64, generator=generator),
        "target_ret": torch.randn(batch_size, 1, generator=generator),
        "target_rv": torch.rand(batch_size, 1, generator=generator).clamp_min(1e-4),
        "target_q": torch.randn(batch_size, 1, generator=generator),
    }
    if mode == "sigma":
        if field == "rv":
            batch["mu_input"] = torch.rand(batch_size, 1, generator=generator).clamp_min(1e-3)
        else:
            batch["mu_input"] = torch.randn(batch_size, 1, generator=generator)

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def _maybe_swap_denoise(model: MultiScaleForecastNetwork, enabled: bool) -> None:
    if not enabled:
        return
    model.denoise_macro = _TailCropDenoise(target_len=64)
    model.denoise_mezzo = _TailCropDenoise(target_len=96)
    model.denoise_micro = _TailCropDenoise(target_len=144)


def _capture_architecture(
    *,
    model: MultiScaleForecastNetwork,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    recorder = _ShapeRecorder()
    try:
        recorder.add_hook("prep_macro", model.dropout_macro_input)
        recorder.add_hook("prep_mezzo", model.dropout_mezzo_input)
        recorder.add_hook("prep_micro", model.dropout_micro_input)
        recorder.add_hook("time_macro", model.scale_macro.time_star)
        recorder.add_hook("time_mezzo", model.scale_mezzo.time_star)
        recorder.add_hook("time_micro", model.scale_micro.time_star)
        recorder.add_hook("wavelet_macro", model.scale_macro.wavelet_encoder)
        recorder.add_hook("wavelet_mezzo", model.scale_mezzo.wavelet_encoder)
        recorder.add_hook("wavelet_micro", model.scale_micro.wavelet_encoder)
        recorder.add_hook("mutual_macro", model.scale_macro)
        recorder.add_hook("mutual_mezzo", model.scale_mezzo)
        recorder.add_hook("mutual_micro", model.scale_micro)
        recorder.add_hook("time_cross_scale_fusion", model.time_cross_scale_fusion)
        recorder.add_hook("wavelet_cross_scale_fusion", model.wavelet_cross_scale_fusion)
        recorder.add_hook("head_tokens", model.prediction_head.concat_proj)
        recorder.add_hook("head_context", model.prediction_head.global_fuse)
        if model.mode == "mu":
            recorder.add_hook("head_readout", model.prediction_head.readout)
            recorder.add_hook("head_output", model.prediction_head.value_head)
        else:
            recorder.add_hook("mu_context", model.prediction_head.mu_encoder)
            recorder.add_hook("head_output", model.prediction_head.sigma_head)

        denoise_shapes = {
            "denoise_macro": _summarize_value(
                model._forward_wavelet_frontend(model.denoise_macro, batch["macro_float_long"])[0]
            ),
            "denoise_mezzo": _summarize_value(
                model._forward_wavelet_frontend(model.denoise_mezzo, batch["mezzo_float_long"])[0]
            ),
            "denoise_micro": _summarize_value(
                model._forward_wavelet_frontend(model.denoise_micro, batch["micro_float_long"])[0]
            ),
        }
        with torch.inference_mode():
            forward_out = model.forward(batch, return_aux=False, return_debug=False)
            loss_out = model.forward_loss(batch, return_aux=False)
        return recorder.records, denoise_shapes, forward_out, loss_out
    finally:
        recorder.clear()


def _module_params(module: nn.Module) -> str:
    total = sum(parameter.numel() for parameter in module.parameters())
    if total >= 1_000_000:
        return f"{total / 1_000_000.0:.2f}M"
    if total >= 1_000:
        return f"{total / 1_000.0:.1f}K"
    return str(total)


def _node_specs(mode: str) -> list[NodeSpec]:
    specs = [
        NodeSpec("input_macro", "Macro Input", 1.8, 9.0, "#dbeafe", width=2.7),
        NodeSpec("input_mezzo", "Mezzo Input", 1.8, 6.1, "#dbeafe", width=2.7),
        NodeSpec("input_micro", "Micro Input", 1.8, 3.2, "#dbeafe", width=2.7),
        NodeSpec("sidechain", "Sidechain", 1.8, 10.8, "#dbeafe", width=2.7),
        NodeSpec("denoise_macro", "Macro Denoise", 5.0, 9.0, "#fef3c7", width=2.7),
        NodeSpec("denoise_mezzo", "Mezzo Denoise", 5.0, 6.1, "#fef3c7", width=2.7),
        NodeSpec("denoise_micro", "Micro Denoise", 5.0, 3.2, "#fef3c7", width=2.7),
        NodeSpec("prep_macro", "Prep", 8.1, 9.0, "#ffe4e6", width=2.65),
        NodeSpec("prep_mezzo", "Prep", 8.1, 6.1, "#ffe4e6", width=2.65),
        NodeSpec("prep_micro", "Prep", 8.1, 3.2, "#ffe4e6", width=2.65),
        NodeSpec("time_macro", "Time Branch", 11.4, 9.55, "#dcfce7", width=2.85),
        NodeSpec("time_mezzo", "Time Branch", 11.4, 6.65, "#dcfce7", width=2.85),
        NodeSpec("time_micro", "Time Branch", 11.4, 3.75, "#dcfce7", width=2.85),
        NodeSpec("wavelet_macro", "Wavelet Branch", 11.4, 8.45, "#d1fae5", width=2.85),
        NodeSpec("wavelet_mezzo", "Wavelet Branch", 11.4, 5.55, "#d1fae5", width=2.85),
        NodeSpec("wavelet_micro", "Wavelet Branch", 11.4, 2.65, "#d1fae5", width=2.85),
        NodeSpec("mutual_macro", "Dual-Domain Mutual", 15.0, 9.0, "#ede9fe", width=3.1, height=1.35),
        NodeSpec("mutual_mezzo", "Dual-Domain Mutual", 15.0, 6.1, "#ede9fe", width=3.1, height=1.35),
        NodeSpec("mutual_micro", "Dual-Domain Mutual", 15.0, 3.2, "#ede9fe", width=3.1, height=1.35),
        NodeSpec("time_cross_scale_fusion", "Cross-Scale Time Fusion", 19.0, 8.55, "#ddd6fe", width=3.25, height=1.45),
        NodeSpec("wavelet_cross_scale_fusion", "Cross-Scale Wavelet Fusion", 19.0, 3.65, "#ddd6fe", width=3.25, height=1.45),
        NodeSpec("head_tokens", "Head Token Mixer", 22.6, 6.15, "#fed7aa", width=3.2, height=1.4),
    ]
    if mode == "mu":
        specs.extend(
            [
                NodeSpec("head_readout", "Task Query Tower", 26.0, 6.85, "#fdba74", width=3.0),
                NodeSpec("head_output", "Value Head", 29.2, 6.85, "#fdba74", width=2.75),
                NodeSpec("loss_fn", "Loss", 32.5, 6.85, "#fecaca", width=2.9),
            ]
        )
    else:
        specs.extend(
            [
                NodeSpec("mu_input", "Frozen mu_input", 22.6, 10.15, "#dbeafe", width=3.2),
                NodeSpec("mu_context", "mu Feature Encoder", 26.0, 10.15, "#bfdbfe", width=3.0),
                NodeSpec("confidence_block", "Confidence Retrieval", 29.7, 6.15, "#fdba74", width=3.35, height=1.5),
                NodeSpec("head_output", "Sigma Head", 33.4, 6.15, "#fdba74", width=2.75),
                NodeSpec("loss_fn", "Loss", 36.6, 6.15, "#fecaca", width=2.9),
            ]
        )
    return specs


def _node_texts(
    *,
    model: MultiScaleForecastNetwork,
    batch: dict[str, torch.Tensor],
    records: dict[str, dict[str, str]],
    denoise_shapes: dict[str, str],
    forward_out: dict[str, torch.Tensor],
    loss_out: dict[str, torch.Tensor],
) -> dict[str, str]:
    mutual_macro_out = _shape_parts(records.get("mutual_macro"))
    mutual_mezzo_out = _shape_parts(records.get("mutual_mezzo"))
    mutual_micro_out = _shape_parts(records.get("mutual_micro"))
    time_fusion_out = _shape_parts(records.get("time_cross_scale_fusion"))
    wavelet_fusion_out = _shape_parts(records.get("wavelet_cross_scale_fusion"))
    texts = {
        "input_macro": f"float { _shape_of(batch['macro_float_long']) }\nindex { _shape_of(batch['macro_i8_long']) }",
        "input_mezzo": f"float { _shape_of(batch['mezzo_float_long']) }\nindex { _shape_of(batch['mezzo_i8_long']) }",
        "input_micro": f"float { _shape_of(batch['micro_float_long']) }\nindex { _shape_of(batch['micro_i8_long']) }",
        "sidechain": f"cond { _shape_of(batch['sidechain_cond']) }",
        "denoise_macro": f"out {denoise_shapes['denoise_macro']}",
        "denoise_mezzo": f"out {denoise_shapes['denoise_mezzo']}",
        "denoise_micro": f"out {denoise_shapes['denoise_micro']}",
        "prep_macro": f"time_x { _shape_from_record(records.get('prep_macro')) }\nstate + pos + cond",
        "prep_mezzo": f"time_x { _shape_from_record(records.get('prep_mezzo')) }\nstate + pos",
        "prep_micro": f"time_x { _shape_from_record(records.get('prep_micro')) }\nstate + pos",
        "time_macro": f"TCN + STAR\ntokens { _shape_part(_shape_parts(records.get('time_macro')), 1) }",
        "time_mezzo": f"TCN + STAR\ntokens { _shape_part(_shape_parts(records.get('time_mezzo')), 1) }",
        "time_micro": f"TCN + STAR\ntokens { _shape_part(_shape_parts(records.get('time_micro')), 1) }",
        "wavelet_macro": f"band encoder\ntokens { _shape_from_record(records.get('wavelet_macro')) }",
        "wavelet_mezzo": f"band encoder\ntokens { _shape_from_record(records.get('wavelet_mezzo')) }",
        "wavelet_micro": f"band encoder\ntokens { _shape_from_record(records.get('wavelet_micro')) }",
        "mutual_macro": f"time { _shape_part(mutual_macro_out, 0) }\nwave { _shape_part(mutual_macro_out, 1) }",
        "mutual_mezzo": f"time { _shape_part(mutual_mezzo_out, 0) }\nwave { _shape_part(mutual_mezzo_out, 1) }",
        "mutual_micro": f"time { _shape_part(mutual_micro_out, 0) }\nwave { _shape_part(mutual_micro_out, 1) }",
        "time_cross_scale_fusion": "\n".join(
            [
                f"macro { _shape_part(time_fusion_out, 0) }",
                f"mezzo { _shape_part(time_fusion_out, 1) }",
                f"micro { _shape_part(time_fusion_out, 2) }",
            ]
        ),
        "wavelet_cross_scale_fusion": "\n".join(
            [
                f"macro { _shape_part(wavelet_fusion_out, 0) }",
                f"mezzo { _shape_part(wavelet_fusion_out, 1) }",
                f"micro { _shape_part(wavelet_fusion_out, 2) }",
            ]
        ),
        "head_tokens": "\n".join(
            [
                f"tokens { _shape_from_record(records.get('head_tokens')) }",
                f"context { _shape_from_record(records.get('head_context')) }",
                f"params { _module_params(model.prediction_head.concat_proj) } + { _module_params(model.prediction_head.global_fuse) }",
            ]
        ),
    }
    if model.mode == "mu":
        texts["head_readout"] = "\n".join(
            [
                f"task_repr { _tensor_shape_or('--', forward_out.get('task_repr')) }",
                f"params { _module_params(model.prediction_head.readout) }",
            ]
        )
        texts["head_output"] = "\n".join(
            [
                f"mu_raw { _shape_from_record(records.get('head_output')) }",
                f"mu_pred { _tensor_shape_or('--', loss_out.get('mu_pred')) }",
            ]
        )
    else:
        texts["mu_input"] = _mu_input_text(model=model, batch=batch)
        texts["mu_context"] = "\n".join(
            [
                f"q { _shape_from_record(records.get('mu_context')) }",
                f"params { _module_params(model.prediction_head.mu_encoder) }",
            ]
        )
        texts["confidence_block"] = "\n".join(
            [
                f"conf_repr { _tensor_shape_or('--', forward_out.get('confidence_repr')) }",
                "query attends to H",
                "attn -> evidence state",
            ]
        )
        texts["head_output"] = "\n".join(
            [
                f"sigma_raw { _shape_from_record(records.get('head_output')) }",
                f"sigma_pred { _tensor_shape_or('--', loss_out.get('sigma_pred')) }",
                "attn entropy / max",
            ]
        )
    texts["loss_fn"] = _loss_text(model=model)
    return texts


def _shape_parts(record: dict[str, str] | None) -> list[str]:
    if record is None:
        return []
    return [part.strip() for part in record["output"].split("|")]


def _shape_part(parts: list[str], index: int) -> str:
    if index >= len(parts):
        return "--"
    return parts[index]


def _shape_from_record(record: dict[str, str] | None) -> str:
    if record is None:
        return "--"
    return record["output"]

def _loss_text(*, model: MultiScaleForecastNetwork) -> str:
    if model.mode == "mu":
        if model.field == "ret":
            return "weighted Huber\noutputs: loss_mu + mu_pred"
        if model.field == "rv":
            return "QLIKE / fixed-gamma mean\noutputs: loss_mu + mu_pred"
        return "pinball\noutputs: loss_mu + mu_pred"
    if model.field == "ret":
        return "Student-t NLL\noutputs: loss_nll + sigma_pred"
    if model.field == "rv":
        return "Gamma NLL\noutputs: loss_nll + sigma_pred"
    return "ALD NLL\noutputs: loss_nll + sigma_pred"


def _mu_input_text(
    *,
    model: MultiScaleForecastNetwork,
    batch: dict[str, torch.Tensor],
) -> str:
    if model.field == "rv":
        features = "mu, log(mu), sqrt(mu)"
    elif model.domain == "q":
        features = "mu, |mu|, sign, log1p, tau"
    else:
        features = "mu, |mu|, sign, log1p"
    return f"{_shape_of(batch['mu_input'])}\n{features}"


def _draw_graph(
    *,
    model: MultiScaleForecastNetwork,
    mode: str,
    field: str,
    output_path: Path,
    texts: dict[str, str],
    dpi: int,
) -> None:
    specs = _node_specs(mode)
    fig, ax = plt.subplots(figsize=(33, 13.5), dpi=dpi)
    ax.set_xlim(0.0, 38.6)
    ax.set_ylim(1.0, 11.9)
    ax.axis("off")

    spec_map = {spec.key: spec for spec in specs}
    lane_specs = [
        ("Macro Lane", 7.95, 10.1, "#f8fbff"),
        ("Mezzo Lane", 5.05, 7.2, "#fbfcff"),
        ("Micro Lane", 2.15, 4.3, "#f8fbff"),
    ]
    for label, y0, y1, color in lane_specs:
        lane = FancyBboxPatch(
            (0.35, y0),
            17.0,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.9,
            edgecolor="#d7dde5",
            facecolor=color,
            zorder=0,
        )
        ax.add_patch(lane)
        ax.text(
            0.55,
            y1 - 0.22,
            label,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#4a5565",
        )

    header_specs = [
        ("Inputs", 1.8),
        ("Preprocess", 6.55),
        ("Per-Scale Encoder", 13.2),
        ("Cross-Scale", 19.0),
        ("Head", 28.5 if mode == "sigma" else 26.0),
        ("Loss", 36.6 if mode == "sigma" else 32.5),
    ]
    for label, x in header_specs:
        ax.text(
            x,
            11.45,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#374151",
        )

    for spec in specs:
        box = FancyBboxPatch(
            (spec.x - spec.width / 2.0, spec.y - spec.height / 2.0),
            spec.width,
            spec.height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#333333",
            facecolor=spec.color,
        )
        ax.add_patch(box)
        ax.text(
            spec.x,
            spec.y + 0.31,
            spec.title,
            ha="center",
            va="center",
            fontsize=12.8,
            fontweight="bold",
        )
        ax.text(
            spec.x,
            spec.y - 0.12,
            texts.get(spec.key, ""),
            ha="center",
            va="center",
            fontsize=10.2,
            family="monospace",
            linespacing=1.18,
        )
    _draw_routes(ax=ax, spec_map=spec_map, mode=mode)

    title = f"MultiScaleForecastNetwork Diagram | mode={mode} field={field}"
    ax.text(18.7, 11.72, title, ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(
        18.7,
        11.08,
        "Submodules are grouped at the architectural-block level. Shapes come from a single dummy forward pass; arrows use explicit routing to avoid box overlap.",
        ha="center",
        va="center",
        fontsize=12,
        color="#444444",
    )

    info_panel = FancyBboxPatch(
        (20.5, 1.18),
        17.2,
        1.52,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#d0d7de",
        facecolor="#ffffff",
    )
    ax.add_patch(info_panel)
    ax.text(20.9, 2.48, "Details", ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(
        20.9,
        2.22,
        "\n".join(
            [
                f"mode={mode}  field={field}  batch=1  model_params={_module_params(model)}",
                f"prediction_head_params={_module_params(model.prediction_head)}  loss={texts['loss_fn'].splitlines()[0]}",
                "Left lanes show per-scale processing; right side shows shared cross-scale fusion and the field-specific head/loss path.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.8,
        family="monospace",
        linespacing=1.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_routes(
    *,
    ax,
    spec_map: dict[str, NodeSpec],
    mode: str,
) -> None:
    lane_names = ("macro", "mezzo", "micro")
    for lane in lane_names:
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"input_{lane}"], "east"),
                _anchor(spec_map[f"denoise_{lane}"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"denoise_{lane}"], "east"),
                _anchor(spec_map[f"prep_{lane}"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"prep_{lane}"], "east"),
                _anchor(spec_map[f"time_{lane}"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"denoise_{lane}"], "east", dy=-0.15),
                (_anchor(spec_map[f"denoise_{lane}"], "east")[0] + 0.9, _anchor(spec_map[f"denoise_{lane}"], "east", dy=-0.15)[1]),
                (_anchor(spec_map[f"denoise_{lane}"], "east")[0] + 0.9, _anchor(spec_map[f"wavelet_{lane}"], "west")[1]),
                _anchor(spec_map[f"wavelet_{lane}"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"time_{lane}"], "east"),
                (_anchor(spec_map[f"time_{lane}"], "east")[0] + 0.85, _anchor(spec_map[f"time_{lane}"], "east")[1]),
                (_anchor(spec_map[f"time_{lane}"], "east")[0] + 0.85, spec_map[f"mutual_{lane}"].y + 0.24),
                _anchor(spec_map[f"mutual_{lane}"], "west", dy=0.24),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map[f"wavelet_{lane}"], "east"),
                (_anchor(spec_map[f"wavelet_{lane}"], "east")[0] + 0.85, _anchor(spec_map[f"wavelet_{lane}"], "east")[1]),
                (_anchor(spec_map[f"wavelet_{lane}"], "east")[0] + 0.85, spec_map[f"mutual_{lane}"].y - 0.24),
                _anchor(spec_map[f"mutual_{lane}"], "west", dy=-0.24),
            ],
        )

    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["sidechain"], "south"),
            (_anchor(spec_map["sidechain"], "south")[0], spec_map["prep_macro"].y + 0.95),
            (_anchor(spec_map["prep_macro"], "west")[0] - 0.65, spec_map["prep_macro"].y + 0.95),
            (_anchor(spec_map["prep_macro"], "west")[0] - 0.65, _anchor(spec_map["prep_macro"], "west")[1]),
            _anchor(spec_map["prep_macro"], "west"),
        ],
    )

    time_targets = {
        "macro": spec_map["time_cross_scale_fusion"].y + 0.34,
        "mezzo": spec_map["time_cross_scale_fusion"].y,
        "micro": spec_map["time_cross_scale_fusion"].y - 0.34,
    }
    wave_targets = {
        "macro": spec_map["wavelet_cross_scale_fusion"].y + 0.34,
        "mezzo": spec_map["wavelet_cross_scale_fusion"].y,
        "micro": spec_map["wavelet_cross_scale_fusion"].y - 0.34,
    }
    for lane in lane_names:
        start = _anchor(spec_map[f"mutual_{lane}"], "east", dy=0.24)
        lane_mid_x = start[0] + 0.85
        _draw_polyline(
            ax=ax,
            points=[
                start,
                (lane_mid_x, start[1]),
                (lane_mid_x, time_targets[lane]),
                _anchor(spec_map["time_cross_scale_fusion"], "west", dy=time_targets[lane] - spec_map["time_cross_scale_fusion"].y),
            ],
        )
        start = _anchor(spec_map[f"mutual_{lane}"], "east", dy=-0.24)
        lane_mid_x = start[0] + 1.15
        _draw_polyline(
            ax=ax,
            points=[
                start,
                (lane_mid_x, start[1]),
                (lane_mid_x, wave_targets[lane]),
                _anchor(spec_map["wavelet_cross_scale_fusion"], "west", dy=wave_targets[lane] - spec_map["wavelet_cross_scale_fusion"].y),
            ],
        )

    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["time_cross_scale_fusion"], "east", dy=0.18),
            (_anchor(spec_map["time_cross_scale_fusion"], "east")[0] + 0.7, _anchor(spec_map["time_cross_scale_fusion"], "east", dy=0.18)[1]),
            (_anchor(spec_map["head_tokens"], "west")[0] - 0.7, _anchor(spec_map["time_cross_scale_fusion"], "east", dy=0.18)[1]),
            (_anchor(spec_map["head_tokens"], "west")[0] - 0.7, spec_map["head_tokens"].y + 0.26),
            _anchor(spec_map["head_tokens"], "west", dy=0.26),
        ],
    )
    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["wavelet_cross_scale_fusion"], "east", dy=-0.18),
            (_anchor(spec_map["wavelet_cross_scale_fusion"], "east")[0] + 1.05, _anchor(spec_map["wavelet_cross_scale_fusion"], "east", dy=-0.18)[1]),
            (_anchor(spec_map["head_tokens"], "west")[0] - 0.4, _anchor(spec_map["wavelet_cross_scale_fusion"], "east", dy=-0.18)[1]),
            (_anchor(spec_map["head_tokens"], "west")[0] - 0.4, spec_map["head_tokens"].y - 0.26),
            _anchor(spec_map["head_tokens"], "west", dy=-0.26),
        ],
    )

    if mode == "mu":
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map["head_tokens"], "east"),
                _anchor(spec_map["head_readout"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map["head_readout"], "east"),
                _anchor(spec_map["head_output"], "west"),
            ],
        )
        _draw_polyline(
            ax=ax,
            points=[
                _anchor(spec_map["head_output"], "east"),
                _anchor(spec_map["loss_fn"], "west"),
            ],
        )
        return

    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["mu_input"], "east"),
            _anchor(spec_map["mu_context"], "west"),
        ],
    )
    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["head_tokens"], "east"),
            (_anchor(spec_map["head_tokens"], "east")[0] + 1.0, _anchor(spec_map["head_tokens"], "east")[1]),
            (_anchor(spec_map["confidence_block"], "west")[0] - 0.9, _anchor(spec_map["head_tokens"], "east")[1]),
            (_anchor(spec_map["confidence_block"], "west")[0] - 0.9, _anchor(spec_map["confidence_block"], "west", dy=-0.24)[1]),
            _anchor(spec_map["confidence_block"], "west", dy=-0.24),
        ],
    )
    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["mu_context"], "south"),
            (_anchor(spec_map["mu_context"], "south")[0], spec_map["confidence_block"].y + 0.95),
            (_anchor(spec_map["confidence_block"], "west")[0] - 0.25, spec_map["confidence_block"].y + 0.95),
            (_anchor(spec_map["confidence_block"], "west")[0] - 0.25, _anchor(spec_map["confidence_block"], "west", dy=0.24)[1]),
            _anchor(spec_map["confidence_block"], "west", dy=0.24),
        ],
    )
    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["confidence_block"], "east"),
            _anchor(spec_map["head_output"], "west"),
        ],
    )
    _draw_polyline(
        ax=ax,
        points=[
            _anchor(spec_map["head_output"], "east"),
            _anchor(spec_map["loss_fn"], "west"),
        ],
    )


def _draw_polyline(
    *,
    ax,
    points: list[tuple[float, float]],
    color: str = "#475569",
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.8, solid_capstyle="round")
    arrow = FancyArrowPatch(
        posA=points[-2],
        posB=points[-1],
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.8,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def _anchor(
    spec: NodeSpec,
    side: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[float, float]:
    if side == "west":
        return (spec.x - spec.width / 2.0 + dx, spec.y + dy)
    if side == "east":
        return (spec.x + spec.width / 2.0 + dx, spec.y + dy)
    if side == "north":
        return (spec.x + dx, spec.y + spec.height / 2.0 + dy)
    if side == "south":
        return (spec.x + dx, spec.y - spec.height / 2.0 + dy)
    raise ValueError(f"Unsupported side: {side!r}")


def _tensor_shape_or(default: str, value: torch.Tensor | None) -> str:
    if value is None:
        return default
    return _shape_of(value)


def _shape_of(value: torch.Tensor) -> str:
    return str(tuple(int(dim) for dim in value.shape))


def _summarize_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _shape_of(value)
    if isinstance(value, dict):
        items = []
        for key, inner in value.items():
            if isinstance(inner, torch.Tensor):
                items.append(f"{key}:{_shape_of(inner)}")
        return ", ".join(items[:4]) if items else "{...}"
    if isinstance(value, tuple | list):
        parts = []
        for inner in value[:4]:
            parts.append(_summarize_value(inner))
        return " | ".join(parts)
    return type(value).__name__


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    mode = canonical_training_task(args.mode)
    field = canonical_task_label(args.field)
    device = _resolve_device(args.device)
    if int(args.batch_size) != 1:
        raise ValueError("Model graph export only supports --batch-size=1 to avoid unnecessary full-batch execution.")

    model = MultiScaleForecastNetwork(mode=mode, field=field)
    _maybe_swap_denoise(model, enabled=bool(args.fast_dummy_denoise))
    model = model.to(device)
    model.eval()

    if args.checkpoint:
        load_checkpoint(
            args.checkpoint,
            model=model,
            map_location=device,
            load_training_state=False,
        )

    batch = _make_dummy_batch(
        batch_size=int(args.batch_size),
        mode=mode,
        field=field,
        device=device,
    )
    records, denoise_shapes, forward_out, loss_out = _capture_architecture(
        model=model,
        batch=batch,
    )
    texts = _node_texts(
        model=model,
        batch=batch,
        records=records,
        denoise_shapes=denoise_shapes,
        forward_out=forward_out,
        loss_out=loss_out,
    )

    output_path = Path(args.output)
    _draw_graph(
        model=model,
        mode=mode,
        field=field,
        output_path=output_path,
        texts=texts,
        dpi=int(args.dpi),
    )
    print({"output": str(output_path), "mode": mode, "field": field})


if __name__ == "__main__":
    main()
