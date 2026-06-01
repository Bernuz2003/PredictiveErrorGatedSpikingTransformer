from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pegst.utils.io import write_csv, write_json


@dataclass
class LayerActivity:
    layer_name: str
    module_type: str
    network_stage: str
    input_shape: str
    output_shape: str
    params: int
    input_numel: int
    input_spike_count: float
    input_firing_rate: float
    input_activity_density: float
    output_numel: int
    output_spike_count: float
    output_firing_rate: float
    output_activity_count: float
    output_activity_density: float
    output_is_spike_like: bool
    dense_macs: float
    estimated_sops: float


def first_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = first_tensor(item)
            if t is not None:
                return t
    return None


def is_spike_like(x: torch.Tensor) -> bool:
    if x.numel() == 0:
        return False
    with torch.no_grad():
        return bool(((x == 0) | (x == 1)).float().mean().item() > 0.995)


def tensor_activity(x: torch.Tensor) -> dict[str, float | bool]:
    with torch.no_grad():
        spike_like = is_spike_like(x)
        nonzero_count = float((x.abs() > 1e-8).sum().item())
        activity_density = nonzero_count / max(1, x.numel())
        if spike_like:
            count = float((x > 0).sum().item())
            firing_rate = count / max(1, x.numel())
        else:
            count = 0.0
            firing_rate = 0.0
        return {
            "numel": x.numel(),
            "spike_count": count,
            "firing_rate": firing_rate,
            "activity_count": nonzero_count,
            "activity_density": activity_density,
            "is_spike_like": spike_like,
        }


def spike_stats(x: torch.Tensor) -> tuple[float, float]:
    stats = tensor_activity(x)
    return float(stats["spike_count"]), float(stats["firing_rate"])


def stage_from_name(name: str) -> str:
    local = name.replace("backbone.", "")
    if local.startswith("predictors."):
        parts = local.split(".")
        return f"predictive_{parts[1]}" if len(parts) > 1 else "predictive"
    if local.startswith("modulators."):
        parts = local.split(".")
        return f"modulation_{parts[1]}" if len(parts) > 1 else "modulation"
    if local.startswith("patch_embed1"):
        return "backbone_patch_embed1"
    if local.startswith("stage1"):
        return "backbone_stage1"
    if local.startswith("patch_embed2"):
        return "backbone_patch_embed2"
    if local.startswith("stage2"):
        return "backbone_stage2"
    if local.startswith("head"):
        return "head"
    return "other"


def dense_macs(module: nn.Module, out: torch.Tensor | None) -> float:
    if out is None:
        return 0.0
    if isinstance(module, nn.Conv2d):
        out_elems = out.numel()
        kh, kw = module.kernel_size
        return float(out_elems * (module.in_channels / module.groups) * kh * kw)
    if isinstance(module, nn.Conv1d):
        out_elems = out.numel()
        (k,) = module.kernel_size
        return float(out_elems * (module.in_channels / module.groups) * k)
    if isinstance(module, nn.Linear):
        return float(out.numel() * module.in_features)
    return 0.0


class ActivityProfiler:
    TRACKED = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.MaxPool1d, nn.MaxPool2d)

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.enabled = False
        self.records: list[LayerActivity] = []
        self.handles: list[Any] = []

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if name and (isinstance(module, self.TRACKED) or "LIF" in module.__class__.__name__):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.records.clear()

    def _hook(self, name: str):
        def fn(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.enabled:
                return
            out = first_tensor(output)
            inp = first_tensor(inputs)
            if out is None:
                return
            out_stats = tensor_activity(out.detach())
            if inp is not None:
                in_stats = tensor_activity(inp.detach())
            else:
                in_stats = {
                    "numel": 0,
                    "spike_count": 0.0,
                    "firing_rate": 0.0,
                    "activity_density": float(out_stats["activity_density"]),
                }
            macs = dense_macs(module, out)
            # SOP estimate follows the common SNN proxy: SOP_l = activity_input * FLOPs_l.
            input_activity = float(in_stats["activity_density"])
            self.records.append(
                LayerActivity(
                    layer_name=name,
                    module_type=module.__class__.__name__,
                    network_stage=stage_from_name(name),
                    input_shape=json.dumps(list(inp.shape)) if inp is not None else "[]",
                    output_shape=json.dumps(list(out.shape)),
                    params=sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad),
                    input_numel=int(in_stats["numel"]),
                    input_spike_count=float(in_stats["spike_count"]),
                    input_firing_rate=float(in_stats["firing_rate"]),
                    input_activity_density=input_activity,
                    output_numel=out.numel(),
                    output_spike_count=float(out_stats["spike_count"]),
                    output_firing_rate=float(out_stats["firing_rate"]),
                    output_activity_count=float(out_stats["activity_count"]),
                    output_activity_density=float(out_stats["activity_density"]),
                    output_is_spike_like=bool(out_stats["is_spike_like"]),
                    dense_macs=macs,
                    estimated_sops=float(macs * input_activity),
                )
            )

        return fn

    def __enter__(self):
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.enabled = False

    def summarize(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        seen_param_layers: set[str] = set()
        layer_counts: dict[str, int] = defaultdict(int)
        for r in self.records:
            s = by_stage[r.network_stage]
            layer_counts[r.layer_name] += 1
            if r.layer_name not in seen_param_layers:
                s["params"] += r.params
                seen_param_layers.add(r.layer_name)
            s["dense_macs"] += r.dense_macs
            s["estimated_sops"] += r.estimated_sops
            s["output_spike_count"] += r.output_spike_count
            s["output_numel"] += r.output_numel
            s["output_activity_count"] += r.output_activity_count
            if r.output_is_spike_like:
                s["spiking_output_spike_count"] += r.output_spike_count
                s["spiking_output_numel"] += r.output_numel
            s["num_records"] += 1
        profiled_forwards = max(layer_counts.values(), default=1)
        stage_rows = []
        total_sops = 0.0
        total_spiking_numel = 0.0
        total_spiking_spikes = 0.0
        for stage, vals in by_stage.items():
            vals["dense_macs"] /= profiled_forwards
            vals["estimated_sops"] /= profiled_forwards
            vals["avg_output_spike_count"] = vals["output_spike_count"] / profiled_forwards
            vals["avg_output_numel"] = vals["output_numel"] / profiled_forwards
            vals["avg_output_activity_count"] = vals["output_activity_count"] / profiled_forwards
            vals["output_activity_density"] = vals["output_activity_count"] / max(1.0, vals["output_numel"])
            vals["weighted_output_firing_rate"] = vals["spiking_output_spike_count"] / max(1.0, vals["spiking_output_numel"])
            vals["network_stage"] = stage
            vals["num_profiled_forwards"] = profiled_forwards
            stage_rows.append(dict(vals))
            total_sops += vals["estimated_sops"]
            total_spiking_numel += vals["spiking_output_numel"]
            total_spiking_spikes += vals["spiking_output_spike_count"]
        return {
            "num_records": len(self.records),
            "num_profiled_forwards": profiled_forwards,
            "total_estimated_sops_per_forward": total_sops,
            "weighted_output_firing_rate": total_spiking_spikes / max(1.0, total_spiking_numel),
            "stage_summary": stage_rows,
        }

    def save(self, out_dir: str | Path) -> dict[str, Any]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "layerwise_activity.csv", [asdict(r) for r in self.records])
        summary = self.summarize()
        write_csv(out_dir / "stage_activity.csv", summary["stage_summary"])
        write_csv(
            out_dir / "sops_estimate.csv",
            [
                {
                    "network_stage": row["network_stage"],
                    "dense_macs": row["dense_macs"],
                    "estimated_sops": row["estimated_sops"],
                    "weighted_output_firing_rate": row["weighted_output_firing_rate"],
                    "output_activity_density": row["output_activity_density"],
                }
                for row in summary["stage_summary"]
            ],
        )
        write_json(out_dir / "activity_summary.json", {k: v for k, v in summary.items() if k != "stage_summary"})
        return summary


def stage_from_parameter_name(name: str) -> str:
    return stage_from_name(name)


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    by_stage: dict[str, dict[str, int]] = defaultdict(lambda: {"total_params": 0, "trainable_params": 0})
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        stage = stage_from_parameter_name(name)
        by_stage[stage]["total_params"] += n
        total_params += n
        if param.requires_grad:
            by_stage[stage]["trainable_params"] += n
            trainable_params += n
    rows = [{"network_stage": stage, **vals} for stage, vals in sorted(by_stage.items())]
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "stage_summary": rows,
    }


def save_parameter_summary(model: nn.Module, out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = parameter_summary(model)
    write_json(out_dir / "params_summary.json", summary)
    return summary
