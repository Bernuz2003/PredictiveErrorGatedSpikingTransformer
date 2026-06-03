from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from pegst.models.snn_layers import DEFAULT_RECORD_TARGETS, lif_modules, record_lif_internal_states
from pegst.utils.io import write_csv, write_json


def stage_from_module_name(name: str) -> str:
    local = name.removeprefix("backbone.")
    if local.startswith("patch_embed1"):
        return "patch_embed1"
    if local.startswith("stage1"):
        return "stage1"
    if local.startswith("patch_embed2"):
        return "patch_embed2"
    if local.startswith("stage2"):
        return "stage2"
    if local.startswith("head"):
        return "head"
    return "other"


def _matches_any(name: str, patterns: Iterable[str] | None) -> bool:
    patterns = list(patterns or [])
    if not patterns:
        return True
    return any(pattern in name or pattern == stage_from_module_name(name) for pattern in patterns)


def collect_recorded_states(
    model: nn.Module,
    *,
    stages: Iterable[str] | None = None,
    layer_patterns: Iterable[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    selected: dict[str, dict[str, torch.Tensor]] = {}
    stage_set = set(stages or [])
    for name, module in lif_modules(model):
        stage = stage_from_module_name(name)
        if stage_set and stage not in stage_set:
            continue
        if not _matches_any(name, layer_patterns):
            continue
        states = getattr(module, "last_internal_states", {})
        if states:
            selected[name] = states
    return selected


def temporal_autocorrelation(x: torch.Tensor, lag: int) -> float:
    if x.shape[0] <= lag:
        return float("nan")
    a = x[:-lag].detach().float().flatten(1)
    b = x[lag:].detach().float().flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a.square().mean(dim=1).sqrt() * b.square().mean(dim=1).sqrt()).clamp_min(1e-12)
    corr = (a * b).mean(dim=1) / denom
    return float(corr.mean().item())


def tensor_stats(
    x: torch.Tensor,
    *,
    target: str,
    threshold: float = 1.0,
    near_zero_eps: float = 1e-5,
    near_threshold_width: float = 0.05,
) -> dict[str, Any]:
    with torch.no_grad():
        y = x.detach().float()
        finite = torch.isfinite(y)
        finite_fraction = float(finite.float().mean().item()) if y.numel() else 1.0
        yf = y[finite] if finite.any() else y.new_zeros(1)
        abs_y = yf.abs()
        if target == "threshold_margin":
            above = yf > 0
            near_threshold = yf.abs() <= near_threshold_width
        elif target in {"pre_membrane", "post_reset_membrane", "input_current"}:
            above = yf > threshold
            near_threshold = (yf - threshold).abs() <= near_threshold_width
        elif target in {"spike", "soft_firing_prob"}:
            above = yf > 0.5
            near_threshold = (yf - 0.5).abs() <= near_threshold_width
        else:
            above = yf > threshold
            near_threshold = (yf - threshold).abs() <= near_threshold_width
        return {
            "shape": json.dumps(list(x.shape)),
            "numel": int(y.numel()),
            "finite_fraction": finite_fraction,
            "mean": float(yf.mean().item()),
            "std": float(yf.std(unbiased=False).item()) if yf.numel() > 1 else 0.0,
            "abs_mean": float(abs_y.mean().item()),
            "abs_max": float(abs_y.max().item()) if abs_y.numel() else 0.0,
            "near_zero_fraction": float((abs_y <= near_zero_eps).float().mean().item()) if abs_y.numel() else 0.0,
            "above_threshold_fraction": float(above.float().mean().item()) if above.numel() else 0.0,
            "near_threshold_fraction": float(near_threshold.float().mean().item()) if near_threshold.numel() else 0.0,
            "temporal_autocorr_lag1": temporal_autocorrelation(y, 1),
            "temporal_autocorr_lag2": temporal_autocorrelation(y, 2),
        }


def recorded_state_stats_rows(
    states: dict[str, dict[str, torch.Tensor]],
    *,
    threshold_by_layer: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold_by_layer = threshold_by_layer or {}
    for layer_name, targets in sorted(states.items()):
        stage = stage_from_module_name(layer_name)
        threshold = float(threshold_by_layer.get(layer_name, 1.0))
        for target, tensor in sorted(targets.items()):
            rows.append(
                {
                    "stage": stage,
                    "layer": layer_name,
                    "target": target,
                    "threshold": threshold,
                    **tensor_stats(tensor, target=target, threshold=threshold),
                }
            )
    return rows


def recorded_state_shapes(states: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, list[int]]]:
    return {
        layer: {target: list(tensor.shape) for target, tensor in targets.items()}
        for layer, targets in sorted(states.items())
    }


@dataclass
class RecordingProfile:
    batches: int = 0
    forwards: int = 0
    forward_time_sec: float = 0.0
    max_cuda_memory_allocated: int = 0
    modules_recorded: int = 0
    targets: list[str] = field(default_factory=list)

    @property
    def forward_time_per_batch_sec(self) -> float:
        return self.forward_time_sec / max(1, self.forwards)


class InternalStateCollector:
    def __init__(
        self,
        model: nn.Module,
        *,
        targets: Iterable[str] | None = None,
        stages: Iterable[str] | None = None,
        layer_patterns: Iterable[str] | None = None,
        detach: bool = True,
        to_cpu: bool = False,
        soft_firing_temperature: float = 0.25,
    ) -> None:
        self.model = model
        self.targets = list(targets or DEFAULT_RECORD_TARGETS)
        self.stages = list(stages or [])
        self.layer_patterns = list(layer_patterns or [])
        self.detach = bool(detach)
        self.to_cpu = bool(to_cpu)
        self.soft_firing_temperature = float(soft_firing_temperature)

    def forward_and_collect(self, x: torch.Tensor, **forward_kwargs: Any) -> tuple[Any, dict[str, dict[str, torch.Tensor]], RecordingProfile]:
        profile = RecordingProfile(targets=self.targets)
        if x.is_cuda:
            torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        with record_lif_internal_states(
            self.model,
            self.targets,
            detach=self.detach,
            to_cpu=self.to_cpu,
            soft_firing_temperature=self.soft_firing_temperature,
        ):
            out = self.model(x, **forward_kwargs)
            states = collect_recorded_states(self.model, stages=self.stages, layer_patterns=self.layer_patterns)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
            profile.max_cuda_memory_allocated = int(torch.cuda.max_memory_allocated(x.device))
        profile.forward_time_sec = time.perf_counter() - start
        profile.batches = 1
        profile.forwards = 1
        profile.modules_recorded = len(states)
        return out, states, profile


class StreamingStats:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.shapes: dict[str, dict[str, list[int]]] = {}
        self.counts: dict[tuple[str, str, str], int] = defaultdict(int)

    def update(self, states: dict[str, dict[str, torch.Tensor]]) -> None:
        for layer, target_map in states.items():
            stage = stage_from_module_name(layer)
            self.shapes.setdefault(layer, {})
            for target, tensor in target_map.items():
                self.shapes[layer][target] = list(tensor.shape)
                stats = tensor_stats(tensor, target=target)
                key = (stage, layer, target)
                self.counts[key] += 1
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.rows[key][stat_name] += float(value)

    def finalize_rows(self) -> list[dict[str, Any]]:
        out = []
        for (stage, layer, target), values in sorted(self.rows.items()):
            n = max(1, self.counts[(stage, layer, target)])
            row: dict[str, Any] = {
                "stage": stage,
                "layer": layer,
                "target": target,
                "batches": self.counts[(stage, layer, target)],
                "shape": json.dumps(self.shapes.get(layer, {}).get(target, [])),
            }
            for name, total in values.items():
                if name in {"numel"}:
                    row[name] = int(round(total / n))
                else:
                    row[name] = total / n
            out.append(row)
        return out

    def finalize_shapes(self) -> dict[str, dict[str, list[int]]]:
        return self.shapes


def write_recording_artifacts(
    out_dir: str | Path,
    *,
    stats_rows: list[dict[str, Any]],
    shapes: dict[str, dict[str, list[int]]],
    profile: dict[str, Any],
    stats_filename: str = "membrane_state_stats.csv",
    shapes_filename: str = "membrane_state_shapes.json",
    profile_filename: str = "membrane_recording_profile.json",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / stats_filename, stats_rows)
    write_json(out / shapes_filename, shapes)
    write_json(out / profile_filename, profile)
