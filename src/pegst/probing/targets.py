from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from pegst.profiling.internal_state_collector import InternalStateCollector, RecordingProfile, stage_from_module_name

INTERNAL_TARGET_MAP = {
    "input_current": "input_current",
    "pre_membrane": "pre_membrane",
    "post_reset_membrane": "post_reset_membrane",
    "threshold_margin": "threshold_margin",
    "soft_firing_prob": "soft_firing_prob",
    "spike_output": "spike",
    "spike": "spike",
}

LATENT_FEATURES = ("patch_embed1", "stage1", "patch_embed2", "stage2")


@dataclass
class TargetTensor:
    target: str
    stage: str
    layer: str
    tensor: torch.Tensor


def predictor_tensor(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 5:
        return x, True
    if x.dim() == 4:
        return x.unsqueeze(-1), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"Unsupported target tensor shape for prediction: {tuple(x.shape)}")


def select_target_item(items: list[TargetTensor], target: str, stage: str, layer: str) -> TargetTensor | None:
    for item in items:
        if item.target == target and item.stage == stage and item.layer == layer:
            return item
    return None


def internal_record_targets(targets: Iterable[str]) -> list[str]:
    out = []
    for target in targets:
        if target in INTERNAL_TARGET_MAP:
            out.append(INTERNAL_TARGET_MAP[target])
    return list(dict.fromkeys(out))


def _stage_allowed(stage: str, stages: set[str]) -> bool:
    return not stages or stage in stages


def collect_targets_from_batch(
    model: nn.Module,
    x: torch.Tensor,
    *,
    targets: Iterable[str],
    stages: Iterable[str] | None = None,
    layer_patterns: Iterable[str] | None = None,
    soft_firing_temperature: float = 0.25,
    to_cpu: bool = False,
) -> tuple[list[TargetTensor], dict[str, Any], RecordingProfile]:
    target_list = list(targets)
    stage_set = set(stages or [])
    need_internal = any(target in INTERNAL_TARGET_MAP for target in target_list)
    need_latent = "latent_post_stage" in target_list
    record_targets = internal_record_targets(target_list)
    collector = InternalStateCollector(
        model,
        targets=record_targets,
        stages=stage_set,
        layer_patterns=layer_patterns,
        to_cpu=to_cpu,
        soft_firing_temperature=soft_firing_temperature,
    )
    if need_internal:
        out, internal_states, profile = collector.forward_and_collect(
            x,
            return_features=need_latent,
            return_timestep_logits=True,
        )
    else:
        out = model(x, return_features=True, return_timestep_logits=True)
        internal_states = {}
        profile = RecordingProfile(targets=[])
    features = out.get("features", {}) if isinstance(out, dict) else {}
    items: list[TargetTensor] = []

    if "latent_post_stage" in target_list:
        for name in LATENT_FEATURES:
            if name not in features or not _stage_allowed(name, stage_set):
                continue
            tensor = features[name].detach()
            if to_cpu:
                tensor = tensor.cpu()
            items.append(TargetTensor(target="latent_post_stage", stage=name, layer=name, tensor=tensor))

    for layer, target_map in sorted(internal_states.items()):
        stage = stage_from_module_name(layer)
        if not _stage_allowed(stage, stage_set):
            continue
        for requested, recorded in INTERNAL_TARGET_MAP.items():
            if requested not in target_list or recorded not in target_map:
                continue
            public_target = "spike_output" if requested == "spike_output" else requested
            items.append(TargetTensor(target=public_target, stage=stage, layer=layer, tensor=target_map[recorded].detach()))
    return items, out, profile
