from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


def compatible_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    if model_keys & state_keys:
        return state
    stripped_backbone = _strip_prefix(state, "backbone.")
    if stripped_backbone and (model_keys & set(stripped_backbone.keys())):
        return stripped_backbone
    prefixed_backbone = {f"backbone.{k}": v for k, v in state.items()}
    if model_keys & set(prefixed_backbone.keys()):
        return prefixed_backbone
    return state


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    raw_state = extract_model_state(ckpt)
    state = compatible_state_dict(model, raw_state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {
        "checkpoint": str(checkpoint_path),
        "strict": strict,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "raw_num_keys": len(raw_state),
        "loaded_num_keys": len(state),
    }
