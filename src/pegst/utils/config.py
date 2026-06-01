from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = normalize_config(cfg)
    return cfg


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if "timesteps" in cfg:
        cfg.setdefault("model", {})
        cfg.setdefault("dataset", {})
        cfg["model"].setdefault("T", cfg["timesteps"])
        cfg["dataset"].setdefault("T", cfg["timesteps"])
    for section in ("predictive", "modulation"):
        if isinstance(cfg.get(section), dict):
            block = cfg[section]
            if "stage" in block and "stages" not in block:
                block["stages"] = [block["stage"]]
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
