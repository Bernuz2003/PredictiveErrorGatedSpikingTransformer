#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pegst.models.qkformer import build_qkformer
from pegst.profiling.internal_state_collector import InternalStateCollector
from pegst.utils.config import load_config
from pegst.utils.progress import log


def main() -> None:
    log("smoke test started")
    cfg = load_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = build_qkformer(cfg["model"])
    x = torch.rand(2, cfg["dataset"]["T"], cfg["model"]["in_channels"], cfg["dataset"]["height"], cfg["dataset"]["width"])
    log("checking QKFormer forward shapes")
    out = model(x, return_features=True, return_timestep_logits=True)
    assert out["logits"].shape == (2, cfg["model"]["num_classes"]), out["logits"].shape
    assert out["timestep_logits"].shape[0] == cfg["dataset"]["T"]
    assert torch.isfinite(out["logits"]).all()

    log("checking opt-in internal-state recorder")
    collector = InternalStateCollector(model, targets=["input_current", "pre_membrane", "spike"], stages=["stage1"])
    _, states, profile = collector.forward_and_collect(x, return_features=True, return_timestep_logits=True)
    assert profile.modules_recorded > 0
    assert any("pre_membrane" in targets for targets in states.values())
    log("smoke_test: ok")


if __name__ == "__main__":
    main()
