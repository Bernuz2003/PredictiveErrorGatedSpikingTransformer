#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pegst.models.predictive_qkformer import build_predictive_qkformer
from pegst.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = build_predictive_qkformer(cfg)
    x = torch.rand(2, cfg["dataset"]["T"], cfg["model"]["in_channels"], cfg["dataset"]["height"], cfg["dataset"]["width"])
    out = model(x, return_aux=True, return_features=True, return_timestep_logits=True)
    assert out["logits"].shape == (2, cfg["model"]["num_classes"]), out["logits"].shape
    assert out["timestep_logits"].shape[0] == cfg["dataset"]["T"]
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["aux_loss"]).all()
    print("smoke_test: ok")


if __name__ == "__main__":
    main()
