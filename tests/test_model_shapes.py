from __future__ import annotations

import torch

from pegst.models.qkformer import build_qkformer
from pegst.profiling.internal_state_collector import InternalStateCollector


def tiny_cfg():
    return {
        "img_size_h": 16,
        "img_size_w": 16,
        "in_channels": 2,
        "num_classes": 4,
        "embed_dims": 16,
        "num_heads": 2,
        "T": 2,
    }


def test_qkformer_model_shapes():
    model = build_qkformer(tiny_cfg())
    x = torch.rand(2, 2, 2, 16, 16)
    out = model(x, return_features=True, return_timestep_logits=True)
    assert out["logits"].shape == (2, 4)
    assert out["timestep_logits"].shape == (2, 2, 4)
    assert "stage1" in out["features"]
    assert "stage2" in out["features"]


def test_internal_state_recorder_is_opt_in():
    model = build_qkformer(tiny_cfg())
    x = torch.rand(2, 2, 2, 16, 16)
    _ = model(x, return_features=True, return_timestep_logits=True)
    assert all(not getattr(m, "last_internal_states", {}) for m in model.modules())

    collector = InternalStateCollector(model, targets=["input_current", "pre_membrane", "spike"], stages=["stage1"])
    _, states, profile = collector.forward_and_collect(x, return_features=True, return_timestep_logits=True)
    assert profile.modules_recorded > 0
    assert any("pre_membrane" in target_map for target_map in states.values())
