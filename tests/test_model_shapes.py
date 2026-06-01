from __future__ import annotations

import torch

from pegst.models.predictive_qkformer import build_predictive_qkformer


def test_predictive_model_shapes():
    cfg = {
        "model": {"img_size_h": 16, "img_size_w": 16, "in_channels": 2, "num_classes": 4, "embed_dims": 16, "num_heads": 2, "T": 2},
        "predictive": {"enabled": True, "stages": ["stage1"], "history": 1, "loss_weight": 0.01},
        "modulation": {"enabled": False},
    }
    model = build_predictive_qkformer(cfg)
    x = torch.rand(2, 2, 2, 16, 16)
    out = model(x, return_aux=True, return_features=True, return_timestep_logits=True)
    assert out["logits"].shape == (2, 4)
    assert out["timestep_logits"].shape == (2, 2, 4)
    assert out["aux_loss"].ndim == 0
    assert "stage1" in out["features"]


def test_disabled_predictive_path_has_no_extra_parameters():
    cfg = {
        "model": {"img_size_h": 16, "img_size_w": 16, "in_channels": 2, "num_classes": 4, "embed_dims": 16, "num_heads": 2, "T": 2},
        "predictive": {"enabled": False, "stage": "stage1", "history": 1},
        "modulation": {"enabled": False, "stage": "stage1"},
    }
    model = build_predictive_qkformer(cfg)
    assert len(model.predictors) == 0
    assert len(model.modulators) == 0
