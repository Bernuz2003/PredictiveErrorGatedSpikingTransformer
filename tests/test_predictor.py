from __future__ import annotations

import torch

from pegst.models.predictive_modules import FutureStatePredictor, ErrorGateModulator


def test_future_state_predictor_sequence():
    x = torch.rand(5, 2, 8, 4, 4)
    pred = FutureStatePredictor(8, history=2, spatial=True)
    batch = pred.forward_sequence(x)
    assert batch.prediction.shape == (4, 2, 8, 4, 4)
    assert batch.target.shape == (4, 2, 8, 4, 4)
    assert batch.loss.ndim == 0


def test_error_gate_modulator():
    x = torch.rand(4, 2, 8, 4, 4)
    mod = ErrorGateModulator(8, spatial=True)
    y, stats = mod(x, x * 0.5)
    assert y.shape == x.shape
    assert "gate_mean" in stats


def test_threshold_modulator_shape():
    x = torch.rand(4, 2, 8, 4, 4)
    mod = ErrorGateModulator(8, spatial=True, mode="threshold_modulation")
    y, stats = mod(x, x * 0.5)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert "error_mean" in stats
