from __future__ import annotations

import pytest
import torch

from pegst.models.temporal_predictors import FutureStatePredictor


def test_future_state_predictor_sequence_spatial():
    x = torch.rand(5, 2, 8, 4, 4)
    pred = FutureStatePredictor(8, history=2, spatial=True)
    batch = pred.forward_sequence(x)
    assert batch.prediction.shape == (4, 2, 8, 4, 4)
    assert batch.target.shape == (4, 2, 8, 4, 4)
    assert batch.loss.ndim == 0
    assert torch.isfinite(batch.loss)


def test_future_state_predictor_sequence_vector():
    x = torch.rand(5, 2, 8)
    pred = FutureStatePredictor(8, history=1, spatial=False)
    batch = pred.forward_sequence(x)
    assert batch.prediction.shape == (4, 2, 8)
    assert batch.target.shape == (4, 2, 8)
    assert torch.isfinite(batch.normalized_error)


def test_motion_extrapolation_requires_history_two():
    with pytest.raises(ValueError, match="history"):
        FutureStatePredictor(8, history=1, spatial=True, predictor_type="motion_extrapolation")
