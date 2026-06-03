from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

PredictionLossType = Literal["l1", "mse", "smooth_l1", "normalized_l1", "normalized_mse", "normalized_smooth_l1"]


@dataclass
class PredictionBatch:
    prediction: torch.Tensor
    target: torch.Tensor
    error: torch.Tensor
    loss: torch.Tensor
    timestep_loss: torch.Tensor
    normalized_error: torch.Tensor
    raw_error_mean: torch.Tensor
    target_abs_mean: torch.Tensor
    prediction_abs_mean: torch.Tensor
    base_loss: torch.Tensor
    amplitude_loss: torch.Tensor
    sample_normalized_error: torch.Tensor
    timestep_normalized_error: torch.Tensor
    timestep_raw_error: torch.Tensor
    timestep_target_abs: torch.Tensor
    timestep_prediction_abs: torch.Tensor


class FutureStatePredictor(nn.Module):
    """Causal predictor for future internal SNN targets.

    The target can be a stage activation, input current, pre-reset membrane,
    threshold margin, soft firing probability, or spike tensor. The module
    consumes only current/past states and predicts the next timestep.
    """

    def __init__(
        self,
        channels: int,
        history: int = 1,
        spatial: bool = True,
        predictor_type: str = "conv1x1",
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if history < 1:
            raise ValueError("history must be >= 1")
        if predictor_type == "motion_extrapolation" and history < 2:
            raise ValueError("motion_extrapolation predictor requires history >= 2")
        self.channels = int(channels)
        self.history = int(history)
        self.spatial = bool(spatial)
        self.predictor_type = predictor_type
        hidden = max(channels, int(channels * hidden_ratio))
        in_channels = channels * history
        if spatial:
            if predictor_type == "motion_extrapolation":
                self.motion_a = nn.Parameter(torch.ones(1, channels, 1, 1))
                self.motion_b = nn.Parameter(torch.zeros(1, channels, 1, 1))
                self.net = nn.Identity()
            elif predictor_type == "depthwise_conv":
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=True),
                )
            elif predictor_type == "mlp_conv":
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
                )
            else:
                self.net = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)
        else:
            if predictor_type == "motion_extrapolation":
                self.motion_a = nn.Parameter(torch.ones(1, channels))
                self.motion_b = nn.Parameter(torch.zeros(1, channels))
                self.net = nn.Identity()
            elif predictor_type == "mlp_conv":
                self.net = nn.Sequential(nn.Linear(in_channels, hidden), nn.SiLU(), nn.Linear(hidden, channels))
            else:
                self.net = nn.Linear(in_channels, channels)

    def make_history(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] < 2:
            raise ValueError("Need at least two timesteps for future prediction")
        xs = []
        for t in range(0, x.shape[0] - 1):
            hist = []
            for h in range(self.history):
                idx = max(0, t - self.history + 1 + h)
                hist.append(x[idx])
            xs.append(torch.cat(hist, dim=1 if self.spatial else -1))
        src = torch.stack(xs, dim=0)
        target = x[1:]
        return src, target

    @staticmethod
    def _normalize_for_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        reduce_dims = tuple(range(2, x.dim()))
        scale = x.square().mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(eps)
        return x / scale

    def _motion_extrapolation(self, src: torch.Tensor) -> torch.Tensor:
        current = src[:, :, -self.channels :]
        previous = src[:, :, -2 * self.channels : -self.channels]
        return self.motion_a * current + self.motion_b * (current - previous)

    def forward_sequence(
        self,
        x: torch.Tensor,
        loss_type: PredictionLossType = "smooth_l1",
        stop_gradient_target: bool = True,
        normalize_error: bool = True,
        normalize_loss: bool = False,
        amplitude_loss_weight: float = 0.0,
    ) -> PredictionBatch:
        src, target = self.make_history(x)
        target_for_loss = target.detach() if stop_gradient_target else target
        if self.predictor_type == "motion_extrapolation":
            pred = self._motion_extrapolation(src)
        elif self.spatial:
            tm1, bsz, _, height, width = src.shape
            pred = self.net(src.flatten(0, 1)).reshape(tm1, bsz, self.channels, height, width)
        else:
            tm1, bsz, c_hist = src.shape
            pred = self.net(src.reshape(tm1 * bsz, c_hist)).reshape(tm1, bsz, self.channels)
        base_loss_type = loss_type
        if str(loss_type).startswith("normalized_"):
            normalize_loss = True
            base_loss_type = str(loss_type).removeprefix("normalized_")  # type: ignore[assignment]
        pred_for_loss = self._normalize_for_loss(pred) if normalize_loss else pred
        target_for_base_loss = self._normalize_for_loss(target_for_loss) if normalize_loss else target_for_loss
        if base_loss_type == "mse":
            raw = F.mse_loss(pred_for_loss, target_for_base_loss, reduction="none")
        elif base_loss_type == "l1":
            raw = F.l1_loss(pred_for_loss, target_for_base_loss, reduction="none")
        else:
            raw = F.smooth_l1_loss(pred_for_loss, target_for_base_loss, reduction="none")
        per_sample = raw.flatten(2).mean(dim=2)
        timestep_loss = per_sample.mean(dim=1)
        base_loss = per_sample.mean()
        pred_amp = pred.abs().flatten(2).mean(dim=2)
        target_amp = target_for_loss.abs().flatten(2).mean(dim=2)
        amplitude_loss = (pred_amp - target_amp).abs().mean()
        loss = base_loss + float(amplitude_loss_weight) * amplitude_loss
        error = (pred - target).detach()
        raw_error = error.abs().flatten(2).mean(dim=2)
        target_abs = target.detach().abs().flatten(2).mean(dim=2)
        prediction_abs = pred.detach().abs().flatten(2).mean(dim=2)
        denom = (target_abs + prediction_abs).clamp_min(1e-6)
        sample_normalized_error = raw_error / denom if normalize_error else raw_error
        normalized_error = sample_normalized_error.mean()
        return PredictionBatch(
            prediction=pred,
            target=target,
            error=error,
            loss=loss,
            timestep_loss=timestep_loss.detach(),
            normalized_error=normalized_error.detach(),
            raw_error_mean=raw_error.mean().detach(),
            target_abs_mean=target_abs.mean().detach(),
            prediction_abs_mean=prediction_abs.mean().detach(),
            base_loss=base_loss.detach(),
            amplitude_loss=amplitude_loss.detach(),
            sample_normalized_error=sample_normalized_error.detach(),
            timestep_normalized_error=sample_normalized_error.mean(dim=1).detach(),
            timestep_raw_error=raw_error.mean(dim=1).detach(),
            timestep_target_abs=target_abs.mean(dim=1).detach(),
            timestep_prediction_abs=prediction_abs.mean(dim=1).detach(),
        )
