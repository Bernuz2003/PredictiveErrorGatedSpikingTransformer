from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


METRIC_NAMES = [
    "loss",
    "base_loss",
    "amplitude_loss",
    "normalized_error",
    "symmetric_normalized_error",
    "raw_error_mean",
    "target_abs_mean",
    "prediction_abs_mean",
    "prediction_abs_ratio",
]


def transform_sequence(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "reverse":
        return torch.flip(x, dims=[0])
    if mode == "shuffle":
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
    if mode != "normal":
        raise ValueError(f"Unknown temporal mode: {mode}")
    return x


def baseline_prediction(x: torch.Tensor, method: str, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    target = x[1:]
    if method == "zero":
        return torch.zeros_like(target), target
    if method == "copy_previous":
        return x[:-1], target
    if method == "linear_extrapolation":
        current = x[:-1]
        previous = torch.cat([x[:1], x[:-2]], dim=0)
        return current + alpha * (current - previous), target
    raise ValueError(f"Unknown baseline method: {method}")


def normalize_for_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    reduce_dims = tuple(range(2, x.dim()))
    scale = x.square().mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(eps)
    return x / scale


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "smooth_l1",
    normalize_loss: bool = False,
    amplitude_loss_weight: float = 0.0,
) -> dict[str, float]:
    base_loss_type = loss_type
    if loss_type.startswith("normalized_"):
        normalize_loss = True
        base_loss_type = loss_type.removeprefix("normalized_")
    pred_for_loss = normalize_for_loss(prediction) if normalize_loss else prediction
    target_for_loss = normalize_for_loss(target) if normalize_loss else target
    if base_loss_type == "mse":
        raw_loss = F.mse_loss(pred_for_loss, target_for_loss, reduction="none")
    elif base_loss_type == "l1":
        raw_loss = F.l1_loss(pred_for_loss, target_for_loss, reduction="none")
    else:
        raw_loss = F.smooth_l1_loss(pred_for_loss, target_for_loss, reduction="none")
    base_loss = raw_loss.flatten(2).mean(dim=2).mean()
    raw_error = (prediction - target).abs().flatten(2).mean(dim=2)
    target_abs = target.abs().flatten(2).mean(dim=2)
    prediction_abs = prediction.abs().flatten(2).mean(dim=2)
    amplitude_loss = (prediction_abs - target_abs).abs().mean()
    loss = base_loss + float(amplitude_loss_weight) * amplitude_loss
    normalized_error = (raw_error / (target_abs + prediction_abs).clamp_min(1e-6)).mean()
    return {
        "loss": float(loss.detach().item()),
        "base_loss": float(base_loss.detach().item()),
        "amplitude_loss": float(amplitude_loss.detach().item()),
        "normalized_error": float(normalized_error.detach().item()),
        "symmetric_normalized_error": float(normalized_error.detach().item()),
        "raw_error_mean": float(raw_error.mean().detach().item()),
        "target_abs_mean": float(target_abs.mean().detach().item()),
        "prediction_abs_mean": float(prediction_abs.mean().detach().item()),
        "prediction_abs_ratio": float((prediction_abs.mean() / target_abs.mean().clamp_min(1e-12)).detach().item()),
    }


def add_weighted(acc: dict[str, float], metrics: dict[str, float], weight: int) -> None:
    acc["_count"] = acc.get("_count", 0.0) + weight
    for key, value in metrics.items():
        acc[key] = acc.get(key, 0.0) + value * weight


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    count = max(1.0, acc.get("_count", 0.0))
    return {key: value / count for key, value in acc.items() if key != "_count"}


def relative_gain(model_error: float, baseline_error: float) -> float:
    if not math.isfinite(model_error) or not math.isfinite(baseline_error):
        return float("nan")
    return 1.0 - model_error / max(baseline_error, 1e-12)


def temporal_autocorrelation(x: torch.Tensor, lag: int = 1) -> float:
    if x.shape[0] <= lag:
        return float("nan")
    a = x[:-lag].detach().float().flatten(1)
    b = x[lag:].detach().float().flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a.square().mean(dim=1).sqrt() * b.square().mean(dim=1).sqrt()).clamp_min(1e-12)
    corr = (a * b).mean(dim=1) / denom
    return float(corr.mean().item())


def tensor_distribution_metrics(x: torch.Tensor, near_zero_eps: float = 1e-5) -> dict[str, float]:
    y = x.detach().float()
    finite = torch.isfinite(y)
    finite_fraction = float(finite.float().mean().item()) if y.numel() else 1.0
    yf = y[finite] if finite.any() else y.new_zeros(1)
    abs_y = yf.abs()
    return {
        "finite_fraction": finite_fraction,
        "mean": float(yf.mean().item()),
        "std": float(yf.std(unbiased=False).item()) if yf.numel() > 1 else 0.0,
        "abs_mean": float(abs_y.mean().item()),
        "abs_max": float(abs_y.max().item()) if abs_y.numel() else 0.0,
        "near_zero_fraction": float((abs_y <= near_zero_eps).float().mean().item()) if abs_y.numel() else 0.0,
    }


def row_has_no_nan(row: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        value = row.get(key)
        if value is None:
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except Exception:
            return False
    return True
