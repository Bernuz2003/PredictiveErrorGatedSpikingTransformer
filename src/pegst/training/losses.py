from __future__ import annotations

import torch
import torch.nn.functional as F


def spike_l1_regularizer(features: dict[str, torch.Tensor], stages: list[str]) -> torch.Tensor:
    vals = []
    for stage in stages:
        if stage in features:
            vals.append(features[stage].abs().mean())
    if not vals:
        device = next(iter(features.values())).device if features else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(vals).mean()


def classification_loss(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    if target.dim() == 2:
        return soft_target_cross_entropy(logits, target)
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def one_hot_with_smoothing(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    out = torch.full((target.shape[0], num_classes), off_value, device=target.device, dtype=torch.float32)
    out.scatter_(1, target.long().view(-1, 1), on_value)
    return out
