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
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)
