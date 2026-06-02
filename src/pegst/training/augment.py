from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from pegst.training.losses import one_hot_with_smoothing


def _torchvision_functional():
    try:
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.transforms import RandomErasing
    except Exception as exc:  # pragma: no cover - optional DVS dependency
        raise ImportError("QKFormer-style augmentation requires torchvision.") from exc
    return F, InterpolationMode, RandomErasing


def _apply_op(img: Tensor, op_name: str, magnitude: float, interpolation: Any, fill: list[float] | None) -> Tensor:
    F, _, _ = _torchvision_functional()
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0], interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0, shear=[0.0, 0.0], interpolation=interpolation, fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0, shear=[0.0, 0.0], interpolation=interpolation, fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"Unknown augmentation op: {op_name}")
    return img


class SNNAugmentWide(nn.Module):
    """QKFormer DVS augmentation copied in spirit from the official repo."""

    def __init__(self, num_magnitude_bins: int = 31, fill: list[float] | None = None) -> None:
        super().__init__()
        _, InterpolationMode, RandomErasing = _torchvision_functional()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = InterpolationMode.NEAREST
        self.fill = fill
        self.cutout = RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))

    def _augmentation_space(self, num_bins: int) -> dict[str, tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True),
        }

    def forward(self, img: Tensor) -> Tensor:
        fill = self.fill
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * img.shape[-3]
        elif fill is not None:
            fill = [float(f) for f in fill]
        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        if op_name == "Cutout":
            return self.cutout(img)
        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


class QKFormerBatchAugment:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.random_horizontal_flip = bool(cfg.get("random_horizontal_flip", False))
        self.flip_prob = float(cfg.get("horizontal_flip_prob", 0.5))
        self.snn_augment_wide = SNNAugmentWide() if bool(cfg.get("snn_augment_wide", False)) else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.random_horizontal_flip and self.snn_augment_wide is None:
            return x
        samples = []
        for sample in x:
            if self.random_horizontal_flip and torch.rand((), device=sample.device).item() < self.flip_prob:
                sample = torch.flip(sample, dims=[-1])
            if self.snn_augment_wide is not None:
                sample = self.snn_augment_wide(sample)
            samples.append(sample)
        return torch.stack(samples, dim=0)


class BatchMixup:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.5,
        prob: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.prob = float(prob)
        self.label_smoothing = float(label_smoothing)
        self.enabled = self.mixup_alpha > 0.0 and self.prob > 0.0

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_soft = one_hot_with_smoothing(target, self.num_classes, self.label_smoothing)
        if not self.enabled or torch.rand((), device=x.device).item() > self.prob:
            return x, target_soft
        beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = beta.sample().to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(x.shape[0], device=x.device)
        mixed_x = x * lam + x[perm] * (1.0 - lam)
        mixed_target = target_soft * lam + target_soft[perm] * (1.0 - lam)
        return mixed_x, mixed_target


def build_batch_augment(cfg: dict[str, Any]) -> QKFormerBatchAugment | None:
    aug = cfg.get("augmentation", {})
    if not (aug.get("random_horizontal_flip", False) or aug.get("snn_augment_wide", False)):
        return None
    return QKFormerBatchAugment(aug)


def build_mixup(cfg: dict[str, Any]) -> BatchMixup | None:
    aug = cfg.get("augmentation", {})
    mixup_alpha = float(aug.get("mixup", 0.0))
    if mixup_alpha <= 0.0:
        return None
    return BatchMixup(
        num_classes=int(cfg.get("model", {}).get("num_classes", 1000)),
        mixup_alpha=mixup_alpha,
        prob=float(aug.get("mixup_prob", 1.0)),
        label_smoothing=float(cfg.get("training", {}).get("label_smoothing", aug.get("smoothing", 0.0))),
    )


def maybe_one_hot_for_soft_target(cfg: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
    loss_cfg = cfg.get("loss", {})
    train_loss = loss_cfg.get("train", "")
    if train_loss != "soft_target_cross_entropy":
        return target
    return one_hot_with_smoothing(
        target,
        int(cfg.get("model", {}).get("num_classes", 1000)),
        float(cfg.get("training", {}).get("label_smoothing", 0.0)),
    )
