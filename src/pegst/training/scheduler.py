from __future__ import annotations

import math
from typing import Any

import torch


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        base_lr: float,
        min_lr: float = 2e-5,
        warmup_epochs: int = 10,
        warmup_lr: float = 1e-5,
    ) -> None:
        self.optimizer = optimizer
        self.epochs = max(1, int(epochs))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_lr = float(warmup_lr)
        self.last_epoch = -1

    def lr_at(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return self.base_lr
            alpha = epoch / float(self.warmup_epochs - 1)
            return self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
        decay_epochs = max(1, self.epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / decay_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    def step(self, epoch: int) -> float:
        self.last_epoch = int(epoch)
        lr = self.lr_at(self.last_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_epoch = int(state.get("last_epoch", -1))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epochs: int) -> WarmupCosineScheduler | None:
    sched_cfg = cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none")
    if sched_type in {None, "none", "constant"}:
        return None
    if sched_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
    opt_cfg = cfg.get("optimizer", {})
    return WarmupCosineScheduler(
        optimizer=optimizer,
        epochs=epochs,
        base_lr=float(opt_cfg.get("lr", 1e-3)),
        min_lr=float(sched_cfg.get("min_lr", 2e-5)),
        warmup_epochs=int(sched_cfg.get("warmup_epochs", 10)),
        warmup_lr=float(sched_cfg.get("warmup_lr", 1e-5)),
    )
