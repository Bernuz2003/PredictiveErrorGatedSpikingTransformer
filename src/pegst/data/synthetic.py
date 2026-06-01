from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticGestureConfig:
    num_samples: int = 256
    T: int = 8
    height: int = 64
    width: int = 64
    num_classes: int = 4
    noise_prob: float = 0.002
    bar_size: int = 5
    seed: int = 2021


class SyntheticGestureDataset(Dataset):
    """Small controllable event-like dataset for smoke tests and pretests.

    Classes encode motion direction of a sparse bar/dot pattern, making it useful
    to validate temporal prediction, reverse-time ablations, and early-exit logic
    without downloading real DVS data.
    """

    def __init__(self, cfg: SyntheticGestureConfig, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        frac = 0.8 if split == "train" else 0.2
        self.n = max(1, int(cfg.num_samples * frac))
        self.offset = 0 if split == "train" else int(cfg.num_samples * 0.8)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.offset + idx
        g = torch.Generator().manual_seed(self.cfg.seed + real_idx)
        label = int(torch.randint(0, self.cfg.num_classes, (1,), generator=g).item())
        x = torch.zeros(self.cfg.T, 2, self.cfg.height, self.cfg.width)
        margin = self.cfg.bar_size + 2
        start_y = int(torch.randint(margin, self.cfg.height - margin, (1,), generator=g).item())
        start_x = int(torch.randint(margin, self.cfg.width - margin, (1,), generator=g).item())
        speed = max(1, min(self.cfg.height, self.cfg.width) // (self.cfg.T + 6))
        for t in range(self.cfg.T):
            if label == 0:  # left -> right
                y, xx = start_y, margin + t * speed
                pol = 1
            elif label == 1:  # right -> left
                y, xx = start_y, self.cfg.width - margin - 1 - t * speed
                pol = 0
            elif label == 2:  # top -> bottom
                y, xx = margin + t * speed, start_x
                pol = 1
            else:  # bottom -> top
                y, xx = self.cfg.height - margin - 1 - t * speed, start_x
                pol = 0
            y = int(max(margin, min(self.cfg.height - margin - 1, y)))
            xx = int(max(margin, min(self.cfg.width - margin - 1, xx)))
            r = self.cfg.bar_size // 2
            x[t, pol, y - r : y + r + 1, xx - r : xx + r + 1] = 1.0
            # opposite-polarity trailing event for DVS-like contrast change
            x[t, 1 - pol, max(0, y - r - 1) : y - r, xx - r : xx + r + 1] = 1.0
        if self.cfg.noise_prob > 0:
            noise = torch.rand(x.shape, generator=g) < self.cfg.noise_prob
            x = torch.clamp(x + noise.float(), 0, 1)
        return x, label
