from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .synthetic import SyntheticGestureConfig, SyntheticGestureDataset


class TensorShapeAdapter(Dataset):
    """Ensure samples are returned as float [T, C, H, W]."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x = torch.as_tensor(x).float()
        # SpikingJelly frame datasets usually return [T, C, H, W].
        if x.dim() != 4:
            raise ValueError(f"Expected [T,C,H,W] sample, got {tuple(x.shape)}")
        return x, int(y)


def _build_spikingjelly_dvs128gesture(root: str, T: int, split: str, split_by: str = "number") -> Dataset:
    try:
        from spikingjelly.datasets import dvs128_gesture
    except Exception as exc:  # pragma: no cover - external dependency
        raise ImportError(
            "DVS128 Gesture requires SpikingJelly. Use dataset.name=synthetic for smoke tests, "
            "or install the Singularity container dependencies."
        ) from exc
    train = split == "train"
    try:
        ds = dvs128_gesture.DVS128Gesture(root=root, train=train, data_type="frame", frames_number=T, split_by=split_by)
    except TypeError:
        ds = dvs128_gesture.DVS128Gesture(root=root, train=train, data_type="frame", frames_number=T)
    return TensorShapeAdapter(ds)


def _build_spikingjelly_cifar10dvs(root: str, T: int, split: str, split_by: str = "number") -> Dataset:
    try:
        from spikingjelly.datasets import cifar10_dvs
    except Exception as exc:  # pragma: no cover
        raise ImportError("CIFAR10-DVS requires SpikingJelly.") from exc
    ds = cifar10_dvs.CIFAR10DVS(root=root, data_type="frame", frames_number=T, split_by=split_by)
    n_train = int(0.9 * len(ds))
    n_test = len(ds) - n_train
    gen = torch.Generator().manual_seed(2021)
    train_set, test_set = random_split(ds, [n_train, n_test], generator=gen)
    return TensorShapeAdapter(train_set if split == "train" else test_set)


def build_dataset(cfg: dict[str, Any], split: str) -> Dataset:
    name = cfg.get("name", "synthetic").lower()
    T = int(cfg.get("T", cfg.get("timesteps", 8)))
    if name == "synthetic":
        scfg = SyntheticGestureConfig(
            num_samples=int(cfg.get("num_samples", 256)),
            T=T,
            height=int(cfg.get("height", 64)),
            width=int(cfg.get("width", 64)),
            num_classes=int(cfg.get("num_classes", 4)),
            noise_prob=float(cfg.get("noise_prob", 0.002)),
            bar_size=int(cfg.get("bar_size", 5)),
            seed=int(cfg.get("seed", 2021)),
        )
        return SyntheticGestureDataset(scfg, split=split)
    root = str(Path(cfg["root"]).expanduser())
    split_by = cfg.get("split_by", "number")
    if name in {"dvs128gesture", "dvs128_gesture", "gesture"}:
        return _build_spikingjelly_dvs128gesture(root, T, split, split_by)
    if name in {"cifar10dvs", "cifar10-dvs", "cifar10_dvs"}:
        return _build_spikingjelly_cifar10dvs(root, T, split, split_by)
    raise ValueError(f"Unknown dataset: {name}")


def build_dataloader(cfg: dict[str, Any], split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=(split == "train"),
        num_workers=int(cfg.get("workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=bool(cfg.get("drop_last", False) and split == "train"),
    )
