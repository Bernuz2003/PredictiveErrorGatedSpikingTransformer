from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda import amp

from pegst.models.snn_layers import reset_spiking_state
from pegst.training.augment import maybe_one_hot_for_soft_target
from pegst.training.losses import classification_loss, spike_l1_regularizer
from pegst.training.metrics import accuracy, confidence_entropy, timestep_accuracy


def forward_qkformer(
    model: nn.Module,
    x: torch.Tensor,
    *,
    return_features: bool = False,
    return_timestep_logits: bool = True,
) -> dict[str, Any]:
    out = model(x, return_features=return_features, return_timestep_logits=return_timestep_logits)
    if isinstance(out, torch.Tensor):
        return {"logits": out}
    return out


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    cfg: dict[str, Any],
    out_dir: str | Path | None = None,
    batch_augment=None,
    mixup_fn=None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_n = 0
    spike_weight = float(cfg.get("training", {}).get("spike_l1_weight", 0.0))
    spike_stages = cfg.get("training", {}).get("spike_l1_stages", ["stage1", "stage2"])
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    use_amp = bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda")
    scaler = amp.GradScaler(enabled=use_amp)
    t_train = cfg.get("training", {}).get("T_train")
    mixup_off_epoch = int(cfg.get("augmentation", {}).get("mixup_off_epoch", 0))
    mixup_enabled = mixup_fn is not None and (mixup_off_epoch <= 0 or epoch < mixup_off_epoch)

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        target_for_loss: torch.Tensor = y
        if train and batch_augment is not None:
            x = batch_augment(x)
        if train and mixup_enabled:
            x, target_for_loss = mixup_fn(x, y)
        elif train:
            target_for_loss = maybe_one_hot_for_soft_target(cfg, y)
        if train and t_train:
            sec_list = torch.randperm(x.shape[1], device=x.device)[: int(t_train)]
            sec_list = torch.sort(sec_list).values
            x = x[:, sec_list]

        reset_spiking_state(model)
        with torch.set_grad_enabled(train), amp.autocast(enabled=use_amp):
            out = forward_qkformer(model, x, return_features=(spike_weight > 0), return_timestep_logits=True)
            logits = out["logits"]
            ce = classification_loss(logits, target_for_loss, label_smoothing=label_smoothing)
            spike_loss = torch.tensor(0.0, device=device)
            if spike_weight > 0:
                spike_loss = spike_l1_regularizer(out.get("features", {}), spike_stages) * spike_weight
            loss = ce + spike_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bsz = y.numel()
        total_n += bsz
        total_loss += float(loss.item()) * bsz
        total_ce += float(ce.item()) * bsz
        total_acc += accuracy(logits.detach(), y, (1,))[0].item() * bsz

    return {
        "epoch": epoch,
        "split": "train" if train else "eval",
        "loss": total_loss / max(1, total_n),
        "ce_loss": total_ce / max(1, total_n),
        "acc1": total_acc / max(1, total_n),
        "num_samples": total_n,
    }


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict[str, Any],
    logits_path: str | Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total_ce = 0.0
    total_n = 0
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    confusion: torch.Tensor | None = None
    per_t_acc_sum: list[float] | None = None
    per_t_conf_sum: list[float] | None = None
    per_t_entropy_sum: list[float] | None = None
    save_logits = bool(cfg.get("profiling", {}).get("save_logits_over_time", False) and logits_path is not None)
    max_logits_batches = int(cfg.get("profiling", {}).get("max_logits_batches", 0))
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device).long()
        reset_spiking_state(model)
        out = forward_qkformer(model, x, return_timestep_logits=True)
        logits = out["logits"]
        ce = classification_loss(logits, y)
        bsz = y.numel()
        total_n += bsz
        total_loss += float(ce.item()) * bsz
        total_ce += float(ce.item()) * bsz
        total_acc += accuracy(logits, y, (1,))[0].item() * bsz
        pred = logits.argmax(dim=1)
        if confusion is None:
            inferred_classes = logits.shape[-1]
            confusion = torch.zeros(num_classes or inferred_classes, num_classes or inferred_classes, dtype=torch.long)
        for true, guess in zip(y.detach().cpu().view(-1), pred.detach().cpu().view(-1), strict=False):
            true_i = int(true.item())
            guess_i = int(guess.item())
            if 0 <= true_i < confusion.shape[0] and 0 <= guess_i < confusion.shape[1]:
                confusion[true_i, guess_i] += 1

        tlog = out["timestep_logits"]
        if save_logits and (max_logits_batches <= 0 or batch_idx < max_logits_batches):
            logits_chunks.append(tlog.detach().cpu())
            label_chunks.append(y.detach().cpu())
        tacc = timestep_accuracy(tlog, y)
        if per_t_acc_sum is None:
            per_t_acc_sum = [0.0] * len(tacc)
            per_t_conf_sum = [0.0] * len(tacc)
            per_t_entropy_sum = [0.0] * len(tacc)
        for t, acc_t in enumerate(tacc):
            per_t_acc_sum[t] += acc_t * bsz
            conf, ent = confidence_entropy(tlog[t])
            per_t_conf_sum[t] += conf.mean().item() * bsz
            per_t_entropy_sum[t] += ent.mean().item() * bsz

    summary = {
        "loss": total_loss / max(1, total_n),
        "ce_loss": total_ce / max(1, total_n),
        "acc1": total_acc / max(1, total_n),
        "num_samples": total_n,
    }
    timestep_rows: list[dict[str, Any]] = []
    if per_t_acc_sum is not None:
        for t in range(len(per_t_acc_sum)):
            timestep_rows.append(
                {
                    "timestep": t + 1,
                    "accuracy": per_t_acc_sum[t] / max(1, total_n),
                    "confidence": per_t_conf_sum[t] / max(1, total_n),
                    "entropy": per_t_entropy_sum[t] / max(1, total_n),
                }
            )

    confusion_rows: list[dict[str, Any]] = []
    if confusion is not None:
        for true_label in range(confusion.shape[0]):
            for pred_label in range(confusion.shape[1]):
                confusion_rows.append(
                    {
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(confusion[true_label, pred_label].item()),
                    }
                )

    if save_logits and logits_chunks:
        logits_path = Path(logits_path)  # type: ignore[arg-type]
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "timestep_logits": torch.cat(logits_chunks, dim=1),
                "labels": torch.cat(label_chunks, dim=0),
            },
            logits_path,
        )
    return summary, timestep_rows, confusion_rows
