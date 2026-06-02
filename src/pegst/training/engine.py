from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda import amp

from pegst.models.snn_layers import reset_spiking_state
from pegst.training.augment import maybe_one_hot_for_soft_target
from pegst.training.losses import classification_loss, spike_l1_regularizer
from pegst.training.metrics import accuracy, confidence_entropy, timestep_accuracy


def prediction_loss_scale(epoch: int, cfg: dict[str, Any]) -> float:
    pred_cfg = cfg.get("predictive", {})
    if not pred_cfg.get("enabled", False):
        return 1.0
    warmup_epochs = int(pred_cfg.get("warmup_epochs", 0) or 0)
    ramp_epochs = int(pred_cfg.get("ramp_epochs", 0) or 0)
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - warmup_epochs) / float(ramp_epochs)
    return max(0.0, min(1.0, progress))


def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norm_sq = torch.tensor(0.0, device=loss.device)
    for grad in grads:
        if grad is not None:
            norm_sq = norm_sq + grad.detach().float().square().sum()
    return float(norm_sq.sqrt().item())


def _shared_grad_params(model: nn.Module) -> list[torch.nn.Parameter]:
    shared = getattr(model, "backbone", model)
    return [p for p in shared.parameters() if p.requires_grad]


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
    train_cfg = cfg.get("training", {})
    if train and (train_cfg.get("freeze_backbone", False) or train_cfg.get("train_predictors_only", False)) and hasattr(model, "backbone"):
        model.backbone.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_acc = 0.0
    total_n = 0
    pred_error_sums: dict[str, float] = defaultdict(float)
    pred_error_counts: dict[str, int] = defaultdict(int)
    pred_loss_sums: dict[str, float] = defaultdict(float)
    pred_loss_counts: dict[str, int] = defaultdict(int)
    pred_base_loss_sums: dict[str, float] = defaultdict(float)
    pred_amplitude_loss_sums: dict[str, float] = defaultdict(float)
    pred_raw_sums: dict[str, float] = defaultdict(float)
    pred_target_abs_sums: dict[str, float] = defaultdict(float)
    pred_prediction_abs_sums: dict[str, float] = defaultdict(float)
    modulation_sums: dict[tuple[str, str], float] = defaultdict(float)
    modulation_counts: dict[tuple[str, str], int] = defaultdict(int)
    spike_weight = float(cfg.get("training", {}).get("spike_l1_weight", 0.0))
    spike_stages = cfg.get("training", {}).get("spike_l1_stages", ["stage1", "stage2"])
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    use_amp = bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda")
    scaler = amp.GradScaler(enabled=use_amp)
    t_train = cfg.get("training", {}).get("T_train")
    mixup_off_epoch = int(cfg.get("augmentation", {}).get("mixup_off_epoch", 0))
    mixup_enabled = mixup_fn is not None and (mixup_off_epoch <= 0 or epoch < mixup_off_epoch)
    pred_loss_scale = prediction_loss_scale(epoch, cfg)
    grad_ratio_every = int(cfg.get("training", {}).get("grad_ratio_every", cfg.get("predictive", {}).get("grad_ratio_every", 0)) or 0)
    grad_params = _shared_grad_params(model) if grad_ratio_every > 0 else []
    grad_norm_ce_sum = 0.0
    grad_norm_pred_sum = 0.0
    grad_ratio_sum = 0.0
    grad_ratio_count = 0

    for batch_idx, (x, y) in enumerate(loader):
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
            out = model(x, return_aux=True, return_features=(spike_weight > 0), return_timestep_logits=True)
            logits = out["logits"]
            ce = classification_loss(logits, target_for_loss, label_smoothing=label_smoothing)
            aux = out.get("aux_loss", torch.tensor(0.0, device=device))
            aux_for_loss = aux * pred_loss_scale
            spike_loss = torch.tensor(0.0, device=device)
            if spike_weight > 0:
                spike_loss = spike_l1_regularizer(out.get("features", {}), spike_stages) * spike_weight
            loss = ce + aux_for_loss + spike_loss
        if train and grad_ratio_every > 0 and grad_params and batch_idx % grad_ratio_every == 0 and aux_for_loss.requires_grad:
            grad_norm_ce = _grad_norm(ce, grad_params)
            grad_norm_pred = _grad_norm(aux_for_loss, grad_params)
            grad_norm_ce_sum += grad_norm_ce
            grad_norm_pred_sum += grad_norm_pred
            grad_ratio_sum += grad_norm_pred / max(grad_norm_ce, 1e-12)
            grad_ratio_count += 1
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
        total_loss += loss.item() * bsz
        total_ce += ce.item() * bsz
        total_aux += float(aux_for_loss.item()) * bsz
        total_acc += accuracy(logits.detach(), y, (1,))[0].item() * bsz
        for stage, err in out.get("prediction_normalized_errors", {}).items():
            pred_error_sums[stage] += float(err.item()) * bsz
            pred_error_counts[stage] += bsz
        for stage, pred_loss in out.get("prediction_stage_losses", {}).items():
            pred_loss_sums[stage] += float(pred_loss.item()) * bsz
            pred_loss_counts[stage] += bsz
        for stage, pred_loss in out.get("prediction_base_losses", {}).items():
            pred_base_loss_sums[stage] += float(pred_loss.item()) * bsz
        for stage, pred_loss in out.get("prediction_amplitude_losses", {}).items():
            pred_amplitude_loss_sums[stage] += float(pred_loss.item()) * bsz
        for stage, batch in out.get("prediction_batches", {}).items():
            pred_raw_sums[stage] += float(batch.raw_error_mean.item()) * bsz
            pred_target_abs_sums[stage] += float(batch.target_abs_mean.item()) * bsz
            pred_prediction_abs_sums[stage] += float(batch.prediction_abs_mean.item()) * bsz
        for stage, stats in out.get("modulation_stats", {}).items():
            for name, value in stats.items():
                key = (stage, name)
                modulation_sums[key] += float(value.item()) * bsz
                modulation_counts[key] += bsz

    pred_error_mean = 0.0
    if pred_error_sums:
        pred_error_mean = sum(pred_error_sums.values()) / max(1, sum(pred_error_counts.values()))
    metrics = {
        "epoch": epoch,
        "split": "train" if train else "eval",
        "loss": total_loss / max(1, total_n),
        "ce_loss": total_ce / max(1, total_n),
        "aux_loss": total_aux / max(1, total_n),
        "acc1": total_acc / max(1, total_n),
        "prediction_error_mean": pred_error_mean,
        "prediction_loss_scale": pred_loss_scale,
        "num_samples": total_n,
    }
    if grad_ratio_count > 0:
        metrics["grad_norm_ce"] = grad_norm_ce_sum / grad_ratio_count
        metrics["grad_norm_pred"] = grad_norm_pred_sum / grad_ratio_count
        metrics["grad_ratio_pred_ce"] = grad_ratio_sum / grad_ratio_count
    for stage, val in pred_error_sums.items():
        metrics[f"prediction_error_{stage}"] = val / max(1, pred_error_counts[stage])
    for stage, val in pred_loss_sums.items():
        metrics[f"prediction_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for stage, val in pred_base_loss_sums.items():
        metrics[f"prediction_base_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for stage, val in pred_amplitude_loss_sums.items():
        metrics[f"prediction_amplitude_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for stage, val in pred_raw_sums.items():
        metrics[f"prediction_raw_error_{stage}"] = val / max(1, pred_loss_counts[stage])
        metrics[f"prediction_target_abs_{stage}"] = pred_target_abs_sums[stage] / max(1, pred_loss_counts[stage])
        metrics[f"prediction_abs_{stage}"] = pred_prediction_abs_sums[stage] / max(1, pred_loss_counts[stage])
        metrics[f"prediction_abs_ratio_{stage}"] = metrics[f"prediction_abs_{stage}"] / max(metrics[f"prediction_target_abs_{stage}"], 1e-12)
    for (stage, name), val in modulation_sums.items():
        metrics[f"modulation_{stage}_{name}"] = val / max(1, modulation_counts[(stage, name)])
    return metrics


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict[str, Any],
    logits_path: str | Path | None = None,
    sample_scores_path: str | Path | None = None,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_n = 0
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    confusion: torch.Tensor | None = None
    pred_error_sums: dict[str, float] = defaultdict(float)
    pred_error_counts: dict[str, int] = defaultdict(int)
    pred_raw_sums: dict[str, float] = defaultdict(float)
    pred_target_abs_sums: dict[str, float] = defaultdict(float)
    pred_prediction_abs_sums: dict[str, float] = defaultdict(float)
    pred_loss_sums: dict[str, float] = defaultdict(float)
    pred_base_loss_sums: dict[str, float] = defaultdict(float)
    pred_amplitude_loss_sums: dict[str, float] = defaultdict(float)
    pred_loss_counts: dict[str, int] = defaultdict(int)
    pred_timestep_sums: dict[tuple[str, int, str], float] = defaultdict(float)
    pred_timestep_counts: dict[tuple[str, int], int] = defaultdict(int)
    modulation_sums: dict[tuple[str, str], float] = defaultdict(float)
    modulation_counts: dict[tuple[str, str], int] = defaultdict(int)
    per_t_acc_sum: list[float] | None = None
    per_t_conf_sum: list[float] | None = None
    per_t_entropy_sum: list[float] | None = None
    save_logits = bool(cfg.get("profiling", {}).get("save_logits_over_time", False) and logits_path is not None)
    max_logits_batches = int(cfg.get("profiling", {}).get("max_logits_batches", 0))
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    sample_score_rows: list[dict[str, Any]] = []
    sample_offset = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device).long()
        reset_spiking_state(model)
        out = model(x, return_aux=True, return_timestep_logits=True)
        logits = out["logits"]
        ce = classification_loss(logits, y)
        aux = out.get("aux_loss", torch.tensor(0.0, device=device))
        loss = ce + aux if cfg.get("loss", {}).get("eval", "cross_entropy") != "cross_entropy" else ce
        bsz = y.numel()
        total_n += bsz
        total_loss += loss.item() * bsz
        total_ce += ce.item() * bsz
        total_aux += float(aux.item()) * bsz
        total_acc += accuracy(logits, y, (1,))[0].item() * bsz
        pred = logits.argmax(dim=1)
        final_conf, final_entropy = confidence_entropy(logits)
        if confusion is None:
            inferred_classes = logits.shape[-1]
            confusion = torch.zeros(num_classes or inferred_classes, num_classes or inferred_classes, dtype=torch.long)
        for true, guess in zip(y.detach().cpu().view(-1), pred.detach().cpu().view(-1), strict=False):
            true_i = int(true.item())
            guess_i = int(guess.item())
            if 0 <= true_i < confusion.shape[0] and 0 <= guess_i < confusion.shape[1]:
                confusion[true_i, guess_i] += 1
        for stage, err in out.get("prediction_normalized_errors", {}).items():
            pred_error_sums[stage] += float(err.item()) * bsz
            pred_error_counts[stage] += bsz
        for stage, pred_loss in out.get("prediction_stage_losses", {}).items():
            pred_loss_sums[stage] += float(pred_loss.item()) * bsz
            pred_loss_counts[stage] += bsz
        for stage, pred_loss in out.get("prediction_base_losses", {}).items():
            pred_base_loss_sums[stage] += float(pred_loss.item()) * bsz
        for stage, pred_loss in out.get("prediction_amplitude_losses", {}).items():
            pred_amplitude_loss_sums[stage] += float(pred_loss.item()) * bsz
        for stage, batch in out.get("prediction_batches", {}).items():
            pred_raw_sums[stage] += float(batch.raw_error_mean.item()) * bsz
            pred_target_abs_sums[stage] += float(batch.target_abs_mean.item()) * bsz
            pred_prediction_abs_sums[stage] += float(batch.prediction_abs_mean.item()) * bsz
            for t in range(batch.timestep_normalized_error.shape[0]):
                target_timestep = t + 2
                count_key = (stage, target_timestep)
                pred_timestep_counts[count_key] += bsz
                timestep_values = {
                    "loss": batch.timestep_loss[t],
                    "raw_error_mean": batch.timestep_raw_error[t],
                    "normalized_error": batch.timestep_normalized_error[t],
                    "symmetric_normalized_error": batch.timestep_normalized_error[t],
                    "target_abs_mean": batch.timestep_target_abs[t],
                    "prediction_abs_mean": batch.timestep_prediction_abs[t],
                }
                for name, value in timestep_values.items():
                    pred_timestep_sums[(stage, target_timestep, name)] += float(value.item()) * bsz
            if sample_scores_path is not None:
                sample_errors = batch.sample_normalized_error.transpose(0, 1).detach().cpu()
                mean_errors = sample_errors.mean(dim=1)
                for j in range(bsz):
                    row: dict[str, Any] = {
                        "sample_id": sample_offset + j,
                        "label": int(y[j].detach().cpu().item()),
                        "predicted_label": int(pred[j].detach().cpu().item()),
                        "correct": int((pred[j] == y[j]).detach().cpu().item()),
                        "final_confidence": float(final_conf[j].detach().cpu().item()),
                        "final_entropy": float(final_entropy[j].detach().cpu().item()),
                        "stage": stage,
                        "mean_prediction_error": float(mean_errors[j].item()),
                    }
                    for t in range(sample_errors.shape[1]):
                        row[f"prediction_error_t{t + 2}"] = float(sample_errors[j, t].item())
                    sample_score_rows.append(row)
        for stage, stats in out.get("modulation_stats", {}).items():
            for name, value in stats.items():
                key = (stage, name)
                modulation_sums[key] += float(value.item()) * bsz
                modulation_counts[key] += bsz
        tlog = out["timestep_logits"]
        if save_logits and (max_logits_batches <= 0 or batch_idx < max_logits_batches):
            logits_chunks.append(tlog.detach().cpu())
            label_chunks.append(y.detach().cpu())
        tacc = timestep_accuracy(tlog, y)
        if per_t_acc_sum is None:
            per_t_acc_sum = [0.0] * len(tacc)
            per_t_conf_sum = [0.0] * len(tacc)
            per_t_entropy_sum = [0.0] * len(tacc)
        for t, a in enumerate(tacc):
            per_t_acc_sum[t] += a * bsz
            conf, ent = confidence_entropy(tlog[t])
            per_t_conf_sum[t] += conf.mean().item() * bsz
            per_t_entropy_sum[t] += ent.mean().item() * bsz
        sample_offset += bsz

    summary = {
        "loss": total_loss / max(1, total_n),
        "ce_loss": total_ce / max(1, total_n),
        "aux_loss": total_aux / max(1, total_n),
        "acc1": total_acc / max(1, total_n),
        "prediction_error_mean": (
            sum(pred_error_sums.values()) / max(1, sum(pred_error_counts.values())) if pred_error_sums else 0.0
        ),
        "num_samples": total_n,
    }
    for stage, val in pred_error_sums.items():
        summary[f"prediction_error_{stage}"] = val / max(1, pred_error_counts[stage])
    for stage, val in pred_loss_sums.items():
        summary[f"prediction_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for stage, val in pred_base_loss_sums.items():
        summary[f"prediction_base_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for stage, val in pred_amplitude_loss_sums.items():
        summary[f"prediction_amplitude_loss_{stage}"] = val / max(1, pred_loss_counts[stage])
    for (stage, name), val in modulation_sums.items():
        summary[f"modulation_{stage}_{name}"] = val / max(1, modulation_counts[(stage, name)])
    rows: list[dict[str, Any]] = []
    if per_t_acc_sum is not None:
        for t in range(len(per_t_acc_sum)):
            rows.append(
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
    prediction_rows = [
        {
            "split": "eval",
            "stage": stage,
            "mode": "end_to_end",
            "normalized_error": val / max(1, pred_error_counts[stage]),
            "symmetric_normalized_error": val / max(1, pred_error_counts[stage]),
            "raw_error_mean": pred_raw_sums[stage] / max(1, pred_error_counts[stage]),
            "target_abs_mean": pred_target_abs_sums[stage] / max(1, pred_error_counts[stage]),
            "prediction_abs_mean": pred_prediction_abs_sums[stage] / max(1, pred_error_counts[stage]),
            "prediction_abs_ratio": pred_prediction_abs_sums[stage] / max(pred_target_abs_sums[stage], 1e-12),
            "loss": pred_loss_sums[stage] / max(1, pred_loss_counts[stage]),
            "base_loss": pred_base_loss_sums[stage] / max(1, pred_loss_counts[stage]),
            "amplitude_loss": pred_amplitude_loss_sums[stage] / max(1, pred_loss_counts[stage]),
        }
        for stage, val in sorted(pred_error_sums.items())
    ]
    prediction_timestep_rows = []
    for stage, timestep in sorted(pred_timestep_counts):
        count = pred_timestep_counts[(stage, timestep)]
        prediction_timestep_rows.append(
            {
                "split": "eval",
                "stage": stage,
                "timestep": timestep,
                "loss": pred_timestep_sums[(stage, timestep, "loss")] / max(1, count),
                "raw_error_mean": pred_timestep_sums[(stage, timestep, "raw_error_mean")] / max(1, count),
                "normalized_error": pred_timestep_sums[(stage, timestep, "normalized_error")] / max(1, count),
                "symmetric_normalized_error": pred_timestep_sums[(stage, timestep, "symmetric_normalized_error")] / max(1, count),
                "target_abs_mean": pred_timestep_sums[(stage, timestep, "target_abs_mean")] / max(1, count),
                "prediction_abs_mean": pred_timestep_sums[(stage, timestep, "prediction_abs_mean")] / max(1, count),
            }
        )
    modulation_rows = [
        {
            "split": "eval",
            "stage": stage,
            "stat": name,
            "value": val / max(1, modulation_counts[(stage, name)]),
        }
        for (stage, name), val in sorted(modulation_sums.items())
    ]
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
    if sample_scores_path is not None and sample_score_rows:
        sample_scores_path = Path(sample_scores_path)
        sample_scores_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": sample_score_rows}, sample_scores_path)
    return summary, rows, confusion_rows, prediction_rows, prediction_timestep_rows, modulation_rows
