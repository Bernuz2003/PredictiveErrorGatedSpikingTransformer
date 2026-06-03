#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.models.temporal_predictors import FutureStatePredictor
from pegst.probing.metrics import baseline_prediction
from pegst.probing.targets import collect_targets_from_batch, predictor_tensor, select_target_item
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.progress import Timer, log, should_log
from pegst.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3b incremental usefulness controls for membrane prediction error.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictor-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batches", type=int, default=0)
    p.add_argument("--baselines", nargs="+", default=["copy_previous", "linear_extrapolation", "zero"])
    p.add_argument("--extrapolation-alpha", type=float, default=1.0)
    p.add_argument("--confidence-thresholds", nargs="+", type=float, default=None)
    p.add_argument("--error-quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--early-exit-signals", nargs="+", default=["learned_error", "copy_previous_error", "linear_extrapolation_error"])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--logreg-epochs", type=int, default=300)
    p.add_argument("--logreg-lr", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float(((x * y).mean() / denom).item())


def auc_score(scores: list[float], labels: list[int]) -> float:
    # labels: 1 is the positive class. Here positive means incorrect sample.
    pairs = sorted(zip(scores, labels, strict=False), key=lambda p: p[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += rank
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def weighted_auc_score(scores: list[float], labels: list[int], weights: list[float]) -> float:
    pos = [(s, w) for s, y, w in zip(scores, labels, weights, strict=False) if y == 1]
    neg = [(s, w) for s, y, w in zip(scores, labels, weights, strict=False) if y == 0]
    denom = sum(wp * wn for _, wp in pos for _, wn in neg)
    if denom <= 0:
        return float("nan")
    total = 0.0
    for sp, wp in pos:
        for sn, wn in neg:
            if sp > sn:
                total += wp * wn
            elif sp == sn:
                total += 0.5 * wp * wn
    return float(total / denom)


def average_precision(scores: list[float], labels: list[int]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels, strict=False), key=lambda p: p[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            precision_sum += tp / rank
    return float(precision_sum / n_pos)


def label_residual_scores(scores: list[float], labels: list[int]) -> list[float]:
    by_label: dict[int, list[float]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=False):
        by_label[int(label)].append(float(score))
    means = {label: sum(vals) / max(1, len(vals)) for label, vals in by_label.items()}
    return [float(score) - means[int(label)] for score, label in zip(scores, labels, strict=False)]


def macro_within_class_auc(scores: list[float], labels: list[int], incorrect: list[int]) -> float:
    values = []
    for label in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == label]
        local_labels = [incorrect[i] for i in idx]
        if sum(local_labels) == 0 or sum(local_labels) == len(local_labels):
            continue
        values.append(auc_score([scores[i] for i in idx], local_labels))
    return float(sum(values) / len(values)) if values else float("nan")


def per_timestep_errors(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = (prediction - target).abs().flatten(2).mean(dim=2)
    target_abs = target.abs().flatten(2).mean(dim=2)
    pred_abs = prediction.abs().flatten(2).mean(dim=2)
    norm = raw / (target_abs + pred_abs).clamp_min(1e-6)
    return norm.detach(), raw.detach()


def padded_error_sequence(per_timestep: torch.Tensor, total_timesteps: int) -> torch.Tensor:
    # per_timestep is [T-1, B] and corresponds to prediction errors for timesteps 2..T in 1-indexed reporting.
    first = torch.full((1, per_timestep.shape[1]), float("inf"), device=per_timestep.device, dtype=per_timestep.dtype)
    seq = torch.cat([first, per_timestep], dim=0)
    if seq.shape[0] != total_timesteps:
        out = torch.full((total_timesteps, per_timestep.shape[1]), float("inf"), device=per_timestep.device, dtype=per_timestep.dtype)
        n = min(total_timesteps, seq.shape[0])
        out[:n] = seq[:n]
        return out
    return seq


def build_predictor_from_checkpoint(path: str | Path, device: torch.device) -> tuple[FutureStatePredictor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    candidate = ckpt["candidate"]
    predictor = FutureStatePredictor(
        channels=int(candidate["channels"]),
        history=int(ckpt["history"]),
        spatial=bool(candidate["spatial"]),
        predictor_type=str(ckpt["predictor_type"]),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    return predictor, ckpt


def deterministic_folds(labels: list[int], folds: int) -> list[list[int]]:
    folds = max(2, int(folds))
    buckets = [[] for _ in range(folds)]
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    for group in (pos, neg):
        for j, idx in enumerate(group):
            buckets[j % folds].append(idx)
    return buckets


def logistic_cv(
    features: list[list[float]],
    labels: list[int],
    *,
    folds: int,
    epochs: int,
    lr: float,
) -> tuple[float, float, list[float], list[int]]:
    if not features:
        return float("nan"), float("nan"), [], []
    x_all = torch.tensor(features, dtype=torch.float32)
    y_all = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    buckets = deterministic_folds(labels, folds)
    scores: list[float] = []
    y_true: list[int] = []
    all_idx = set(range(len(labels)))
    for test_idx in buckets:
        if not test_idx:
            continue
        train_idx = sorted(all_idx - set(test_idx))
        if not train_idx:
            continue
        y_train = y_all[train_idx]
        y_test = [labels[i] for i in test_idx]
        if int(y_train.sum().item()) == 0 or int(y_train.sum().item()) == len(train_idx):
            continue
        if sum(y_test) == 0 or sum(y_test) == len(y_test):
            continue
        x_train = x_all[train_idx]
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_all[test_idx] - mean) / std
        clf = nn.Linear(x_train.shape[1], 1)
        pos = float(y_train.sum().item())
        neg = float(len(train_idx) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(max(1, epochs)):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(clf(x_train), y_train)
            loss.backward()
            opt.step()
        with torch.no_grad():
            fold_scores = torch.sigmoid(clf(x_test)).flatten().tolist()
        scores.extend(float(s) for s in fold_scores)
        y_true.extend(y_test)
    return auc_score(scores, y_true), average_precision(scores, y_true), scores, y_true


def signal_metrics(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    scores = [float(r[signal]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    incorrect = [int(r["incorrect"]) for r in rows]
    confidence = [float(r["confidence"]) for r in rows]
    entropy = [float(r["entropy"]) for r in rows]
    correct = [int(r["correct"]) for r in rows]
    label_counts: dict[int, int] = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    weights = [1.0 / max(1, label_counts[label]) for label in labels]
    residual = label_residual_scores(scores, labels)
    sorted_rows = sorted(rows, key=lambda r: float(r[signal]))
    q = max(1, len(sorted_rows) // 4)
    low = sorted_rows[:q]
    high = sorted_rows[-q:]
    class_rows = []
    for label in sorted(label_counts):
        sub = [r for r in rows if int(r["label"]) == label]
        class_rows.append(
            {
                "label": label,
                "accuracy": sum(int(r["correct"]) for r in sub) / max(1, len(sub)),
                "error_rate": sum(int(r["incorrect"]) for r in sub) / max(1, len(sub)),
                "mean_signal": sum(float(r[signal]) for r in sub) / max(1, len(sub)),
            }
        )
    return {
        "signal": signal,
        "auc_incorrect": auc_score(scores, incorrect),
        "average_precision_incorrect": average_precision(scores, incorrect),
        "label_balanced_auc": weighted_auc_score(scores, incorrect, weights),
        "within_class_auc": macro_within_class_auc(scores, labels, incorrect),
        "label_residual_auc": auc_score(residual, incorrect),
        "corr_signal_confidence": pearson(scores, confidence),
        "corr_signal_entropy": pearson(scores, entropy),
        "corr_signal_incorrect": pearson(scores, incorrect),
        "label_residual_corr_signal_incorrect": pearson(residual, incorrect),
        "low_q25_accuracy": sum(int(r["correct"]) for r in low) / max(1, len(low)),
        "high_q75_accuracy": sum(int(r["correct"]) for r in high) / max(1, len(high)),
        "high_minus_low_error_rate": (
            sum(int(r["incorrect"]) for r in high) / max(1, len(high))
            - sum(int(r["incorrect"]) for r in low) / max(1, len(low))
        ),
        "corr_class_mean_signal_vs_class_accuracy": pearson(
            [float(r["mean_signal"]) for r in class_rows],
            [float(r["accuracy"]) for r in class_rows],
        ),
        "corr_class_mean_signal_vs_class_error_rate": pearson(
            [float(r["mean_signal"]) for r in class_rows],
            [float(r["error_rate"]) for r in class_rows],
        ),
    }


def early_exit_rows(
    records: list[dict[str, Any]],
    *,
    confidence_thresholds: list[float],
    error_thresholds: dict[str, list[tuple[float, float]]],
    signals: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not records:
        return rows
    total_timesteps = len(records[0]["confidence_seq"])
    for conf_tau in confidence_thresholds:
        rules: list[tuple[str, str, float | str, float | None]] = [("confidence_only", "", "", None)]
        for signal in signals:
            for quantile, threshold in error_thresholds.get(signal, []):
                rules.append((f"confidence_and_{signal}", signal, quantile, threshold))
        local_rows: list[dict[str, Any]] = []
        for rule_name, signal, quantile, err_tau in rules:
            correct = 0
            total = 0
            exit_sum = 0.0
            early_count = 0
            for record in records:
                conf_seq = record["confidence_seq"]
                pred_seq = record["prediction_seq"]
                err_seq = record["error_seq"].get(signal, [])
                exit_t = total_timesteps - 1
                for t in range(total_timesteps):
                    if conf_seq[t] < conf_tau:
                        continue
                    if err_tau is not None:
                        if t >= len(err_seq) or not math.isfinite(float(err_seq[t])) or float(err_seq[t]) > err_tau:
                            continue
                    exit_t = t
                    break
                pred = int(pred_seq[exit_t])
                label = int(record["label"])
                correct += int(pred == label)
                total += 1
                exit_sum += exit_t + 1
                early_count += int(exit_t < total_timesteps - 1)
            local_rows.append(
                {
                    "rule": rule_name,
                    "signal": signal,
                    "confidence_threshold": conf_tau,
                    "error_quantile": quantile,
                    "error_threshold": "" if err_tau is None else err_tau,
                    "accuracy": correct / max(1, total),
                    "avg_exit_timestep": exit_sum / max(1, total),
                    "coverage": early_count / max(1, total),
                    "expected_sops_fraction": (exit_sum / max(1, total)) / max(1, total_timesteps),
                    "num_samples": total,
                }
            )
        baseline = local_rows[0]
        for row in local_rows:
            row["accuracy_delta_vs_confidence_only"] = float(row["accuracy"]) - float(baseline["accuracy"])
            row["error_rate_delta_vs_confidence_only"] = (1.0 - float(row["accuracy"])) - (1.0 - float(baseline["accuracy"]))
            row["coverage_delta_vs_confidence_only"] = float(row["coverage"]) - float(baseline["coverage"])
            row["expected_sops_fraction_delta_vs_confidence_only"] = float(row["expected_sops_fraction"]) - float(
                baseline["expected_sops_fraction"]
            )
        rows.extend(local_rows)
    return rows


def main() -> None:
    args = parse_args()
    timer = Timer()
    log("M3b incremental usefulness controls started")
    log(f"config={args.config}")
    log(f"checkpoint={args.checkpoint}")
    log(f"predictor_checkpoint={args.predictor_checkpoint}")
    log(f"output_dir={args.output_dir}")
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    log("loading baseline checkpoint")
    load_info = load_model_checkpoint(model, args.checkpoint, strict=False)
    log(
        "baseline loaded: "
        f"missing={len(load_info.get('missing_keys', []))}, "
        f"unexpected={len(load_info.get('unexpected_keys', []))}"
    )
    model.eval()
    log("loading M2 predictor checkpoint")
    predictor, pred_ckpt = build_predictor_from_checkpoint(args.predictor_checkpoint, device)
    candidate = pred_ckpt["candidate"]
    log(
        "predictor ready: "
        f"target={candidate['target']}, stage={candidate['stage']}, layer={candidate['layer']}, "
        f"history={pred_ckpt['history']}, predictor={pred_ckpt['predictor_type']}, "
        f"amp={pred_ckpt.get('amplitude_loss_weight', 0.0)}"
    )

    confidence_thresholds = args.confidence_thresholds or cfg.get("early_exit", {}).get(
        "confidence_thresholds", [0.7, 0.8, 0.9, 0.95]
    )
    log(
        "settings: "
        f"split={args.split}, batches={'all' if args.batches <= 0 else args.batches}, "
        f"baselines={args.baselines}, confidence_thresholds={confidence_thresholds}, "
        f"error_quantiles={args.error_quantiles}, cv_folds={args.cv_folds}"
    )
    loader = build_dataloader(cfg["dataset"], args.split)

    sample_rows: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    sample_id = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if args.batches > 0 and batch_idx >= args.batches:
                break
            batch_no = batch_idx + 1
            x = x.to(device).float()
            y = y.to(device).long()
            reset_spiking_state(model)
            items, out, _ = collect_targets_from_batch(
                model,
                x,
                targets=[candidate["target"]],
                stages=[candidate["stage"]],
                layer_patterns=[candidate["layer"]],
                soft_firing_temperature=float(pred_ckpt.get("soft_firing_temperature", 0.25)),
            )
            item = select_target_item(items, candidate["target"], candidate["stage"], candidate["layer"])
            if item is None:
                continue
            z, _ = predictor_tensor(item.tensor.detach())
            learned = predictor.forward_sequence(
                z,
                loss_type=str(pred_ckpt.get("loss_type", "smooth_l1")),  # type: ignore[arg-type]
                normalize_loss=bool(pred_ckpt.get("normalize_loss", False)),
                amplitude_loss_weight=float(pred_ckpt.get("amplitude_loss_weight", 0.0)),
            )
            logits = out["logits"]
            timestep_logits = out["timestep_logits"]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            ent = entropy_from_logits(logits)
            timestep_probs = torch.softmax(timestep_logits, dim=-1)
            timestep_conf, timestep_pred = timestep_probs.max(dim=-1)
            logit_stability = (timestep_probs[1:] - timestep_probs[:-1]).abs().sum(dim=-1).mean(dim=0)
            total_timesteps = timestep_logits.shape[0]

            signal_timestep_errors: dict[str, torch.Tensor] = {
                "learned_error": padded_error_sequence(learned.sample_normalized_error, total_timesteps)
            }
            learned_raw_t = learned.error.abs().flatten(2).mean(dim=2)
            raw_timestep_errors: dict[str, torch.Tensor] = {
                "learned_raw_error": padded_error_sequence(learned_raw_t, total_timesteps)
            }
            for baseline in args.baselines:
                base_pred, base_target = baseline_prediction(z, baseline, alpha=args.extrapolation_alpha)
                norm_t, raw_t = per_timestep_errors(base_pred, base_target)
                signal_timestep_errors[f"{baseline}_error"] = padded_error_sequence(norm_t, total_timesteps)
                raw_timestep_errors[f"{baseline}_raw_error"] = padded_error_sequence(raw_t, total_timesteps)

            for j in range(y.numel()):
                correct = int((pred[j] == y[j]).item())
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "split": args.split,
                    "label": int(y[j].item()),
                    "predicted_label": int(pred[j].item()),
                    "correct": correct,
                    "incorrect": 1 - correct,
                    "confidence": float(conf[j].item()),
                    "confidence_error": float(1.0 - conf[j].item()),
                    "entropy": float(ent[j].item()),
                    "logit_stability_l1": float(logit_stability[j].item()),
                    "target": candidate["target"],
                    "stage": candidate["stage"],
                    "layer": candidate["layer"],
                    "history": int(pred_ckpt["history"]),
                    "predictor_type": str(pred_ckpt["predictor_type"]),
                    "amplitude_loss_weight": float(pred_ckpt.get("amplitude_loss_weight", 0.0)),
                }
                error_seq: dict[str, list[float]] = {}
                for signal, seq in signal_timestep_errors.items():
                    finite_seq = seq[:, j]
                    finite_vals = finite_seq[torch.isfinite(finite_seq)]
                    row[signal] = float(finite_vals.mean().item()) if finite_vals.numel() else float("nan")
                    error_seq[signal] = [float(v) for v in finite_seq.detach().cpu().tolist()]
                for signal, seq in raw_timestep_errors.items():
                    finite_seq = seq[:, j]
                    finite_vals = finite_seq[torch.isfinite(finite_seq)]
                    row[signal] = float(finite_vals.mean().item()) if finite_vals.numel() else float("nan")
                sample_rows.append(row)
                sequence_records.append(
                    {
                        "sample_id": sample_id,
                        "label": int(y[j].item()),
                        "confidence_seq": [float(v) for v in timestep_conf[:, j].detach().cpu().tolist()],
                        "prediction_seq": [int(v) for v in timestep_pred[:, j].detach().cpu().tolist()],
                        "error_seq": error_seq,
                    }
                )
                sample_id += 1
            if should_log(batch_no, args.batches if args.batches > 0 else None, args.log_every):
                log(f"M3b split={args.split}: processed batch {batch_no}, samples={len(sample_rows)}, elapsed={timer.elapsed_str()}")

    if not sample_rows:
        raise RuntimeError("No samples collected for M3b.")

    log(f"M3b collected {len(sample_rows)} samples; computing signal metrics")
    signal_names = ["learned_error", *[f"{b}_error" for b in args.baselines], "confidence_error", "entropy", "logit_stability_l1"]
    signal_rows = [signal_metrics(sample_rows, signal) for signal in signal_names if signal in sample_rows[0]]

    class_rows: list[dict[str, Any]] = []
    for signal in signal_names:
        if signal not in sample_rows[0]:
            continue
        for label in sorted({int(r["label"]) for r in sample_rows}):
            sub = [r for r in sample_rows if int(r["label"]) == label]
            class_rows.append(
                {
                    "signal": signal,
                    "label": label,
                    "num_samples": len(sub),
                    "accuracy": sum(int(r["correct"]) for r in sub) / max(1, len(sub)),
                    "error_rate": sum(int(r["incorrect"]) for r in sub) / max(1, len(sub)),
                    "mean_signal": sum(float(r[signal]) for r in sub) / max(1, len(sub)),
                    "mean_confidence": sum(float(r["confidence"]) for r in sub) / max(1, len(sub)),
                    "mean_entropy": sum(float(r["entropy"]) for r in sub) / max(1, len(sub)),
                }
            )

    log("M3b computing incremental cross-validated AUC/AP")
    labels_incorrect = [int(r["incorrect"]) for r in sample_rows]
    incremental_rows: list[dict[str, Any]] = []
    base_features = [[float(r["confidence_error"]), float(r["entropy"])] for r in sample_rows]
    base_auc, base_ap, _, _ = logistic_cv(
        base_features,
        labels_incorrect,
        folds=args.cv_folds,
        epochs=args.logreg_epochs,
        lr=args.logreg_lr,
    )
    for signal in signal_names:
        if signal not in sample_rows[0]:
            continue
        signal_features = [[float(r[signal])] for r in sample_rows]
        signal_auc, signal_ap, _, _ = logistic_cv(
            signal_features,
            labels_incorrect,
            folds=args.cv_folds,
            epochs=args.logreg_epochs,
            lr=args.logreg_lr,
        )
        combo_features = [[float(r["confidence_error"]), float(r["entropy"]), float(r[signal])] for r in sample_rows]
        combo_auc, combo_ap, _, _ = logistic_cv(
            combo_features,
            labels_incorrect,
            folds=args.cv_folds,
            epochs=args.logreg_epochs,
            lr=args.logreg_lr,
        )
        incremental_rows.append(
            {
                "signal": signal,
                "auc_signal_only": signal_auc,
                "ap_signal_only": signal_ap,
                "auc_confidence_entropy": base_auc,
                "ap_confidence_entropy": base_ap,
                "auc_confidence_entropy_signal": combo_auc,
                "ap_confidence_entropy_signal": combo_ap,
                "delta_auc": combo_auc - base_auc if math.isfinite(combo_auc) and math.isfinite(base_auc) else float("nan"),
                "delta_ap": combo_ap - base_ap if math.isfinite(combo_ap) and math.isfinite(base_ap) else float("nan"),
                "passes_incremental_auc_0p02": math.isfinite(combo_auc) and math.isfinite(base_auc) and combo_auc - base_auc >= 0.02,
                "passes_incremental_ap_0p02": math.isfinite(combo_ap) and math.isfinite(base_ap) and combo_ap - base_ap >= 0.02,
            }
        )

    log("M3b computing early-exit controls")
    finite_error_values: dict[str, list[float]] = {}
    for signal in args.early_exit_signals:
        vals = []
        for record in sequence_records:
            vals.extend(v for v in record["error_seq"].get(signal, []) if math.isfinite(float(v)))
        finite_error_values[signal] = vals
    error_thresholds: dict[str, list[tuple[float, float]]] = {}
    for signal, vals in finite_error_values.items():
        if not vals:
            continue
        t = torch.tensor(vals, dtype=torch.float32)
        error_thresholds[signal] = [(float(q), float(torch.quantile(t, float(q)).item())) for q in args.error_quantiles]
    exit_rows = early_exit_rows(
        sequence_records,
        confidence_thresholds=[float(v) for v in confidence_thresholds],
        error_thresholds=error_thresholds,
        signals=args.early_exit_signals,
    )

    learned_row = next((row for row in signal_rows if row["signal"] == "learned_error"), {})
    copy_row = next((row for row in signal_rows if row["signal"] == "copy_previous_error"), {})
    learned_inc = next((row for row in incremental_rows if row["signal"] == "learned_error"), {})
    summary = {
        "num_samples": len(sample_rows),
        "target": candidate["target"],
        "stage": candidate["stage"],
        "layer": candidate["layer"],
        "history": int(pred_ckpt["history"]),
        "predictor_type": str(pred_ckpt["predictor_type"]),
        "amplitude_loss_weight": float(pred_ckpt.get("amplitude_loss_weight", 0.0)),
        "learned_auc_incorrect": learned_row.get("auc_incorrect", float("nan")),
        "copy_auc_incorrect": copy_row.get("auc_incorrect", float("nan")),
        "learned_minus_copy_auc": (
            learned_row.get("auc_incorrect", float("nan")) - copy_row.get("auc_incorrect", float("nan"))
            if learned_row and copy_row
            else float("nan")
        ),
        "learned_label_residual_auc": learned_row.get("label_residual_auc", float("nan")),
        "learned_within_class_auc": learned_row.get("within_class_auc", float("nan")),
        "learned_delta_auc_over_conf_entropy": learned_inc.get("delta_auc", float("nan")),
        "learned_delta_ap_over_conf_entropy": learned_inc.get("delta_ap", float("nan")),
        "passes_learned_beats_copy_auc_0p02": (
            learned_row.get("auc_incorrect", float("nan")) - copy_row.get("auc_incorrect", float("nan")) >= 0.02
            if learned_row and copy_row
            else False
        ),
        "passes_incremental_auc_0p02": bool(learned_inc.get("passes_incremental_auc_0p02", False)),
        "passes_incremental_ap_0p02": bool(learned_inc.get("passes_incremental_ap_0p02", False)),
    }
    summary["passes_m3b"] = (
        summary["passes_learned_beats_copy_auc_0p02"]
        or summary["passes_incremental_auc_0p02"]
        or summary["passes_incremental_ap_0p02"]
    )

    write_csv(out_dir / "m3b_samples.csv", sample_rows)
    write_csv(out_dir / "m3b_signal_metrics.csv", signal_rows)
    write_csv(out_dir / "m3b_class_signal_summary.csv", class_rows)
    write_csv(out_dir / "m3b_incremental_auc.csv", incremental_rows)
    write_csv(out_dir / "m3b_early_exit.csv", exit_rows)
    write_csv(out_dir / "m3b_summary.csv", [summary])
    write_json(
        out_dir / "m3b_summary.json",
        {
            **summary,
            "checkpoint_load_info": load_info,
            "predictor_checkpoint": str(args.predictor_checkpoint),
            "confidence_thresholds": confidence_thresholds,
            "error_quantiles": args.error_quantiles,
            "criteria": {
                "passes_learned_beats_copy_auc_0p02": "AUC(learned_error -> incorrect) - AUC(copy_error -> incorrect) >= 0.02",
                "passes_incremental_auc_0p02": "AUC(confidence_error + entropy + learned_error) - AUC(confidence_error + entropy) >= 0.02",
                "passes_incremental_ap_0p02": "AP(confidence_error + entropy + learned_error) - AP(confidence_error + entropy) >= 0.02",
                "passes_m3b": "any criterion above",
            },
        },
    )
    log(
        "M3b summary: "
        f"learned_auc={summary['learned_auc_incorrect']:.4f}, "
        f"copy_auc={summary['copy_auc_incorrect']:.4f}, "
        f"delta_auc_conf_entropy={summary['learned_delta_auc_over_conf_entropy']:.4f}, "
        f"label_residual_auc={summary['learned_label_residual_auc']:.4f}, "
        f"passes_m3b={summary['passes_m3b']}"
    )
    log(f"M3b finished in {timer.elapsed_str()} -> {out_dir}")


if __name__ == "__main__":
    main()
