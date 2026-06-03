#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.models.temporal_predictors import FutureStatePredictor
from pegst.probing.targets import collect_targets_from_batch, predictor_tensor, select_target_item
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 usefulness probe for membrane prediction error.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictor-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batches", type=int, default=0)
    return p.parse_args()


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
    # labels: 1 = positive class, here incorrect prediction.
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


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    load_info = load_model_checkpoint(model, args.checkpoint, strict=False)
    model.eval()
    predictor, pred_ckpt = build_predictor_from_checkpoint(args.predictor_checkpoint, device)
    candidate = pred_ckpt["candidate"]
    loader = build_dataloader(cfg["dataset"], args.split)

    sample_rows: list[dict[str, Any]] = []
    sample_id = 0
    for batch_idx, (x, y) in enumerate(loader):
        if args.batches > 0 and batch_idx >= args.batches:
            break
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
        batch = predictor.forward_sequence(
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
        logit_stability = (timestep_probs[1:] - timestep_probs[:-1]).abs().sum(dim=-1).mean(dim=0)
        sample_error = batch.sample_normalized_error.mean(dim=0)
        raw_error = batch.error.abs().flatten(2).mean(dim=2).mean(dim=0)
        for j in range(y.numel()):
            correct = int((pred[j] == y[j]).item())
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "split": args.split,
                    "label": int(y[j].item()),
                    "predicted_label": int(pred[j].item()),
                    "correct": correct,
                    "incorrect": 1 - correct,
                    "confidence": float(conf[j].item()),
                    "entropy": float(ent[j].item()),
                    "prediction_error": float(sample_error[j].item()),
                    "raw_prediction_error": float(raw_error[j].item()),
                    "logit_stability_l1": float(logit_stability[j].item()),
                    "target": candidate["target"],
                    "stage": candidate["stage"],
                    "layer": candidate["layer"],
                    "history": int(pred_ckpt["history"]),
                    "predictor_type": str(pred_ckpt["predictor_type"]),
                    "amplitude_loss_weight": float(pred_ckpt.get("amplitude_loss_weight", 0.0)),
                }
            )
            sample_id += 1

    errors = [float(r["prediction_error"]) for r in sample_rows]
    raw_errors = [float(r["raw_prediction_error"]) for r in sample_rows]
    confidence = [float(r["confidence"]) for r in sample_rows]
    entropy = [float(r["entropy"]) for r in sample_rows]
    incorrect = [int(r["incorrect"]) for r in sample_rows]
    correct = [int(r["correct"]) for r in sample_rows]
    logit_stability = [float(r["logit_stability_l1"]) for r in sample_rows]
    summary: dict[str, Any] = {
        "num_samples": len(sample_rows),
        "target": candidate["target"],
        "stage": candidate["stage"],
        "layer": candidate["layer"],
        "history": int(pred_ckpt["history"]),
        "predictor_type": str(pred_ckpt["predictor_type"]),
        "amplitude_loss_weight": float(pred_ckpt.get("amplitude_loss_weight", 0.0)),
        "corr_error_confidence": pearson(errors, confidence),
        "corr_error_entropy": pearson(errors, entropy),
        "corr_error_logit_stability": pearson(errors, logit_stability),
        "auc_error_detects_incorrect": auc_score(errors, incorrect),
        "auc_raw_error_detects_incorrect": auc_score(raw_errors, incorrect),
        "overall_accuracy": sum(correct) / max(1, len(correct)),
    }
    if sample_rows:
        sorted_rows = sorted(sample_rows, key=lambda r: float(r["prediction_error"]))
        q = max(1, len(sorted_rows) // 4)
        low = sorted_rows[:q]
        high = sorted_rows[-q:]
        summary["low_error_q25_accuracy"] = sum(int(r["correct"]) for r in low) / max(1, len(low))
        summary["high_error_q75_accuracy"] = sum(int(r["correct"]) for r in high) / max(1, len(high))
        summary["high_minus_low_error_rate"] = (
            sum(int(r["incorrect"]) for r in high) / max(1, len(high))
            - sum(int(r["incorrect"]) for r in low) / max(1, len(low))
        )
    summary["passes_auc_correctness"] = math.isfinite(float(summary["auc_error_detects_incorrect"])) and summary["auc_error_detects_incorrect"] >= 0.65
    summary["passes_entropy_corr"] = math.isfinite(float(summary["corr_error_entropy"])) and summary["corr_error_entropy"] > 0.25
    summary["passes_confidence_corr"] = math.isfinite(float(summary["corr_error_confidence"])) and summary["corr_error_confidence"] < -0.25
    summary["passes_quartile_accuracy_gap"] = summary.get("low_error_q25_accuracy", 0.0) > summary.get("high_error_q75_accuracy", 0.0)
    summary["passes_m3"] = (
        summary["passes_auc_correctness"]
        or summary["passes_entropy_corr"]
        or summary["passes_confidence_corr"]
        or summary["passes_quartile_accuracy_gap"]
    )
    write_csv(out_dir / "usefulness_samples.csv", sample_rows)
    write_csv(out_dir / "usefulness_summary.csv", [summary])
    write_json(
        out_dir / "usefulness_summary.json",
        {
            **summary,
            "checkpoint_load_info": load_info,
            "predictor_checkpoint": str(args.predictor_checkpoint),
            "criteria": {
                "passes_auc_correctness": "AUC(error -> incorrect) >= 0.65",
                "passes_entropy_corr": "corr(error, entropy) > 0.25",
                "passes_confidence_corr": "corr(error, confidence) < -0.25",
                "passes_quartile_accuracy_gap": "low-error quartile accuracy > high-error quartile accuracy",
                "passes_m3": "any criterion above",
            },
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
