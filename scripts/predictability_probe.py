#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pegst.data.dvs import build_dataloader
from pegst.models.predictive_modules import FutureStatePredictor
from pegst.models.predictive_qkformer import build_predictive_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.seed import seed_everything


METRIC_NAMES = [
    "loss",
    "base_loss",
    "amplitude_loss",
    "normalized_error",
    "symmetric_normalized_error",
    "raw_error_mean",
    "target_abs_mean",
    "prediction_abs_mean",
    "prediction_abs_ratio",
]


def transform_feature(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "reverse":
        return torch.flip(x, dims=[0])
    if mode == "shuffle":
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
    if mode != "normal":
        raise ValueError(f"Unknown temporal mode: {mode}")
    return x


def baseline_prediction(x: torch.Tensor, method: str, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    target = x[1:]
    if method == "zero":
        return torch.zeros_like(target), target
    if method == "copy_previous":
        return x[:-1], target
    if method == "linear_extrapolation":
        current = x[:-1]
        previous = torch.cat([x[:1], x[:-2]], dim=0)
        return current + alpha * (current - previous), target
    raise ValueError(f"Unknown baseline method: {method}")


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
    normalize_loss: bool = False,
    amplitude_loss_weight: float = 0.0,
) -> dict[str, float]:
    base_loss_type = loss_type
    if loss_type.startswith("normalized_"):
        normalize_loss = True
        base_loss_type = loss_type.removeprefix("normalized_")
    pred_for_loss = FutureStatePredictor._normalize_for_loss(prediction) if normalize_loss else prediction
    target_for_loss = FutureStatePredictor._normalize_for_loss(target) if normalize_loss else target
    if base_loss_type == "mse":
        raw_loss = F.mse_loss(pred_for_loss, target_for_loss, reduction="none")
    elif base_loss_type == "smooth_l1":
        raw_loss = F.smooth_l1_loss(pred_for_loss, target_for_loss, reduction="none")
    else:
        raw_loss = F.l1_loss(pred_for_loss, target_for_loss, reduction="none")
    base_loss = raw_loss.flatten(2).mean(dim=2).mean()
    raw_error = (prediction - target).abs().flatten(2).mean(dim=2)
    target_abs = target.abs().flatten(2).mean(dim=2)
    prediction_abs = prediction.abs().flatten(2).mean(dim=2)
    amplitude_loss = (prediction_abs - target_abs).abs().mean()
    loss = base_loss + float(amplitude_loss_weight) * amplitude_loss
    normalized_error = (raw_error / (target_abs + prediction_abs).clamp_min(1e-6)).mean()
    return {
        "loss": float(loss.detach().item()),
        "base_loss": float(base_loss.detach().item()),
        "amplitude_loss": float(amplitude_loss.detach().item()),
        "normalized_error": float(normalized_error.detach().item()),
        "symmetric_normalized_error": float(normalized_error.detach().item()),
        "raw_error_mean": float(raw_error.mean().detach().item()),
        "target_abs_mean": float(target_abs.mean().detach().item()),
        "prediction_abs_mean": float(prediction_abs.mean().detach().item()),
        "prediction_abs_ratio": float((prediction_abs.mean() / target_abs.mean().clamp_min(1e-12)).detach().item()),
    }


def add_weighted(acc: dict[str, float], metrics: dict[str, float], weight: int) -> None:
    acc["_count"] = acc.get("_count", 0.0) + weight
    for key, value in metrics.items():
        acc[key] = acc.get(key, 0.0) + value * weight


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    count = max(1.0, acc.get("_count", 0.0))
    return {key: value / count for key, value in acc.items() if key != "_count"}


def feature_batches(
    model,
    loader,
    device: torch.device,
    stage: str,
    mode: str,
    max_batches: int,
) -> Iterable[torch.Tensor]:
    with torch.no_grad():
        for step, (x, _) in enumerate(loader):
            if step >= max_batches:
                break
            reset_spiking_state(model)
            out = model(x.to(device).float(), return_features=True, return_timestep_logits=False, return_aux=True)
            yield transform_feature(out["features"][stage].detach(), mode)


def evaluate_predictor(
    model,
    predictor: FutureStatePredictor,
    loader,
    device: torch.device,
    stage: str,
    mode: str,
    batches: int,
    loss_type: str,
    normalize_loss: bool,
    amplitude_loss_weight: float,
    baselines: list[str],
    extrapolation_alpha: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    learned_acc: dict[str, float] = {}
    baseline_accs: dict[str, dict[str, float]] = {name: {} for name in baselines}
    with torch.no_grad():
        for feat in feature_batches(model, loader, device, stage, mode, batches):
            batch = predictor.forward_sequence(
                feat,
                loss_type=loss_type,  # type: ignore[arg-type]
                normalize_loss=normalize_loss,
                amplitude_loss_weight=amplitude_loss_weight,
            )
            add_weighted(
                learned_acc,
                prediction_metrics(
                    batch.prediction,
                    batch.target,
                    loss_type=loss_type,
                    normalize_loss=normalize_loss,
                    amplitude_loss_weight=amplitude_loss_weight,
                ),
                batch.target.shape[1],
            )
            for baseline in baselines:
                pred, target = baseline_prediction(feat, baseline, alpha=extrapolation_alpha)
                add_weighted(
                    baseline_accs[baseline],
                    prediction_metrics(
                        pred,
                        target,
                        loss_type=loss_type,
                        normalize_loss=normalize_loss,
                        amplitude_loss_weight=amplitude_loss_weight,
                    ),
                    target.shape[1],
                )
    return finalize_metrics(learned_acc), {name: finalize_metrics(acc) for name, acc in baseline_accs.items()}


def row_from_metrics(
    *,
    stage: str,
    history: int,
    train_mode: str,
    eval_split: str,
    eval_mode: str,
    method: str,
    predictor_type: str,
    amplitude_loss_weight: float,
    steps: int,
    last_train_loss: float | str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "history": history,
        "train_mode": train_mode,
        "eval_split": eval_split,
        "eval_mode": eval_mode,
        "mode": eval_mode,  # Backward-compatible column name.
        "method": method,
        "predictor_type": predictor_type,
        "amplitude_loss_weight": amplitude_loss_weight,
        "steps": steps,
        "last_train_loss": last_train_loss,
        **metrics,
    }


def find_row(rows: list[dict[str, Any]], **criteria) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def relative_gain(model_error: float, baseline_error: float) -> float:
    return 1.0 - model_error / max(baseline_error, 1e-12)


def build_decision_rows(rows: list[dict[str, Any]], baselines: list[str], eval_modes: list[str]) -> list[dict[str, Any]]:
    decision_rows: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                row["stage"],
                int(row["history"]),
                row["train_mode"],
                row["eval_split"],
                row["predictor_type"],
                float(row["amplitude_loss_weight"]),
                int(row["steps"]),
            )
            for row in rows
            if row["method"] == "learned_predictor"
        }
    )
    for stage, history, train_mode, eval_split, predictor_type, amp_weight, steps in group_keys:
        normal = find_row(
            rows,
            stage=stage,
            history=history,
            train_mode=train_mode,
            eval_split=eval_split,
            eval_mode="normal",
            method="learned_predictor",
            predictor_type=predictor_type,
            amplitude_loss_weight=amp_weight,
            steps=steps,
        )
        if normal is None:
            continue
        decision: dict[str, Any] = {
            "stage": stage,
            "history": history,
            "train_mode": train_mode,
            "eval_split": eval_split,
            "predictor_type": predictor_type,
            "amplitude_loss_weight": amp_weight,
            "steps": steps,
            "normal_loss": normal["loss"],
            "normal_normalized_error": normal["normalized_error"],
            "normal_raw_error_mean": normal["raw_error_mean"],
            "normal_prediction_abs_ratio": normal["prediction_abs_ratio"],
            "amplitude_in_0p8_1p2": 0.8 <= float(normal["prediction_abs_ratio"]) <= 1.2,
        }
        for baseline in baselines:
            base = find_row(
                rows,
                stage=stage,
                history=history,
                train_mode=train_mode,
                eval_split=eval_split,
                eval_mode="normal",
                method=baseline,
                predictor_type="analytic",
                amplitude_loss_weight=amp_weight,
                steps=steps,
            )
            if base is None:
                continue
            for metric in ("loss", "normalized_error", "raw_error_mean"):
                decision[f"relative_gain_vs_{baseline}_{metric}"] = relative_gain(float(normal[metric]), float(base[metric]))
        for mode in eval_modes:
            if mode == "normal":
                continue
            row = find_row(
                rows,
                stage=stage,
                history=history,
                train_mode=train_mode,
                eval_split=eval_split,
                eval_mode=mode,
                method="learned_predictor",
                predictor_type=predictor_type,
                amplitude_loss_weight=amp_weight,
                steps=steps,
            )
            if row is None:
                continue
            for metric in ("loss", "normalized_error", "raw_error_mean"):
                decision[f"temporal_selectivity_{mode}_{metric}"] = float(row[metric]) / max(float(normal[metric]), 1e-12)
            decision[f"{mode}_prediction_abs_ratio"] = row["prediction_abs_ratio"]
        gain_copy = float(decision.get("relative_gain_vs_copy_previous_normalized_error", 0.0))
        gain_zero_raw = float(decision.get("relative_gain_vs_zero_raw_error_mean", 0.0))
        shuffle_sel = float(decision.get("temporal_selectivity_shuffle_normalized_error", 1.0))
        reverse_sel = float(decision.get("temporal_selectivity_reverse_normalized_error", 1.0))
        decision["passes_copy_gain_5pct"] = gain_copy >= 0.05
        decision["passes_zero_raw_gain"] = gain_zero_raw > 0.0
        decision["passes_temporal_specificity"] = shuffle_sel > 1.05 and reverse_sel > 1.05
        decision["passes_amplitude"] = decision["amplitude_in_0p8_1p2"]
        decision["passes_strict_probe"] = (
            decision["passes_copy_gain_5pct"]
            and decision["passes_zero_raw_gain"]
            and decision["passes_temporal_specificity"]
            and decision["passes_amplitude"]
        )
        decision_rows.append(decision)
    return decision_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline causal predictability probe for latent QKFormer states.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stages", nargs="+", default=["stage1", "stage2"])
    p.add_argument("--modes", nargs="+", default=["normal", "shuffle", "reverse"])
    p.add_argument("--train-mode", default="normal", choices=["normal", "shuffle", "reverse"])
    p.add_argument("--eval-splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--histories", nargs="+", type=int, default=None)
    p.add_argument("--baselines", nargs="+", default=["zero", "copy_previous", "linear_extrapolation"])
    p.add_argument("--predictor-types", nargs="+", default=None)
    p.add_argument("--amplitude-loss-weights", nargs="+", type=float, default=None)
    p.add_argument("--extrapolation-alpha", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--curve-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pred_cfg: dict[str, Any] = cfg.get("predictive", {})
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_predictive_qkformer(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    train_loader = build_dataloader(cfg["dataset"], "train")
    eval_loaders = {split: build_dataloader(cfg["dataset"], split) for split in args.eval_splits}
    histories = args.histories or [int(pred_cfg.get("history", 1))]
    loss_type = str(pred_cfg.get("loss_type", "l1"))
    normalize_loss = bool(pred_cfg.get("normalize_loss", False))
    amplitude_weights = args.amplitude_loss_weights or [float(pred_cfg.get("amplitude_loss_weight", 0.0))]
    predictor_types = args.predictor_types or [str(pred_cfg.get("predictor_type", "conv1x1"))]
    hidden_ratio = float(pred_cfg.get("hidden_ratio", 1.0))

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for stage in args.stages:
        if stage not in model.backbone.stage_channels:
            skipped.append({"stage": stage, "reason": "unknown_stage"})
            continue
        channels = model.backbone.stage_channels[stage]
        for history in histories:
            for predictor_type in predictor_types:
                if predictor_type == "motion_extrapolation" and history < 2:
                    skipped.append({"stage": stage, "history": history, "predictor_type": predictor_type, "reason": "requires_history_ge_2"})
                    continue
                for amplitude_loss_weight in amplitude_weights:
                    predictor = FutureStatePredictor(
                        channels,
                        history=history,
                        spatial=True,
                        predictor_type=predictor_type,
                        hidden_ratio=hidden_ratio,
                    ).to(device)
                    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    last_loss = None
                    train_iter = iter(train_loader)
                    eval_points = set()
                    if args.eval_every > 0:
                        eval_points.update(range(args.eval_every, args.steps + 1, args.eval_every))
                    eval_points.add(args.steps)
                    for step_idx in range(1, args.steps + 1):
                        try:
                            x, _ = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            x, _ = next(train_iter)
                        x = x.to(device).float()
                        reset_spiking_state(model)
                        with torch.no_grad():
                            out = model(x, return_features=True, return_timestep_logits=False, return_aux=True)
                            feat = transform_feature(out["features"][stage].detach(), args.train_mode)
                        batch = predictor.forward_sequence(
                            feat,
                            loss_type=loss_type,  # type: ignore[arg-type]
                            normalize_loss=normalize_loss,
                            amplitude_loss_weight=amplitude_loss_weight,
                        )
                        opt.zero_grad(set_to_none=True)
                        batch.loss.backward()
                        opt.step()
                        last_loss = batch.loss.item()
                        if step_idx in eval_points:
                            curve_metrics, _ = evaluate_predictor(
                                model,
                                predictor,
                                eval_loaders["test"] if "test" in eval_loaders else train_loader,
                                device,
                                stage,
                                "normal",
                                args.curve_batches,
                                loss_type,
                                normalize_loss,
                                amplitude_loss_weight,
                                [],
                                args.extrapolation_alpha,
                            )
                            curve_rows.append(
                                {
                                    "stage": stage,
                                    "history": history,
                                    "train_mode": args.train_mode,
                                    "eval_split": "test" if "test" in eval_loaders else "train",
                                    "eval_mode": "normal",
                                    "method": "learned_predictor",
                                    "predictor_type": predictor_type,
                                    "amplitude_loss_weight": amplitude_loss_weight,
                                    "step": step_idx,
                                    "train_loss": last_loss,
                                    **curve_metrics,
                                }
                            )

                    for eval_split, loader in eval_loaders.items():
                        for eval_mode in args.modes:
                            learned_metrics, baseline_metrics = evaluate_predictor(
                                model,
                                predictor,
                                loader,
                                device,
                                stage,
                                eval_mode,
                                args.batches,
                                loss_type,
                                normalize_loss,
                                amplitude_loss_weight,
                                args.baselines,
                                args.extrapolation_alpha,
                            )
                            rows.append(
                                row_from_metrics(
                                    stage=stage,
                                    history=history,
                                    train_mode=args.train_mode,
                                    eval_split=eval_split,
                                    eval_mode=eval_mode,
                                    method="learned_predictor",
                                    predictor_type=predictor_type,
                                    amplitude_loss_weight=amplitude_loss_weight,
                                    steps=args.steps,
                                    last_train_loss=last_loss if last_loss is not None else "",
                                    metrics=learned_metrics,
                                )
                            )
                            print(rows[-1])
                            for baseline, metrics in baseline_metrics.items():
                                rows.append(
                                    row_from_metrics(
                                        stage=stage,
                                        history=history,
                                        train_mode=args.train_mode,
                                        eval_split=eval_split,
                                        eval_mode=eval_mode,
                                        method=baseline,
                                        predictor_type="analytic",
                                        amplitude_loss_weight=amplitude_loss_weight,
                                        steps=args.steps,
                                        last_train_loss="",
                                        metrics=metrics,
                                    )
                                )
                                print(rows[-1])

    decision_rows = build_decision_rows(rows, args.baselines, args.modes)
    write_csv(out_dir / "prediction_error.csv", rows)
    write_csv(out_dir / "probe_learning_curve.csv", curve_rows)
    write_csv(out_dir / "decision_metrics.csv", decision_rows)
    write_json(
        out_dir / "probe_summary.json",
        {
            "rows": rows,
            "decision_rows": decision_rows,
            "learning_curve_rows": curve_rows,
            "skipped": skipped,
            "train_mode": args.train_mode,
            "eval_splits": args.eval_splits,
            "eval_modes": args.modes,
            "baselines": args.baselines,
            "predictor_types": predictor_types,
            "histories": histories,
            "amplitude_loss_weights": amplitude_weights,
            "extrapolation_alpha": args.extrapolation_alpha,
            "steps": args.steps,
            "batches": args.batches,
            "eval_every": args.eval_every,
            "curve_batches": args.curve_batches,
            "loss_type": loss_type,
            "normalize_loss": normalize_loss,
            "decision_criteria": {
                "passes_copy_gain_5pct": "relative_gain_vs_copy_previous_normalized_error >= 0.05",
                "passes_zero_raw_gain": "relative_gain_vs_zero_raw_error_mean > 0",
                "passes_temporal_specificity": "shuffle/reverse normalized_error at least 5% worse than normal",
                "passes_amplitude": "0.8 <= normal_prediction_abs_ratio <= 1.2",
                "passes_strict_probe": "all criteria above",
            },
        },
    )


if __name__ == "__main__":
    main()
