#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.models.temporal_predictors import FutureStatePredictor
from pegst.probing.metrics import (
    add_weighted,
    baseline_prediction,
    finalize_metrics,
    prediction_metrics,
    relative_gain,
    transform_sequence,
)
from pegst.probing.targets import collect_targets_from_batch, predictor_tensor, select_target_item
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.seed import seed_everything


DEFAULT_TARGETS = ["pre_membrane", "threshold_margin", "soft_firing_prob", "input_current"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M2 causal learned probe for membrane-aware internal targets.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--stages", nargs="+", default=["patch_embed1", "stage1", "patch_embed2", "stage2"])
    p.add_argument("--layer-patterns", nargs="*", default=None)
    p.add_argument("--audit-csv", default="")
    p.add_argument("--require-passes-m1", action="store_true")
    p.add_argument("--audit-split", default="test", choices=["train", "test", "both", "any"])
    p.add_argument("--modes", nargs="+", default=["normal", "shuffle", "reverse"], choices=["normal", "shuffle", "reverse"])
    p.add_argument("--train-mode", default="normal", choices=["normal", "shuffle", "reverse"])
    p.add_argument("--eval-splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--histories", nargs="+", type=int, default=[1, 2])
    p.add_argument("--baselines", nargs="+", default=["zero", "copy_previous", "linear_extrapolation"])
    p.add_argument("--predictor-types", nargs="+", default=["motion_extrapolation", "depthwise_conv"])
    p.add_argument("--amplitude-loss-weights", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--extrapolation-alpha", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--curve-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-type", default="smooth_l1", choices=["l1", "mse", "smooth_l1", "normalized_l1", "normalized_mse", "normalized_smooth_l1"])
    p.add_argument("--normalize-loss", action="store_true")
    p.add_argument("--soft-firing-temperature", type=float, default=0.25)
    return p.parse_args()


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes", "y"}


def audit_candidates(path: str | Path, require_pass: bool, audit_split: str = "test") -> set[tuple[str, str, str]]:
    if not path:
        return set()
    by_split: dict[tuple[str, str, str], set[str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if require_pass and not truthy(row.get("passes_m1", "")):
                continue
            key = (row["target"], row["stage"], row["layer"])
            by_split.setdefault(key, set()).add(row.get("split", ""))
    if audit_split == "any":
        return set(by_split)
    if audit_split == "both":
        return {key for key, splits in by_split.items() if {"train", "test"}.issubset(splits)}
    return {key for key, splits in by_split.items() if audit_split in splits}


@torch.no_grad()
def discover_candidates(
    model,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    allowed: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    x, _ = next(iter(loader))
    reset_spiking_state(model)
    items, _, _ = collect_targets_from_batch(
        model,
        x.to(device).float(),
        targets=args.targets,
        stages=args.stages,
        layer_patterns=args.layer_patterns,
        soft_firing_temperature=args.soft_firing_temperature,
    )
    candidates = []
    for item in items:
        key = (item.target, item.stage, item.layer)
        if allowed and key not in allowed:
            continue
        z, spatial = predictor_tensor(item.tensor)
        candidates.append(
            {
                "target": item.target,
                "stage": item.stage,
                "layer": item.layer,
                "channels": int(z.shape[2]),
                "spatial": spatial,
                "shape": list(item.tensor.shape),
            }
        )
    return candidates


def target_batches(
    model,
    loader,
    device: torch.device,
    candidate: dict[str, Any],
    mode: str,
    max_batches: int,
    args: argparse.Namespace,
) -> Iterable[torch.Tensor]:
    with torch.no_grad():
        for step, (x, _) in enumerate(loader):
            if step >= max_batches:
                break
            reset_spiking_state(model)
            items, _, _ = collect_targets_from_batch(
                model,
                x.to(device).float(),
                targets=[candidate["target"]],
                stages=[candidate["stage"]],
                layer_patterns=[candidate["layer"]],
                soft_firing_temperature=args.soft_firing_temperature,
            )
            item = select_target_item(items, candidate["target"], candidate["stage"], candidate["layer"])
            if item is None:
                continue
            z, _ = predictor_tensor(item.tensor.detach())
            yield transform_sequence(z, mode)


def evaluate_predictor(
    model,
    predictor: FutureStatePredictor,
    loader,
    device: torch.device,
    candidate: dict[str, Any],
    mode: str,
    batches: int,
    args: argparse.Namespace,
    amplitude_loss_weight: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    learned_acc: dict[str, float] = {}
    baseline_accs: dict[str, dict[str, float]] = {name: {} for name in args.baselines}
    with torch.no_grad():
        for z in target_batches(model, loader, device, candidate, mode, batches, args):
            if z.shape[0] < 2:
                continue
            batch = predictor.forward_sequence(
                z,
                loss_type=args.loss_type,  # type: ignore[arg-type]
                normalize_loss=args.normalize_loss,
                amplitude_loss_weight=amplitude_loss_weight,
            )
            add_weighted(
                learned_acc,
                prediction_metrics(
                    batch.prediction,
                    batch.target,
                    loss_type=args.loss_type,
                    normalize_loss=args.normalize_loss,
                    amplitude_loss_weight=amplitude_loss_weight,
                ),
                batch.target.shape[1],
            )
            for baseline in args.baselines:
                pred, target = baseline_prediction(z, baseline, alpha=args.extrapolation_alpha)
                add_weighted(
                    baseline_accs[baseline],
                    prediction_metrics(
                        pred,
                        target,
                        loss_type=args.loss_type,
                        normalize_loss=args.normalize_loss,
                        amplitude_loss_weight=amplitude_loss_weight,
                    ),
                    target.shape[1],
                )
    return finalize_metrics(learned_acc), {name: finalize_metrics(acc) for name, acc in baseline_accs.items()}


def row_from_metrics(
    *,
    candidate: dict[str, Any],
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
        "target": candidate["target"],
        "stage": candidate["stage"],
        "layer": candidate["layer"],
        "history": history,
        "train_mode": train_mode,
        "eval_split": eval_split,
        "eval_mode": eval_mode,
        "mode": eval_mode,
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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def build_decision_rows(rows: list[dict[str, Any]], baselines: list[str], eval_modes: list[str]) -> list[dict[str, Any]]:
    decision_rows: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                row["target"],
                row["stage"],
                row["layer"],
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
    for target, stage, layer, history, train_mode, eval_split, predictor_type, amp_weight, steps in group_keys:
        normal = find_row(
            rows,
            target=target,
            stage=stage,
            layer=layer,
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
            "target": target,
            "stage": stage,
            "layer": layer,
            "history": history,
            "train_mode": train_mode,
            "eval_split": eval_split,
            "predictor_type": predictor_type,
            "amplitude_loss_weight": amp_weight,
            "steps": steps,
            "normal_loss": normal.get("loss", float("nan")),
            "normal_normalized_error": normal.get("normalized_error", float("nan")),
            "normal_raw_error_mean": normal.get("raw_error_mean", float("nan")),
            "normal_prediction_abs_ratio": normal.get("prediction_abs_ratio", float("nan")),
        }
        amp_ratio = float(decision["normal_prediction_abs_ratio"]) if finite(decision["normal_prediction_abs_ratio"]) else float("nan")
        decision["amplitude_in_0p8_1p2"] = 0.8 <= amp_ratio <= 1.2
        for baseline in baselines:
            base = find_row(
                rows,
                target=target,
                stage=stage,
                layer=layer,
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
                target=target,
                stage=stage,
                layer=layer,
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
                if finite(row.get(metric)) and finite(normal.get(metric)):
                    decision[f"temporal_selectivity_{mode}_{metric}"] = float(row[metric]) / max(float(normal[metric]), 1e-12)
                else:
                    decision[f"temporal_selectivity_{mode}_{metric}"] = float("nan")
            decision[f"{mode}_prediction_abs_ratio"] = row.get("prediction_abs_ratio", float("nan"))
        gain_copy = float(decision.get("relative_gain_vs_copy_previous_normalized_error", float("nan")))
        gain_zero_raw = float(decision.get("relative_gain_vs_zero_raw_error_mean", float("nan")))
        shuffle_sel = float(decision.get("temporal_selectivity_shuffle_normalized_error", float("nan")))
        reverse_sel = float(decision.get("temporal_selectivity_reverse_normalized_error", float("nan")))
        decision["passes_copy_gain_5pct"] = finite(gain_copy) and gain_copy >= 0.05
        decision["passes_zero_raw_gain"] = finite(gain_zero_raw) and gain_zero_raw > 0.0
        decision["passes_temporal_specificity"] = finite(shuffle_sel) and finite(reverse_sel) and shuffle_sel > 1.05 and reverse_sel > 1.05
        decision["passes_amplitude"] = bool(decision["amplitude_in_0p8_1p2"])
        decision["passes_no_nan"] = all(finite(decision.get(k)) for k in ["normal_loss", "normal_normalized_error", "normal_raw_error_mean"])
        decision["passes_strict_probe"] = (
            decision["passes_copy_gain_5pct"]
            and decision["passes_zero_raw_gain"]
            and decision["passes_temporal_specificity"]
            and decision["passes_amplitude"]
            and decision["passes_no_nan"]
        )
        decision_rows.append(decision)
    return decision_rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    load_info = load_model_checkpoint(model, args.checkpoint, strict=False)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    train_loader = build_dataloader(cfg["dataset"], "train")
    eval_loaders = {split: build_dataloader(cfg["dataset"], split) for split in args.eval_splits}
    allowed = audit_candidates(args.audit_csv, args.require_passes_m1, args.audit_split)
    candidates = discover_candidates(model, train_loader, device, args, allowed)
    if not candidates:
        raise RuntimeError("No target candidates discovered. Check --targets, --stages, --layer-patterns, or --audit-csv.")

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    predictor_dir = out_dir / "predictors"
    predictor_dir.mkdir(exist_ok=True)

    for candidate in candidates:
        for history in args.histories:
            for predictor_type in args.predictor_types:
                if predictor_type == "motion_extrapolation" and history < 2:
                    skipped.append({**candidate, "history": history, "predictor_type": predictor_type, "reason": "requires_history_ge_2"})
                    continue
                for amplitude_loss_weight in args.amplitude_loss_weights:
                    predictor = FutureStatePredictor(
                        candidate["channels"],
                        history=history,
                        spatial=bool(candidate["spatial"]),
                        predictor_type=predictor_type,
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
                        reset_spiking_state(model)
                        with torch.no_grad():
                            items, _, _ = collect_targets_from_batch(
                                model,
                                x.to(device).float(),
                                targets=[candidate["target"]],
                                stages=[candidate["stage"]],
                                layer_patterns=[candidate["layer"]],
                                soft_firing_temperature=args.soft_firing_temperature,
                            )
                            item = select_target_item(items, candidate["target"], candidate["stage"], candidate["layer"])
                            if item is None:
                                continue
                            z, _ = predictor_tensor(item.tensor.detach())
                            z = transform_sequence(z, args.train_mode)
                        batch = predictor.forward_sequence(
                            z,
                            loss_type=args.loss_type,  # type: ignore[arg-type]
                            normalize_loss=args.normalize_loss,
                            amplitude_loss_weight=amplitude_loss_weight,
                        )
                        opt.zero_grad(set_to_none=True)
                        batch.loss.backward()
                        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=5.0)
                        opt.step()
                        last_loss = float(batch.loss.item())
                        if step_idx in eval_points:
                            curve_metrics, _ = evaluate_predictor(
                                model,
                                predictor,
                                eval_loaders["test"] if "test" in eval_loaders else train_loader,
                                device,
                                candidate,
                                "normal",
                                args.curve_batches,
                                args,
                                amplitude_loss_weight,
                            )
                            curve_rows.append(
                                {
                                    **{k: candidate[k] for k in ["target", "stage", "layer"]},
                                    "history": history,
                                    "train_mode": args.train_mode,
                                    "eval_split": "test" if "test" in eval_loaders else "train",
                                    "eval_mode": "normal",
                                    "method": "learned_predictor",
                                    "predictor_type": predictor_type,
                                    "amplitude_loss_weight": amplitude_loss_weight,
                                    "step": step_idx,
                                    "train_loss": last_loss if last_loss is not None else "",
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
                                candidate,
                                eval_mode,
                                args.batches,
                                args,
                                amplitude_loss_weight,
                            )
                            rows.append(
                                row_from_metrics(
                                    candidate=candidate,
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
                                        candidate=candidate,
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

                    safe_name = (
                        f"{candidate['target']}__{candidate['stage']}__"
                        f"{candidate['layer'].replace('.', '_')}__h{history}__"
                        f"{predictor_type}__amp{str(amplitude_loss_weight).replace('.', 'p')}.pt"
                    )
                    torch.save(
                        {
                            "predictor": predictor.state_dict(),
                            "candidate": candidate,
                            "history": history,
                            "predictor_type": predictor_type,
                            "amplitude_loss_weight": amplitude_loss_weight,
                            "loss_type": args.loss_type,
                            "normalize_loss": args.normalize_loss,
                            "soft_firing_temperature": args.soft_firing_temperature,
                        },
                        predictor_dir / safe_name,
                    )

    decision_rows = build_decision_rows(rows, args.baselines, args.modes)
    write_csv(out_dir / "prediction_error.csv", rows)
    write_csv(out_dir / "probe_learning_curve.csv", curve_rows)
    write_csv(out_dir / "decision_metrics.csv", decision_rows)
    write_json(
        out_dir / "probe_summary.json",
        {
            "candidates": candidates,
            "rows": rows,
            "decision_rows": decision_rows,
            "learning_curve_rows": curve_rows,
            "skipped": skipped,
            "checkpoint_load_info": load_info,
            "train_mode": args.train_mode,
            "eval_splits": args.eval_splits,
            "eval_modes": args.modes,
            "baselines": args.baselines,
            "predictor_types": args.predictor_types,
            "histories": args.histories,
            "amplitude_loss_weights": args.amplitude_loss_weights,
            "extrapolation_alpha": args.extrapolation_alpha,
            "steps": args.steps,
            "batches": args.batches,
            "eval_every": args.eval_every,
            "curve_batches": args.curve_batches,
            "loss_type": args.loss_type,
            "normalize_loss": args.normalize_loss,
            "decision_criteria": {
                "passes_copy_gain_5pct": "relative_gain_vs_copy_previous_normalized_error >= 0.05",
                "passes_zero_raw_gain": "relative_gain_vs_zero_raw_error_mean > 0",
                "passes_temporal_specificity": "shuffle/reverse normalized_error at least 5% worse than normal",
                "passes_amplitude": "0.8 <= normal_prediction_abs_ratio <= 1.2",
                "passes_no_nan": "normal learned metrics finite",
                "passes_strict_probe": "all criteria above",
            },
        },
    )


if __name__ == "__main__":
    main()
