#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.probing.targets import collect_targets_from_batch, predictor_tensor
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.progress import Timer, log, should_log
from pegst.utils.seed import seed_everything


DEFAULT_TARGETS = ["pre_membrane", "threshold_margin", "soft_firing_prob", "post_reset_membrane"]
DEFAULT_LAYERS = [
    "patch_embed1.proj_lif",
    "patch_embed1.proj_res_lif",
    "patch_embed2.proj_lif",
    "patch_embed2.proj_res_lif",
    "stage1.0.tssa.proj_lif",
    "stage1.0.mlp.mlp2_lif",
    "stage2.0.ssa.proj_lif",
    "stage2.0.mlp.mlp2_lif",
]
DEFAULT_ERROR_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
DEFAULT_SIGNAL_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E temporal-consistency-gated early-exit sweep.")
    p.add_argument("--config", nargs="+", required=True, help="One or more baseline configs.")
    p.add_argument("--checkpoint", nargs="+", required=True, help="One checkpoint per config.")
    p.add_argument("--run-names", nargs="+", default=None, help="Optional names, one per config/checkpoint pair.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batches", type=int, default=0)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--stages", nargs="+", default=["patch_embed1", "stage1", "patch_embed2", "stage2"])
    p.add_argument("--error-types", nargs="+", default=["copy", "linear"])
    p.add_argument("--linear-alphas", nargs="+", type=float, default=[1.0, 0.5])
    p.add_argument("--aggregations", nargs="+", default=["mean", "max", "topk_mean"])
    p.add_argument("--topk-fraction", type=float, default=0.10)
    p.add_argument("--confidence-signals", nargs="+", default=["confidence", "entropy", "logit_margin"])
    p.add_argument("--confidence-thresholds", nargs="+", type=float, default=DEFAULT_CONFIDENCE_THRESHOLDS)
    p.add_argument("--signal-quantiles", nargs="+", type=float, default=DEFAULT_SIGNAL_QUANTILES)
    p.add_argument("--error-quantiles", nargs="+", type=float, default=DEFAULT_ERROR_QUANTILES)
    p.add_argument("--soft-firing-temperature", type=float, default=0.25)
    p.add_argument("--max-accuracy-drop", type=float, default=0.5, help="Percent points.")
    p.add_argument("--target-sops-fraction", type=float, default=0.70)
    p.add_argument("--include-confidence-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--classwise-for", choices=["all", "pareto", "none"], default="all")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    if len(args.config) != len(args.checkpoint):
        raise ValueError("--config and --checkpoint must have the same length.")
    if args.run_names is not None and len(args.run_names) != len(args.config):
        raise ValueError("--run-names must have the same length as --config.")
    return args


def canonical_layer(layer: str) -> str:
    # The project uses SSA in stage2; allow the document's tssa spelling as a convenience.
    return layer.replace("stage2.0.tssa.", "stage2.0.ssa.")


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def logit_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def finite_quantiles(values: torch.Tensor, quantiles: list[float]) -> list[tuple[float, float]]:
    flat = values.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return []
    out = []
    for q in quantiles:
        qf = min(1.0, max(0.0, float(q)))
        out.append((qf, float(torch.quantile(flat, qf).item())))
    return out


def aggregate_error(diff: torch.Tensor, reference: torch.Tensor, aggregation: str, topk_fraction: float) -> torch.Tensor:
    flat = diff.abs().flatten(2)
    if aggregation == "mean":
        return flat.mean(dim=2)
    if aggregation == "max":
        return flat.max(dim=2).values
    if aggregation == "topk_mean":
        k = max(1, int(math.ceil(flat.shape[2] * max(0.0, min(1.0, topk_fraction)))))
        return torch.topk(flat, k=k, dim=2).values.mean(dim=2)
    if aggregation == "foreground_weighted_mean":
        weights = reference.detach().abs().flatten(2)
        return (flat * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def temporal_error_sequence(
    z: torch.Tensor,
    *,
    error_type: str,
    alpha: float,
    aggregation: str,
    topk_fraction: float,
) -> torch.Tensor:
    T, B = z.shape[:2]
    out = torch.full((T, B), float("inf"), dtype=torch.float32, device=z.device)
    if error_type == "copy":
        if T < 2:
            return out
        err = aggregate_error(z[1:] - z[:-1], z[1:], aggregation, topk_fraction)
        out[1:] = err.float()
        return out
    if error_type == "linear":
        if T < 3:
            return out
        pred = z[1:-1] + float(alpha) * (z[1:-1] - z[:-2])
        err = aggregate_error(z[2:] - pred, z[2:], aggregation, topk_fraction)
        out[2:] = err.float()
        return out
    raise ValueError(f"Unknown error type: {error_type}")


def threshold_direction(signal: str) -> str:
    if signal == "entropy":
        return "le"
    if signal in {"confidence", "logit_margin"}:
        return "ge"
    raise ValueError(f"Unknown confidence signal: {signal}")


def decision_mask(values: torch.Tensor, signal: str, threshold: float) -> torch.Tensor:
    if threshold_direction(signal) == "le":
        return values <= float(threshold)
    return values >= float(threshold)


def sanitize_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return "nan"
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def make_rule_id(row: dict[str, Any]) -> str:
    parts = [
        str(row["run_name"]).replace("/", "_"),
        f"T{row['T']}",
        str(row["target"]),
        str(row["layer"]).replace(".", "_"),
        str(row["error_type"]),
        f"a{sanitize_float(row['alpha'])}",
        str(row["aggregation"]),
        str(row["confidence_signal"]),
        f"ct{sanitize_float(row['confidence_threshold'])}",
        f"eq{sanitize_float(row['error_quantile'])}",
    ]
    return "__".join(parts)


def threshold_key(value: Any) -> str:
    return f"{float(value):.12g}"


def infer_run_name(config_path: str, checkpoint_path: str, cfg: dict[str, Any], index: int) -> str:
    t = cfg.get("model", {}).get("T", cfg.get("timesteps", "?"))
    ckpt_parent = Path(checkpoint_path).parent.name
    return ckpt_parent if ckpt_parent else f"T{t}_{index}"


def classwise_full_stats(labels: torch.Tensor, full_pred: torch.Tensor) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for label in sorted(int(v) for v in labels.unique().tolist()):
        mask = labels == label
        total = int(mask.sum().item())
        correct = int((full_pred[mask] == labels[mask]).sum().item())
        stats[label] = {
            "num_samples": float(total),
            "full_accuracy": 100.0 * correct / max(1, total),
        }
    return stats


def evaluate_rule(
    *,
    labels: torch.Tensor,
    full_pred: torch.Tensor,
    timestep_pred: torch.Tensor,
    decision_values: torch.Tensor,
    decision_signal: str,
    decision_threshold: float,
    error_values: torch.Tensor | None,
    error_threshold: float | None,
    full_accuracy: float,
    full_class_stats: dict[int, dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    T, N = timestep_pred.shape
    exit_t = torch.full((N,), T - 1, dtype=torch.long)
    final_pred = full_pred.detach().cpu().clone()
    exited = torch.zeros(N, dtype=torch.bool)
    decision_cpu = decision_values.detach().cpu()
    error_cpu = error_values.detach().cpu() if error_values is not None else None
    for t in range(T):
        mask = (~exited) & decision_mask(decision_cpu[t], decision_signal, decision_threshold)
        if error_cpu is not None:
            err_t = error_cpu[t]
            mask = mask & torch.isfinite(err_t) & (err_t <= float(error_threshold))
        if mask.any():
            final_pred[mask] = timestep_pred[t, mask]
            exit_t[mask] = t
            exited |= mask
    correct = final_pred == labels
    accuracy = 100.0 * float(correct.float().mean().item())
    avg_exit = float((exit_t.float() + 1.0).mean().item())
    early_exit_coverage = float((exit_t < T - 1).float().mean().item())
    dist_counts = Counter(int(v.item()) for v in exit_t)
    dist_rows = [
        {
            "timestep": t + 1,
            "count": int(dist_counts.get(t, 0)),
            "fraction": int(dist_counts.get(t, 0)) / max(1, N),
        }
        for t in range(T)
    ]
    class_rows = []
    worst_drop = float("-inf")
    for label, full_stats in full_class_stats.items():
        mask = labels == int(label)
        total = int(mask.sum().item())
        if total == 0:
            continue
        exit_acc = 100.0 * float((final_pred[mask] == labels[mask]).float().mean().item())
        avg_exit_label = float((exit_t[mask].float() + 1.0).mean().item())
        drop = float(full_stats["full_accuracy"]) - exit_acc
        worst_drop = max(worst_drop, drop)
        class_rows.append(
            {
                "label": int(label),
                "full_accuracy": float(full_stats["full_accuracy"]),
                "exit_accuracy": exit_acc,
                "accuracy_drop": drop,
                "avg_exit_timestep": avg_exit_label,
                "num_samples": total,
            }
        )
    summary = {
        "accuracy": accuracy,
        "full_accuracy": full_accuracy,
        "accuracy_drop": full_accuracy - accuracy,
        "avg_exit_timestep": avg_exit,
        "expected_sops_fraction": avg_exit / max(1, T),
        "early_exit_coverage": early_exit_coverage,
        "num_samples": int(N),
        "worst_class_drop": 0.0 if worst_drop == float("-inf") else worst_drop,
    }
    return summary, class_rows, dist_rows


def add_confidence_only_comparison(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["error_type"] != "none":
            continue
        key = (
            str(row["run_name"]),
            int(row["T"]),
            str(row["confidence_signal"]),
            threshold_key(row["confidence_threshold"]),
        )
        baselines[key] = row

    metric_names = [
        "accuracy",
        "accuracy_drop",
        "avg_exit_timestep",
        "expected_sops_fraction",
        "early_exit_coverage",
        "worst_class_drop",
    ]
    for row in rows:
        key = (
            str(row["run_name"]),
            int(row["T"]),
            str(row["confidence_signal"]),
            threshold_key(row["confidence_threshold"]),
        )
        baseline = baselines.get(key)
        row["confidence_only_rule_id"] = baseline["rule_id"] if baseline else ""
        for name in metric_names:
            row[f"confidence_only_{name}"] = baseline[name] if baseline else ""
            row[f"{name}_delta_vs_confidence_only"] = (
                float(row[name]) - float(baseline[name]) if baseline else float("nan")
            )
        if baseline is None or row["error_type"] == "none":
            row["dominates_confidence_only"] = False
            row["dominated_by_confidence_only"] = False
            row["temporal_stability_adds_accuracy"] = False
            continue
        accuracy_delta = float(row["accuracy_delta_vs_confidence_only"])
        sops_delta = float(row["expected_sops_fraction_delta_vs_confidence_only"])
        row["dominates_confidence_only"] = (
            accuracy_delta >= 0.0
            and sops_delta <= 0.0
            and (accuracy_delta > 1e-9 or sops_delta < -1e-9)
        )
        row["dominated_by_confidence_only"] = (
            accuracy_delta <= 0.0
            and sops_delta >= 0.0
            and (accuracy_delta < -1e-9 or sops_delta > 1e-9)
        )
        row["temporal_stability_adds_accuracy"] = accuracy_delta > 0.0


def confidence_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["error_type"] != "none" and row.get("confidence_only_rule_id")]


def collect_run_data(
    *,
    config_path: str,
    checkpoint_path: str,
    run_name: str,
    args: argparse.Namespace,
    output_dir: Path,
    run_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    timer = Timer()
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 2021)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    log(f"[{run_name}] loading checkpoint")
    load_info = load_model_checkpoint(model, checkpoint_path, strict=bool(cfg.get("eval_strict", False)))
    model.eval()
    loader = build_dataloader(cfg["dataset"], args.split)
    T_cfg = int(cfg.get("model", {}).get("T", cfg.get("timesteps", cfg.get("dataset", {}).get("T", 0))))
    targets = list(dict.fromkeys(args.targets))
    layers = [canonical_layer(v) for v in args.layers]
    layer_set = set(layers)
    layer_patterns = layers
    log(
        f"[{run_name}] collecting split={args.split}, T={T_cfg}, "
        f"targets={targets}, layers={len(layers)}, batches={'all' if args.batches <= 0 else args.batches}"
    )

    labels_chunks: list[torch.Tensor] = []
    full_pred_chunks: list[torch.Tensor] = []
    timestep_pred_chunks: list[torch.Tensor] = []
    decision_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    error_chunks: dict[tuple[str, str, str, str, float | str, str], list[torch.Tensor]] = defaultdict(list)
    inventory: dict[tuple[str, str, str], int] = defaultdict(int)

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
                targets=targets,
                stages=args.stages,
                layer_patterns=layer_patterns,
                soft_firing_temperature=args.soft_firing_temperature,
            )
            logits = out["logits"]
            timestep_logits = out["timestep_logits"]
            probs_t = torch.softmax(timestep_logits, dim=-1)
            confidence_t, pred_t = probs_t.max(dim=-1)
            full_pred = logits.argmax(dim=-1)
            labels_chunks.append(y.detach().cpu())
            full_pred_chunks.append(full_pred.detach().cpu())
            timestep_pred_chunks.append(pred_t.detach().cpu())
            if "confidence" in args.confidence_signals:
                decision_chunks["confidence"].append(confidence_t.detach().cpu())
            if "entropy" in args.confidence_signals:
                decision_chunks["entropy"].append(entropy_from_probs(probs_t).detach().cpu())
            if "logit_margin" in args.confidence_signals:
                decision_chunks["logit_margin"].append(logit_margin(timestep_logits).detach().cpu())

            for item in items:
                layer = canonical_layer(item.layer)
                if item.target != "latent_post_stage" and layer not in layer_set:
                    continue
                z, _ = predictor_tensor(item.tensor.detach())
                z = z.float()
                inventory[(item.target, item.stage, layer)] += int(y.numel())
                for aggregation in args.aggregations:
                    if "copy" in args.error_types:
                        seq = temporal_error_sequence(
                            z,
                            error_type="copy",
                            alpha=0.0,
                            aggregation=aggregation,
                            topk_fraction=args.topk_fraction,
                        )
                        error_chunks[(item.target, item.stage, layer, "copy", "", aggregation)].append(seq.detach().cpu())
                    if "linear" in args.error_types:
                        for alpha in args.linear_alphas:
                            seq = temporal_error_sequence(
                                z,
                                error_type="linear",
                                alpha=float(alpha),
                                aggregation=aggregation,
                                topk_fraction=args.topk_fraction,
                            )
                            error_chunks[(item.target, item.stage, layer, "linear", float(alpha), aggregation)].append(
                                seq.detach().cpu()
                            )
            if should_log(batch_no, args.batches if args.batches > 0 else None, args.log_every):
                log(
                    f"[{run_name}] processed batch {batch_no}, "
                    f"samples={sum(int(v.numel()) for v in labels_chunks)}, "
                    f"candidate_signals={len(error_chunks)}, elapsed={timer.elapsed_str()}"
                )

    if not labels_chunks:
        raise RuntimeError(f"[{run_name}] no samples collected.")
    labels = torch.cat(labels_chunks, dim=0)
    full_pred = torch.cat(full_pred_chunks, dim=0)
    timestep_pred = torch.cat(timestep_pred_chunks, dim=1)
    decision_values = {name: torch.cat(chunks, dim=1) for name, chunks in decision_chunks.items()}
    error_values = {key: torch.cat(chunks, dim=1) for key, chunks in error_chunks.items() if chunks}
    T = int(timestep_pred.shape[0])
    full_accuracy = 100.0 * float((full_pred == labels).float().mean().item())
    full_class_stats = classwise_full_stats(labels, full_pred)
    log(f"[{run_name}] collected {labels.numel()} samples, T={T}, full_accuracy={full_accuracy:.3f}%")

    sweep_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    rule_index = 0

    def append_rule(
        base: dict[str, Any],
        decision_signal: str,
        decision_threshold: float,
        error_seq: torch.Tensor | None,
        error_threshold: float | None,
        error_quantile: float | str,
    ) -> None:
        nonlocal rule_index
        metric_row, class_rows, dist_rows = evaluate_rule(
            labels=labels,
            full_pred=full_pred,
            timestep_pred=timestep_pred,
            decision_values=decision_values[decision_signal],
            decision_signal=decision_signal,
            decision_threshold=decision_threshold,
            error_values=error_seq,
            error_threshold=error_threshold,
            full_accuracy=full_accuracy,
            full_class_stats=full_class_stats,
        )
        row = {
            "rule_id": "",
            "run_name": run_name,
            "run_index": run_index,
            "T": T,
            "checkpoint": checkpoint_path,
            "config": config_path,
            **base,
            "confidence_signal": decision_signal,
            "confidence_threshold": float(decision_threshold),
            "confidence_threshold_source": "fixed" if decision_signal == "confidence" else "quantile",
            "error_quantile": error_quantile,
            "error_threshold": "" if error_threshold is None else float(error_threshold),
            **metric_row,
        }
        row["passes_phase_e_target"] = (
            float(row["accuracy_drop"]) <= float(args.max_accuracy_drop)
            and float(row["expected_sops_fraction"]) <= float(args.target_sops_fraction)
        )
        row["rule_id"] = make_rule_id(row)
        row["rule_index"] = rule_index
        rule_index += 1
        sweep_rows.append(row)
        for class_row in class_rows:
            classwise_rows.append({"rule_id": row["rule_id"], **class_row})
        for dist_row in dist_rows:
            distribution_rows.append({"rule_id": row["rule_id"], **dist_row})

    decision_thresholds: dict[str, list[tuple[float | str, float]]] = {}
    for signal in args.confidence_signals:
        if signal not in decision_values:
            continue
        if signal == "confidence":
            decision_thresholds[signal] = [("fixed", float(v)) for v in args.confidence_thresholds]
        else:
            decision_thresholds[signal] = finite_quantiles(decision_values[signal], args.signal_quantiles)

    if args.include_confidence_only:
        log(f"[{run_name}] evaluating confidence-only baselines")
        for signal, thresholds in decision_thresholds.items():
            for _, threshold in thresholds:
                append_rule(
                    {
                        "target": "none",
                        "stage": "none",
                        "layer": "none",
                        "error_type": "none",
                        "alpha": "",
                        "aggregation": "none",
                    },
                    signal,
                    threshold,
                    None,
                    None,
                    "",
                )

    log(f"[{run_name}] evaluating temporal consistency rules: {len(error_values)} candidate signals")
    for candidate_index, (key, error_seq) in enumerate(sorted(error_values.items()), start=1):
        target, stage, layer, error_type, alpha, aggregation = key
        thresholds = finite_quantiles(error_seq, args.error_quantiles)
        if not thresholds:
            continue
        if should_log(candidate_index, len(error_values), max(1, args.log_every)):
            log(
                f"[{run_name}] candidate {candidate_index}/{len(error_values)}: "
                f"{target}/{layer}/{error_type}/alpha={alpha}/agg={aggregation}"
            )
        for signal, signal_thresholds in decision_thresholds.items():
            for _, signal_threshold in signal_thresholds:
                for error_quantile, error_threshold in thresholds:
                    append_rule(
                        {
                            "target": target,
                            "stage": stage,
                            "layer": layer,
                            "error_type": error_type,
                            "alpha": alpha,
                            "aggregation": aggregation,
                        },
                        signal,
                        signal_threshold,
                        error_seq,
                        error_threshold,
                        error_quantile,
                    )

    inventory_rows = [
        {
            "run_name": run_name,
            "T": T,
            "target": target,
            "stage": stage,
            "layer": layer,
            "num_samples_seen": count,
        }
        for (target, stage, layer), count in sorted(inventory.items())
    ]
    write_csv(output_dir / f"target_layer_inventory_{run_name}.csv", inventory_rows)
    add_confidence_only_comparison(sweep_rows)
    run_summary = {
        "run_name": run_name,
        "run_index": run_index,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "T": T,
        "num_samples": int(labels.numel()),
        "full_accuracy": full_accuracy,
        "num_rules": len(sweep_rows),
        "num_temporal_candidate_signals": len(error_values),
        "targets": targets,
        "layers": layers,
        "checkpoint_load_info": load_info,
        "elapsed_sec": timer.elapsed(),
    }
    log(f"[{run_name}] finished sweep: rules={len(sweep_rows)}, elapsed={timer.elapsed_str()}")
    return sweep_rows, classwise_rows, distribution_rows, run_summary


def pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            float(r["avg_exit_timestep"]),
            float(r["expected_sops_fraction"]),
            -float(r["accuracy"]),
            float(r["worst_class_drop"]),
        ),
    )
    pareto = []
    best_accuracy = float("-inf")
    for row in sorted_rows:
        acc = float(row["accuracy"])
        if acc > best_accuracy + 1e-9:
            pareto.append(row)
            best_accuracy = acc
    return pareto


def choose_best(rows: list[dict[str, Any]], *, key: str, reverse: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda r: float(r[key]), reverse=reverse)[0]


def choose_best_tradeoff(rows: list[dict[str, Any]], *, max_drop: float, max_sops: float) -> dict[str, Any] | None:
    feasible = [
        r
        for r in rows
        if float(r["accuracy_drop"]) <= max_drop and float(r["expected_sops_fraction"]) <= max_sops
    ]
    if feasible:
        return sorted(
            feasible,
            key=lambda r: (
                float(r["expected_sops_fraction"]),
                float(r["worst_class_drop"]),
                -float(r["accuracy"]),
            ),
        )[0]
    return None


def best_rules(rows: list[dict[str, Any]], run_summaries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    savings_30 = [r for r in rows if float(r["expected_sops_fraction"]) <= args.target_sops_fraction]
    within_drop = [r for r in rows if float(r["accuracy_drop"]) <= args.max_accuracy_drop]
    temporal_rows = [r for r in rows if r["error_type"] != "none"]
    hardware = [
        r
        for r in rows
        if r["target"] == "threshold_margin"
        and r["error_type"] == "copy"
        and r["confidence_signal"] == "logit_margin"
        and r["aggregation"] in {"mean", "max", "topk_mean"}
    ]
    hardware_feasible = [
        r
        for r in hardware
        if float(r["accuracy_drop"]) <= args.max_accuracy_drop
        and float(r["expected_sops_fraction"]) <= args.target_sops_fraction
    ]
    t8_full = max((float(s["full_accuracy"]) for s in run_summaries if int(s["T"]) == 8), default=float("nan"))
    t16_candidates = [
        r
        for r in rows
        if int(r["T"]) == 16
        and math.isfinite(t8_full)
        and float(r["accuracy"]) >= t8_full
        and float(r["expected_sops_fraction"]) <= 0.50
    ]
    return {
        "criteria": {
            "accuracy_units": "percent",
            "accuracy_drop_units": "percentage_points",
            "phase_e_promising": f"accuracy_drop <= {args.max_accuracy_drop} and expected_sops_fraction <= {args.target_sops_fraction}",
            "best_T16_vs_T8_tradeoff": "T16 exit accuracy >= best T8 full accuracy and expected_sops_fraction <= 0.50",
        },
        "best_accuracy_with_30pct_savings": choose_best(savings_30, key="accuracy", reverse=True),
        "best_savings_within_0p5_acc_drop": choose_best(within_drop, key="expected_sops_fraction", reverse=False),
        "best_phase_e_tradeoff": choose_best_tradeoff(rows, max_drop=args.max_accuracy_drop, max_sops=args.target_sops_fraction),
        "best_temporal_stability_accuracy_gain_vs_confidence_only": choose_best(
            temporal_rows,
            key="accuracy_delta_vs_confidence_only",
            reverse=True,
        ),
        "best_temporal_stability_sops_gain_vs_confidence_only": choose_best(
            temporal_rows,
            key="expected_sops_fraction_delta_vs_confidence_only",
            reverse=False,
        ),
        "best_temporal_stability_non_dominated_vs_confidence_only": choose_best(
            [r for r in temporal_rows if not bool(r.get("dominated_by_confidence_only", False))],
            key="accuracy_delta_vs_confidence_only",
            reverse=True,
        ),
        "best_hardware_friendly_rule": choose_best_tradeoff(
            hardware_feasible or hardware,
            max_drop=args.max_accuracy_drop if hardware_feasible else 1e9,
            max_sops=args.target_sops_fraction if hardware_feasible else 1e9,
        ),
        "best_T16_vs_T8_tradeoff": choose_best(t16_candidates, key="expected_sops_fraction", reverse=False),
        "best_T8_full_accuracy": None if not math.isfinite(t8_full) else t8_full,
    }


def main() -> None:
    args = parse_args()
    timer = Timer()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("Phase E temporal consistency early-exit sweep started")
    log(f"output_dir={out_dir}")
    log(f"configs={args.config}")
    log(f"checkpoints={args.checkpoint}")
    all_sweep_rows: list[dict[str, Any]] = []
    all_classwise_rows: list[dict[str, Any]] = []
    all_distribution_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for i, (config_path, checkpoint_path) in enumerate(zip(args.config, args.checkpoint, strict=True)):
        cfg = load_config(config_path)
        run_name = args.run_names[i] if args.run_names else infer_run_name(config_path, checkpoint_path, cfg, i)
        run_rows, class_rows, dist_rows, summary = collect_run_data(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            args=args,
            output_dir=out_dir,
            run_index=i,
        )
        all_sweep_rows.extend(run_rows)
        all_classwise_rows.extend(class_rows)
        all_distribution_rows.extend(dist_rows)
        run_summaries.append(summary)

    pareto = pareto_rows(all_sweep_rows)
    pareto_rule_ids = {row["rule_id"] for row in pareto}
    classwise_out = all_classwise_rows
    if args.classwise_for == "pareto":
        classwise_out = [row for row in all_classwise_rows if row["rule_id"] in pareto_rule_ids]
    elif args.classwise_for == "none":
        classwise_out = []
    best = best_rules(all_sweep_rows, run_summaries, args)
    confidence_comparison = confidence_comparison_rows(all_sweep_rows)
    promising = [
        row
        for row in all_sweep_rows
        if float(row["accuracy_drop"]) <= args.max_accuracy_drop
        and float(row["expected_sops_fraction"]) <= args.target_sops_fraction
    ]
    summary = {
        "num_runs": len(run_summaries),
        "num_rules": len(all_sweep_rows),
        "num_temporal_stability_rules": len(confidence_comparison),
        "num_temporal_rules_dominated_by_confidence_only": sum(
            1 for row in confidence_comparison if bool(row.get("dominated_by_confidence_only", False))
        ),
        "num_temporal_rules_dominating_confidence_only": sum(
            1 for row in confidence_comparison if bool(row.get("dominates_confidence_only", False))
        ),
        "num_temporal_rules_with_accuracy_gain_vs_confidence_only": sum(
            1 for row in confidence_comparison if bool(row.get("temporal_stability_adds_accuracy", False))
        ),
        "num_pareto_rules": len(pareto),
        "num_promising_rules": len(promising),
        "max_accuracy_drop": args.max_accuracy_drop,
        "target_sops_fraction": args.target_sops_fraction,
        "targets": args.targets,
        "layers": [canonical_layer(v) for v in args.layers],
        "error_types": args.error_types,
        "linear_alphas": args.linear_alphas,
        "aggregations": args.aggregations,
        "confidence_signals": args.confidence_signals,
        "run_summaries": run_summaries,
        "best_rules": best,
        "elapsed_sec": timer.elapsed(),
    }
    write_csv(out_dir / "early_exit_sweep.csv", all_sweep_rows)
    write_csv(out_dir / "confidence_vs_temporal_stability.csv", confidence_comparison)
    write_csv(out_dir / "early_exit_pareto.csv", pareto)
    write_csv(out_dir / "classwise_exit_analysis.csv", classwise_out)
    write_csv(out_dir / "exit_timestep_distribution.csv", all_distribution_rows)
    write_json(out_dir / "best_rules.json", best)
    write_json(out_dir / "summary.json", summary)
    log(
        "Phase E summary: "
        f"rules={len(all_sweep_rows)}, pareto={len(pareto)}, promising={len(promising)}, "
        f"elapsed={timer.elapsed_str()}"
    )
    log(f"Phase E finished -> {out_dir}")


if __name__ == "__main__":
    main()
