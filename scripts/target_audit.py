#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.probing.metrics import (
    add_weighted,
    baseline_prediction,
    finalize_metrics,
    prediction_metrics,
    relative_gain,
    temporal_autocorrelation,
    tensor_distribution_metrics,
    transform_sequence,
)
from pegst.probing.targets import collect_targets_from_batch
from pegst.profiling.internal_state_collector import StreamingStats, write_recording_artifacts
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.seed import seed_everything


DEFAULT_TARGETS = [
    "latent_post_stage",
    "input_current",
    "pre_membrane",
    "post_reset_membrane",
    "threshold_margin",
    "soft_firing_prob",
    "spike_output",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 no-training audit for internal QKFormer/SNN targets.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--stages", nargs="+", default=["patch_embed1", "stage1", "patch_embed2", "stage2"])
    p.add_argument("--layer-patterns", nargs="*", default=None)
    p.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--modes", nargs="+", default=["normal", "shuffle", "reverse"], choices=["normal", "shuffle", "reverse"])
    p.add_argument("--baselines", nargs="+", default=["zero", "copy_previous", "linear_extrapolation"])
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--extrapolation-alpha", type=float, default=1.0)
    p.add_argument("--soft-firing-temperature", type=float, default=0.25)
    p.add_argument("--copy-gain-threshold", type=float, default=0.05)
    p.add_argument("--autocorr-threshold", type=float, default=0.2)
    p.add_argument("--temporal-selectivity-threshold", type=float, default=1.03)
    p.add_argument("--reverse-sanity-min", type=float, default=0.95)
    p.add_argument("--reverse-sanity-max", type=float, default=1.05)
    p.add_argument("--near-zero-max", type=float, default=0.95)
    return p.parse_args()


def avg_rows(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        out = grouped.setdefault(key, {col: row[col] for col in group_cols})
        counts[key] += 1
        for col, value in row.items():
            if col in group_cols:
                continue
            if isinstance(value, bool):
                out[col] = out.get(col, 0.0) + float(value)
            else:
                try:
                    out[col] = out.get(col, 0.0) + float(value)
                except Exception:
                    pass
    finalized = []
    for key, out in grouped.items():
        n = max(1, counts[key])
        row = dict(out)
        for col, value in list(row.items()):
            if col in group_cols:
                continue
            row[col] = value / n
            if col.startswith("passes_"):
                row[col] = row[col] >= 0.5
        row["num_layers"] = n
        finalized.append(row)
    return sorted(finalized, key=lambda r: tuple(r[c] for c in group_cols))


def build_decision_rows(
    by_layer_base: dict[tuple[str, str, str, str], dict[str, float]],
    distribution: dict[tuple[str, str, str, str], dict[str, float]],
    autocorr: dict[tuple[str, str, str, str], dict[str, float]],
    *,
    split: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    keys = sorted({(target, stage, layer) for (target, stage, layer, mode) in by_layer_base if mode == "normal"})
    rows: list[dict[str, Any]] = []
    for target, stage, layer in keys:
        normal = by_layer_base.get((target, stage, layer, "normal"), {})
        shuffle = by_layer_base.get((target, stage, layer, "shuffle"), {})
        reverse = by_layer_base.get((target, stage, layer, "reverse"), {})
        zero_raw = normal.get("zero_raw_error_mean", float("nan"))
        copy_raw = normal.get("copy_previous_raw_error_mean", float("nan"))
        linear_raw = normal.get("linear_extrapolation_raw_error_mean", float("nan"))
        copy_gain = relative_gain(copy_raw, zero_raw)
        linear_gain = relative_gain(linear_raw, zero_raw)
        shuffle_sel = shuffle.get("copy_previous_raw_error_mean", float("nan")) / max(copy_raw, 1e-12)
        reverse_sel = reverse.get("copy_previous_raw_error_mean", float("nan")) / max(copy_raw, 1e-12)
        dist = distribution.get((target, stage, layer, split), {})
        corr = autocorr.get((target, stage, layer, split), {})
        finite_ok = dist.get("finite_fraction", 0.0) >= 1.0
        amplitude_ok = dist.get("abs_mean", 0.0) > 1e-6
        not_zero_dominated = dist.get("near_zero_fraction", 1.0) <= args.near_zero_max
        passes_copy = copy_gain >= args.copy_gain_threshold
        passes_autocorr = corr.get("autocorr_lag1", float("nan")) >= args.autocorr_threshold
        passes_temporal_shuffle = shuffle_sel >= args.temporal_selectivity_threshold
        passes_reverse_sanity = args.reverse_sanity_min <= reverse_sel <= args.reverse_sanity_max
        row = {
            "split": split,
            "target": target,
            "stage": stage,
            "layer": layer,
            "zero_raw_error_mean": zero_raw,
            "copy_raw_error_mean": copy_raw,
            "linear_raw_error_mean": linear_raw,
            "relative_gain_copy_vs_zero_raw_error_mean": copy_gain,
            "relative_gain_linear_vs_zero_raw_error_mean": linear_gain,
            "temporal_selectivity_shuffle_copy_raw_error": shuffle_sel,
            "temporal_selectivity_reverse_copy_raw_error": reverse_sel,
            "autocorr_lag1": corr.get("autocorr_lag1", float("nan")),
            "autocorr_lag2": corr.get("autocorr_lag2", float("nan")),
            "abs_mean": dist.get("abs_mean", float("nan")),
            "std": dist.get("std", float("nan")),
            "near_zero_fraction": dist.get("near_zero_fraction", float("nan")),
            "finite_fraction": dist.get("finite_fraction", float("nan")),
            "passes_copy_beats_zero": passes_copy,
            "passes_autocorr": passes_autocorr,
            "passes_temporal_shuffle": passes_temporal_shuffle,
            "passes_reverse_sanity": passes_reverse_sanity,
            "passes_temporal_order": passes_temporal_shuffle,
            "passes_amplitude": amplitude_ok,
            "passes_not_zero_dominated": not_zero_dominated,
            "passes_no_nan": finite_ok,
        }
        row["passes_m1"] = (
            row["passes_copy_beats_zero"]
            and row["passes_autocorr"]
            and row["passes_temporal_order"]
            and row["passes_amplitude"]
            and row["passes_not_zero_dominated"]
            and row["passes_no_nan"]
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    load_info = {}
    if args.checkpoint:
        load_info = load_model_checkpoint(model, args.checkpoint, strict=False)
    model.eval()

    baseline_accs: dict[tuple[str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    distribution_accs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    autocorr_accs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    recording_stats = StreamingStats()
    profile = {
        "batches": 0,
        "forwards": 0,
        "forward_time_sec": 0.0,
        "max_cuda_memory_allocated": 0,
        "targets": args.targets,
        "stages": args.stages,
        "layer_patterns": args.layer_patterns or [],
        "checkpoint_load_info": load_info,
    }

    for split in args.splits:
        loader = build_dataloader(cfg["dataset"], split)
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= args.batches:
                break
            x = x.to(device).float()
            reset_spiking_state(model)
            with torch.no_grad():
                target_items, _, rec_profile = collect_targets_from_batch(
                    model,
                    x,
                    targets=args.targets,
                    stages=args.stages,
                    layer_patterns=args.layer_patterns,
                    soft_firing_temperature=args.soft_firing_temperature,
                )
            profile["batches"] += 1
            profile["forwards"] += rec_profile.forwards
            profile["forward_time_sec"] += rec_profile.forward_time_sec
            profile["max_cuda_memory_allocated"] = max(
                int(profile["max_cuda_memory_allocated"]),
                rec_profile.max_cuda_memory_allocated,
            )
            state_map: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for item in target_items:
                if item.target != "latent_post_stage":
                    state_map[item.layer][item.target] = item.tensor
            recording_stats.update(state_map)

            for item in target_items:
                z_normal = item.tensor.detach()
                dist_key = (item.target, item.stage, item.layer, split)
                dist = tensor_distribution_metrics(z_normal)
                add_weighted(distribution_accs[dist_key], dist, z_normal.shape[1])
                autocorr_key = (item.target, item.stage, item.layer, split)
                add_weighted(
                    autocorr_accs[autocorr_key],
                    {
                        "autocorr_lag1": temporal_autocorrelation(z_normal, 1),
                        "autocorr_lag2": temporal_autocorrelation(z_normal, 2),
                    },
                    z_normal.shape[1],
                )
                for mode in args.modes:
                    z = transform_sequence(z_normal, mode)
                    if z.shape[0] < 2:
                        continue
                    for baseline in args.baselines:
                        pred, target = baseline_prediction(z, baseline, alpha=args.extrapolation_alpha)
                        metrics = prediction_metrics(pred, target, loss_type="l1")
                        key = (item.target, item.stage, item.layer, split, mode, baseline)
                        add_weighted(baseline_accs[key], metrics, target.shape[1])

    baseline_rows = []
    for (target, stage, layer, split, mode, baseline), acc in sorted(baseline_accs.items()):
        row = {
            "target": target,
            "stage": stage,
            "layer": layer,
            "split": split,
            "mode": mode,
            "method": baseline,
            **finalize_metrics(acc),
        }
        baseline_rows.append(row)

    autocorr_rows = [
        {"target": target, "stage": stage, "layer": layer, "split": split, **finalize_metrics(acc)}
        for (target, stage, layer, split), acc in sorted(autocorr_accs.items())
    ]
    distribution_rows = [
        {"target": target, "stage": stage, "layer": layer, "split": split, **finalize_metrics(acc)}
        for (target, stage, layer, split), acc in sorted(distribution_accs.items())
    ]

    decision_by_split: list[dict[str, Any]] = []
    for split in args.splits:
        base_lookup: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
        for row in baseline_rows:
            if row["split"] != split:
                continue
            key = (row["target"], row["stage"], row["layer"], row["mode"])
            for metric_name in ("loss", "normalized_error", "raw_error_mean", "prediction_abs_ratio"):
                base_lookup[key][f"{row['method']}_{metric_name}"] = row.get(metric_name, float("nan"))
        dist_lookup = {
            (row["target"], row["stage"], row["layer"], row["split"]): row
            for row in distribution_rows
            if row["split"] == split
        }
        corr_lookup = {
            (row["target"], row["stage"], row["layer"], row["split"]): row
            for row in autocorr_rows
            if row["split"] == split
        }
        decision_by_split.extend(build_decision_rows(base_lookup, dist_lookup, corr_lookup, split=split, args=args))

    by_stage = avg_rows(decision_by_split, ["split", "target", "stage"])
    summary = avg_rows(decision_by_split, ["split", "target"])
    stats_rows = recording_stats.finalize_rows()
    shapes = recording_stats.finalize_shapes()
    profile["forward_time_per_batch_sec"] = profile["forward_time_sec"] / max(1, profile["forwards"])

    write_csv(out_dir / "target_baseline_errors.csv", baseline_rows)
    write_csv(out_dir / "target_autocorrelation.csv", autocorr_rows)
    write_csv(out_dir / "target_distribution.csv", distribution_rows)
    write_csv(out_dir / "target_audit_by_layer.csv", decision_by_split)
    write_csv(out_dir / "target_audit_by_stage.csv", by_stage)
    write_csv(out_dir / "target_audit_summary.csv", summary)
    write_recording_artifacts(out_dir, stats_rows=stats_rows, shapes=shapes, profile=profile)
    write_json(
        out_dir / "target_audit_criteria.json",
        {
            "copy_beats_zero": f"relative_gain_copy_vs_zero_raw_error_mean >= {args.copy_gain_threshold}",
            "autocorr": f"autocorr_lag1 >= {args.autocorr_threshold}",
            "temporal_order": f"shuffle copy raw-error selectivity >= {args.temporal_selectivity_threshold}",
            "reverse_sanity": f"{args.reverse_sanity_min} <= reverse copy raw-error selectivity <= {args.reverse_sanity_max}",
            "not_zero_dominated": f"near_zero_fraction <= {args.near_zero_max}",
            "amplitude": "abs_mean > 1e-6",
            "no_nan": "finite_fraction == 1.0",
        },
    )
    print({"output_dir": str(out_dir), "summary_rows": len(summary), "by_layer_rows": len(decision_by_split)})


if __name__ == "__main__":
    main()
