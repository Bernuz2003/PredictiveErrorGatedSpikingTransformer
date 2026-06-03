#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.progress import Timer, log


@torch.no_grad()
def main() -> None:
    timer = Timer()
    p = argparse.ArgumentParser(description="Evaluate confidence-based early exit over QKFormer timestep logits.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--confidence-thresholds", nargs="+", type=float, default=None)
    args = p.parse_args()
    log("early-exit evaluation started")
    log(f"config={args.config}")
    log(f"checkpoint={args.checkpoint}")
    log(f"output_dir={args.output_dir}")
    cfg = load_config(args.config)
    early_cfg = cfg.get("early_exit", {})
    confidence_thresholds = args.confidence_thresholds or early_cfg.get("confidence_thresholds", [0.7, 0.8, 0.9, 0.95])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    log("loading checkpoint")
    load_info = load_model_checkpoint(model, args.checkpoint, strict=bool(cfg.get("eval_strict", False)))
    model.eval()
    log(f"checkpoint loaded; thresholds={confidence_thresholds}")
    log("building test dataloader")
    loader = build_dataloader(cfg["dataset"], "test")
    rows = []
    distribution_rows = []
    for tau in confidence_thresholds:
        tau_timer = Timer()
        log(f"threshold tau={tau}: evaluation started")
        correct = 0
        total = 0
        exit_sum = 0.0
        early_count = 0
        dist_counts: list[int] | None = None
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).long()
            reset_spiking_state(model)
            out = model(x, return_timestep_logits=True)
            tlog = out["timestep_logits"]
            probs = torch.softmax(tlog, dim=-1)
            conf, pred = probs.max(dim=-1)
            T, B = conf.shape
            if dist_counts is None:
                dist_counts = [0] * T
            exited = torch.zeros(B, dtype=torch.bool, device=device)
            exit_t = torch.full((B,), T - 1, dtype=torch.long, device=device)
            final_pred = pred[-1].clone()
            for t in range(T):
                mask = (~exited) & (conf[t] >= float(tau))
                final_pred[mask] = pred[t, mask]
                exit_t[mask] = t
                exited |= mask
            correct += (final_pred == y).sum().item()
            total += B
            exit_sum += (exit_t.float() + 1).sum().item()
            early_count += (exit_t < T - 1).sum().item()
            for t in range(T):
                dist_counts[t] += int((exit_t == t).sum().item())
        avg_exit = exit_sum / max(1, total)
        row = {
            "confidence_threshold": float(tau),
            "accuracy": 100.0 * correct / max(1, total),
            "avg_exit_timestep": avg_exit,
            "coverage": early_count / max(1, total),
            "expected_sops_fraction": avg_exit / max(1, len(dist_counts or [])),
        }
        rows.append(row)
        log(
            f"threshold tau={tau}: accuracy={row['accuracy']:.2f}, "
            f"avg_exit={row['avg_exit_timestep']:.2f}, coverage={row['coverage']:.3f}, "
            f"elapsed={tau_timer.elapsed_str()}"
        )
        for t, count in enumerate(dist_counts or []):
            distribution_rows.append(
                {
                    "confidence_threshold": float(tau),
                    "timestep": t + 1,
                    "count": count,
                }
            )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "checkpoint_load_info.json", load_info)
    write_csv(out_dir / "early_exit_summary.csv", rows)
    write_csv(out_dir / "accuracy_vs_exit_threshold.csv", rows)
    write_csv(out_dir / "exit_timestep_distribution.csv", distribution_rows)
    log(f"early-exit evaluation finished in {timer.elapsed_str()} -> {out_dir}")


if __name__ == "__main__":
    main()
