#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.predictive_qkformer import build_predictive_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.utils.config import load_config
from pegst.utils.io import write_csv


def prediction_error_scores(out: dict, timesteps: int, batch_size: int, device: torch.device) -> torch.Tensor | None:
    scores = []
    for batch in out.get("prediction_batches", {}).values():
        norm_err = batch.sample_normalized_error
        first = torch.full((1, batch_size), float("inf"), device=device, dtype=norm_err.dtype)
        scores.append(torch.cat([first, norm_err], dim=0))
    if not scores:
        return None
    score = torch.stack(scores, dim=0).mean(dim=0)
    if score.shape[0] != timesteps:
        padded = torch.full((timesteps, batch_size), float("inf"), device=device, dtype=score.dtype)
        n = min(timesteps, score.shape[0])
        padded[:n] = score[:n]
        score = padded
    return score


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate confidence-based early exit over timestep logits.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--confidence-thresholds", nargs="+", type=float, default=None)
    p.add_argument("--prediction-error-thresholds", nargs="+", type=float, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    early_cfg = cfg.get("early_exit", {})
    confidence_thresholds = args.confidence_thresholds or early_cfg.get("confidence_thresholds", [0.7, 0.8, 0.9, 0.95])
    pe_thresholds = args.prediction_error_thresholds
    if pe_thresholds is None:
        pe_thresholds = early_cfg.get("prediction_error_thresholds", [])
    pe_thresholds_with_none: list[float | None] = [None] if not pe_thresholds else [float(v) for v in pe_thresholds]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_predictive_qkformer(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    loader = build_dataloader(cfg["dataset"], "test")
    rows = []
    distribution_rows = []
    for tau in confidence_thresholds:
        for pe_tau in pe_thresholds_with_none:
            correct = 0
            total = 0
            exit_sum = 0.0
            early_count = 0
            dist_counts: list[int] | None = None
            pe_available = False
            for x, y in loader:
                x, y = x.to(device).float(), y.to(device).long()
                reset_spiking_state(model)
                out = model(x, return_aux=True, return_timestep_logits=True)
                tlog = out["timestep_logits"]
                probs = torch.softmax(tlog, dim=-1)
                conf, pred = probs.max(dim=-1)  # [T,B]
                T, B = conf.shape
                if dist_counts is None:
                    dist_counts = [0] * T
                pe_scores = prediction_error_scores(out, T, B, device)
                pe_available = pe_available or pe_scores is not None
                exited = torch.zeros(B, dtype=torch.bool, device=device)
                exit_t = torch.full((B,), T - 1, dtype=torch.long, device=device)
                final_pred = pred[-1].clone()
                for t in range(T):
                    mask = (~exited) & (conf[t] >= float(tau))
                    if pe_tau is not None:
                        if pe_scores is None:
                            mask = torch.zeros_like(mask)
                        else:
                            mask = mask & (pe_scores[t] <= pe_tau)
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
                "prediction_error_threshold": "" if pe_tau is None else pe_tau,
                "prediction_error_available": pe_available,
                "accuracy": 100.0 * correct / max(1, total),
                "avg_exit_timestep": avg_exit,
                "coverage": early_count / max(1, total),
                "expected_sops_fraction": avg_exit / max(1, len(dist_counts or [])),
            }
            rows.append(row)
            for t, count in enumerate(dist_counts or []):
                distribution_rows.append(
                    {
                        "confidence_threshold": float(tau),
                        "prediction_error_threshold": "" if pe_tau is None else pe_tau,
                        "timestep": t + 1,
                        "count": count,
                    }
                )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "early_exit_summary.csv", rows)
    write_csv(out_dir / "accuracy_vs_exit_threshold.csv", rows)
    write_csv(out_dir / "exit_timestep_distribution.csv", distribution_rows)
    print(rows)


if __name__ == "__main__":
    main()
