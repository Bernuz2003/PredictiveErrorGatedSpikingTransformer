#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from pegst.data.dvs import build_dataloader
from pegst.models.predictive_modules import FutureStatePredictor
from pegst.models.predictive_qkformer import build_predictive_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.seed import seed_everything


def transform_feature(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "reverse":
        return torch.flip(x, dims=[0])
    if mode == "shuffle":
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
    return x


def main() -> None:
    p = argparse.ArgumentParser(description="Offline predictability probe for latent QKFormer states.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stages", nargs="+", default=["patch_embed1", "stage1", "patch_embed2", "stage2"])
    p.add_argument("--modes", nargs="+", default=["normal", "shuffle", "reverse"])
    p.add_argument("--histories", nargs="+", type=int, default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batches", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 2021)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_predictive_qkformer(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    loader = build_dataloader(cfg["dataset"], "train")
    rows = []
    histories = args.histories or [int(cfg.get("predictive", {}).get("history", 1))]
    for stage in args.stages:
        if stage not in model.backbone.stage_channels:
            continue
        channels = model.backbone.stage_channels[stage]
        for history in histories:
            for mode in args.modes:
                predictor = FutureStatePredictor(channels, history=history, spatial=True).to(device)
                opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
                last_loss = None
                train_iter = iter(loader)
                for _ in range(args.steps):
                    try:
                        x, _ = next(train_iter)
                    except StopIteration:
                        train_iter = iter(loader)
                        x, _ = next(train_iter)
                    x = x.to(device).float()
                    reset_spiking_state(model)
                    with torch.no_grad():
                        out = model(x, return_features=True, return_timestep_logits=False, return_aux=True)
                        feat = out["features"][stage].detach()
                        feat = transform_feature(feat, mode)
                    batch = predictor.forward_sequence(feat)
                    opt.zero_grad(set_to_none=True)
                    batch.loss.backward()
                    opt.step()
                    last_loss = batch.loss.item()
                # Evaluate on held-out pass order after the tiny fit.
                losses = []
                norm_errors = []
                raw_errors = []
                target_abs = []
                pred_abs = []
                with torch.no_grad():
                    for step, (x, _) in enumerate(loader):
                        if step >= args.batches:
                            break
                        reset_spiking_state(model)
                        out = model(x.to(device).float(), return_features=True, return_timestep_logits=False, return_aux=True)
                        feat = transform_feature(out["features"][stage].detach(), mode)
                        batch = predictor.forward_sequence(feat)
                        losses.append(batch.loss.item())
                        norm_errors.append(batch.normalized_error.item())
                        raw_errors.append(batch.raw_error_mean.item())
                        target_abs.append(batch.target_abs_mean.item())
                        pred_abs.append(batch.prediction_abs_mean.item())
                rows.append(
                    {
                        "stage": stage,
                        "history": history,
                        "mode": mode,
                        "loss": sum(losses) / max(1, len(losses)),
                        "normalized_error": sum(norm_errors) / max(1, len(norm_errors)),
                        "symmetric_normalized_error": sum(norm_errors) / max(1, len(norm_errors)),
                        "raw_error_mean": sum(raw_errors) / max(1, len(raw_errors)),
                        "target_abs_mean": sum(target_abs) / max(1, len(target_abs)),
                        "prediction_abs_mean": sum(pred_abs) / max(1, len(pred_abs)),
                        "last_train_loss": last_loss,
                    }
                )
                print(rows[-1])
    write_csv(out_dir / "prediction_error.csv", rows)
    write_json(out_dir / "probe_summary.json", {"rows": rows})


if __name__ == "__main__":
    main()
