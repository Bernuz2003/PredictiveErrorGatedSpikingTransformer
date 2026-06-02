#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def transform_feature(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "reverse":
        return torch.flip(x, dims=[0])
    if mode == "shuffle":
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
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


def feature_batches(model, loader, device: torch.device, stage: str, mode: str, max_batches: int):
    with torch.no_grad():
        for step, (x, _) in enumerate(loader):
            if step >= max_batches:
                break
            reset_spiking_state(model)
            out = model(x.to(device).float(), return_features=True, return_timestep_logits=False, return_aux=True)
            yield transform_feature(out["features"][stage].detach(), mode)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline predictability probe for latent QKFormer states.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stages", nargs="+", default=["stage1", "stage2"])
    p.add_argument("--modes", nargs="+", default=["normal", "shuffle", "reverse"])
    p.add_argument("--histories", nargs="+", type=int, default=None)
    p.add_argument("--baselines", nargs="+", default=["zero", "copy_previous", "linear_extrapolation"])
    p.add_argument("--extrapolation-alpha", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batches", type=int, default=8)
    args = p.parse_args()
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
    loader = build_dataloader(cfg["dataset"], "train")
    rows = []
    histories = args.histories or [int(pred_cfg.get("history", 1))]
    loss_type = str(pred_cfg.get("loss_type", "l1"))
    normalize_loss = bool(pred_cfg.get("normalize_loss", False))
    amplitude_loss_weight = float(pred_cfg.get("amplitude_loss_weight", 0.0))
    predictor_type = str(pred_cfg.get("predictor_type", "conv1x1"))
    hidden_ratio = float(pred_cfg.get("hidden_ratio", 1.0))
    for stage in args.stages:
        if stage not in model.backbone.stage_channels:
            continue
        channels = model.backbone.stage_channels[stage]
        for history in histories:
            for mode in args.modes:
                predictor = FutureStatePredictor(
                    channels,
                    history=history,
                    spatial=True,
                    predictor_type=predictor_type,
                    hidden_ratio=hidden_ratio,
                ).to(device)
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
                        feat = transform_feature(out["features"][stage].detach(), mode)
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

                learned_acc: dict[str, float] = {}
                baseline_accs: dict[str, dict[str, float]] = {name: {} for name in args.baselines}
                with torch.no_grad():
                    for feat in feature_batches(model, loader, device, stage, mode, args.batches):
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
                        for baseline in args.baselines:
                            pred, target = baseline_prediction(feat, baseline, alpha=args.extrapolation_alpha)
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
                row = {
                    "stage": stage,
                    "history": history,
                    "mode": mode,
                    "method": "learned_predictor",
                    "last_train_loss": last_loss,
                    **finalize_metrics(learned_acc),
                }
                rows.append(row)
                print(row)
                for baseline, acc in baseline_accs.items():
                    row = {
                        "stage": stage,
                        "history": history,
                        "mode": mode,
                        "method": baseline,
                        "last_train_loss": "",
                        **finalize_metrics(acc),
                    }
                    rows.append(row)
                    print(row)
    write_csv(out_dir / "prediction_error.csv", rows)
    write_json(
        out_dir / "probe_summary.json",
        {
            "rows": rows,
            "baselines": args.baselines,
            "extrapolation_alpha": args.extrapolation_alpha,
            "loss_type": loss_type,
            "normalize_loss": normalize_loss,
            "amplitude_loss_weight": amplitude_loss_weight,
        },
    )


if __name__ == "__main__":
    main()
