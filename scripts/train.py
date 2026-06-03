#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.profiling.activity_profiler import ActivityProfiler, save_parameter_summary
from pegst.training.augment import build_batch_augment, build_mixup
from pegst.training.engine import evaluate_detailed, run_epoch
from pegst.training.scheduler import build_scheduler
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config, save_config
from pegst.utils.io import append_csv, write_json, write_csv
from pegst.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline QKFormer on DVS datasets.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--profile-batches", type=int, default=0)
    return p.parse_args()


def configure_trainable_parameters(model, cfg: dict) -> list[torch.nn.Parameter]:
    train_cfg = cfg.get("training", {})
    if train_cfg.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad_(False)
    if train_cfg.get("freeze_classifier", False):
        classifier = getattr(model, "head", None)
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad_(False)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain after applying training freeze options.")
    return params


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    seed_everything(int(cfg.get("seed", 2021)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_qkformer(cfg.get("model", {})).to(device)
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        load_model_checkpoint(model, args.resume, strict=True)
    else:
        init_checkpoint = args.init_checkpoint or cfg.get("training", {}).get("init_checkpoint", "")
        if init_checkpoint:
            load_model_checkpoint(model, init_checkpoint, strict=bool(cfg.get("training", {}).get("init_strict", False)))

    trainable_params = configure_trainable_parameters(model, cfg)
    save_parameter_summary(model, out_dir)
    train_loader = build_dataloader(cfg["dataset"], "train")
    val_loader = build_dataloader(cfg["dataset"], "test")
    opt_cfg = cfg.get("optimizer", {})
    opt_type = opt_cfg.get("type", opt_cfg.get("opt", "adamw"))
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=float(opt_cfg.get("lr", 1e-3)),
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.06)),
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.06)),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    epochs = int(cfg.get("training", {}).get("epochs", 1))
    scheduler = build_scheduler(optimizer, cfg, epochs)
    start_epoch = 0
    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scheduler is not None and resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
    best_acc = -1.0
    metric_fields = [
        "epoch",
        "split",
        "loss",
        "ce_loss",
        "acc1",
        "lr",
        "num_samples",
        "val_loss",
        "val_ce_loss",
        "val_acc1",
    ]
    batch_augment = build_batch_augment(cfg)
    mixup_fn = build_mixup(cfg)
    for epoch in range(start_epoch, epochs):
        if scheduler is not None:
            scheduler.step(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            cfg,
            out_dir,
            batch_augment=batch_augment,
            mixup_fn=mixup_fn,
        )
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        val_summary, timestep_rows, confusion_rows = evaluate_detailed(
            model,
            val_loader,
            device,
            cfg,
            logits_path=out_dir / "logits_over_time.pt",
        )
        row = {
            **train_metrics,
            "val_loss": val_summary["loss"],
            "val_ce_loss": val_summary.get("ce_loss", 0.0),
            "val_acc1": val_summary["acc1"],
        }
        append_csv(out_dir / "metrics.csv", row, metric_fields)
        write_csv(out_dir / "timestep_metrics.csv", timestep_rows)
        write_csv(out_dir / "confusion_matrix.csv", confusion_rows)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "epoch": epoch,
            "val_acc1": val_summary["acc1"],
        }
        torch.save(state, out_dir / "checkpoint_last.pt")
        if val_summary["acc1"] > best_acc:
            best_acc = val_summary["acc1"]
            torch.save(state, out_dir / "checkpoint_best.pt")
            write_json(out_dir / "metrics_best.json", row)
            write_csv(out_dir / "timestep_metrics_best.csv", timestep_rows)
            write_csv(out_dir / "confusion_matrix_best.csv", confusion_rows)
        print(f"epoch={epoch} train_acc={train_metrics['acc1']:.2f} val_acc={val_summary['acc1']:.2f}")

    profile_batches = args.profile_batches
    if profile_batches <= 0 and cfg.get("profiling", {}).get("enabled", False):
        profile_batches = int(cfg.get("profiling", {}).get("profile_batches", 0))
    if profile_batches > 0:
        profiler = ActivityProfiler(model)
        profiler.attach()
        model.eval()
        with torch.no_grad(), profiler:
            for i, (x, _) in enumerate(val_loader):
                if i >= profile_batches:
                    break
                x = x.to(device).float()
                reset_spiking_state(model)
                _ = model(x, return_timestep_logits=True)
        summary = profiler.save(out_dir)
        profiler.close()
        write_json(out_dir / "profile_summary.json", summary)


if __name__ == "__main__":
    main()
