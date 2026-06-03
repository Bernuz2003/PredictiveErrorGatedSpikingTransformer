#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pegst.data.dvs import build_dataloader
from pegst.models.qkformer import build_qkformer
from pegst.models.snn_layers import reset_spiking_state
from pegst.profiling.activity_profiler import ActivityProfiler, save_parameter_summary
from pegst.training.engine import evaluate_detailed
from pegst.utils.checkpoint import load_model_checkpoint
from pegst.utils.config import load_config
from pegst.utils.io import write_csv, write_json
from pegst.utils.progress import Timer, log


def main() -> None:
    timer = Timer()
    p = argparse.ArgumentParser(description="Evaluate a baseline QKFormer checkpoint.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--profile-batches", type=int, default=0)
    args = p.parse_args()
    log("baseline evaluation started")
    log(f"config={args.config}")
    log(f"checkpoint={args.checkpoint}")
    log(f"output_dir={args.output_dir}")
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_qkformer(cfg.get("model", {})).to(device)
    log("loading checkpoint")
    load_info = load_model_checkpoint(model, args.checkpoint, strict=bool(cfg.get("eval_strict", False)))
    log(
        "checkpoint loaded: "
        f"missing={len(load_info.get('missing_keys', []))}, "
        f"unexpected={len(load_info.get('unexpected_keys', []))}"
    )
    write_json(out / "checkpoint_load_info.json", load_info)
    save_parameter_summary(model, out)
    log("building test dataloader")
    loader = build_dataloader(cfg["dataset"], "test")
    log("evaluation pass started")
    summary, timestep_rows, confusion_rows = evaluate_detailed(
        model,
        loader,
        device,
        cfg,
        logits_path=out / "logits_over_time.pt",
    )
    write_json(out / "eval_summary.json", summary)
    write_csv(out / "timestep_metrics.csv", timestep_rows)
    write_csv(out / "confusion_matrix.csv", confusion_rows)
    log(f"evaluation pass completed: acc1={summary['acc1']:.2f}, samples={summary['num_samples']}")
    if args.profile_batches > 0:
        log(f"activity profiling started: batches={args.profile_batches}")
        profiler = ActivityProfiler(model)
        profiler.attach()
        model.eval()
        with torch.no_grad(), profiler:
            for i, (x, _) in enumerate(loader):
                if i >= args.profile_batches:
                    break
                reset_spiking_state(model)
                _ = model(x.to(device).float(), return_timestep_logits=True)
        profiler.save(out)
        profiler.close()
        log("activity profiling completed")
    log(f"baseline evaluation finished in {timer.elapsed_str()} -> {out}")


if __name__ == "__main__":
    main()
