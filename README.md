# Predictive Error-Gated Spiking Transformer

Research repository for a predictive-coding inspired Spiking Transformer on event-based vision.

The starting backbone is a compact two-stage QKFormer/Mini-QKFormer-style model adapted from the provided QKFormer pipeline. The new contribution is an incremental predictive path:

1. **Phase 0:** baseline + activity/SOP profiling;
2. **Phase 1:** offline predictability probe on latent stages;
3. **Phase 2:** auxiliary future-feature prediction loss;
4. **Phase 3:** prediction-error gated latent/membrane modulation;
5. **Phase 5:** timestep-wise classification and early exit.

The first intended dataset is **DVS128 Gesture**, because gesture classes are defined by real motion dynamics. `synthetic` is included only for smoke tests and controlled debugging.

## Repository structure

```text
configs/                         YAML configurations
scripts/train.py                  training entrypoint
scripts/evaluate.py               evaluation + optional profiler
scripts/predictability_probe.py   offline latent predictability analysis
scripts/early_exit_eval.py        confidence-threshold early exit
scripts/smoke_test.py             fast CPU shape test
src/pegst/models/                 QKFormer + predictive modules
src/pegst/data/                   DVS and synthetic datasets
src/pegst/profiling/              activity/firing/SOP profiler
src/pegst/training/               train/eval loops and metrics
src/pegst/utils/                  config, I/O, seeds
```

## Installation

On the SMILIES server, use a Singularity container with PyTorch, SpikingJelly, torchvision, pandas, pyyaml, and matplotlib. For local smoke tests, only PyTorch and PyYAML are required.

```bash
cd PredictiveErrorGatedSpikingTransformer
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For real DVS datasets and analysis notebooks:

```bash
pip install -e ".[dvs,analysis,dev]"
```

If the CUDA/CuPy backend is available on the server, install the matching optional package, for example:

```bash
pip install -e ".[cuda]"
```

The code tries to use `spikingjelly.clock_driven` when available; otherwise, it falls back to a minimal PyTorch LIF node for smoke tests.

## Quick smoke test

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

A one-epoch synthetic run:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/synthetic_smoke.yaml \
  --output-dir runs/synthetic_smoke \
  --profile-batches 1
```

## Baseline DVS128 Gesture run

Edit `dataset.root` in `configs/baseline_dvs128gesture_T16.yaml`, then:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/baseline_dvs128gesture_T16.yaml \
  --output-dir runs/dvs128gesture/baseline_T16 \
  --profile-batches 2
```

Expected outputs include:

```text
config.yaml
metrics.csv
timestep_metrics.csv
prediction_error.csv
prediction_timestep_error.csv
prediction_sample_scores.pt
modulation_stats.csv
confusion_matrix.csv
params_summary.json
checkpoint_last.pt
checkpoint_best.pt
layerwise_activity.csv
stage_activity.csv
sops_estimate.csv
activity_summary.json
profile_summary.json
```

## Phase 1: predictability probe

```bash
PYTHONPATH=src python scripts/predictability_probe.py \
  --config configs/predictability_probe_dvs128gesture.yaml \
  --checkpoint runs/dvs128gesture/baseline_T16/checkpoint_best.pt \
  --output-dir runs/dvs128gesture/predictability_probe \
  --stages patch_embed1 stage1 patch_embed2 stage2 \
  --modes normal shuffle reverse
```

The key file is:

```text
prediction_error.csv
```

The normal temporal order should ideally have lower normalized prediction error than shuffle/reverse for at least one stage.

The normalized error is the symmetric normalized prediction error:

```text
SNPE = |U - U_hat|_1 / (|U|_1 + |U_hat|_1 + eps)
```

## Phase 2: auxiliary future-feature prediction

Edit `dataset.root`, then:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/predictive_aux_dvs128gesture_T8.yaml \
  --output-dir runs/dvs128gesture/predictive_aux_T8 \
  --profile-batches 2
```

This optimizes:

\[
\mathcal{L}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{pred}
\]

where the prediction target is the selected latent stage, by default `stage1`.

Stage ablation configs are provided for `stage1`, `stage2`, and `stage1+stage2`:

```text
configs/predictive_aux_stage1_dvs128gesture_T8.yaml
configs/predictive_aux_stage2_dvs128gesture_T8.yaml
configs/predictive_aux_stage1_stage2_dvs128gesture_T8.yaml
```

When multiple stages are enabled, the prediction loss is reduced with `loss_reduction: mean` by default so the auxiliary-loss scale remains comparable.

## Phase 3: error-gated modulation

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/error_gated_dvs128gesture_T8.yaml \
  --output-dir runs/dvs128gesture/error_gated_T8 \
  --profile-batches 2
```

The default modulation is:

\[
\tilde{U}_t^l=U_t^l\odot(1+\alpha G_t^l)
\]

where the gate is computed from the causal prediction error.

Error-gated stage ablation configs are provided:

```text
configs/error_gated_stage1_dvs128gesture_T8.yaml
configs/error_gated_stage2_dvs128gesture_T8.yaml
configs/error_gated_stage1_stage2_dvs128gesture_T8.yaml
```

## Early exit evaluation

```bash
PYTHONPATH=src python scripts/early_exit_eval.py \
  --config configs/predictive_aux_dvs128gesture_T8.yaml \
  --checkpoint runs/dvs128gesture/predictive_aux_T8/checkpoint_best.pt \
  --output-dir runs/dvs128gesture/predictive_aux_T8/early_exit
```

To include calibrated prediction-error thresholds:

```bash
PYTHONPATH=src python scripts/early_exit_eval.py \
  --config configs/predictive_aux_dvs128gesture_T8.yaml \
  --checkpoint runs/dvs128gesture/predictive_aux_T8/checkpoint_best.pt \
  --output-dir runs/dvs128gesture/predictive_aux_T8/early_exit_pe \
  --confidence-thresholds 0.7 0.8 0.9 \
  --prediction-error-thresholds 0.25 0.5 1.0
```

Early-exit outputs include `early_exit_summary.csv`, `accuracy_vs_exit_threshold.csv`, and `exit_timestep_distribution.csv`.

## Analysis notebook

Open `notebooks/pegst_analysis.ipynb`, set `RUN_DIR`, and run the cells to plot accuracy, timestep metrics, firing activity, SOP estimates, prediction error, and early-exit trade-offs.

## Current predictive scope

The current implementation predicts latent future features, not true membrane potentials. This is an intentional first-phase limitation. See `docs/phase1_latent_scope_and_pre_membrane_next.md` for the planned next step, which adds only `target: pre_membrane`.

## Notes

This repository intentionally prioritizes algorithmic testability over FPGA constraints. Hardware considerations can be reintroduced after the predictive mechanism shows measurable value in accuracy, low-timestep behavior, firing rate, SOPs, or early-exit reliability.
