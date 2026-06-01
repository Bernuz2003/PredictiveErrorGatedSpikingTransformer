from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .qkformer import QKFormerConfig, QKFormerNet
from .predictive_modules import ErrorGateModulator, FutureStatePredictor


@dataclass
class PredictiveConfig:
    enabled: bool = False
    stage: str | None = None
    stages: list[str] = field(default_factory=lambda: ["stage1"])
    history: int = 1
    predictor_type: str = "conv1x1"
    hidden_ratio: float = 1.0
    loss_type: str = "l1"
    loss_weight: float = 0.01
    loss_reduction: str = "mean"
    stage_loss_weights: dict[str, float] = field(default_factory=dict)
    target: str = "latent"
    stop_gradient_target: bool = True


@dataclass
class ModulationConfig:
    enabled: bool = False
    stage: str | None = None
    stages: list[str] = field(default_factory=lambda: ["stage1"])
    mode: str = "membrane_gain"
    alpha: float = 0.1


class PredictiveQKFormer(nn.Module):
    """QKFormer with optional future-state prediction and error-gated modulation.

    The auxiliary prediction path is always causal and consumes only current/past
    stage states. In Phase 2 it can be used as a pure auxiliary loss. In Phase 3
    it can modulate a selected stage through prediction-error gates.
    """

    def __init__(
        self,
        backbone_cfg: QKFormerConfig,
        predictive_cfg: PredictiveConfig | None = None,
        modulation_cfg: ModulationConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = QKFormerNet(backbone_cfg)
        self.predictive_cfg = predictive_cfg or PredictiveConfig()
        self.modulation_cfg = modulation_cfg or ModulationConfig()
        self.predictors = nn.ModuleDict()
        self.modulators = nn.ModuleDict()

        predictor_stages: list[str] = []
        if self.predictive_cfg.enabled:
            predictor_stages.extend(self.predictive_cfg.stages)
        if self.modulation_cfg.enabled:
            predictor_stages.extend(self.modulation_cfg.stages)

        for stage in dict.fromkeys(predictor_stages):
            channels = self._stage_channels(stage, backbone_cfg)
            self.predictors[stage] = FutureStatePredictor(
                channels=channels,
                history=self.predictive_cfg.history,
                spatial=(stage != "pooled"),
                predictor_type=self.predictive_cfg.predictor_type,
                hidden_ratio=self.predictive_cfg.hidden_ratio,
            )
        if self.modulation_cfg.enabled:
            for stage in self.modulation_cfg.stages:
                channels = self._stage_channels(stage, backbone_cfg)
                self.modulators[stage] = ErrorGateModulator(
                    channels=channels,
                    spatial=(stage != "pooled"),
                    mode=self.modulation_cfg.mode,
                    alpha=self.modulation_cfg.alpha,
                )

    def _stage_channels(self, stage: str, backbone_cfg: QKFormerConfig) -> int:
        if stage == "pooled":
            return backbone_cfg.embed_dims
        if stage not in self.backbone.stage_channels:
            allowed = ", ".join([*self.backbone.stage_channels.keys(), "pooled"])
            raise ValueError(f"Unknown predictive/modulation stage '{stage}'. Expected one of: {allowed}")
        return self.backbone.stage_channels[stage]

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
        return_timestep_logits: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        collected: dict[str, torch.Tensor] = {}
        mod_stats: dict[str, dict[str, torch.Tensor]] = {}

        def cb(name: str, z: torch.Tensor) -> torch.Tensor:
            collected[name] = z
            if self.modulation_cfg.enabled and name in self.modulators and name in self.predictors:
                pred_current = self.predictors[name].predict_current_from_past(z)
                z_mod, stats = self.modulators[name](z, pred_current)
                mod_stats[name] = stats
                collected[f"{name}_modulated"] = z_mod
                return z_mod
            return z

        pooled, features = self.backbone.forward_features(
            x,
            feature_callback=cb,
            collect_features=return_features,
        )
        collected["pooled"] = pooled
        logits, timestep_logits = self.backbone.logits_from_pooled(pooled)

        prediction_batches = {}
        pred_loss = torch.tensor(0.0, device=x.device)
        pred_loss_unweighted = torch.tensor(0.0, device=x.device)
        pred_norm_errors: dict[str, torch.Tensor] = {}
        pred_stage_losses: dict[str, torch.Tensor] = {}
        if self.predictive_cfg.enabled:
            weight_sum = 0.0
            for stage, predictor in self.predictors.items():
                if stage not in collected:
                    continue
                batch = predictor.forward_sequence(
                    collected[stage],
                    loss_type=self.predictive_cfg.loss_type,  # type: ignore[arg-type]
                    stop_gradient_target=self.predictive_cfg.stop_gradient_target,
                )
                prediction_batches[stage] = batch
                stage_weight = float(self.predictive_cfg.stage_loss_weights.get(stage, 1.0))
                pred_loss_unweighted = pred_loss_unweighted + batch.loss * stage_weight
                weight_sum += stage_weight
                pred_stage_losses[stage] = batch.loss.detach()
                pred_norm_errors[stage] = batch.normalized_error
            if self.predictive_cfg.loss_reduction == "mean" and weight_sum > 0:
                pred_loss_unweighted = pred_loss_unweighted / weight_sum
            pred_loss = pred_loss_unweighted
            pred_loss = pred_loss * self.predictive_cfg.loss_weight

        if not (return_aux or return_features or return_timestep_logits):
            return logits

        out: dict[str, Any] = {
            "logits": logits,
            "aux_loss": pred_loss,
            "prediction_loss": pred_loss,
            "prediction_loss_unweighted": pred_loss_unweighted.detach(),
            "prediction_stage_losses": pred_stage_losses,
            "prediction_normalized_errors": pred_norm_errors,
            "modulation_stats": mod_stats,
        }
        if return_timestep_logits:
            out["timestep_logits"] = timestep_logits
        if return_features:
            out["features"] = {**features, **collected}
        if return_aux:
            out["prediction_batches"] = prediction_batches
        return out


def _coerce_dataclass(cls, value: dict | Any | None):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    value = dict(value)
    if "stage" in value and "stages" not in value:
        value["stages"] = [value["stage"]]
    return cls(**value)


def build_predictive_qkformer(cfg: dict[str, Any]) -> PredictiveQKFormer:
    model_cfg_dict = dict(cfg.get("model", {}))
    if "timesteps" in cfg and "T" not in model_cfg_dict:
        model_cfg_dict["T"] = cfg["timesteps"]
    model_cfg = QKFormerConfig(**model_cfg_dict)
    pred_cfg = _coerce_dataclass(PredictiveConfig, cfg.get("predictive"))
    mod_cfg = _coerce_dataclass(ModulationConfig, cfg.get("modulation"))
    return PredictiveQKFormer(model_cfg, pred_cfg, mod_cfg)
