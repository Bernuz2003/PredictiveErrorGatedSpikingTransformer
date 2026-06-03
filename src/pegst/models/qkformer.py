from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch import nn

from .snn_layers import MultiStepLIFNode, SPIKING_BACKEND

FeatureCallback = Callable[[str, torch.Tensor], torch.Tensor]


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    try:
        nn.init.trunc_normal_(tensor, std=std)
    except AttributeError:  # pragma: no cover
        nn.init.normal_(tensor, std=std)


@dataclass
class QKFormerConfig:
    img_size_h: int = 128
    img_size_w: int = 128
    in_channels: int = 2
    num_classes: int = 11
    embed_dims: int = 128
    mlp_ratio: float = 1.0
    num_heads: int = 16
    T: int = 16


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)
        return x


class TokenQKAttention(nn.Module):
    """QKFormer token Q-K attention from the provided baseline.

    Input and output shape: [T, B, C, H, W].
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        _, _, _, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q = self.q_bn(self.q_conv(x_for_qkv)).reshape(T, B, C, N)
        q = self.q_lif(q).unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        k = self.k_bn(self.k_conv(x_for_qkv)).reshape(T, B, C, N)
        k = self.k_lif(k).unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)
        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        return self.proj_lif(x)


class SpikingSelfAttention(nn.Module):
    """Spikformer-style SSA block retained for the second QKFormer stage."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        _, _, _, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q = self.q_bn(self.q_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        q = self.q_lif(q).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_bn(self.k_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        k = self.k_lif(k).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_bn(self.v_conv(x_for_qkv)).reshape(T, B, C, N).contiguous()
        v = self.v_lif(v).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        return self.proj_lif(x)


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.0) -> None:
        super().__init__()
        self.tssa = TokenQKAttention(dim, num_heads)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.0) -> None:
        super().__init__()
        self.ssa = SpikingSelfAttention(dim, num_heads)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssa(x)
        x = x + self.mlp(x)
        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, in_channels: int = 2, embed_dims: int = 64) -> None:
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.shape
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        x = self.proj1_bn(self.proj1_conv(x))
        x = self.maxpool1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()
        x_feat = x
        x = self.proj2_bn(self.proj2_conv(x))
        x = self.maxpool2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()
        x = self.proj3_bn(self.proj3_conv(x))
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)
        x_feat = self.proj_res_bn(self.proj_res_conv(x_feat)).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        return x + x_feat


class PatchEmbeddingStage(nn.Module):
    def __init__(self, embed_dims: int = 128) -> None:
        super().__init__()
        self.proj_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)
        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=SPIKING_BACKEND)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.shape
        x = x.flatten(0, 1).contiguous()
        x_feat = x
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        x = self.proj4_bn(self.proj4_conv(x))
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)
        x_feat = self.proj_res_bn(self.proj_res_conv(x_feat)).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        return x + x_feat


class QKFormerNet(nn.Module):
    """Two-stage QKFormer/Mini-QKFormer adapted for DVS128 Gesture.

    The implementation is based on the uploaded QKFormer pipeline, but removes
    timm registration and exposes stage activations, timestep logits, and feature
    callbacks required by membrane-aware audits and causal probes.
    """

    def __init__(self, cfg: QKFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.T = cfg.T
        stage1_dim = cfg.embed_dims // 2
        stage2_dim = cfg.embed_dims
        heads = max(1, min(cfg.num_heads, stage1_dim))
        while stage1_dim % heads != 0 or stage2_dim % heads != 0:
            heads -= 1
        self.patch_embed1 = PatchEmbedInit(in_channels=cfg.in_channels, embed_dims=stage1_dim)
        self.stage1 = nn.ModuleList([TokenSpikingTransformer(stage1_dim, heads, cfg.mlp_ratio)])
        self.patch_embed2 = PatchEmbeddingStage(embed_dims=stage2_dim)
        self.stage2 = nn.ModuleList([SpikingTransformer(stage2_dim, heads, cfg.mlp_ratio)])
        self.head = nn.Linear(stage2_dim, cfg.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @property
    def stage_channels(self) -> dict[str, int]:
        return {
            "patch_embed1": self.cfg.embed_dims // 2,
            "stage1": self.cfg.embed_dims // 2,
            "patch_embed2": self.cfg.embed_dims,
            "stage2": self.cfg.embed_dims,
        }

    def forward_features(
        self,
        x: torch.Tensor,
        feature_callback: FeatureCallback | None = None,
        collect_features: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # External convention: [B, T, C, H, W]. Internal SNN convention: [T, B, C, H, W].
        if x.dim() != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        features: dict[str, torch.Tensor] = {}

        def maybe(name: str, z: torch.Tensor) -> torch.Tensor:
            if collect_features:
                features[name] = z
            if feature_callback is not None:
                z = feature_callback(name, z)
            return z

        x = maybe("patch_embed1", self.patch_embed1(x))
        for i, blk in enumerate(self.stage1):
            x = blk(x)
            x = maybe("stage1", x) if i == len(self.stage1) - 1 else x
        x = maybe("patch_embed2", self.patch_embed2(x))
        for i, blk in enumerate(self.stage2):
            x = blk(x)
            x = maybe("stage2", x) if i == len(self.stage2) - 1 else x
        pooled = x.flatten(3).mean(3)  # [T, B, C]
        if collect_features:
            features["pooled"] = pooled
        return pooled, features

    def logits_from_pooled(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_t_logits = self.head(pooled)  # [T, B, num_classes]
        cumulative_features = torch.cumsum(pooled, dim=0) / torch.arange(
            1, pooled.shape[0] + 1, device=pooled.device, dtype=pooled.dtype
        ).view(-1, 1, 1)
        timestep_logits = self.head(cumulative_features)  # [T, B, num_classes]
        final_logits = self.head(pooled.mean(0))
        return final_logits, timestep_logits

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_timestep_logits: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        pooled, features = self.forward_features(x, collect_features=return_features)
        logits, timestep_logits = self.logits_from_pooled(pooled)
        if not (return_features or return_timestep_logits):
            return logits
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"logits": logits}
        if return_timestep_logits:
            out["timestep_logits"] = timestep_logits
        if return_features:
            out["features"] = features
        return out


def build_qkformer(cfg: dict | QKFormerConfig) -> QKFormerNet:
    if isinstance(cfg, dict):
        cfg = QKFormerConfig(**cfg)
    return QKFormerNet(cfg)
