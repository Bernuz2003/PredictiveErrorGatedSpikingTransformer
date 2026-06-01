from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _select_backend() -> str:
    try:
        import cupy  # noqa: F401

        return "cupy"
    except Exception:
        return "torch"


SPIKING_BACKEND = _select_backend()


class FallbackMultiStepLIFNode(nn.Module):
    """Small PyTorch fallback for SpikingJelly's MultiStepLIFNode.

    It is intentionally simple, deterministic, CPU-friendly, and sufficient for
    unit tests, synthetic smoke runs, and environments where SpikingJelly is not
    installed. In the SMILIES/Singularity training container the real
    SpikingJelly neuron will be used automatically.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        backend: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.backend = backend or "torch"
        self.register_buffer("v", torch.tensor(0.0), persistent=False)
        self.last_v_seq: torch.Tensor | None = None

    def reset(self) -> None:
        self.v = torch.tensor(0.0, device=self.v.device if isinstance(self.v, torch.Tensor) else None)
        self.last_v_seq = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() < 2:
            raise ValueError("Multi-step LIF expects [T, B, ...] input")
        v = torch.zeros_like(x_seq[0])
        spikes = []
        membranes = []
        decay = 1.0 - 1.0 / self.tau
        for t in range(x_seq.shape[0]):
            v = decay * v + x_seq[t]
            spike = (v >= self.v_threshold).to(x_seq.dtype)
            if self.v_reset is None:
                reset_v = v - spike * self.v_threshold
            else:
                hard_reset = torch.full_like(v, float(self.v_reset))
                reset_v = torch.where(spike.bool(), hard_reset, v)
            if self.detach_reset:
                v = reset_v.detach() + (reset_v - reset_v.detach())
            else:
                v = reset_v
            membranes.append(v)
            spikes.append(spike)
        self.v = v.detach()
        self.last_v_seq = torch.stack(membranes, dim=0).detach()
        return torch.stack(spikes, dim=0)


try:  # pragma: no cover - depends on external training container
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as _SJMultiStepLIFNode
    from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode as _SJMultiStepParametricLIFNode

    class MultiStepLIFNode(_SJMultiStepLIFNode):  # type: ignore[misc]
        pass

    class MultiStepParametricLIFNode(_SJMultiStepParametricLIFNode):  # type: ignore[misc]
        pass

except Exception:  # pragma: no cover - exercised in lightweight CI/sandbox
    MultiStepLIFNode = FallbackMultiStepLIFNode
    MultiStepParametricLIFNode = FallbackMultiStepLIFNode


def reset_spiking_state(module: nn.Module) -> None:
    """Reset stateful SNN modules, supporting SpikingJelly and the fallback."""
    try:  # pragma: no cover - external dependency
        from spikingjelly.clock_driven import functional

        functional.reset_net(module)
        return
    except Exception:
        pass

    for m in module.modules():
        if hasattr(m, "reset") and callable(getattr(m, "reset")):
            try:
                m.reset()
            except TypeError:
                continue
