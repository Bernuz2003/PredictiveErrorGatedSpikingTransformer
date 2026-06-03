from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import torch
from torch import nn


def _select_backend() -> str:
    try:
        import cupy  # noqa: F401

        return "cupy"
    except Exception:
        return "torch"


SPIKING_BACKEND = _select_backend()

DEFAULT_RECORD_TARGETS = (
    "input_current",
    "pre_membrane",
    "post_reset_membrane",
    "spike",
    "threshold_margin",
    "soft_firing_prob",
)


class InternalStateRecorderMixin:
    """Optional recorder for multi-step LIF internals.

    Recording is disabled by default and therefore does not change the standard
    QKFormer forward path. Probe scripts enable it only inside short audit
    contexts, where the extra tensors are needed for target analysis.
    """

    record_internal_states: bool
    record_targets: set[str]
    record_detach: bool
    record_to_cpu: bool
    soft_firing_temperature: float
    last_internal_states: dict[str, torch.Tensor]

    def _init_internal_state_recorder(self) -> None:
        self.record_internal_states = False
        self.record_targets = set(DEFAULT_RECORD_TARGETS)
        self.record_detach = True
        self.record_to_cpu = False
        self.soft_firing_temperature = 0.25
        self.last_internal_states = {}

    def enable_internal_state_recording(
        self,
        targets: Iterable[str] | None = None,
        *,
        detach: bool = True,
        to_cpu: bool = False,
        soft_firing_temperature: float = 0.25,
    ) -> None:
        self.record_internal_states = True
        self.record_targets = set(targets or DEFAULT_RECORD_TARGETS)
        self.record_detach = bool(detach)
        self.record_to_cpu = bool(to_cpu)
        self.soft_firing_temperature = float(soft_firing_temperature)
        self.last_internal_states = {}

    def disable_internal_state_recording(self) -> None:
        self.record_internal_states = False
        self.last_internal_states = {}

    def _store_recorded_states(self, states: dict[str, list[torch.Tensor]]) -> None:
        out: dict[str, torch.Tensor] = {}
        for name, chunks in states.items():
            if not chunks or name not in self.record_targets:
                continue
            value = torch.stack(chunks, dim=0)
            if self.record_detach:
                value = value.detach()
            if self.record_to_cpu:
                value = value.cpu()
            out[name] = value
        self.last_internal_states = out

    def _record_common_targets(
        self,
        states: dict[str, list[torch.Tensor]],
        *,
        input_current: torch.Tensor,
        pre_membrane: torch.Tensor,
        post_reset_membrane: torch.Tensor,
        spike: torch.Tensor,
    ) -> None:
        if "input_current" in self.record_targets:
            states.setdefault("input_current", []).append(input_current)
        if "pre_membrane" in self.record_targets:
            states.setdefault("pre_membrane", []).append(pre_membrane)
        if "post_reset_membrane" in self.record_targets:
            states.setdefault("post_reset_membrane", []).append(post_reset_membrane)
        if "spike" in self.record_targets:
            states.setdefault("spike", []).append(spike)
        threshold = float(getattr(self, "v_threshold", 1.0))
        margin = pre_membrane - threshold
        if "threshold_margin" in self.record_targets:
            states.setdefault("threshold_margin", []).append(margin)
        if "soft_firing_prob" in self.record_targets:
            temp = max(float(self.soft_firing_temperature), 1e-6)
            states.setdefault("soft_firing_prob", []).append(torch.sigmoid(margin / temp))


class FallbackMultiStepLIFNode(InternalStateRecorderMixin, nn.Module):
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
        InternalStateRecorderMixin._init_internal_state_recorder(self)
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
        self.last_internal_states = {}

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() < 2:
            raise ValueError("Multi-step LIF expects [T, B, ...] input")
        v = torch.zeros_like(x_seq[0])
        spikes = []
        membranes = []
        states: dict[str, list[torch.Tensor]] = {}
        decay = 1.0 - 1.0 / self.tau
        for t in range(x_seq.shape[0]):
            pre_v = decay * v + x_seq[t]
            spike = (pre_v >= self.v_threshold).to(x_seq.dtype)
            if self.v_reset is None:
                reset_v = pre_v - spike * self.v_threshold
            else:
                hard_reset = torch.full_like(pre_v, float(self.v_reset))
                reset_v = torch.where(spike.bool(), hard_reset, pre_v)
            if self.detach_reset:
                v = reset_v.detach() + (reset_v - reset_v.detach())
            else:
                v = reset_v
            if self.record_internal_states:
                InternalStateRecorderMixin._record_common_targets(
                    self,
                    states,
                    input_current=x_seq[t],
                    pre_membrane=pre_v,
                    post_reset_membrane=v,
                    spike=spike,
                )
            membranes.append(v)
            spikes.append(spike)
        self.v = v.detach()
        self.last_v_seq = torch.stack(membranes, dim=0).detach()
        if self.record_internal_states:
            InternalStateRecorderMixin._store_recorded_states(self, states)
        return torch.stack(spikes, dim=0)


try:  # pragma: no cover - depends on external training container
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as _SJMultiStepLIFNode
    from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode as _SJMultiStepParametricLIFNode

    class MultiStepLIFNode(_SJMultiStepLIFNode, InternalStateRecorderMixin):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._init_internal_state_recorder()

        def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
            if not self.record_internal_states:
                return super().forward(x_seq)
            if x_seq.dim() < 2:
                raise ValueError("Multi-step LIF expects [T, B, ...] input")
            self.last_internal_states = {}
            states: dict[str, list[torch.Tensor]] = {}
            spikes: list[torch.Tensor] = []
            for t in range(x_seq.shape[0]):
                if hasattr(self, "v_float_to_tensor"):
                    self.v_float_to_tensor(x_seq[t])
                self.neuronal_charge(x_seq[t])
                pre_v = self.v
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                post_v = self.v
                self._record_common_targets(
                    states,
                    input_current=x_seq[t],
                    pre_membrane=pre_v,
                    post_reset_membrane=post_v,
                    spike=spike,
                )
                spikes.append(spike)
            self._store_recorded_states(states)
            return torch.stack(spikes, dim=0)

    class MultiStepParametricLIFNode(_SJMultiStepParametricLIFNode, InternalStateRecorderMixin):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._init_internal_state_recorder()

        def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
            if not self.record_internal_states:
                return super().forward(x_seq)
            if x_seq.dim() < 2:
                raise ValueError("Multi-step LIF expects [T, B, ...] input")
            self.last_internal_states = {}
            states: dict[str, list[torch.Tensor]] = {}
            spikes: list[torch.Tensor] = []
            for t in range(x_seq.shape[0]):
                if hasattr(self, "v_float_to_tensor"):
                    self.v_float_to_tensor(x_seq[t])
                self.neuronal_charge(x_seq[t])
                pre_v = self.v
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                post_v = self.v
                self._record_common_targets(
                    states,
                    input_current=x_seq[t],
                    pre_membrane=pre_v,
                    post_reset_membrane=post_v,
                    spike=spike,
                )
                spikes.append(spike)
            self._store_recorded_states(states)
            return torch.stack(spikes, dim=0)

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


def lif_modules(module: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, child in module.named_modules():
        if name and hasattr(child, "enable_internal_state_recording") and hasattr(child, "last_internal_states"):
            yield name, child


@contextmanager
def record_lif_internal_states(
    module: nn.Module,
    targets: Iterable[str] | None = None,
    *,
    detach: bool = True,
    to_cpu: bool = False,
    soft_firing_temperature: float = 0.25,
):
    modules = list(lif_modules(module))
    for _, child in modules:
        child.enable_internal_state_recording(
            targets,
            detach=detach,
            to_cpu=to_cpu,
            soft_firing_temperature=soft_firing_temperature,
        )
    try:
        yield
    finally:
        for _, child in modules:
            child.disable_internal_state_recording()
