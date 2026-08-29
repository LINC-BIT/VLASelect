"""Online diagonal Elastic Weight Consolidation for continual RL updates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EWCState:
    """Track an online diagonal Fisher estimate and its parameter anchor."""

    def __init__(self, enabled: bool = True, strength: float = 1000.0, decay: float = 1.0):
        if strength < 0:
            raise ValueError("EWC strength must be non-negative")
        if not 0 <= decay <= 1:
            raise ValueError("EWC decay must be in [0, 1]")
        self.enabled = enabled
        self.strength = float(strength)
        self.decay = float(decay)
        self.anchor: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}
        self._current_fisher: dict[str, torch.Tensor] = {}
        self._fisher_batches = 0

    @staticmethod
    def _parameters(model: nn.Module) -> dict[str, nn.Parameter]:
        return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        parameters = self._parameters(model)
        if not self.enabled or not self.anchor:
            parameter = next(iter(parameters.values()), None)
            return torch.zeros((), device=parameter.device if parameter is not None else None)
        penalty = None
        for name, parameter in parameters.items():
            anchor = self.anchor.get(name)
            fisher = self.fisher.get(name)
            if anchor is None or fisher is None:
                continue
            term = (fisher.to(device=parameter.device, dtype=parameter.dtype) * (parameter - anchor.to(device=parameter.device, dtype=parameter.dtype)).pow(2)).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            parameter = next(iter(parameters.values()), None)
            return torch.zeros((), device=parameter.device if parameter is not None else None)
        return 0.5 * self.strength * penalty

    def observe(self, loss: torch.Tensor, model: nn.Module) -> None:
        """Accumulate squared gradients of the task loss for the current Env."""
        if not self.enabled:
            return
        parameters = self._parameters(model)
        if not parameters:
            return
        grads = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True, allow_unused=True)
        for (name, parameter), grad in zip(parameters.items(), grads):
            if grad is None:
                continue
            value = grad.detach().pow(2)
            if name not in self._current_fisher:
                self._current_fisher[name] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            self._current_fisher[name].add_(value)
        self._fisher_batches += 1

    def consolidate(self, model: nn.Module) -> None:
        """Merge the completed Env Fisher into online history and anchor params."""
        if not self.enabled:
            return
        parameters = self._parameters(model)
        if self._fisher_batches:
            scale = 1.0 / self._fisher_batches
            for name, value in self._current_fisher.items():
                current = value * scale
                previous = self.fisher.get(name)
                self.fisher[name] = current if previous is None else self.decay * previous + current
        self.anchor = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        self._current_fisher = {}
        self._fisher_batches = 0

    def mean_fisher(self) -> float:
        if not self.fisher:
            return 0.0
        total = 0.0
        count = 0
        for value in self.fisher.values():
            total += float(value.detach().float().sum().item())
            count += value.numel()
        return total / count if count else 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strength": self.strength,
            "decay": self.decay,
            "anchor": {name: value.detach().cpu() for name, value in self.anchor.items()},
            "fisher": {name: value.detach().cpu() for name, value in self.fisher.items()},
            "current_fisher": {name: value.detach().cpu() for name, value in self._current_fisher.items()},
            "fisher_batches": self._fisher_batches,
        }

    def load_state_dict(self, state: dict[str, Any], model: nn.Module) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.strength = float(state.get("strength", self.strength))
        self.decay = float(state.get("decay", self.decay))
        parameters = self._parameters(model)
        self.anchor = {name: value.to(parameters[name].device) for name, value in state.get("anchor", {}).items() if name in parameters}
        self.fisher = {name: value.to(parameters[name].device) for name, value in state.get("fisher", {}).items() if name in parameters}
        self._current_fisher = {name: value.to(parameters[name].device) for name, value in state.get("current_fisher", {}).items() if name in parameters}
        self._fisher_batches = int(state.get("fisher_batches", 0))
