"""State-only PPO MLP policy used by ``mlp.sh``."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch
from torch import nn
from torch.distributions.normal import Normal

from ours.libs.train_with_fbs.lib_cnn import Linear2DWithFBS
from ours.libs.train_with_fbs.lib import set_sparsity


class MLPAgent(nn.Module):
    """Official ManiSkill PPO MLP with FBS on hidden linear layers 1 and 3."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_logstd: float = -0.5,
        state_max: torch.Tensor | None = None,
        state_min: torch.Tensor | None = None,
        normalize_states: bool = False,
    ):
        super().__init__()
        def init(linear: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
            nn.init.orthogonal_(linear.weight, std)
            nn.init.constant_(linear.bias, 0.0)
            return linear

        self.critic = nn.Sequential(
            Linear2DWithFBS(init(nn.Linear(int(state_dim), 256)), k=0.0, r=8),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            Linear2DWithFBS(init(nn.Linear(256, 256)), k=0.0, r=8),
            nn.Tanh(),
            init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            Linear2DWithFBS(init(nn.Linear(int(state_dim), 256)), k=0.0, r=8),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            Linear2DWithFBS(init(nn.Linear(256, 256)), k=0.0, r=8),
            nn.Tanh(),
            init(nn.Linear(256, int(action_dim)), std=0.01 * math.sqrt(2.0)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, int(action_dim)) * float(actor_logstd))
        self.action_dim = int(action_dim)
        self.normalize_states = bool(normalize_states and state_max is not None and state_min is not None)
        self.register_buffer(
            "state_max",
            torch.as_tensor(state_max).float() if state_max is not None else torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "state_min",
            torch.as_tensor(state_min).float() if state_min is not None else torch.empty(0),
            persistent=False,
        )

    def _state(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            state = batch["state"]
        else:
            state = batch["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        state = state.float()
        if self.normalize_states:
            state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return state

    def get_action_and_value(self, batch: Any, action: torch.Tensor | None = None, return_action_mean: bool = False):
        state = self._state(batch)
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        value = self.critic(state).squeeze(-1)
        result = (action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value)
        if return_action_mean:
            return (*result, action_mean)
        return result

    def get_action(self, batch: Any, deterministic: bool = False) -> torch.Tensor:
        action_mean = self.actor_mean(self._state(batch))
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return Normal(action_mean, torch.exp(action_logstd)).sample()

    def get_value(self, batch: Any) -> torch.Tensor:
        return self.critic(self._state(batch)).squeeze(-1)

    def forward(self, batch: Any) -> torch.Tensor:
        """Return the action mean, matching the callable policy behavior used by FBS."""
        return self.actor_mean(self._state(batch))


def generate_small_mlp_with_verify(
    model: MLPAgent,
    sparsity: float,
    *_args: Any,
    return_pruning_info: bool = False,
    **_kwargs: Any,
):
    """Create a compact-runtime copy by changing the two FBS gates' sparsity."""
    # FBS caches activations during rollouts; clear them before cloning because
    # PyTorch intentionally rejects deepcopy of non-leaf cached tensors.
    for module in model.modules():
        if hasattr(module, "cached_raw_w"):
            module.cached_raw_w = None
        if hasattr(module, "cached_w"):
            module.cached_w = None
        if hasattr(module, "l1_reg_of_raw_w"):
            module.l1_reg_of_raw_w = None
        if hasattr(module, "cached_i"):
            module.cached_i = None
    small_model = deepcopy(model)
    set_sparsity(small_model, float(sparsity))
    pruning_info = {
        "model_type": "mlp",
        "fbs_layers": ["actor_mean.0", "actor_mean.4", "critic.0", "critic.4"],
        "sparsity": float(sparsity),
    }
    if return_pruning_info:
        return small_model, pruning_info
    return small_model


@torch.no_grad()
def feedback_small_mlp_to_large_model(large_model: MLPAgent, small_model: MLPAgent, alpha: float) -> None:
    """Blend the small policy into the large policy using the existing feedback alpha."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"feedback alpha must be in [0, 1], got {alpha}")
    for large_param, small_param in zip(large_model.parameters(), small_model.parameters()):
        if large_param.shape == small_param.shape:
            large_param.data.lerp_(small_param.data.to(large_param.device), alpha)
