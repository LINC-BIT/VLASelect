"""Model plug-in contract for the unified online-RL continual-learning runner.

The interface intentionally contains only model/environment adaptation points.  Rollout,
GAE, PPO updates, evaluation, checkpointing, and continual-environment scheduling belong
to the shared runner and therefore do not need to be reimplemented by a model author.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class EnvironmentContract:
    """Dimensions and mapping required to translate policy actions to env actions."""

    state_dim: int
    action_dim: int
    env_action_dim: int
    controlled_action_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0 or self.env_action_dim <= 0:
            raise ValueError("environment dimensions must be positive")
        if len(self.controlled_action_indices) != self.action_dim:
            raise ValueError("controlled_action_indices must have one entry per policy action")
        if len(set(self.controlled_action_indices)) != len(self.controlled_action_indices):
            raise ValueError("controlled_action_indices must be unique")
        if not all(0 <= i < self.env_action_dim for i in self.controlled_action_indices):
            raise ValueError("controlled action index is outside env_action_dim")


@dataclass
class PolicyBatch:
    """Canonical observation batch passed from an adapter to a policy."""

    rgbs: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class FBSLayerGroups:
    """Declared transformer layer names used by dynamic and static FBS conversion."""

    vision_qkv: Tuple[str, ...] = ()
    vision_proj: Tuple[str, ...] = ()
    vision_ff1: Tuple[str, ...] = ()
    vision_ff2: Tuple[str, ...] = ()
    language_qkv: Tuple[Tuple[str, ...], ...] = ()
    language_proj: Tuple[str, ...] = ()
    language_ff1: Tuple[Tuple[str, ...], ...] = ()
    language_ff2: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Architecture declaration independent of whether checkpoint weights are available."""

    architecture_name: str
    policy_class_name: str
    state_dim: int
    action_dim: int
    env_action_dim: int
    fbs_layers: FBSLayerGroups
    architecture_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.architecture_name.strip():
            raise ValueError("architecture_name must not be empty")
        if not self.policy_class_name.strip():
            raise ValueError("policy_class_name must not be empty")
        if not any(
            (
                self.fbs_layers.vision_qkv,
                self.fbs_layers.language_qkv,
                self.fbs_layers.vision_proj,
                self.fbs_layers.language_proj,
            )
        ):
            raise ValueError(
                "an architecture declaration must include at least one transformer FBS layer"
            )


class VLAModelInterface(ABC):
    """Abstract adapter that makes one VLA implementation usable by the shared trainer.

    A new model should implement every method below. The unified runner keeps the
    reference continual-learning algorithm intact; an adapter only supplies the model,
    observation, action-space, FBS, and workload-specific behavior.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a stable, filesystem-safe model name for logs and checkpoint metadata."""

    @property
    @abstractmethod
    def architecture_spec(self) -> ModelArchitectureSpec:
        """Declare policy class identity and all FBS layer paths without inspecting weights."""

    @property
    @abstractmethod
    def policy_class(self) -> type[nn.Module]:
        """Return the concrete actor-critic class used by this adapter."""

    @abstractmethod
    def register_workloads(self) -> None:
        """Register/import model-specific environment workloads before creating environments."""

    @abstractmethod
    def inspect_environment_contract(
        self, env_id: str, *, device: torch.device, config: Mapping[str, Any]
    ) -> EnvironmentContract:
        """Probe an environment and return state/action dimensions and controlled indices."""

    @abstractmethod
    def apply_environment_contract(self, args: Any, *, env_id: str, device: torch.device) -> None:
        """Write the resolved state/action dimensions and action mapping into runner args.

        Implement this when a policy action space differs from the environment action
        space, for example TinyVLA's controlled action channels.
        """

    @abstractmethod
    def make_vector_env(
        self,
        args: Any,
        *,
        device: torch.device,
        env_id: str,
        num_envs: int,
        record_metrics: bool = True,
        video_output_dir: Optional[Path] = None,
        video_max_steps: Optional[int] = None,
    ) -> Any:
        """Create a vectorized environment matching the adapter's observation/action schema."""

    @abstractmethod
    def extract_observations(self, obs: Any) -> PolicyBatch:
        """Convert raw environment observations into uint8 RGB and float32 state batches."""

    @abstractmethod
    def extract_rgb_batch_from_obs(self, obs: Any) -> torch.Tensor:
        """Extract a CPU uint8 ``[batch, height, width, 3]`` image tensor from env observations."""

    @abstractmethod
    def extract_state_batch_from_obs(self, obs: Any) -> Any:
        """Extract a float32 ``[batch, state_dim]`` NumPy-compatible state array."""

    @abstractmethod
    def build_policy(
        self,
        model_dir: Path,
        *,
        args: Any,
        device: torch.device,
    ) -> nn.Module:
        """Construct the full actor-critic policy for the resolved runner arguments."""

    @abstractmethod
    def convert_to_fbs_policy(
        self,
        policy: nn.Module,
        *,
        device: torch.device,
        max_sparsity: float,
    ) -> nn.Module:
        """Insert the real FBS modules into a full policy and return that instrumented policy.

        This is the dynamic FBS stage from the reference implementation (SVD decomposition,
        FBS module insertion, sparsity initialization, and forward-equivalence verification).
        Static small-policy generation is intentionally a separate shared-runner step.
        """

    @abstractmethod
    def restore_policy_after_fbs(self, policy: nn.Module, *, device: torch.device) -> nn.Module:
        """Restore device, dtype, and model-specific runtime attributes after FBS conversion."""

    @abstractmethod
    def configure_trainable_modules(self, policy: nn.Module, *, train_backbone: bool) -> None:
        """Set ``requires_grad`` for backbone and task heads according to the training schedule."""

    @abstractmethod
    def build_optimizer(
        self, policy: nn.Module, *, config: Mapping[str, Any]
    ) -> Optimizer:
        """Create named optimizer parameter groups for backbone, heads, state, and value modules."""

    @abstractmethod
    def set_backbone_learning_rate(self, optimizer: Optimizer, learning_rate: float) -> None:
        """Update the adapter's backbone optimizer group when a warmup period ends."""

    @abstractmethod
    def get_action_and_value(
        self,
        policy: nn.Module,
        batch: PolicyBatch,
        *,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return env actions, summed log-probability, entropy, value, and discrete action bins."""

    @abstractmethod
    def get_value(self, policy: nn.Module, batch: PolicyBatch) -> torch.Tensor:
        """Return one scalar bootstrap value per observation in ``batch``."""

    @abstractmethod
    def prepare_policy_for_checkpoint_load(self, policy: nn.Module) -> None:
        """Perform model-specific preparation before loading a shared checkpoint state dict."""
