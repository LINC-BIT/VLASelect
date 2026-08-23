"""Reference-backed adapters used by the two verification entry points.

All imports below resolve to copies under ``api/vendor``.  This keeps the API examples
self-contained while preserving the model/environment implementations from the reference
launch scripts.
"""

from __future__ import annotations

import sys
import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

API_DIR = Path(__file__).resolve().parents[1]
VENDOR = API_DIR / "vendor"
EVAL_ROOT = API_DIR.parent / "eval"
for path in (
    VENDOR,
    VENDOR / "ours",
    VENDOR / "train" / "vla_adapter_new" / "model_impl",
    VENDOR / "train" / "tinyvla" / "model_impl",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.vla_model_interface import (
    EnvironmentContract,
    FBSLayerGroups,
    ModelArchitectureSpec,
    PolicyBatch,
    VLAModelInterface,
)


class ReferenceAdapter(VLAModelInterface):
    """Base class that delegates shared model mechanics to the copied reference modules."""

    reference_api: Any
    policy_type: Any
    model_name = "reference"
    state_dim = 105
    action_dim = 16
    env_action_dim = 16
    controlled_action_indices: Tuple[int, ...] = tuple(range(16))
    workload_module: Any = None

    @property
    def architecture_spec(self) -> ModelArchitectureSpec:
        return self._architecture_spec

    @property
    def policy_class(self) -> type[nn.Module]:
        return self.policy_type

    def register_workloads(self) -> None:
        # Importing the workload is the registration side effect used by the reference code.
        if self.workload_module is not None:
            return None

    def inspect_environment_contract(self, env_id: str, *, device: torch.device, config: Mapping[str, Any]) -> EnvironmentContract:
        del env_id, device, config
        return EnvironmentContract(self.state_dim, self.action_dim, self.env_action_dim, self.controlled_action_indices)

    def apply_environment_contract(self, args: Any, *, env_id: str, device: torch.device) -> None:
        del env_id, device
        args.state_dim = self.state_dim
        args.action_dim = self.action_dim
        if hasattr(args, "env_action_dim"):
            args.env_action_dim = self.env_action_dim
        if hasattr(args, "controlled_action_indices"):
            args.controlled_action_indices = self.controlled_action_indices

    def make_vector_env(self, args: Any, *, device: torch.device, env_id: str, num_envs: int, record_metrics: bool = True, video_output_dir: Optional[Path] = None, video_max_steps: Optional[int] = None) -> Any:
        # Keep the reference environment construction, including RecordEpisode and its
        # custom vector wrapper.  The unified runner calls this for train and eval envs.
        local_args = args
        original = local_args.env_id
        local_args.env_id = env_id
        try:
            return self.reference_api.make_vector_env(
                local_args,
                device,
                num_envs,
                record_metrics=record_metrics,
                video_output_dir=video_output_dir,
                video_max_steps=video_max_steps,
            )
        finally:
            local_args.env_id = original

    def extract_rgb_batch_from_obs(self, obs: Any) -> torch.Tensor:
        return self.reference_api.extract_rgb_batch_from_obs(obs)

    def extract_state_batch_from_obs(self, obs: Any) -> np.ndarray:
        return self.reference_api.extract_state_batch_from_obs(obs)

    def extract_observations(self, obs: Any) -> PolicyBatch:
        return PolicyBatch(self.extract_rgb_batch_from_obs(obs).cpu().numpy(), self.extract_state_batch_from_obs(obs))

    def build_policy(self, model_dir: Path, *, args: Any, device: torch.device) -> nn.Module:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"VLA model directory does not exist: {model_dir}. "
                "The API adapter requires the original checkpoint-backed model."
            )
        kwargs = {"state_dim": args.state_dim, "action_dim": args.action_dim}
        if self.model_name == "tinyvla":
            kwargs.update({"env_action_dim": args.env_action_dim, "controlled_action_indices": args.controlled_action_indices})
        policy = self.policy_class(model_dir, device=device, **kwargs)
        expected = self.architecture_spec.policy_class_name
        actual = f"{type(policy).__module__}.{type(policy).__name__}"
        if actual != expected:
            raise TypeError(f"adapter {self.model_name} declared {expected}, built {actual}")
        return policy

    def convert_to_fbs_policy(self, policy: nn.Module, *, device: torch.device, max_sparsity: float) -> nn.Module:
        # This calls the actual SVD decomposition + dynamic FBS insertion implementation.
        # The reference converter already discovers every transformer block from the
        # constructed policy.  Passing the interface declaration here is unsafe for
        # checkpoint-backed models because the declaration may contain fewer blocks
        # than the loaded architecture;
        # static small-model generation would then look for an unconverted ``.0`` FBS
        # child (for example ``...blocks.2.attn.qkv.0``).
        kwargs = {"max_sparsity": max_sparsity}
        parameters = inspect.signature(self.fbs_converter).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return self.fbs_converter(policy, device, **kwargs)

    def restore_policy_after_fbs(self, policy: nn.Module, *, device: torch.device) -> nn.Module:
        policy.to(device=device)
        policy.device = device
        if hasattr(policy, "vla"):
            policy.vla.to(device=device, dtype=torch.bfloat16)
        for name in ("state_projector", "context_projector", "actor_head", "value_head"):
            module = getattr(policy, name, None)
            if module is not None:
                module.to(device=device, dtype=torch.float32)
        if hasattr(policy, "action_bin_centers"):
            policy._buffers["action_bin_centers"] = policy.action_bin_centers.to(device=device, dtype=torch.float32)
        policy.eval_micro_batch_size = getattr(policy, "eval_micro_batch_size", 32)
        return policy

    def configure_trainable_modules(self, policy: nn.Module, *, train_backbone: bool) -> None:
        policy.configure_trainable_modules(train_backbone=train_backbone)

    def build_optimizer(self, policy: nn.Module, *, config: Mapping[str, Any]) -> Any:
        class ArgsProxy:
            pass
        args = ArgsProxy()
        for key, value in config.items():
            setattr(args, key, value)
        return self.reference_api.build_optimizer(args, policy)

    def set_backbone_learning_rate(self, optimizer: Any, learning_rate: float) -> None:
        self.reference_api.set_optimizer_group_lr(optimizer, "vla", learning_rate)

    def get_action_and_value(self, policy: nn.Module, batch: Any, *, action_bins: Optional[torch.Tensor] = None, deterministic: bool = False):
        return self.reference_api.batched_get_action_and_value_no_grad(
            policy, batch.rgbs, batch.states,
            action_bins=action_bins,
            micro_batch_size=getattr(policy, "eval_micro_batch_size", 32),
            deterministic=deterministic,
        )

    def get_value(self, policy: nn.Module, batch: Any) -> torch.Tensor:
        return self.reference_api.batched_get_value_no_grad(
            policy, batch.rgbs, batch.states,
            micro_batch_size=getattr(policy, "eval_micro_batch_size", 32),
        )

    def prepare_policy_for_checkpoint_load(self, policy: nn.Module) -> None:
        policy.eval()


def make_vla_adapter() -> ReferenceAdapter:
    from train.vla_adapter_new.model_impl import online_rl_hold_cube_in_hand as reference_api
    from train.vla_adapter_new.ours.model_with_fbs_test import convert_to_fbs_model

    import env as workload_module  # noqa: F401
    import workloads.hold_in_hand as workload_module  # noqa: F811

    VLAAdapterImplementation.policy_type = reference_api.HandVLAAdapterActorCritic
    VLAAdapterImplementation.fbs_converter = staticmethod(convert_to_fbs_model)
    VLAAdapterImplementation.reference_api = reference_api
    VLAAdapterImplementation.workload_module = workload_module
    VLAAdapterImplementation._architecture_spec = ModelArchitectureSpec(
        architecture_name="VLA-Adapter / HandVLAAdapterActorCritic",
        policy_class_name=f"{reference_api.HandVLAAdapterActorCritic.__module__}.{reference_api.HandVLAAdapterActorCritic.__name__}",
        state_dim=105,
        action_dim=16,
        env_action_dim=16,
        fbs_layers=_vla_adapter_fbs_layers(),
        architecture_config={},
    )
    return VLAAdapterImplementation()


def make_tinyvla() -> ReferenceAdapter:
    # TinyVLA follows the OpenCabinet continual-learning reference workload.
    from train.tinyvla.model_impl import online_rl_open_cabinet_drawer as reference_api
    from train.tinyvla.model_impl import online_rl_open_cabinet_drawer as tiny_reference
    from train.tinyvla.ours.model_with_fbs import convert_to_fbs_model

    import workloads.mobile_arm as workload_module

    TinyVLAImplementation.policy_type = tiny_reference.EdgeVLAActorCritic
    TinyVLAImplementation.fbs_converter = staticmethod(convert_to_fbs_model)
    TinyVLAImplementation.reference_api = reference_api
    TinyVLAImplementation.workload_module = workload_module
    TinyVLAImplementation._architecture_spec = ModelArchitectureSpec(
        architecture_name="TinyVLA / EdgeVLAActorCritic",
        policy_class_name=f"{tiny_reference.EdgeVLAActorCritic.__module__}.{tiny_reference.EdgeVLAActorCritic.__name__}",
        state_dim=44,
        action_dim=8,
        env_action_dim=13,
        fbs_layers=_tinyvla_fbs_layers(),
        architecture_config={},
    )
    return TinyVLAImplementation()


def make_edgevla() -> ReferenceAdapter:
    """Build the EdgeVLA adapter for the Unitree G1 continual-learning workload."""
    from train.tinyvla.ours.model_with_fbs import convert_to_fbs_model
    from train.tinyvla.model_impl import online_rl_open_cabinet_drawer as edge_reference

    if str(EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EVAL_ROOT))
    from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task

    import workloads.human as workload_module

    human_task.patch_reference_for_humanoid_env()
    EdgeVLAImplementation.policy_type = edge_reference.EdgeVLAActorCritic
    EdgeVLAImplementation.fbs_converter = staticmethod(convert_to_fbs_model)
    EdgeVLAImplementation.reference_api = edge_reference
    EdgeVLAImplementation.workload_module = workload_module
    EdgeVLAImplementation._architecture_spec = ModelArchitectureSpec(
        architecture_name="EdgeVLA / EdgeVLAActorCritic",
        policy_class_name=(
            f"{edge_reference.EdgeVLAActorCritic.__module__}."
            f"{edge_reference.EdgeVLAActorCritic.__name__}"
        ),
        state_dim=73,
        action_dim=12,
        env_action_dim=25,
        fbs_layers=_tinyvla_fbs_layers(),
        architecture_config={"task": "UnitreeG1LiftApple-v1"},
    )
    return EdgeVLAImplementation()


def _vla_adapter_fbs_layers() -> FBSLayerGroups:
    # These paths describe the original checkpoint-backed VLA-Adapter architecture.
    # The actual converter discovers all blocks from the loaded model; this declaration
    # records the native fused-vision/Qwen module layout without a random-init surrogate.
    vision_layers = 1
    language_layers = 24
    vision = tuple(
        f"vla.vision_backbone.fused_featurizer.blocks.{index}.attn.qkv" for index in range(vision_layers)
    )
    vision_proj = tuple(
        f"vla.vision_backbone.fused_featurizer.blocks.{index}.attn.proj" for index in range(vision_layers)
    )
    vision_ff1 = tuple(
        f"vla.vision_backbone.fused_featurizer.blocks.{index}.mlp.fc1" for index in range(vision_layers)
    )
    vision_ff2 = tuple(
        f"vla.vision_backbone.fused_featurizer.blocks.{index}.mlp.fc2" for index in range(vision_layers)
    )
    language_qkv = tuple(
        tuple(f"model.layers.{index}.self_attn.{name}_proj" for name in ("q", "k", "v"))
        for index in range(language_layers)
    )
    language_proj = tuple(f"model.layers.{index}.self_attn.o_proj" for index in range(language_layers))
    language_ff1 = tuple(
        tuple(
            f"model.layers.{index}.mlp.{name}_proj"
            for name in ("gate", "up")
        )
        for index in range(language_layers)
    )
    language_ff2 = tuple(f"model.layers.{index}.mlp.down_proj" for index in range(language_layers))
    return FBSLayerGroups(vision, vision_proj, vision_ff1, vision_ff2, language_qkv, language_proj, language_ff1, language_ff2)


def _tinyvla_fbs_layers() -> FBSLayerGroups:
    return _vla_adapter_fbs_layers()


class VLAAdapterImplementation(ReferenceAdapter):
    """VLA-Adapter: native 105-state, 16-action HoldCube policy."""

    model_name = "vla_adapter"
    state_dim = 105
    action_dim = 16
    env_action_dim = 16
    controlled_action_indices = tuple(range(16))

    def extract_state_batch_from_obs(self, obs: Any) -> np.ndarray:
        return self.reference_api.extract_hand_state_batch_from_obs(obs)


class TinyVLAImplementation(ReferenceAdapter):
    """TinyVLA: 44-state, 8-policy-action adapter for OpenCabinet environments."""

    model_name = "tinyvla"
    state_dim = 44
    action_dim = 8
    env_action_dim = 13
    controlled_action_indices = tuple(range(8))

    def extract_state_batch_from_obs(self, obs: Any) -> np.ndarray:
        return self.reference_api.extract_cabinet_state_batch_from_obs(obs)[:, : self.state_dim]


class EdgeVLAImplementation(ReferenceAdapter):
    """EdgeVLA: 73-state, 12-policy-action Unitree G1 adapter."""

    model_name = "edgevla"
    state_dim = 73
    action_dim = 12
    env_action_dim = 25
    controlled_action_indices = (2, 4, 6, 8, 10, 14, 15, 16, 20, 21, 22, 24)

    def build_policy(self, model_dir: Path, *, args: Any, device: torch.device) -> nn.Module:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"VLA model directory does not exist: {model_dir}. "
                "The API adapter requires the original checkpoint-backed model."
            )
        policy = self.policy_class(
            model_dir,
            device=device,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            env_action_dim=args.env_action_dim,
            controlled_action_indices=args.controlled_action_indices,
        )
        expected = self.architecture_spec.policy_class_name
        actual = f"{type(policy).__module__}.{type(policy).__name__}"
        if actual != expected:
            raise TypeError(f"adapter {self.model_name} declared {expected}, built {actual}")
        return policy

    def extract_state_batch_from_obs(self, obs: Any) -> np.ndarray:
        return self.reference_api.extract_cabinet_state_batch_from_obs(obs)
