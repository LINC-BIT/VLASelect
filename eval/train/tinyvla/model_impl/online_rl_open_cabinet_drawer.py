import argparse
import importlib.util
import json
import math
import os
import shutil
import sys; sys.path.append('.')
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, Union
import workloads.mobile_arm as mobile_arm_workload
from workloads.mobile_arm import *

THIS_DIR = Path(__file__).resolve().parent
VLA_ADAPTER_IMPL_DIR = THIS_DIR.parent.parent / "vla_adapter_new" / "model_impl"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(VLA_ADAPTER_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(VLA_ADAPTER_IMPL_DIR))

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv, torch_clone_dict
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor

from online_rl import (
    broadcast_object,
    cleanup_runtime,
    distributed_barrier,
    distributed_max,
    distributed_mean,
    ensure_package,
    explained_variance,
    gather_metric_summary,
    get_rank,
    get_world_size,
    init_runtime,
    is_distributed,
    is_main_process,
    iter_slices,
    load_module_from_path,
    mkdir,
    parse_bool,
    plot_metrics_history,
    save_json,
    save_metrics_history,
    save_rollout_progress,
    set_seed,
    strip_module_prefix,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from train.common.random_init_vla import maybe_build_random_init_vla_bundle
from train.common.mwe_eval import use_train_success_only


TASK_PROMPT = "open the cabinet drawer."
DEFAULT_MODEL_DIR = "ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_INIT_POLICY = "ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt"
DEFAULT_WORKDIR = "train/tinyvla/model_impl/outputs/ppo_open_cabinet_drawer"


def resolve_model_dir_path(model_dir: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir
    candidates = [
        Path.cwd() / model_dir,
        THIS_DIR.parents[3] / model_dir,
    ]
    model_dir_str = model_dir.as_posix()
    if model_dir_str.startswith("eval/"):
        trimmed = Path(model_dir_str[len("eval/"):])
        candidates.append(Path.cwd() / trimmed)
        candidates.append(THIS_DIR.parents[2] / trimmed)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / model_dir).resolve()


def backup_run_sources(output_dir: Path) -> None:
    code_dir = mkdir(output_dir / "code")
    sources = {
        "online_rl_open_cabinet_drawer.py": Path(__file__).resolve(),
        "workloads_mobile_arm__init__.py": Path(mobile_arm_workload.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {
            "source": str(source_path),
            "backup": str(destination),
        }
    save_json(code_dir / "source_manifest.json", manifest)


def get_attention_implementation() -> str:
    requested = os.environ.get("EDGEVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[setup] flash_attn is not installed; falling back to SDPA attention")
        return "sdpa"
    print(f"[setup] using {requested} attention implementation for EdgeVLAActorCritic")
    return requested


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> torch.Tensor:
    sensor_data = obs["sensor_data"]
    head_rgb = sensor_data["fetch_head"]["rgb"]
    hand_rgb = sensor_data["fetch_hand"]["rgb"]
    if not isinstance(head_rgb, torch.Tensor):
        head_rgb = torch.from_numpy(np.asarray(head_rgb)[..., :3].astype(np.uint8, copy=False))
    else:
        head_rgb = head_rgb[..., :3].detach().to(device="cpu", dtype=torch.uint8)
    if not isinstance(hand_rgb, torch.Tensor):
        hand_rgb = torch.from_numpy(np.asarray(hand_rgb)[..., :3].astype(np.uint8, copy=False))
    else:
        hand_rgb = hand_rgb[..., :3].detach().to(device="cpu", dtype=torch.uint8)
    if head_rgb.ndim == 3:
        head_rgb = head_rgb.unsqueeze(0)
    if hand_rgb.ndim == 3:
        hand_rgb = hand_rgb.unsqueeze(0)
    return torch.cat([head_rgb.contiguous(), hand_rgb.contiguous()], dim=2).contiguous()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_cabinet_state_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    agent = obs["agent"]
    extra = obs["extra"]

    qpos = _to_numpy(agent["qpos"]).astype(np.float32)
    qvel = _to_numpy(agent["qvel"]).astype(np.float32)
    tcp_pose = _to_numpy(extra["tcp_pose"]).astype(np.float32)
    tcp_to_handle_pos = _to_numpy(extra["tcp_to_handle_pos"]).astype(np.float32)
    target_link_qpos = _to_numpy(extra["target_link_qpos"]).astype(np.float32)
    target_handle_pos = _to_numpy(extra["target_handle_pos"]).astype(np.float32)

    if qpos.ndim == 1:
        qpos = qpos[None, :]
        qvel = qvel[None, :]
        tcp_pose = tcp_pose[None, :]
        tcp_to_handle_pos = tcp_to_handle_pos[None, :]
        target_link_qpos = target_link_qpos.reshape(1, -1)
        target_handle_pos = target_handle_pos[None, :]
    elif target_link_qpos.ndim == 1:
        target_link_qpos = target_link_qpos[:, None]

    qvel = np.clip(qvel, -10.0, 10.0) / 10.0
    tcp_to_handle_pos = np.clip(tcp_to_handle_pos, -1.0, 1.0)
    target_handle_pos = np.clip(target_handle_pos, -2.0, 2.0) / 2.0

    return np.concatenate(
        [qpos, qvel, tcp_pose, tcp_to_handle_pos, target_link_qpos, target_handle_pos],
        axis=-1,
    ).astype(np.float32)


def get_controlled_action_indices(action_mapping: Dict[str, Tuple[int, int]]) -> Tuple[int, ...]:
    controlled_indices: List[int] = []
    for controller_name in ("arm", "gripper"):
        action_slice = action_mapping.get(controller_name)
        if action_slice is None:
            raise KeyError(f"Expected controller {controller_name!r} in action_mapping, got {action_mapping}")
        start, end = action_slice
        controlled_indices.extend(range(int(start), int(end)))
    return tuple(controlled_indices)


def inspect_env_contract(args: "Args", device: torch.device) -> Tuple[int, int, Tuple[int, ...]]:
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        **backend_kwargs,
    )
    obs, _ = env.reset(seed=args.seed)
    action_dim = int(env.action_space.shape[-1])
    state_dim = int(extract_cabinet_state_batch_from_obs(obs).shape[-1])
    action_mapping = getattr(env.unwrapped.agent.controller, "action_mapping", None)
    if not isinstance(action_mapping, dict):
        raise RuntimeError(f"Unable to inspect controller action mapping for env {args.env_id}")
    controlled_action_indices = get_controlled_action_indices(action_mapping)
    env.close()
    return action_dim, state_dim, controlled_action_indices


def get_maniskill_backend_kwargs(device: torch.device) -> Dict[str, str]:
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        return {
            "sim_backend": f"physx_cuda:{device_index}",
            "render_backend": f"cuda:{device_index}",
        }
    return {
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
    }


class HandSafeManiSkillVectorEnv(ManiSkillVectorEnv):
    """Mirror ManiSkill's partial auto-reset behavior for hand PPO training."""

    def step(self, actions):  # type: ignore[override]
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        episode_info: Optional[dict] = None
        if self.record_metrics:
            episode_info = dict()
            self.returns += rew
            if "success" in infos:
                self.success_once = self.success_once | infos["success"]
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.base_env.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]

        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)

        if self.ignore_terminations:
            terminations[:] = False
            if episode_info:
                if "success" in infos:
                    episode_info["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    episode_info["fail_at_end"] = infos["fail"].clone()
        if self.record_metrics:
            infos["episode"] = episode_info

        dones = torch.logical_or(terminations, truncations)
        if dones.any() and self.auto_reset:
            final_obs = torch_clone_dict(obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = torch_clone_dict(infos)
            obs, infos = self.reset(options=dict(env_idx=env_idx))
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones

        return obs, rew, terminations, truncations, infos


def make_vector_env(
    args: "Args",
    device: torch.device,
    num_envs: int,
    record_metrics: bool = True,
    video_output_dir: Optional[Path] = None,
    video_max_steps: Optional[int] = None,
) -> ManiSkillVectorEnv:
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        **backend_kwargs,
    )
    if video_output_dir is not None:
        resolved_video_max_steps = video_max_steps
        if resolved_video_max_steps is None:
            resolved_video_max_steps = args.max_episode_steps or gym_utils.find_max_episode_steps_value(env)
        env = RecordEpisode(
            env,
            output_dir=str(video_output_dir),
            save_trajectory=False,
            max_steps_per_video=resolved_video_max_steps,
            video_fps=30,
        )
    return HandSafeManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False, record_metrics=record_metrics)


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelActionTokenHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.query_tokens = nn.Parameter(torch.randn(action_dim, hidden_dim, dtype=torch.float32) * 0.02)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        prompt_feature: torch.Tensor,
        state_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prompt_feature.shape[0]
        seq_len = self.action_dim
        action_features = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_feature = prompt_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_state = state_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_context = context_feature.unsqueeze(1).expand(-1, seq_len, -1)
        action_features = self.context_encoder(action_features + prompt_feature + expanded_context)
        fused = torch.cat([action_features, prompt_feature, expanded_state, expanded_context], dim=-1)
        return self.logit_head(fused) * self.residual_scale


class EdgeVLAActorCritic(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        state_dim: int = 44,
        action_dim: int = 8,
        env_action_dim: int = 13,
        controlled_action_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.model_dir = resolve_model_dir_path(model_dir)
        self.device = device
        self.state_dim = state_dim
        self.policy_action_dim = action_dim
        self.env_action_dim = env_action_dim
        if controlled_action_indices is None:
            controlled_action_indices = tuple(range(action_dim))
        if len(controlled_action_indices) != action_dim:
            raise ValueError(
                f"Expected {action_dim} controlled action indices, got {len(controlled_action_indices)}"
            )
        self.controlled_action_indices = tuple(int(index) for index in controlled_action_indices)
        self.prompt = f"In: What action should the robot take to {TASK_PROMPT}\nOut: "

        fallback_bundle = maybe_build_random_init_vla_bundle(
            model_dir=self.model_dir,
            prompt=self.prompt,
            device=device,
            num_action_tokens=action_dim,
            action_stats_dim=action_dim,
        )
        if fallback_bundle is None:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir), trust_remote_code=True)
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            prompt_tokens = self.processor.tokenizer(self.prompt, return_tensors="pt")
            self.register_buffer("prompt_input_ids", prompt_tokens["input_ids"], persistent=False)
            self.register_buffer("prompt_attention_mask", prompt_tokens["attention_mask"], persistent=False)
            ensure_package("local_edge_vla_pkg", self.model_dir)
            config_mod = load_module_from_path(
                "local_edge_vla_pkg.configuration_prismatic",
                self.model_dir / "configuration_prismatic.py",
            )
            model_mod = load_module_from_path(
                "local_edge_vla_pkg.modeling_prismatic",
                self.model_dir / "modeling_prismatic.py",
            )
            self.ignore_index = int(getattr(model_mod, "IGNORE_INDEX", -100))
            self.num_tokens = int(getattr(model_mod, "NUM_TOKENS", 64))

            self.vla = model_mod.OpenVLAForActionPrediction.from_pretrained(
                str(self.model_dir),
                config=config_mod.OpenVLAConfig.from_pretrained(str(self.model_dir)),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=get_attention_implementation(),
            ).to(device)
            self.vla.set_version("v1")
            self.full_vocab_size = int(self.vla.vocab_size)
        else:
            self.processor = fallback_bundle["processor"]
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            self.register_buffer("prompt_input_ids", fallback_bundle["prompt_tokens"]["input_ids"], persistent=False)
            self.register_buffer("prompt_attention_mask", fallback_bundle["prompt_tokens"]["attention_mask"], persistent=False)
            self.ignore_index = int(fallback_bundle["ignore_index"])
            self.num_tokens = int(fallback_bundle["num_tokens"])
            self.vla = fallback_bundle["vla"]
            self.full_vocab_size = int(self.vla.vocab_size)

        self.action_token_end_idx = int(self.action_tokenizer.action_token_end_idx)
        self.num_action_bins = int(self.action_tokenizer.vocab_size)
        self.hidden_dim = int(self.vla.llm_dim)

        self.register_buffer(
            "action_bin_centers",
            torch.from_numpy(self.action_tokenizer.bin_centers.astype(np.float32)),
            persistent=False,
        )

        self.state_projector = MLPProjector(state_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim).to(
            device=device,
            dtype=torch.float32,
        )
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(device=device, dtype=torch.float32)
        self.actor_head = ParallelActionTokenHead(
            hidden_dim=self.hidden_dim,
            action_dim=self.policy_action_dim,
            num_bins=self.num_action_bins,
        ).to(device=device, dtype=torch.float32)

        critic_input_dim = self.hidden_dim * 3
        critic_hidden_dim = max(256, self.hidden_dim // 2)
        self.value_head = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.GELU(),
            nn.Linear(critic_hidden_dim, 1),
        ).to(device=device, dtype=torch.float32)

        self.eval_micro_batch_size = 32
        self._vla_trainable = True
        self._set_backbone_trainable(True)

    def _set_backbone_trainable(self, trainable: bool) -> None:
        self._vla_trainable = trainable
        for parameter in self.vla.parameters():
            parameter.requires_grad = trainable

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self._set_backbone_trainable(train_backbone)
        for module in [self.state_projector, self.context_projector, self.actor_head, self.value_head]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def trainable_parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        modules = {
            "vla": self.vla,
            "state_projector": self.state_projector,
            "context_projector": self.context_projector,
            "actor_head": self.actor_head,
            "value_head": self.value_head,
        }
        summary: Dict[str, Tuple[int, int]] = {}
        for name, module in modules.items():
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            summary[name] = (total, trainable)
        return summary

    @staticmethod
    def _prepare_image(rgb: Union[np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(rgb, torch.Tensor):
            if rgb.device.type != "cpu" or rgb.dtype != torch.uint8 or not rgb.is_contiguous():
                rgb = rgb.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            return Image.fromarray(rgb.numpy(), mode="RGB").convert("RGB")
        return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGB")

    def _prepare_policy_inputs(self, rgbs: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach()
            if rgb_batch.device.type != "cpu" or rgb_batch.dtype != torch.uint8 or not rgb_batch.is_contiguous():
                rgb_batch = rgb_batch.to(device="cpu", dtype=torch.uint8).contiguous()
        else:
            rgb_batch = torch.from_numpy(np.asarray(rgbs)[..., :3].astype(np.uint8, copy=False)).contiguous()

        images = [self._prepare_image(rgb) for rgb in rgb_batch]
        batch_size = len(images)
        pixel_values = self.processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return {
            "input_ids": self.prompt_input_ids.expand(batch_size, -1).to(self.device, non_blocking=True),
            "attention_mask": self.prompt_attention_mask.expand(batch_size, -1).to(self.device, non_blocking=True),
            "pixel_values": pixel_values.to(self.device, dtype=torch.bfloat16, non_blocking=True),
        }

    def _action_bins_to_token_ids(self, action_bins: torch.Tensor) -> torch.Tensor:
        action_bins = action_bins.to(self.device, dtype=torch.long)
        action_bins = torch.clamp(action_bins, 0, self.num_action_bins - 1)
        return self.action_token_end_idx - action_bins - 1

    def _is_vla_trainable(self) -> bool:
        return self._vla_trainable

    def env_actions_to_bin_indices(self, env_actions: np.ndarray) -> torch.Tensor:
        env_actions = np.asarray(env_actions, dtype=np.float32)
        if env_actions.ndim == 1:
            env_actions = env_actions[None, :]
        if env_actions.shape[-1] == self.env_action_dim:
            env_actions = env_actions[:, self.controlled_action_indices]
        elif env_actions.shape[-1] != self.policy_action_dim:
            raise ValueError(
                f"Expected env_actions with shape [batch, {self.env_action_dim}] or "
                f"[batch, {self.policy_action_dim}], got {tuple(env_actions.shape)}"
            )
        env_actions = np.clip(env_actions, -1.0, 1.0)
        token_ids = np.asarray(self.action_tokenizer(env_actions, use_minivlm=True), dtype=np.int64)
        bin_indices = self.action_token_end_idx - token_ids - 1
        bin_indices = np.clip(bin_indices, 0, self.action_bin_centers.numel() - 1)
        return torch.from_numpy(bin_indices.astype(np.int64))

    def _compose_env_actions(self, controlled_actions: torch.Tensor) -> torch.Tensor:
        if controlled_actions.ndim == 1:
            controlled_actions = controlled_actions.unsqueeze(0)
        batch_size = controlled_actions.shape[0]
        env_actions = torch.zeros((batch_size, self.env_action_dim), device=self.device, dtype=torch.float32)
        env_actions[:, self.controlled_action_indices] = controlled_actions.to(self.device, dtype=torch.float32)
        return env_actions

    def bin_indices_to_env_actions(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if bin_indices.ndim == 1:
            bin_indices = bin_indices.unsqueeze(0)
        bin_indices = bin_indices.to(self.device, dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, 0, self.action_bin_centers.numel() - 1)
        controlled_actions = self.action_bin_centers.to(self.device)[bin_indices].to(torch.float32)
        controlled_actions = torch.nan_to_num(controlled_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        return self._compose_env_actions(controlled_actions)

    def _project_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._is_vla_trainable():
            return self.vla._process_vision_features(pixel_values, language_embeddings=None, use_film=False)
        with torch.no_grad():
            return self.vla._process_vision_features(pixel_values, language_embeddings=None, use_film=False)

    @staticmethod
    def _append_state_token_to_patches(
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        state_token = state_feature.to(dtype=projected_patch_embeddings.dtype).unsqueeze(1)
        return torch.cat([projected_patch_embeddings, state_token], dim=1)

    def _language_model_from_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
    ):
        def run_language_model():
            input_embeddings = self.vla.get_input_embeddings()(input_ids)
            multimodal_embeddings, multimodal_attention_mask = self.vla._build_multimodal_attention(
                input_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )
            return self.vla.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        if self._is_vla_trainable():
            return run_language_model()
        with torch.no_grad():
            return run_language_model()

    def _compute_policy_logits_and_value(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_feature = self.state_projector(state_tensor)
        projected_patch_embeddings = self._project_vision_features(pixel_values)
        projected_patch_embeddings = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)
        output = self._language_model_from_prefix(input_ids, attention_mask, projected_patch_embeddings)
        final_hidden = output.hidden_states[-1].to(torch.float32)
        prompt_hidden = final_hidden[:, -1, :]
        context_feature = self.context_projector(final_hidden.mean(dim=1))
        logits = torch.nan_to_num(
            self.actor_head(prompt_hidden, state_feature, context_feature),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        value_input = torch.cat([prompt_hidden, state_feature, context_feature], dim=-1)
        value = torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)
        return logits, value

    def _parallel_action_and_value(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self._compute_policy_logits_and_value(rgbs=rgbs, states=states)
        categorical = torch.distributions.Categorical(logits=logits)
        if action_bins is not None:
            action_bins = action_bins.to(self.device, dtype=torch.long)
            if action_bins.ndim != 2 or action_bins.shape[1] != self.policy_action_dim:
                raise ValueError(
                    f"action_bins must have shape [batch, {self.policy_action_dim}], got {tuple(action_bins.shape)}"
                )
            selected_bins = action_bins
        else:
            selected_bins = logits.argmax(dim=-1) if deterministic else categorical.sample()
        log_prob = categorical.log_prob(selected_bins).sum(dim=-1)
        entropy = categorical.entropy().mean(dim=-1)
        env_actions = self.bin_indices_to_env_actions(selected_bins)
        return env_actions, log_prob, entropy, value, selected_bins

    def forward_policy(self, rgbs: np.ndarray, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._compute_policy_logits_and_value(rgbs=rgbs, states=states)

    def get_value(self, rgbs: np.ndarray, states: np.ndarray) -> torch.Tensor:
        _, value = self._compute_policy_logits_and_value(rgbs=rgbs, states=states)
        return value

    def get_action_and_value(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._parallel_action_and_value(
            rgbs=rgbs,
            states=states,
            action_bins=action_bins,
            deterministic=deterministic,
        )

    def forward(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        mode: str = "action_and_value",
    ):
        if mode == "action_and_value":
            return self.get_action_and_value(
                rgbs=rgbs,
                states=states,
                action_bins=action_bins,
                deterministic=deterministic,
            )
        if mode == "value":
            return self.get_value(rgbs=rgbs, states=states)
        if mode == "policy":
            return self.forward_policy(rgbs=rgbs, states=states)
        raise ValueError(f"Unsupported forward mode: {mode}")

    def predict_action(self, rgb: Union[np.ndarray, torch.Tensor], state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            action, _, _, _, _ = self.get_action_and_value(
                rgbs=rgb.unsqueeze(0) if isinstance(rgb, torch.Tensor) else np.expand_dims(rgb, axis=0),
                states=np.expand_dims(state, axis=0),
                deterministic=True,
            )
        return action[0].detach().cpu().numpy().astype(np.float32)


def build_optimizer(args: "Args", policy: EdgeVLAActorCritic) -> optim.Optimizer:
    param_groups = [
        {
            "params": list(policy.vla.parameters()),
            "lr": 0.0 if args.freeze_vla_backbone or args.backbone_warmup_updates > 0 else args.backbone_learning_rate,
            "group_name": "vla",
        },
        {
            "params": list(policy.state_projector.parameters()),
            "lr": args.state_learning_rate,
            "group_name": "state_projector",
        },
        {
            "params": list(policy.context_projector.parameters()),
            "lr": args.head_learning_rate,
            "group_name": "context_projector",
        },
        {
            "params": list(policy.actor_head.parameters()),
            "lr": args.head_learning_rate,
            "group_name": "actor_head",
        },
        {
            "params": list(policy.value_head.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "value_head",
        },
    ]
    return optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def set_optimizer_group_lr(optimizer: optim.Optimizer, group_name: str, lr: float) -> None:
    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == group_name:
            param_group["lr"] = lr
            return
    raise KeyError(f"Optimizer param group {group_name!r} not found")


@dataclass
class Args:
    mode: str = "train"
    seed: int = 1
    env_id: str = "OpenCabinetDrawerEasyLevel0-v1"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    output_dir: str = DEFAULT_WORKDIR
    num_envs: int = 128
    num_eval_envs: int = 32
    num_steps: int = 50
    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-6
    head_learning_rate: float = 5e-5
    state_learning_rate: float = 5e-5
    value_head_learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 1
    num_minibatches: int = 8
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    minibatch_target_kl_factor: float = 1.5
    eval_episodes: int = 40
    eval_every_updates: int = 10
    max_episode_steps: Optional[int] = None
    cuda_device: str = "0"
    smoke_steps: int = 32
    save_video: bool = False
    save_train_video_freq: int = 20
    train_video_num_envs: int = 4
    test_video_num_envs: int = 4
    test_video_episodes: int = 4
    run_setup_smoke: bool = True
    max_runtime_hours: float = 8.0
    rollout_micro_batch_size: int = 32
    eval_micro_batch_size: int = 32
    update_micro_batch_size: int = 16
    rollout_progress_log_interval: int = 5
    freeze_vla_backbone: bool = False
    backbone_warmup_updates: int = 20
    resume_from: Optional[str] = None
    init_from_policy: Optional[str] = DEFAULT_INIT_POLICY
    action_dim: int = 8
    env_action_dim: int = 13
    state_dim: int = 44


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        field_type = field_def.type
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=parse_bool, default=default)
        elif default is None:
            arg_type = str
            origin = get_origin(field_type)
            if origin is not None:
                candidate_types = [candidate for candidate in get_args(field_type) if candidate is not type(None)]
                if len(candidate_types) == 1 and isinstance(candidate_types[0], type):
                    arg_type = candidate_types[0]
            parser.add_argument(arg_name, type=arg_type, default=None)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    namespace = parser.parse_args()
    return Args(**vars(namespace))


def unwrap_policy(policy: nn.Module) -> EdgeVLAActorCritic:
    return policy.module if isinstance(policy, DDP) else policy


def sync_trainable_parameters(module: nn.Module) -> None:
    if not is_distributed():
        return
    for parameter in module.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)


def average_trainable_gradients(module: nn.Module) -> None:
    if not is_distributed():
        return
    world_size = get_world_size()
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def policy_get_action_and_value(
    policy: nn.Module,
    rgbs: np.ndarray,
    states: np.ndarray,
    action_bins: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(policy, DDP):
        return policy(rgbs, states, action_bins=action_bins, deterministic=deterministic, mode="action_and_value")
    return policy.get_action_and_value(rgbs=rgbs, states=states, action_bins=action_bins, deterministic=deterministic)


def policy_get_value(policy: nn.Module, rgbs: np.ndarray, states: np.ndarray) -> torch.Tensor:
    if isinstance(policy, DDP):
        return policy(rgbs, states, mode="value")
    return policy.get_value(rgbs=rgbs, states=states)


def get_completed_episode_metrics(infos: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    final_info = infos.get("final_info")
    done_mask = infos.get("_final_info")
    if final_info is None or done_mask is None:
        return None, {}
    episode_metrics = final_info.get("episode")
    if not isinstance(episode_metrics, dict):
        return done_mask, {}
    return done_mask, episode_metrics


def batched_get_action_and_value_no_grad(
    policy: nn.Module,
    rgbs: np.ndarray,
    states: np.ndarray,
    micro_batch_size: int,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = []
    log_probs = []
    entropies = []
    values = []
    action_bins = []
    with torch.no_grad():
        for start, end in iter_slices(len(rgbs), micro_batch_size):
            action, log_prob, entropy, value, bins = policy_get_action_and_value(
                policy,
                rgbs=rgbs[start:end],
                states=states[start:end],
                deterministic=deterministic,
            )
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            action_bins.append(bins)
    return (
        torch.cat(actions, dim=0),
        torch.cat(log_probs, dim=0),
        torch.cat(entropies, dim=0),
        torch.cat(values, dim=0),
        torch.cat(action_bins, dim=0),
    )


def batched_get_value_no_grad(
    policy: nn.Module,
    rgbs: np.ndarray,
    states: np.ndarray,
    micro_batch_size: int,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start, end in iter_slices(len(rgbs), micro_batch_size):
            values.append(policy_get_value(policy, rgbs[start:end], states[start:end]))
    return torch.cat(values, dim=0)


def evaluate_policy(policy: nn.Module, envs: ManiSkillVectorEnv, target_episodes: int) -> Dict[str, float]:
    if use_train_success_only():
        return {}
    if target_episodes <= 0:
        return {}
    metrics = defaultdict(list)
    obs, _ = envs.reset(seed=0)
    episodes = 0
    raw_policy = unwrap_policy(policy)
    raw_policy.eval()
    with torch.no_grad():
        while episodes < target_episodes:
            rgbs = extract_rgb_batch_from_obs(obs)
            states = extract_cabinet_state_batch_from_obs(obs)
            action_chunks = []
            for start, end in iter_slices(len(rgbs), raw_policy.eval_micro_batch_size):
                action_chunk, _, _, _, _ = policy_get_action_and_value(
                    raw_policy,
                    rgbs=rgbs[start:end],
                    states=states[start:end],
                    deterministic=True,
                )
                action_chunks.append(action_chunk)
            action = torch.cat(action_chunks, dim=0)
            obs, _, _, _, infos = envs.step(action)
            done_mask, episode_metrics = get_completed_episode_metrics(infos)
            if done_mask is None:
                continue
            episodes += int(done_mask.sum().item())
            for key, value in episode_metrics.items():
                metrics[key].append(value[done_mask].float().detach().cpu())
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = torch.cat(values).mean().item()
    return result


def record_policy_rollout_video(policy: nn.Module, envs: ManiSkillVectorEnv, num_steps: int, seed: int) -> None:
    obs, _ = envs.reset(seed=seed)
    raw_policy = unwrap_policy(policy)
    raw_policy.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            rgbs = extract_rgb_batch_from_obs(obs)
            states = extract_cabinet_state_batch_from_obs(obs)
            action_chunks = []
            for start, end in iter_slices(len(rgbs), raw_policy.eval_micro_batch_size):
                action_chunk, _, _, _, _ = policy_get_action_and_value(
                    raw_policy,
                    rgbs=rgbs[start:end],
                    states=states[start:end],
                    deterministic=False,
                )
                action_chunks.append(action_chunk)
            action = torch.cat(action_chunks, dim=0)
            obs, _, _, _, _ = envs.step(action)


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_error = error.abs()
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=error.device, dtype=error.dtype))
    linear = abs_error - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def normalize_advantages(advantages: torch.Tensor, device: torch.device) -> torch.Tensor:
    flat = advantages.detach().reshape(-1).to(device=device, dtype=torch.float64)
    stats = torch.stack(
        (
            torch.tensor(float(flat.numel()), device=device, dtype=torch.float64),
            flat.sum(),
            (flat * flat).sum(),
        )
    )
    if is_distributed():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count = torch.clamp(stats[0], min=1.0)
    mean = stats[1] / count
    variance = torch.clamp(stats[2] / count - mean * mean, min=0.0)
    std = torch.sqrt(variance)
    return ((advantages - mean.to(advantages.dtype)) / (std.to(advantages.dtype) + 1e-8)).to(advantages.dtype)


def ppo_update_with_micro_batches(
    args: Args,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    b_rgbs: np.ndarray,
    b_states: np.ndarray,
    b_action_bins: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_values: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    minibatch_inds: np.ndarray,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    local_advantages = b_advantages[minibatch_inds]
    total = len(minibatch_inds)
    stats = defaultdict(float)
    skipped_on_kl = False
    micro_slices = list(iter_slices(total, args.update_micro_batch_size))

    for slice_idx, (local_start, local_end) in enumerate(micro_slices):
        micro_inds = minibatch_inds[local_start:local_end]
        micro_weight = (local_end - local_start) / total
        context = policy.no_sync() if isinstance(policy, DDP) and slice_idx != len(micro_slices) - 1 else torch.enable_grad()
        with context:
            _, newlogprob, entropy, newvalue, _ = policy_get_action_and_value(
                policy,
                b_rgbs[micro_inds],
                b_states[micro_inds],
                b_action_bins[micro_inds],
            )
            logratio = newlogprob - b_logprobs[micro_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                micro_approx_kl = ((ratio - 1) - logratio).mean().item()
                stats["approx_kl"] += micro_approx_kl * micro_weight
                stats["clipfrac"] += (((ratio - 1.0).abs() > args.clip_coef).float().mean().item()) * micro_weight

            global_micro_approx_kl = distributed_max(micro_approx_kl, unwrap_policy(policy).device)
            if args.target_kl is not None and global_micro_approx_kl > args.target_kl * args.minibatch_target_kl_factor:
                skipped_on_kl = True
                break

            micro_adv = local_advantages[local_start:local_end]
            pg_loss1 = -micro_adv * ratio
            pg_loss2 = -micro_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            newvalue = newvalue.view(-1)
            oldvalue = b_values[micro_inds].view(-1)
            returns = b_returns[micro_inds].view(-1)
            value_pred_clipped = oldvalue + (newvalue - oldvalue).clamp(-args.clip_coef, args.clip_coef)
            value_loss_original = huber_loss(newvalue - returns, delta=10.0)
            value_loss_clipped = huber_loss(value_pred_clipped - returns, delta=10.0)
            v_loss = torch.max(value_loss_original, value_loss_clipped).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            (loss * micro_weight).backward()

        stats["pg_loss"] += float(pg_loss.detach().item()) * micro_weight
        stats["v_loss"] += float(v_loss.detach().item()) * micro_weight
        stats["entropy"] += float(entropy_loss.detach().item()) * micro_weight

    if skipped_on_kl:
        optimizer.zero_grad(set_to_none=True)
        stats["skipped_on_kl"] = 1.0
        return dict(stats)

    average_trainable_gradients(unwrap_policy(policy))
    nn.utils.clip_grad_norm_(unwrap_policy(policy).parameters(), args.max_grad_norm)
    optimizer.step()
    stats["skipped_on_kl"] = 0.0
    return dict(stats)


def run_vla_inference_smoke(
    args: Args,
    device: torch.device,
    output_dir: Path,
    policy: Optional[EdgeVLAActorCritic] = None,
) -> None:
    if policy is None:
        inferred_env_action_dim, inferred_state_dim, controlled_action_indices = inspect_env_contract(args, device)
        policy = EdgeVLAActorCritic(
            Path(args.model_dir),
            device=device,
            state_dim=inferred_state_dim,
            action_dim=len(controlled_action_indices),
            env_action_dim=inferred_env_action_dim,
            controlled_action_indices=controlled_action_indices,
        ).to(device)
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        **backend_kwargs,
    )
    obs, _ = env.reset(seed=args.seed)
    returns = 0.0
    success = False
    for step in range(args.smoke_steps):
        rgb = extract_rgb_batch_from_obs(obs)[0]
        state = extract_cabinet_state_batch_from_obs(obs)[0]
        action = policy.predict_action(rgb=rgb, state=state)
        obs, reward, terminated, truncated, info = env.step(
            torch.from_numpy(action).view(1, -1).to(device=device, dtype=torch.float32)
        )
        returns += float(reward.item())
        success = bool(info["success"].item())
        if terminated.item() or truncated.item():
            break
    env.close()
    payload = {"smoke_return": returns, "smoke_success": success, "steps": step + 1}
    print("[edgevla-smoke]", payload)
    save_json(output_dir / "vla_smoke.json", payload)


def maybe_warm_start_from_policy(raw_policy: EdgeVLAActorCritic, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=raw_policy.device)
    source_state = checkpoint["policy"] if isinstance(checkpoint, dict) and "policy" in checkpoint else checkpoint
    source_state = strip_module_prefix(source_state)
    target_state = raw_policy.state_dict()

    matched = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)

    target_state.update(matched)
    raw_policy.load_state_dict(target_state, strict=True)
    print(
        f"[setup] warm-started from {checkpoint_path} matched={len(matched)} skipped={len(skipped)}"
    )


def maybe_load_checkpoint(
    args: Args,
    raw_policy: EdgeVLAActorCritic,
    optimizer: optim.Optimizer,
) -> Tuple[int, int, float]:
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=raw_policy.device)
        policy_state = strip_module_prefix(checkpoint["policy"])
        raw_policy.load_state_dict(policy_state, strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint.get("update", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_success = float(checkpoint.get("best_success_once", -1.0))
        return start_update, global_step, best_success

    if args.init_from_policy and Path(args.init_from_policy).is_file():
        maybe_warm_start_from_policy(raw_policy, args.init_from_policy)
    return 1, 0, -1.0


def train(args: Args) -> None:
    device, rank, world_size = init_runtime(args)
    set_seed(args.seed + rank)

    if args.num_envs % world_size != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by world_size={world_size}")
    local_num_envs = args.num_envs // world_size

    timestamp = broadcast_object(time.strftime("%Y%m%d-%H%M%S") if is_main_process() else None)
    output_dir = mkdir(Path(args.output_dir) / timestamp)
    if is_main_process():
        backup_run_sources(output_dir)
        print("[setup] loading EdgeVLA policy")

    inferred_env_action_dim, inferred_state_dim, controlled_action_indices = inspect_env_contract(args, device)
    inferred_action_dim = len(controlled_action_indices)
    if inferred_env_action_dim != args.env_action_dim:
        if is_main_process():
            print(
                f"[setup] overriding env_action_dim from {args.env_action_dim} "
                f"to env-probed {inferred_env_action_dim}"
            )
        args.env_action_dim = inferred_env_action_dim
    if inferred_action_dim != args.action_dim:
        if is_main_process():
            print(f"[setup] overriding action_dim from {args.action_dim} to env-probed {inferred_action_dim}")
        args.action_dim = inferred_action_dim
    if inferred_state_dim != args.state_dim:
        if is_main_process():
            print(f"[setup] overriding state_dim from {args.state_dim} to env-probed {inferred_state_dim}")
        args.state_dim = inferred_state_dim
    if is_main_process():
        args_payload = asdict(args)
        args_payload.update({"world_size": world_size, "local_num_envs": local_num_envs})
        save_json(output_dir / "args.json", args_payload)

    raw_policy = EdgeVLAActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        env_action_dim=args.env_action_dim,
        controlled_action_indices=controlled_action_indices,
    ).to(device)
    train_backbone_now = not args.freeze_vla_backbone and args.backbone_warmup_updates <= 0
    raw_policy.configure_trainable_modules(train_backbone=train_backbone_now)
    raw_policy.eval_micro_batch_size = args.eval_micro_batch_size
    optimizer = build_optimizer(args, raw_policy)
    start_update, global_step, best_success_once = maybe_load_checkpoint(args, raw_policy, optimizer)

    if is_main_process():
        summary = raw_policy.trainable_parameter_summary()
        summary_text = " ".join(f"{name}={trainable}/{total}" for name, (total, trainable) in summary.items())
        print(f"[setup] trainable_params {summary_text}")

    if args.run_setup_smoke and is_main_process() and world_size == 1:
        run_vla_inference_smoke(args, device, output_dir, policy=raw_policy)
    elif is_main_process():
        print("[setup] distributed mode: skip VLA smoke")
    distributed_barrier()

    sync_trainable_parameters(raw_policy)
    policy: nn.Module = raw_policy
    if world_size > 1 and is_main_process():
        print("[setup] distributed mode: using manual gradient all-reduce instead of DDP forward synchronization")

    envs = make_vector_env(args, device, local_num_envs, record_metrics=True)
    eval_envs = make_vector_env(args, device, args.num_eval_envs, record_metrics=True) if is_main_process() else None
    train_video_envs = None
    test_video_envs = None
    if args.save_video and is_main_process():
        train_video_envs = make_vector_env(
            args,
            device,
            min(args.train_video_num_envs, args.num_envs),
            record_metrics=False,
            video_output_dir=output_dir / "train_videos",
            video_max_steps=args.num_steps,
        )
        test_video_envs = make_vector_env(
            args,
            device,
            min(args.test_video_num_envs, args.num_eval_envs),
            record_metrics=False,
            video_output_dir=output_dir / "test_videos",
        )

    global_batch_size = args.num_envs * args.num_steps
    local_batch_size = local_num_envs * args.num_steps
    global_minibatch_size = max(1, global_batch_size // args.num_minibatches)
    if global_minibatch_size % world_size != 0:
        raise ValueError(
            f"global minibatch size {global_minibatch_size} must be divisible by world_size={world_size}"
        )
    local_minibatch_size = max(1, global_minibatch_size // world_size)
    num_updates = max(1, args.total_timesteps // global_batch_size)

    env_actions_buf = torch.zeros((args.num_steps, local_num_envs, args.env_action_dim), device=device)
    action_bins_buf = torch.zeros((args.num_steps, local_num_envs, args.action_dim), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    final_values = torch.zeros((args.num_steps, local_num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed + rank)
    next_done = torch.zeros(local_num_envs, device=device)
    metrics_history: List[Dict[str, Any]] = []
    train_start_time = time.time()
    if is_main_process():
        print(
            f"[setup] world_size={world_size} local_num_envs={local_num_envs} num_updates={num_updates} "
            f"global_batch_size={global_batch_size} local_batch_size={local_batch_size} "
            f"global_minibatch_size={global_minibatch_size} local_minibatch_size={local_minibatch_size}"
        )

    for update in range(start_update, num_updates + 1):
        if not args.freeze_vla_backbone and args.backbone_warmup_updates > 0 and update == args.backbone_warmup_updates + 1:
            raw_policy.configure_trainable_modules(train_backbone=True)
            set_optimizer_group_lr(optimizer, "vla", args.backbone_learning_rate)
            if is_main_process():
                print(f"[setup] unfreezing VLA backbone at update={update} lr={args.backbone_learning_rate}")

        raw_policy.eval()
        final_values.zero_()
        rollout_rgbs: List[torch.Tensor] = []
        rollout_states: List[np.ndarray] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            step_rgbs = extract_rgb_batch_from_obs(next_obs)
            step_states = extract_cabinet_state_batch_from_obs(next_obs)
            rollout_rgbs.append(step_rgbs.clone())
            rollout_states.append(step_states.copy())
            dones_buf[step] = next_done

            action, logprob, _, value, action_bins = batched_get_action_and_value_no_grad(
                policy,
                step_rgbs,
                step_states,
                micro_batch_size=args.rollout_micro_batch_size,
            )
            env_actions_buf[step] = action
            action_bins_buf[step] = action_bins
            logprobs_buf[step] = logprob
            values_buf[step] = value

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            truncation_mask = truncations.to(torch.bool)
            next_done = (terminations | truncations).to(torch.float32)
            rewards_buf[step] = reward.view(-1)
            partial_reward_means.append(float(rewards_buf[: step + 1].mean().item()))

            if (
                (step + 1) % args.rollout_progress_log_interval == 0
                or step == 0
                or step + 1 == args.num_steps
            ):
                elapsed_hours = (time.time() - train_start_time) / 3600.0
                reward_mean_so_far = distributed_mean(partial_reward_means[-1], device)
                if is_main_process():
                    logged_partial_reward_means.append(reward_mean_so_far)
                    print(
                        f"[rollout] update={update}/{num_updates} step={step + 1}/{args.num_steps} "
                        f"reward_mean_so_far={reward_mean_so_far:.4f} elapsed_h={elapsed_hours:.2f}"
                    )
                    save_rollout_progress(
                        output_dir=output_dir,
                        update=update,
                        num_updates=num_updates,
                        rollout_step=step + 1,
                        num_steps=args.num_steps,
                        elapsed_hours=elapsed_hours,
                        partial_reward_means=logged_partial_reward_means,
                    )

            done_mask, episode_metrics = get_completed_episode_metrics(infos)
            if done_mask is not None:
                if done_mask.any():
                    for key, value_tensor in episode_metrics.items():
                        train_episode_metrics[key].append(value_tensor[done_mask].float().detach().cpu())
                bootstrap_mask = truncation_mask
                if "final_observation" in infos and bootstrap_mask.any():
                    final_obs = infos["final_observation"]
                    bootstrap_idx = bootstrap_mask.detach().cpu().numpy().astype(bool)
                    final_rgbs = extract_rgb_batch_from_obs(final_obs)[bootstrap_idx]
                    final_states = extract_cabinet_state_batch_from_obs(final_obs)[bootstrap_idx]
                    final_values[step, bootstrap_mask] = batched_get_value_no_grad(
                        policy,
                        final_rgbs,
                        final_states,
                        micro_batch_size=args.eval_micro_batch_size,
                    ).view(-1)

        with torch.no_grad():
            next_value = batched_get_value_no_grad(
                policy,
                extract_rgb_batch_from_obs(next_obs),
                extract_cabinet_state_batch_from_obs(next_obs),
                micro_batch_size=args.eval_micro_batch_size,
            ).view(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                real_next_values = next_nonterminal * next_values + final_values[t]
                delta = rewards_buf[t] + args.gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            returns = advantages + values_buf

        b_rgbs = torch.cat(rollout_rgbs, dim=0)
        b_states = np.concatenate(rollout_states, axis=0)
        b_action_bins = action_bins_buf.reshape(-1, args.action_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = normalize_advantages(advantages, device).reshape(-1)
        b_returns = returns.reshape(-1)
        ev = explained_variance(values_buf, returns, device)

        inds = np.arange(local_batch_size)
        approx_kl = 0.0
        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_value = 0.0
        clipfrac_value = 0.0
        stopped_on_minibatch_kl = False
        skipped_updates_on_kl = 0

        update_i = 0
        # Keep dropout/other train-time stochastic layers disabled so PPO ratios compare
        # the updated policy against the same distribution used during rollout.
        raw_policy.eval()

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            epoch_stats = defaultdict(list)
            for start in range(0, local_batch_size, local_minibatch_size):

                if is_main_process():
                    print(
                        f"[update] {update_i}/{args.update_epochs * (local_batch_size // local_minibatch_size)} "
                    )
                    update_i += 1

                end = start + local_minibatch_size
                mb_inds = inds[start:end]
                mb_stats = ppo_update_with_micro_batches(
                    args=args,
                    policy=policy,
                    optimizer=optimizer,
                    b_rgbs=b_rgbs,
                    b_states=b_states,
                    b_action_bins=b_action_bins,
                    b_logprobs=b_logprobs,
                    b_values=b_values,
                    b_advantages=b_advantages,
                    b_returns=b_returns,
                    minibatch_inds=mb_inds,
                )
                for key, value in mb_stats.items():
                    epoch_stats[key].append(value)
                skipped_updates_on_kl += int(mb_stats.get("skipped_on_kl", 0.0) > 0.0)
                minibatch_kl = distributed_max(float(mb_stats.get("approx_kl", 0.0)), device)
                if mb_stats.get("skipped_on_kl", 0.0) > 0.0 or minibatch_kl > args.target_kl * args.minibatch_target_kl_factor:
                    stopped_on_minibatch_kl = True
                    break

            approx_kl = distributed_mean(
                float(np.mean(epoch_stats["approx_kl"])) if epoch_stats["approx_kl"] else 0.0,
                device,
            )
            clipfrac_value = distributed_mean(
                float(np.mean(epoch_stats["clipfrac"])) if epoch_stats["clipfrac"] else 0.0,
                device,
            )
            pg_loss_value = distributed_mean(
                float(np.mean(epoch_stats["pg_loss"])) if epoch_stats["pg_loss"] else 0.0,
                device,
            )
            v_loss_value = distributed_mean(
                float(np.mean(epoch_stats["v_loss"])) if epoch_stats["v_loss"] else 0.0,
                device,
            )
            entropy_value = distributed_mean(
                float(np.mean(epoch_stats["entropy"])) if epoch_stats["entropy"] else 0.0,
                device,
            )
            if stopped_on_minibatch_kl or approx_kl > args.target_kl:
                break

        gae_return_mean = distributed_mean(returns.mean().item(), device)
        metric = {
            "update": update,
            "global_step": global_step,
            "reward_mean": distributed_mean(rewards_buf.mean().item(), device),
            "return_mean": gae_return_mean,
            "gae_return_mean": gae_return_mean,
            "value_mean": distributed_mean(values_buf.mean().item(), device),
            "explained_variance": ev,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac_value,
            "pg_loss": pg_loss_value,
            "v_loss": v_loss_value,
            "entropy": entropy_value,
            "stopped_on_minibatch_kl": stopped_on_minibatch_kl,
            "skipped_updates_on_kl": skipped_updates_on_kl,
            "elapsed_hours": (time.time() - train_start_time) / 3600.0,
        }

        local_train_summary = {}
        for key, values in train_episode_metrics.items():
            if values:
                cat = torch.cat(values)
                local_train_summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
        metric.update(gather_metric_summary(local_train_summary))
        if use_train_success_only():
            for source_key, target_key in (
                ("train_success_once", "eval_success_once"),
                ("train_success_at_end", "eval_success_at_end"),
                ("train_success", "eval_success"),
            ):
                value = metric.get(source_key)
                if value is not None:
                    metric[target_key] = value

        if (
            is_main_process()
            and train_video_envs is not None
            and args.save_train_video_freq > 0
            and update % args.save_train_video_freq == 0
        ):
            record_policy_rollout_video(raw_policy, train_video_envs, num_steps=args.num_steps, seed=args.seed + update)

        if update % args.eval_every_updates == 0 or update == num_updates:
            if is_main_process() and eval_envs is not None:
                eval_metrics = evaluate_policy(raw_policy, eval_envs, args.eval_episodes)
                if not eval_metrics and use_train_success_only():
                    for source_key, target_key in (
                        ("train_success_once", "success_once"),
                        ("train_success_at_end", "success_at_end"),
                        ("train_success", "success"),
                    ):
                        value = metric.get(source_key)
                        if value is not None:
                            eval_metrics[target_key] = value
                metric.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                if test_video_envs is not None:
                    evaluate_policy(
                        raw_policy,
                        test_video_envs,
                        min(args.test_video_episodes, max(1, args.test_video_num_envs)),
                    )
                success_once = eval_metrics.get("success_once", eval_metrics.get("success", 0.0))
                if success_once >= best_success_once:
                    best_success_once = success_once
                    torch.save(
                        {
                            "policy": raw_policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "update": update,
                            "global_step": global_step,
                            "best_success_once": best_success_once,
                        },
                        output_dir / "best_policy.pt",
                    )

        if is_main_process():
            metrics_history.append(metric)
            train_return = metric.get("train_return", float("nan"))
            print(
                f"[train] update={update}/{num_updates} step={global_step} "
                f"reward={metric['reward_mean']:.4f} train_return={train_return:.4f} "
                f"gae_return={metric['gae_return_mean']:.4f} "
                f"value_mean={metric['value_mean']:.4f} explained_variance={metric['explained_variance']:.4f} "
                f"approx_kl={metric['approx_kl']:.5f} eval_success_once={metric.get('eval_success_once', float('nan')):.4f} "
                f"elapsed_h={metric['elapsed_hours']:.2f}"
            )
            save_json(output_dir / "latest_metrics.json", metric)
            save_metrics_history(output_dir, metrics_history)
            plot_metrics_history(output_dir, metrics_history)
            if update % 10 == 0 or update == num_updates:
                torch.save(
                    {
                        "policy": raw_policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "global_step": global_step,
                        "best_success_once": best_success_once,
                    },
                    output_dir / "latest_policy.pt",
                )

        reached_time_limit = metric["elapsed_hours"] >= args.max_runtime_hours
        reached_time_limit = broadcast_object(reached_time_limit)
        if reached_time_limit:
            if is_main_process():
                print(f"[train] reached time limit: {metric['elapsed_hours']:.2f}h >= {args.max_runtime_hours:.2f}h")
            break

    if is_main_process():
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    if train_video_envs is not None:
        train_video_envs.close()
    if test_video_envs is not None:
        test_video_envs.close()
    distributed_barrier()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        try:
            train(args)
        finally:
            cleanup_runtime()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = mkdir(Path(args.output_dir) / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    backup_run_sources(output_dir)
    save_json(output_dir / "args.json", asdict(args))
    if args.mode == "vla_smoke":
        run_vla_inference_smoke(args, device, output_dir)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
