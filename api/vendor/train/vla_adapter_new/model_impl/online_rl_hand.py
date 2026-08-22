import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

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

import env as easier_rotate_env  # noqa: F401
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


TASK_PROMPT = "rotate the object in hand quickly without dropping it."
DEFAULT_MODEL_DIR = "eval/ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_WORKDIR = "train/vla_adapter_new/model_impl/outputs/ppo_rotate_hand"


def get_attention_implementation() -> str:
    requested = os.environ.get("HAND_VLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[setup] flash_attn is not installed; falling back to SDPA attention")
        return "sdpa"
    print(f"[setup] using {requested} attention implementation for HandVLAAdapterActorCritic")
    return requested


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb[..., :3].detach().cpu().numpy()
    else:
        rgb = np.asarray(rgb)[..., :3]
    return rgb.astype(np.uint8)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_hand_state_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    agent = obs["agent"]
    extra = obs["extra"]

    qpos = _to_numpy(agent["qpos"]).astype(np.float32)
    qvel = _to_numpy(agent["qvel"]).astype(np.float32)
    palm_pose = _to_numpy(agent["palm_pose"]).astype(np.float32)
    tip_poses = _to_numpy(agent["tip_poses"]).astype(np.float32)
    fsr_impulse = _to_numpy(agent["fsr_impulse"]).astype(np.float32)
    rotate_dir = _to_numpy(extra["rotate_dir"]).astype(np.float32)
    obj_pose = _to_numpy(extra["obj_pose"]).astype(np.float32)
    obj_tip_vec = _to_numpy(extra["obj_tip_vec"]).astype(np.float32)

    if qpos.ndim == 1:
        qpos = qpos[None, :]
        qvel = qvel[None, :]
        palm_pose = palm_pose[None, :]
        tip_poses = tip_poses[None, :]
        fsr_impulse = fsr_impulse[None, :]
        rotate_dir = rotate_dir[None, :]
        obj_pose = obj_pose[None, :]
        obj_tip_vec = obj_tip_vec[None, :]

    qvel = np.clip(qvel, -20.0, 20.0) / 20.0
    fsr_impulse = np.log1p(np.clip(fsr_impulse, 0.0, None))

    return np.concatenate(
        [qpos, qvel, palm_pose, tip_poses, fsr_impulse, rotate_dir, obj_pose, obj_tip_vec],
        axis=-1,
    ).astype(np.float32)


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


class ResidualDiscreteActorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
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
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        action_features: torch.Tensor,
        state_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = action_features.shape
        # Encode each next-token position independently. During teacher forcing,
        # cross-position attention here would leak future action tokens.
        action_features = self.context_encoder(action_features.reshape(batch_size * seq_len, 1, hidden_dim)).reshape(
            batch_size,
            seq_len,
            hidden_dim,
        )
        seq_len = action_features.shape[1]
        expanded_state = state_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_context = context_feature if context_feature.ndim == 3 else context_feature.unsqueeze(1)
        expanded_context = expanded_context.expand(-1, seq_len, -1)
        fused = torch.cat([action_features, expanded_state, expanded_context], dim=-1)
        return self.logit_head(fused) * self.residual_scale


class HandVLAAdapterActorCritic(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device, state_dim: int = 105, action_dim: int = 16):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        self.state_dim = state_dim
        self.env_action_dim = action_dim
        self.prompt = f"In: What action should the robot take to {TASK_PROMPT}\nOut: "

        fallback_bundle = maybe_build_random_init_vla_bundle(
            model_dir=model_dir,
            prompt=self.prompt,
            device=device,
            num_action_tokens=action_dim,
            action_stats_dim=action_dim,
        )
        if fallback_bundle is None:
            self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            ensure_package("local_hand_vla_pkg", model_dir)
            config_mod = load_module_from_path(
                "local_hand_vla_pkg.configuration_prismatic",
                model_dir / "configuration_prismatic.py",
            )
            model_mod = load_module_from_path(
                "local_hand_vla_pkg.modeling_prismatic",
                model_dir / "modeling_prismatic.py",
            )
            self.ignore_index = int(getattr(model_mod, "IGNORE_INDEX", -100))
            self.num_tokens = int(getattr(model_mod, "NUM_TOKENS", 64))

            self.vla = model_mod.OpenVLAForActionPrediction.from_pretrained(
                str(model_dir),
                config=config_mod.OpenVLAConfig.from_pretrained(str(model_dir)),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=get_attention_implementation(),
            ).to(device)
            self.vla.set_version("v1")
            self.full_vocab_size = int(self.vla.vocab_size)
        else:
            self.processor = fallback_bundle["processor"]
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            self.ignore_index = int(fallback_bundle["ignore_index"])
            self.num_tokens = int(fallback_bundle["num_tokens"])
            self.vla = fallback_bundle["vla"]
            self.full_vocab_size = int(self.vla.vocab_size)

        self.action_token_start_idx = int(self.action_tokenizer.action_token_begin_idx + 1)
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
        self.actor_head = ResidualDiscreteActorHead(
            hidden_dim=self.hidden_dim,
            action_dim=self.env_action_dim,
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
    def _prepare_image(rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")

    def _prepare_policy_inputs(self, rgbs: np.ndarray) -> Dict[str, torch.Tensor]:
        images = [self._prepare_image(rgb[..., :3]) for rgb in np.asarray(rgbs)]
        prompts = [self.prompt] * len(images)
        processor_outputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        return {
            "input_ids": processor_outputs["input_ids"].to(self.device),
            "attention_mask": processor_outputs["attention_mask"].to(self.device),
            "pixel_values": processor_outputs["pixel_values"].to(self.device, dtype=torch.bfloat16),
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
        env_actions = np.clip(env_actions, -1.0, 1.0)
        token_ids = np.asarray(self.action_tokenizer(env_actions, use_minivlm=True), dtype=np.int64)
        bin_indices = self.action_token_end_idx - token_ids - 1
        bin_indices = np.clip(bin_indices, 0, self.action_bin_centers.numel() - 1)
        return torch.from_numpy(bin_indices.astype(np.int64))

    def bin_indices_to_env_actions(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if bin_indices.ndim == 1:
            bin_indices = bin_indices.unsqueeze(0)
        bin_indices = bin_indices.to(self.device, dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, 0, self.action_bin_centers.numel() - 1)
        env_actions = self.action_bin_centers.to(self.device)[bin_indices].to(torch.float32)
        return torch.nan_to_num(env_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

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

    def _language_model_from_prefix_with_cache(
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
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        if self._is_vla_trainable():
            return run_language_model()
        with torch.no_grad():
            return run_language_model()

    def _language_model_next_token_from_cache(
        self,
        token_id: torch.Tensor,
        past_key_values,
    ):
        def run_language_model():
            return self.vla.language_model(
                input_ids=token_id,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        if self._is_vla_trainable():
            return run_language_model()
        with torch.no_grad():
            return run_language_model()

    def _next_token_logits_and_value_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._language_model_from_prefix(input_ids, attention_mask, projected_patch_embeddings)
        final_hidden = output.hidden_states[-1].to(torch.float32)
        next_hidden = final_hidden[:, -1, :]
        context_feature = self.context_projector(final_hidden.mean(dim=1))

        action_token_logits = output.logits[:, -1, self.action_token_start_idx : self.action_token_end_idx].to(
            torch.float32
        )
        base_bin_logits = torch.flip(action_token_logits, dims=[-1])
        residual_logits = self.actor_head(next_hidden.unsqueeze(1), state_feature, context_feature).squeeze(1)
        logits = torch.nan_to_num(base_bin_logits + residual_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return logits, next_hidden, context_feature

    def _compute_prompt_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        _, prompt_hidden, context_feature = self._next_token_logits_and_value_features(
            input_ids,
            attention_mask,
            projected_patch_embeddings,
            state_feature,
        )
        value_input = torch.cat([prompt_hidden, state_feature, context_feature], dim=-1)
        return torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)

    def _autoregressive_action_and_value(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_feature = self.state_projector(state_tensor)
        projected_patch_embeddings = self._project_vision_features(pixel_values)
        projected_patch_embeddings = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)

        if action_bins is not None:
            action_bins = action_bins.to(self.device, dtype=torch.long)
            if action_bins.ndim != 2 or action_bins.shape[1] != self.env_action_dim:
                raise ValueError(
                    f"action_bins must have shape [batch, {self.env_action_dim}], got {tuple(action_bins.shape)}"
                )
            action_token_ids = self._action_bins_to_token_ids(action_bins)
            teacher_input_ids = torch.cat([input_ids, action_token_ids], dim=1)
            teacher_attention_mask = torch.cat([attention_mask, torch.ones_like(action_token_ids)], dim=1)
            output = self._language_model_from_prefix(
                teacher_input_ids,
                teacher_attention_mask,
                projected_patch_embeddings,
            )
            final_hidden = output.hidden_states[-1].to(torch.float32)
            num_patches = projected_patch_embeddings.shape[1]
            prompt_len = input_ids.shape[1]
            pred_start = num_patches + prompt_len - 1
            pred_end = pred_start + self.env_action_dim

            action_token_logits = output.logits[
                :,
                pred_start:pred_end,
                self.action_token_start_idx : self.action_token_end_idx,
            ].to(torch.float32)
            base_bin_logits = torch.flip(action_token_logits, dims=[-1])
            prediction_features = final_hidden[:, pred_start:pred_end, :]
            cumulative_context = final_hidden.cumsum(dim=1) / torch.arange(
                1,
                final_hidden.shape[1] + 1,
                device=final_hidden.device,
                dtype=final_hidden.dtype,
            ).view(1, -1, 1)
            context_features = self.context_projector(cumulative_context[:, pred_start:pred_end, :])
            residual_logits = self.actor_head(prediction_features, state_feature, context_features)
            logits = torch.nan_to_num(base_bin_logits + residual_logits, nan=0.0, posinf=20.0, neginf=-20.0)

            categorical = torch.distributions.Categorical(logits=logits)
            log_prob = categorical.log_prob(action_bins).sum(dim=-1)
            entropy = categorical.entropy().mean(dim=-1)
            prompt_hidden = final_hidden[:, pred_start, :]
            prompt_context = self.context_projector(cumulative_context[:, pred_start, :])
            value_input = torch.cat([prompt_hidden, state_feature, prompt_context], dim=-1)
            value = torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)
            return self.bin_indices_to_env_actions(action_bins), log_prob, entropy, value, action_bins

        generated_bins: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        value: Optional[torch.Tensor] = None
        output = self._language_model_from_prefix_with_cache(input_ids, attention_mask, projected_patch_embeddings)
        final_hidden = output.hidden_states[-1].to(torch.float32)
        hidden_sum = final_hidden.sum(dim=1)
        hidden_count = final_hidden.shape[1]
        past_key_values = output.past_key_values

        if past_key_values is None:
            prefix_ids = input_ids
            prefix_attention_mask = attention_mask

            for action_idx in range(self.env_action_dim):
                logits, prompt_hidden, context_feature = self._next_token_logits_and_value_features(
                    prefix_ids,
                    prefix_attention_mask,
                    projected_patch_embeddings,
                    state_feature,
                )
                if action_idx == 0:
                    value_input = torch.cat([prompt_hidden, state_feature, context_feature], dim=-1)
                    value = torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)

                categorical = torch.distributions.Categorical(logits=logits)
                selected_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
                log_probs.append(categorical.log_prob(selected_bin))
                entropies.append(categorical.entropy())
                generated_bins.append(selected_bin)

                next_token_id = self._action_bins_to_token_ids(selected_bin).unsqueeze(1)
                prefix_ids = torch.cat([prefix_ids, next_token_id], dim=1)
                prefix_attention_mask = torch.cat([prefix_attention_mask, torch.ones_like(next_token_id)], dim=1)
        else:
            for action_idx in range(self.env_action_dim):
                prompt_hidden = output.hidden_states[-1].to(torch.float32)[:, -1, :]
                context_feature = self.context_projector(hidden_sum / hidden_count)
                action_token_logits = output.logits[:, -1, self.action_token_start_idx : self.action_token_end_idx].to(
                    torch.float32
                )
                base_bin_logits = torch.flip(action_token_logits, dims=[-1])
                residual_logits = self.actor_head(prompt_hidden.unsqueeze(1), state_feature, context_feature).squeeze(1)
                logits = torch.nan_to_num(base_bin_logits + residual_logits, nan=0.0, posinf=20.0, neginf=-20.0)

                if action_idx == 0:
                    value_input = torch.cat([prompt_hidden, state_feature, context_feature], dim=-1)
                    value = torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)

                categorical = torch.distributions.Categorical(logits=logits)
                selected_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
                log_probs.append(categorical.log_prob(selected_bin))
                entropies.append(categorical.entropy())
                generated_bins.append(selected_bin)

                if action_idx + 1 < self.env_action_dim:
                    next_token_id = self._action_bins_to_token_ids(selected_bin).unsqueeze(1)
                    output = self._language_model_next_token_from_cache(next_token_id, output.past_key_values)
                    last_hidden = output.hidden_states[-1].to(torch.float32)[:, -1, :]
                    hidden_sum = hidden_sum + last_hidden
                    hidden_count += 1

        selected_bins = torch.stack(generated_bins, dim=1)
        log_prob = torch.stack(log_probs, dim=1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=1).mean(dim=-1)
        env_actions = self.bin_indices_to_env_actions(selected_bins)
        if value is None:
            value = self._compute_prompt_value(input_ids, attention_mask, projected_patch_embeddings, state_feature)
        return env_actions, log_prob, entropy, value, selected_bins

    def forward_policy(self, rgbs: np.ndarray, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_feature = self.state_projector(state_tensor)
        projected_patch_embeddings = self._project_vision_features(pixel_values)
        projected_patch_embeddings = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)
        logits, prompt_hidden, context_feature = self._next_token_logits_and_value_features(
            input_ids,
            attention_mask,
            projected_patch_embeddings,
            state_feature,
        )
        value_input = torch.cat([prompt_hidden, state_feature, context_feature], dim=-1)
        value = torch.nan_to_num(self.value_head(value_input).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)
        return logits.unsqueeze(1), value

    def get_value(self, rgbs: np.ndarray, states: np.ndarray) -> torch.Tensor:
        model_inputs = self._prepare_policy_inputs(rgbs)
        state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_feature = self.state_projector(state_tensor)
        projected_patch_embeddings = self._project_vision_features(model_inputs["pixel_values"])
        projected_patch_embeddings = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)
        return self._compute_prompt_value(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            projected_patch_embeddings,
            state_feature,
        )

    def get_action_and_value(
        self,
        rgbs: np.ndarray,
        states: np.ndarray,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._autoregressive_action_and_value(
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

    def predict_action(self, rgb: np.ndarray, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            action, _, _, _, _ = self.get_action_and_value(
                rgbs=np.expand_dims(rgb, axis=0),
                states=np.expand_dims(state, axis=0),
                deterministic=True,
            )
        return action[0].detach().cpu().numpy().astype(np.float32)


def build_optimizer(args: "Args", policy: HandVLAAdapterActorCritic) -> optim.Optimizer:
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
    env_id: str = "EasierRotateSingleObjectInHandLevel0-v1"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    output_dir: str = DEFAULT_WORKDIR
    num_envs: int = 128
    num_eval_envs: int = 32
    num_steps: int = 60
    total_timesteps: int = 30_000_000
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
    max_episode_steps: Optional[int] = 300
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
    action_dim: int = 16
    state_dim: int = 105


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


def unwrap_policy(policy: nn.Module) -> HandVLAAdapterActorCritic:
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
            states = extract_hand_state_batch_from_obs(obs)
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
            states = extract_hand_state_batch_from_obs(obs)
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
    policy: Optional[HandVLAAdapterActorCritic] = None,
) -> None:
    if policy is None:
        policy = HandVLAAdapterActorCritic(Path(args.model_dir), device=device, state_dim=args.state_dim, action_dim=args.action_dim)
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
        state = extract_hand_state_batch_from_obs(obs)[0]
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
    print("[hand-vla-smoke]", payload)
    save_json(output_dir / "vla_smoke.json", payload)


def maybe_load_checkpoint(
    args: Args,
    raw_policy: HandVLAAdapterActorCritic,
    optimizer: optim.Optimizer,
) -> Tuple[int, int, float]:
    if not args.resume_from:
        return 1, 0, -1.0
    checkpoint = torch.load(args.resume_from, map_location=raw_policy.device)
    policy_state = strip_module_prefix(checkpoint["policy"])
    raw_policy.load_state_dict(policy_state, strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_update = int(checkpoint.get("update", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_success = float(checkpoint.get("best_success_once", -1.0))
    return start_update, global_step, best_success


def train(args: Args) -> None:
    device, rank, world_size = init_runtime(args)
    set_seed(args.seed + rank)

    if args.num_envs % world_size != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by world_size={world_size}")
    local_num_envs = args.num_envs // world_size

    timestamp = broadcast_object(time.strftime("%Y%m%d-%H%M%S") if is_main_process() else None)
    output_dir = mkdir(Path(args.output_dir) / timestamp)
    if is_main_process():
        args_payload = asdict(args)
        args_payload.update({"world_size": world_size, "local_num_envs": local_num_envs})
        save_json(output_dir / "args.json", args_payload)
        print("[setup] loading hand VLA-Adapter policy")

    raw_policy = HandVLAAdapterActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
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

    env_actions_buf = torch.zeros((args.num_steps, local_num_envs, args.action_dim), device=device)
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
        rollout_rgbs: List[np.ndarray] = []
        rollout_states: List[np.ndarray] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            step_rgbs = extract_rgb_batch_from_obs(next_obs)
            step_states = extract_hand_state_batch_from_obs(next_obs)
            rollout_rgbs.append(step_rgbs.copy())
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
                    final_states = extract_hand_state_batch_from_obs(final_obs)[bootstrap_idx]
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
                extract_hand_state_batch_from_obs(next_obs),
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

        b_rgbs = np.concatenate(rollout_rgbs, axis=0)
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
    save_json(output_dir / "args.json", asdict(args))
    if args.mode == "vla_smoke":
        run_vla_inference_smoke(args, device, output_dir)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
