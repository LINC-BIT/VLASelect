from __future__ import annotations

import ast
import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional
import json
import os
import random
import shutil
import time

from train.common.mwe_runtime import ActiveRuntimeTracker
from train.common.mwe_checkpoint import maybe_save_model_checkpoint
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs
from train.common.checkpoint_noise import maybe_apply_checkpoint_noise_to_state_dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
import workloads.table_top
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ours.libs.train_with_fbs.lib import set_sparsity
from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn, generate_small_cnn_with_verify
from ours.utils.dl.common.model import get_module, set_module
from train.octo.model import Actor
from train.octo.metrics_json import JsonMetricsLogger, build_metric_entry
from train.common.mwe_eval import append_episode_metric_batch, summarize_episode_metric_tensors, use_train_success_only
from train.octo.ours.deft_multiple_models.online_rl import Agent, DictArray
from train.octo.ours.evolving_envs import PickCubeEnvMutable
from train.octo.world_env.draw_online_rl_acc import draw_success_curve
from train.octo.world_env.pretrain_world_model import DynamicsWorldModel, StateNormalizer
from forgetting.past_env_eval import record_past_env_snapshot


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "EuroSys2027"
    wandb_entity: Optional[str] = None
    wandb_group: str = "world_env"
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: str = "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
    world_model_checkpoint: str = ""
    render_mode: str = "all"

    env_id: str = "PickCube-v1-mutable"
    envs_id: Optional[str] = None
    env_change_time_points: Optional[str] = None
    include_state: bool = True
    normalize_states: bool = True
    env_config_path: str = "datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json"
    state_norm_stats_path: str = "ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth"

    total_timesteps: int = 100000000
    max_time: Optional[float] = None
    learning_rate: float = 2e-5
    num_envs: int = 128
    num_eval_envs: int = 32
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 16
    update_epochs: int = 2
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    reward_scale: float = 1.0
    eval_freq: int = 1
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    actor_logstd: float = -0.5
    max_sparsity: float = 0.8
    small_model_generation_strategy: str = "source"

    real_reward_weight: float = 0.3
    verified_reward_weight: float = 0.7
    wm_reward_weight: float = 0.45
    wm_success_weight: float = 0.4
    wm_reference_weight: float = 0.15
    wm_state_temperature: float = 0.25
    wm_reward_clip: float = 2.0
    success_termination_threshold: float = 0.6
    success_bonus: float = 0.2

    curve_output: Optional[str] = None
    curve_tag: str = "eval/success_at_end"
    curve_title: str = "World-Env Success Curve"
    tag: Optional[str] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _ensure_2d_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    return value


def _slice_xyz(value):
    value = _ensure_2d_tensor(value)
    return value[:, :3]


def _resolve_extra_tensor(
    extra,
    batch_size,
    key,
    *,
    aliases=(),
    default_dim,
    device=None,
    derive=None,
):
    candidate_keys = (key, *aliases)
    for candidate_key in candidate_keys:
        if candidate_key in extra:
            value = _ensure_2d_tensor(extra[candidate_key])
            if candidate_key == "is_grasped":
                value = value.to(torch.float32)
            if value.shape[1] > default_dim:
                value = value[:, :default_dim]
            elif value.shape[1] < default_dim:
                pad = torch.zeros(
                    (value.shape[0], default_dim - value.shape[1]),
                    dtype=value.dtype,
                    device=value.device,
                )
                value = torch.cat([value, pad], dim=1)
            return value
    if derive is not None:
        derived_value = derive(extra)
        if derived_value is None:
            return torch.zeros((batch_size, default_dim), dtype=torch.float32, device=device)
        value = _ensure_2d_tensor(derived_value)
        if value.shape[1] > default_dim:
            value = value[:, :default_dim]
        elif value.shape[1] < default_dim:
            pad = torch.zeros(
                (value.shape[0], default_dim - value.shape[1]),
                dtype=value.dtype,
                device=value.device,
            )
            value = torch.cat([value, pad], dim=1)
        return value
    return torch.zeros((batch_size, default_dim), dtype=torch.float32, device=device)


class Logger:
    def __init__(self, writer: Optional[SummaryWriter], log_wandb: bool = False):
        self.writer = writer
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb

            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


class FlattenRGBDObservationWrapper2(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.sep_depth = sep_depth

        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        observation = dict(observation)
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images, depth_images = [], []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)

        agent = observation["agent"]
        extra = observation["extra"]
        state_parts = []
        for key in ["qpos", "qvel"]:
            state_parts.append(_ensure_2d_tensor(agent[key]).to(torch.float32))

        batch_size = state_parts[0].shape[0]
        device = state_parts[0].device
        tcp_pose = _resolve_extra_tensor(
            extra,
            batch_size,
            "tcp_pose",
            default_dim=7,
            device=device,
        )
        goal_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "goal_pos",
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(obs_extra["cubeB_pose"])
            if "cubeB_pose" in obs_extra
            else None,
        )
        obj_pose = _resolve_extra_tensor(
            extra,
            batch_size,
            "obj_pose",
            aliases=("cube_pose", "ball_pose", "cubeA_pose"),
            default_dim=7,
            device=device,
        )
        tcp_to_obj_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "tcp_to_obj_pos",
            aliases=("tcp_to_ball_pos", "tcp_to_cubeA_pos", "peghead_to_cube_pos", "tcp_to_peg_pos"),
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(obj_pose) - _slice_xyz(tcp_pose),
        )
        obj_to_goal_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "obj_to_goal_pos",
            aliases=("ball_to_goal_pos", "cubeA_to_cubeB_pos", "cube_to_goal_pos"),
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(goal_pos) - _slice_xyz(obj_pose),
        )
        is_grasped = _resolve_extra_tensor(
            extra,
            batch_size,
            "is_grasped",
            default_dim=1,
            device=device,
        )

        state_parts.extend(
            [
                is_grasped.to(torch.float32),
                tcp_pose.to(torch.float32),
                goal_pos.to(torch.float32),
                obj_pose.to(torch.float32),
                tcp_to_obj_pos.to(torch.float32),
                obj_to_goal_pos.to(torch.float32),
            ]
        )
        state_tensor = torch.cat(state_parts, dim=1)

        ret = {}
        if self.include_state:
            ret["state"] = state_tensor
        if self.include_rgb and self.include_depth:
            if self.sep_depth:
                ret["rgb"] = rgb_images
                ret["depth"] = depth_images
            else:
                ret["rgbd"] = torch.concat([rgb_images, depth_images], axis=-1)
        elif self.include_rgb:
            ret["rgb"] = rgb_images
        elif self.include_depth:
            ret["depth"] = depth_images
        return ret


@dataclass
class ContinualEnvSchedule:
    env_ids: List[str]
    change_time_points: List[float]


def _parse_cli_sequence(raw_value, arg_name, cast_fn):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raw_text = str(raw_value).strip()
        if raw_text == "":
            return []
        try:
            parsed_value = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed_value = [item.strip() for item in raw_text.split(",") if item.strip()]
        if isinstance(parsed_value, (list, tuple)):
            values = list(parsed_value)
        else:
            values = [parsed_value]
    try:
        return [cast_fn(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse `{arg_name}` from {raw_value!r}") from exc


def build_continual_env_schedule(args: Args) -> Optional[ContinualEnvSchedule]:
    env_ids = _parse_cli_sequence(args.envs_id, "envs_id", str)
    time_points = _parse_cli_sequence(args.env_change_time_points, "env_change_time_points", float)
    if env_ids is None and time_points is None:
        return None
    if env_ids is None or time_points is None:
        raise ValueError("`envs_id` and `env_change_time_points` must be provided together")
    if len(env_ids) == 0:
        raise ValueError("`envs_id` must contain at least one environment")
    if len(env_ids) != len(time_points):
        raise ValueError(
            f"`envs_id` and `env_change_time_points` must have the same length, "
            f"got {len(env_ids)} and {len(time_points)}"
        )
    last_time_point = None
    for time_point in time_points:
        if time_point <= 0:
            raise ValueError("All `env_change_time_points` must be positive")
        if last_time_point is not None and time_point <= last_time_point:
            raise ValueError("`env_change_time_points` must be strictly increasing")
        last_time_point = time_point
    return ContinualEnvSchedule(env_ids=env_ids, change_time_points=time_points)


def _sanitize_env_name_for_path(env_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", env_id)


def collect_sample_for_small_model_generation_source(args: Args, env_kwargs: dict, device: torch.device):
    source_eval_envs = gym.make(
        "PickCube-v1",
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **dict(env_kwargs),
    )
    try:
        source_eval_envs = FlattenRGBDObservationWrapper2(
            source_eval_envs,
            rgb=True,
            depth=True,
            state=args.include_state,
        )
        if isinstance(source_eval_envs.action_space, gym.spaces.Dict):
            source_eval_envs = FlattenActionSpaceWrapper(source_eval_envs)
        source_eval_envs = ManiSkillVectorEnv(
            source_eval_envs,
            args.num_eval_envs,
            ignore_terminations=not args.eval_partial_reset,
            record_metrics=True,
        )
        source_eval_obs, _ = source_eval_envs.reset()
        print("use source domain data for small model generation")
        return {
            "rgb": source_eval_obs["rgb"].to(device)[0:1],
            "depth": source_eval_obs["depth"].to(device)[0:1],
            "state": source_eval_obs["state"].to(device)[0:1],
        }
    finally:
        source_eval_envs.close()


class WorldEnvReflector:
    def __init__(self, ckpt_path: str, device: torch.device):
        if not ckpt_path:
            raise ValueError("--world-model-checkpoint is required for World-Env online training")
        payload = torch.load(ckpt_path, map_location="cpu")
        config = payload["model_config"]
        self.model = DynamicsWorldModel(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            latent_dim=config["latent_dim"],
        ).to(device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.device = device
        self.normalizer = StateNormalizer(payload["state_max"], payload["state_min"])
        reference_bank = payload.get("reference_bank")
        if reference_bank is None:
            raise RuntimeError("World model checkpoint does not contain reference_bank.")
        self.reference_latent = F.normalize(reference_bank["latent"].float().to(device), dim=1)
        self.reference_state = reference_bank["state"].float().to(device)

    @torch.no_grad()
    def compute_reward(self, obs, action, args: Args):
        model_out = self.model.imagine_next(
            {"rgb": obs["rgb"], "depth": obs["depth"], "state": obs["state"]},
            action,
            self.normalizer,
        )
        pred_latent = F.normalize(model_out["pred_next_latent"], dim=1)
        pred_state = model_out["pred_next_state"]
        pred_reward = torch.tanh(model_out["pred_reward"])
        pred_success = torch.sigmoid(model_out["pred_success_logit"])

        similarity = pred_latent @ self.reference_latent.t()
        max_similarity = similarity.max(dim=1).values

        distances = torch.cdist(pred_state, self.reference_state)
        nearest_distance = distances.min(dim=1).values
        state_score = torch.exp(-nearest_distance / max(args.wm_state_temperature, 1e-6))

        reference_score = 0.5 * (max_similarity + 1.0)
        reference_score = 0.5 * (reference_score + state_score)

        termination_bonus = (pred_success > args.success_termination_threshold).float() * args.success_bonus
        verified_reward = (
            args.wm_reward_weight * pred_reward
            + args.wm_success_weight * pred_success
            + args.wm_reference_weight * reference_score
            + termination_bonus
        )
        verified_reward = torch.clamp(verified_reward, 0.0, args.wm_reward_clip)
        return verified_reward.view(-1), {
            "reward_pred": pred_reward.mean().item(),
            "success_pred": pred_success.mean().item(),
            "reference_score": reference_score.mean().item(),
            "state_score": state_score.mean().item(),
            "termination_bonus": termination_bonus.mean().item(),
            "virtual_done_rate": (pred_success > args.success_termination_threshold).float().mean().item(),
        }


def build_agent_from_checkpoint(args: Args, device: torch.device, env_kwargs: dict):
    if args.small_model_generation_strategy != "source":
        raise ValueError("World-Env baseline fixes small_model_generation_strategy to 'source'")

    actor = Actor(42, 4, 1, False).to(device=device)
    set_module(actor, "rgb_encoder.fc.0", svd_decompose_linear(get_module(actor, "rgb_encoder.fc.0")))
    set_module(actor, "depth_encoder.fc.0", svd_decompose_linear(get_module(actor, "depth_encoder.fc.0")))

    actor_example = {
        "rgb": torch.rand((1, 3, 128, 128), device=device),
        "depth": torch.rand((1, 1, 128, 128), device=device),
        "state": torch.rand((1, 42), device=device),
    }
    add_FBS_into_cnn(
        actor,
        [f"rgb_encoder.cnn.{i}" for i in [0, 6, 12]] + [f"depth_encoder.cnn.{i}" for i in [0, 6, 12]],
        ["decoder.0", "rgb_encoder.fc.0.0", "depth_encoder.fc.0.0"],
        actor_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample["rgb"], sample["depth"], sample["state"]),
    )

    state_max, state_min = torch.load(args.state_norm_stats_path, map_location="cpu")
    state_max = state_max.to(device)
    state_min = state_min.to(device)
    agent = Agent(actor, 256 * 3, state_max, state_min, args.normalize_states, args.actor_logstd).to(device)
    actor.decoder = nn.Identity()

    agent_example = {
        "rgb": torch.rand((1, 128, 128, 3)),
        "depth": torch.rand((1, 128, 128, 1)),
        "state": torch.rand((1, 42)),
    }
    add_FBS_into_cnn(
        agent,
        [],
        ["actor_mean.0", "critic.0"],
        agent_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample),
    )

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["agent"] = maybe_apply_checkpoint_noise_to_state_dict(
            checkpoint["agent"],
            checkpoint_path=checkpoint_path,
            state_label="agent",
        )
        print(agent.load_state_dict(checkpoint["agent"], strict=True))
    else:
        print(f"checkpoint not found at {checkpoint_path}; keep current initialization")
    for module in agent.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    set_sparsity(agent, args.max_sparsity)
    sample = collect_sample_for_small_model_generation_source(args, env_kwargs, device)
    print(f"compress loaded model into source-generated small model (max_sparsity={args.max_sparsity})")
    small_agent = generate_small_cnn_with_verify(
        agent,
        args.max_sparsity,
        sample,
        lambda model, sample_input: model(sample_input),
    )
    for module in small_agent.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    del agent
    return small_agent.to(device)


def load_env_kwargs(args: Args):
    with open(args.env_config_path, "r") as f:
        demo_info = json.load(f)
        env_kwargs = demo_info["env_info"]["env_kwargs"]

    env_kwargs["sim_backend"] = "physx_cuda"
    del env_kwargs["num_envs"]
    del env_kwargs["reward_mode"]
    return env_kwargs


def make_envs_for_env_id(args: Args, env_id: str, env_kwargs: dict, run_name: str, env_index: int):
    print(f"making gym for env[{env_index}]={env_id}...")
    eval_envs = gym.make(
        env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **dict(env_kwargs),
    )
    envs = gym.make(
        env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **dict(env_kwargs),
    )

    envs = FlattenRGBDObservationWrapper2(envs, rgb=True, depth=True, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper2(eval_envs, rgb=True, depth=True, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        env_dir_name = f"env{env_index:02d}-{_sanitize_env_name_for_path(env_id)}"
        eval_output_dir = f"ckpt/{run_name}/videos/{env_dir_name}"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos/{env_dir_name}"
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"ckpt/{run_name}/train_videos/{env_dir_name}",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs if not args.evaluate else 1,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    return envs, eval_envs


@torch.no_grad()
def evaluate(agent, eval_envs, args: Args, logger: Optional[Logger], global_step: int, train_episode_metrics=None):
    if use_train_success_only():
        mean_metrics = summarize_episode_metric_tensors(train_episode_metrics or {})
        if "success_at_end" in mean_metrics:
            print(
                "[eval] "
                f"global_step={global_step} "
                f"success_at_end={mean_metrics['success_at_end']:.4f}"
            )
        return mean_metrics
    set_sparsity(agent, args.max_sparsity)
    agent.eval()
    metrics = defaultdict(list)
    eval_obs, _ = eval_envs.reset()
    for _ in range(args.num_eval_steps):
        eval_obs, _, _, _, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
        if "final_info" not in eval_infos:
            continue
        done_mask = eval_infos["_final_info"]
        for key, value in eval_infos["final_info"]["episode"].items():
            metrics[key].append(value[done_mask].float().detach().cpu())

    mean_metrics = {}
    for key, values in metrics.items():
        if len(values) == 0:
            continue
        mean_value = torch.cat(values).mean().item()
        mean_metrics[key] = mean_value
        if logger is not None:
            logger.add_scalar(f"eval/{key}", mean_value, global_step)
    if "success_at_end" in mean_metrics:
        print(
            "[eval] "
            f"global_step={global_step} "
            f"success_at_end={mean_metrics['success_at_end']:.4f}"
        )
    return mean_metrics


def save_checkpoint(run_name, filename, agent, optimizer, iteration, extra_metrics=None):
    os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
    payload = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    maybe_save_model_checkpoint(payload, f"ckpt/{run_name}/checkpoints/{filename}")


def copy_run_metadata(run_name: str, args: Args):
    os.makedirs(f"ckpt/{run_name}/code", exist_ok=True)
    shutil.copyfile(__file__, f"ckpt/{run_name}/code/script.py")
    with open(f"ckpt/{run_name}/code/args.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def maybe_draw_success_curve(run_name: str, args: Args, logger: Optional[Logger]):
    if logger is not None:
        logger.flush()
    run_dir = Path("ckpt") / run_name
    output_path = run_dir / "success_curve.png" if args.curve_output is None else Path(args.curve_output)
    try:
        draw_success_curve(
            run_dir=run_dir,
            output=output_path,
            tag=args.curve_tag,
            title=args.curve_title,
        )
    except Exception as exc:
        print(f"warning: failed to update success curve at {output_path}: {exc}")
    return output_path


def main():
    args = tyro.cli(Args)
    if args.batch_size == 0:
        args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("batch_size must be divisible by num_minibatches")
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.exp_name is None:
        run_name = f"{args.env_id}/baselines/world_env/{Path(__file__).stem}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.tag is not None:
            run_name += f"-{args.tag}"
    else:
        run_name = args.exp_name

    copy_run_metadata(run_name, args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device: {device}")

    env_kwargs = load_env_kwargs(args)
    continual_env_schedule = build_continual_env_schedule(args)
    current_env_index = 0
    current_env_id = args.env_id if continual_env_schedule is None else continual_env_schedule.env_ids[0]
    envs, eval_envs = make_envs_for_env_id(
        args,
        current_env_id,
        env_kwargs,
        run_name,
        current_env_index,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    logger = None
    if not args.evaluate:
        if args.track:
            import wandb

            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=current_env_id, env_horizon=max_episode_steps)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=current_env_id, env_horizon=max_episode_steps)
            if continual_env_schedule is not None:
                config["continual_env_ids"] = list(continual_env_schedule.env_ids)
                config["continual_env_change_time_points"] = list(continual_env_schedule.change_time_points)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=config,
                name=run_name,
                save_code=True,
                group=f"{args.wandb_group}/{current_env_id}",
            )
        writer = SummaryWriter(f"ckpt/{run_name}/tb")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(writer=writer, log_wandb=args.track)

    agent = build_agent_from_checkpoint(args, device, env_kwargs)
    reflector = WorldEnvReflector(args.world_model_checkpoint, device)
    optimizer = optim.Adam([parameter for parameter in agent.parameters() if parameter.requires_grad], lr=args.learning_rate, eps=1e-5)

    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    env_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    verified_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    best_success_once = 0.0
    best_success_end = 0.0
    start_time = time.time()
    training_start_time = time.monotonic()
    runtime_tracker = ActiveRuntimeTracker.from_env(wall_clock_start_time=training_start_time)
    output_dir = Path(f"ckpt/{run_name}")
    json_metrics = JsonMetricsLogger(output_dir)

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    initial_metrics = evaluate(agent, eval_envs, args, logger, global_step)
    curve_output_path = maybe_draw_success_curve(run_name, args, logger)
    last_eval_metrics = dict(initial_metrics)
    json_metrics.append(
        build_metric_entry(
            update=0,
            global_step=global_step,
            current_env_id=current_env_id,
            current_env_index=current_env_index,
            elapsed_minutes=0.0,
            eval_metrics=initial_metrics,
        )
    )
    print(f"initial eval: {initial_metrics}")
    print(f"success curve path: {curve_output_path}")
    if continual_env_schedule is not None:
        print(
            f"continual env schedule enabled: envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )

    if args.evaluate:
        json_metrics.save_final_eval(last_eval_metrics)
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        if logger is not None:
            logger.close()
        return

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_id, current_env_index
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = runtime_tracker.current_minutes()
        scheduled_env_index = bisect.bisect_right(
            continual_env_schedule.change_time_points,
            elapsed_minutes,
        )
        if scheduled_env_index >= len(continual_env_schedule.env_ids):
            record_past_env_snapshot(
                agent=agent, args=args, env_ids=continual_env_schedule.env_ids,
                completed_env_index=current_env_index, elapsed_minutes=elapsed_minutes,
                global_step=global_step, update=last_iteration, json_metrics=json_metrics,
                make_env_pair=lambda env_id, env_index: make_envs_for_env_id(
                    args, env_id, env_kwargs, run_name, env_index
                ),
            )
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes

        previous_env_id = current_env_id
        record_past_env_snapshot(
            agent=agent, args=args, env_ids=continual_env_schedule.env_ids,
            completed_env_index=current_env_index, elapsed_minutes=elapsed_minutes,
            global_step=global_step, update=last_iteration, json_metrics=json_metrics,
            make_env_pair=lambda env_id, env_index: make_envs_for_env_id(
                args, env_id, env_kwargs, run_name, env_index
            ),
        )
        current_env_index = scheduled_env_index
        current_env_id = continual_env_schedule.env_ids[current_env_index]
        print(
            f"switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        envs, eval_envs = make_envs_for_env_id(
            args,
            current_env_id,
            env_kwargs,
            run_name,
            current_env_index,
        )
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        next_done = torch.zeros(args.num_envs, device=device)
        return True, False, elapsed_minutes

    last_iteration = 0
    for iteration in range(1, args.num_iterations + 1):
        last_iteration = iteration
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/current_env_index", current_env_index, global_step)
        if should_stop_for_schedule:
            print(f"reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes, stopping training.")
            break
        if args.max_time is not None and elapsed_minutes is not None and elapsed_minutes >= args.max_time:
            print(f"Reached max_time={args.max_time} minutes, stopping.")
            break
        if args.max_time is not None and elapsed_minutes is None:
            elapsed_minutes = runtime_tracker.current_minutes()
            if elapsed_minutes >= args.max_time:
                print(f"Reached max_time={args.max_time} minutes, stopping.")
                break

        agent.train()
        set_sparsity(agent, args.max_sparsity)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lr_now

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        train_episode_metrics = defaultdict(list)
        rollout_start = time.perf_counter()
        reward_info_last = {}
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                verified_reward, reward_info_last = reflector.compute_reward(next_obs, action, args)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).float()
            mixed_reward = args.real_reward_weight * reward.view(-1) + args.verified_reward_weight * verified_reward
            rewards[step] = mixed_reward * args.reward_scale
            env_rewards[step] = reward.view(-1)
            verified_rewards[step] = verified_reward

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                append_episode_metric_batch(train_episode_metrics, infos["final_info"]["episode"], done_mask)
                done_indices = torch.arange(args.num_envs, device=device)[done_mask]
                if logger is not None:
                    for key, value in infos["final_info"]["episode"].items():
                        logger.add_scalar(f"train/{key}", value[done_mask].float().mean().item(), global_step)

                for key in infos["final_observation"]:
                    infos["final_observation"][key] = infos["final_observation"][key][done_mask]
                with torch.no_grad():
                    final_values[step, done_indices] = agent.get_value(infos["final_observation"]).view(-1)

        rollout_time = time.perf_counter() - rollout_start
        runtime_tracker.add_active_seconds(rollout_time)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    next_values = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                real_next_values = next_not_done * next_values + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_start = time.perf_counter()
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            early_stop = False
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    early_stop = True
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if early_stop:
                break

        update_time = time.perf_counter() - update_start
        runtime_tracker.add_active_seconds(update_time)
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/current_env_index", current_env_index, global_step)
        if should_stop_for_schedule:
            print(
                f"reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes "
                f"after rollout, stopping before evaluation."
            )
            break
        should_run_eval = iteration % args.eval_freq == 0 or iteration == args.num_iterations
        if not should_run_eval:
            continue
        metrics = evaluate(agent, eval_envs, args, logger, global_step, train_episode_metrics=train_episode_metrics)
        curve_output_path = maybe_draw_success_curve(run_name, args, logger)
        success_once = metrics.get("success_once")
        success_end = metrics.get("success_at_end")
        success_once_display = float(success_once) if success_once is not None else float("nan")
        success_end_display = float(success_end) if success_end is not None else float("nan")
        last_eval_metrics = dict(metrics)
        json_metrics.append(
            build_metric_entry(
                update=iteration,
                global_step=global_step,
                current_env_id=current_env_id,
                current_env_index=current_env_index,
                elapsed_minutes=runtime_tracker.current_minutes(),
                eval_metrics=metrics,
                extras={
                    "best_success_once": best_success_once,
                    "best_success_at_end": best_success_end,
                    "rollout_time": rollout_time,
                    "update_time": update_time,
                    "env_reward_mean": env_rewards.mean().item(),
                    "verified_reward_mean": verified_rewards.mean().item(),
                },
            )
        )
        print(
            f"iter={iteration} success_once={success_once_display:.4f} success_end={success_end_display:.4f} "
            f"env_reward={env_rewards.mean().item():.4f} verified_reward={verified_rewards.mean().item():.4f} "
            f"curve={curve_output_path}"
        )

        if logger is not None:
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("reward/env_mean", env_rewards.mean().item(), global_step)
            logger.add_scalar("reward/verified_mean", verified_rewards.mean().item(), global_step)
            logger.add_scalar("reward/mixed_mean", rewards.mean().item(), global_step)
            for key, value in reward_info_last.items():
                logger.add_scalar(f"world_env/{key}", value, global_step)

        if success_once is not None and success_once >= best_success_once:
            best_success_once = success_once
            save_checkpoint(
                run_name,
                "best_success_once.pt",
                agent,
                optimizer,
                iteration,
                {k: v for k, v in {"success_once": best_success_once, "success_at_end": success_end}.items() if v is not None},
            )
        if success_end is not None and success_end >= best_success_end:
            best_success_end = success_end
            save_checkpoint(
                run_name,
                "best_success_end.pt",
                agent,
                optimizer,
                iteration,
                {k: v for k, v in {"success_once": success_once, "success_at_end": best_success_end}.items() if v is not None},
            )

        if args.save_model and iteration % args.eval_freq == 0:
            save_checkpoint(
                run_name,
                "last.pt",
                agent,
                optimizer,
                iteration,
                {k: v for k, v in {"success_once": success_once, "success_at_end": success_end}.items() if v is not None},
            )

    if args.save_model:
        save_checkpoint(run_name, "last.pt", agent, optimizer, last_iteration, None)

    json_metrics.save_final_eval(last_eval_metrics)
    close_envs(envs, eval_envs)
    envs = None
    eval_envs = None
    clear_torch_cuda_cache()
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
