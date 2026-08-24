import ast
import bisect
from collections import defaultdict
from copy import deepcopy
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
from train.common.checkpoint_noise import maybe_apply_checkpoint_noise_to_state_dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
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
from train.octo.ours.evolving_envs import PickCubeEnvMutable


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "EuroSys2026"
    wandb_entity: Optional[str] = None
    wandb_group: str = "self_improv"
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: str = "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
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
    learning_rate: float = 3e-5
    num_envs: int = 256
    num_eval_envs: int = 32
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.9
    num_minibatches: int = 16
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    reward_scale: float = 1.0
    eval_freq: int = 1
    save_train_video_freq: Optional[int] = None

    actor_logstd: float = -0.5
    max_sparsity: float = 0.8

    reinforce_coef: float = 5e-2
    normalize_returns: bool = True
    freeze_feature_net: bool = False
    success_threshold_quantile: float = 0.25
    success_calibration_episodes: int = 64

    small_model_generation_strategy: str = "source"
    tag: Optional[str] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@dataclass
class ContinualEnvSchedule:
    env_ids: List[str]
    change_time_points: List[float]


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


class DictArray:
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict is not None:
            self.data = data_dict
            return

        assert isinstance(element_space, gym.spaces.dict.Dict)
        self.data = {}
        for key, value in element_space.items():
            if isinstance(value, gym.spaces.dict.Dict):
                self.data[key] = DictArray(buffer_shape, value, device=device)
            else:
                dtype = (
                    torch.float32 if value.dtype in (np.float32, np.float64)
                    else torch.uint8 if value.dtype == np.uint8
                    else torch.int16 if value.dtype == np.int16
                    else torch.int32 if value.dtype == np.int32
                    else value.dtype
                )
                self.data[key] = torch.zeros(buffer_shape + value.shape, dtype=dtype, device=device)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {key: value[index] for key, value in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        for key, item in value.items():
            self.data[key][index] = item

    def reshape(self, shape):
        prefix_len = len(self.buffer_shape)
        new_dict = {}
        for key, value in self.data.items():
            if isinstance(value, DictArray):
                new_dict[key] = value.reshape(shape)
            else:
                new_dict[key] = value.reshape(shape + value.shape[prefix_len:])
        return DictArray(next(iter(new_dict.values())).shape[: len(shape)], None, data_dict=new_dict)


class Agent(nn.Module):
    def __init__(self, feature_net, latent_size, state_max=None, state_min=None, normalize_states=True, actor_logstd=-0.5):
        super().__init__()
        self.feature_net = feature_net
        self.normalize_states = normalize_states
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 4), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 4) * actor_logstd)
        self.state_max = state_max
        self.state_min = state_min

    def preprocess(self, obs):
        device = self.actor_logstd.device
        if isinstance(obs["rgb"], np.ndarray):
            rgb = torch.from_numpy(obs["rgb"])
            depth = torch.from_numpy(obs["depth"])
            state = torch.from_numpy(obs["state"])
        else:
            rgb = obs["rgb"]
            depth = obs["depth"]
            state = obs["state"]

        rgb = rgb.permute(0, 3, 1, 2)[:, 0:3].float() / 255.0
        depth = depth.permute(0, 3, 1, 2)[:, 0:1].float() / 1024.0
        rgb = F.interpolate(rgb, size=128, mode="bilinear", align_corners=False).to(device)
        depth = F.interpolate(depth, size=128, mode="bilinear", align_corners=False).to(device)
        state = state.to(device)

        if self.normalize_states:
            state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return {"rgb": rgb, "depth": depth, "state": state}

    def get_features(self, obs):
        return self.feature_net(self.preprocess(obs))

    def get_value(self, obs):
        return self.critic(self.get_features(obs))

    def get_action(self, obs, deterministic=False):
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, action=None, return_action_mean=False):
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self.critic(features)
        if return_action_mean:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def forward(self, obs):
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return torch.cat([action_mean, value], dim=1).sum()


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


def load_env_kwargs(args: Args):
    with open(args.env_config_path, "r") as f:
        demo_info = json.load(f)
        env_kwargs = demo_info["env_info"]["env_kwargs"]

    env_kwargs["sim_backend"] = "physx_cuda"
    del env_kwargs["num_envs"]
    del env_kwargs["reward_mode"]
    return env_kwargs


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


def build_small_agent(args: Args, device: torch.device, env_kwargs: dict):
    if args.small_model_generation_strategy != "source":
        raise ValueError("self_improv baseline fixes small_model_generation_strategy to 'source'")

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
    sample_for_gen_small_model = collect_sample_for_small_model_generation_source(args, env_kwargs, device)
    print(f"compress loaded model into small model (max_sparsity={args.max_sparsity})")
    small_agent = generate_small_cnn_with_verify(
        agent,
        args.max_sparsity,
        sample_for_gen_small_model,
        lambda model, sample: model(sample),
    )
    for module in small_agent.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    del agent
    return small_agent.to(device)


def make_envs_for_env_id(args: Args, env_id: str, env_kwargs: dict, run_name: str, env_index: int, evaluate_only: bool):
    print(f"making gym for env[{env_index}]={env_id}...")
    eval_envs = gym.make(
        env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    envs = gym.make(
        env_id,
        num_envs=args.num_envs if not evaluate_only else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    print("gym made!")

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
        args.num_envs if not evaluate_only else 1,
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


def make_envs(args: Args, env_kwargs: dict, run_name: str, evaluate_only: bool, continual_env_schedule: Optional[ContinualEnvSchedule]):
    initial_env_id = continual_env_schedule.env_ids[0] if continual_env_schedule is not None else args.env_id
    envs, eval_envs = make_envs_for_env_id(
        args,
        initial_env_id,
        env_kwargs,
        run_name,
        env_index=0,
        evaluate_only=evaluate_only,
    )
    return envs, eval_envs, initial_env_id


def freeze_reward_model(agent: Agent):
    agent.eval()
    for parameter in agent.parameters():
        parameter.requires_grad = False


def configure_trainable_policy(agent: Agent, args: Args):
    for parameter in agent.feature_net.parameters():
        parameter.requires_grad = not args.freeze_feature_net
    for parameter in agent.actor_mean.parameters():
        parameter.requires_grad = True
    agent.actor_logstd.requires_grad_(True)
    for parameter in agent.critic.parameters():
        parameter.requires_grad = False
    trainable_parameters = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters left for self-improvement policy updates")
    return trainable_parameters


def clone_obs(obs):
    return {key: value.clone() for key, value in obs.items()}


def merge_next_obs_with_final_obs(next_obs, final_obs, done_mask):
    reward_obs = clone_obs(next_obs)
    if final_obs is None or done_mask is None or not done_mask.any():
        return reward_obs
    for key in reward_obs:
        reward_obs[key][done_mask] = final_obs[key]
    return reward_obs


def filter_final_observation(infos):
    if "final_info" not in infos:
        return None, None, None
    done_mask = infos["_final_info"]
    final_obs = {}
    for key, value in infos["final_observation"].items():
        final_obs[key] = value[done_mask]
    return done_mask, final_obs, infos["final_info"]["episode"]


def compute_progress_score(agent: Agent, obs):
    return agent.get_value(obs).view(-1)


def calibrate_success_threshold(reward_agent: Agent, eval_envs, args: Args):
    reward_agent.eval()
    success_scores = []
    all_scores = []
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    max_steps = max(args.num_eval_steps * max(2, args.success_calibration_episodes // max(1, args.num_eval_envs)), args.num_eval_steps)
    with torch.no_grad():
        for _ in range(max_steps):
            eval_obs, _, _, _, eval_infos = eval_envs.step(reward_agent.get_action(eval_obs, deterministic=True))
            done_mask, final_obs, final_episode = filter_final_observation(eval_infos)
            if done_mask is None:
                continue
            final_scores = compute_progress_score(reward_agent, final_obs).detach().cpu()
            all_scores.extend(final_scores.tolist())
            success_once = final_episode["success_once"][done_mask].float().detach().cpu()
            if success_once.numel() > 0:
                success_scores.extend(final_scores[success_once > 0].tolist())
            if len(all_scores) >= args.success_calibration_episodes:
                break

    if success_scores:
        threshold = float(np.quantile(np.array(success_scores), args.success_threshold_quantile))
        source = "successful terminal states"
    elif all_scores:
        threshold = float(np.quantile(np.array(all_scores), 0.9))
        source = "all terminal states fallback"
    else:
        threshold = 0.0
        source = "empty fallback"
    print(f"calibrated success threshold={threshold:.6f} from {source}")
    return threshold


def evaluate(agent, eval_envs, args: Args, logger: Optional[Logger], global_step: int, reward_agent: Optional[Agent] = None, success_threshold: Optional[float] = None):
    set_sparsity(agent, args.max_sparsity)
    agent.eval()
    metrics = defaultdict(list)
    eval_obs, _ = eval_envs.reset()
    with torch.no_grad():
        for _ in range(args.num_eval_steps):
            eval_obs, _, _, _, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
            done_mask, final_obs, final_episode = filter_final_observation(eval_infos)
            if done_mask is None:
                continue
            for key, value in final_episode.items():
                metrics[key].append(value[done_mask].float().detach().cpu())
            if reward_agent is not None and success_threshold is not None:
                predicted_success = (compute_progress_score(reward_agent, final_obs) >= success_threshold).float().detach().cpu()
                metrics["predicted_success_at_end"].append(predicted_success)

    mean_metrics = {}
    for key, values in metrics.items():
        if len(values) == 0:
            continue
        mean_value = torch.cat(values).mean().item()
        mean_metrics[key] = mean_value
        if logger is not None:
            logger.add_scalar(f"eval/{key}", mean_value, global_step)
    return mean_metrics


def save_checkpoint(run_name, filename, agent, optimizer, iteration, success_threshold, extra_metrics=None):
    os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
    payload = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "success_threshold": success_threshold,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    torch.save(payload, f"ckpt/{run_name}/checkpoints/{filename}")


def copy_run_metadata(run_name: str, args: Args):
    os.makedirs(f"ckpt/{run_name}/code", exist_ok=True)
    shutil.copyfile(__file__, f"ckpt/{run_name}/code/script.py")
    with open(f"ckpt/{run_name}/code/args.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


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

    continual_env_schedule = build_continual_env_schedule(args)

    if args.exp_name is None:
        args.exp_name = Path(__file__).stem
        run_env_id = continual_env_schedule.env_ids[0] if continual_env_schedule is not None else args.env_id
        run_name = f"{run_env_id}/baselines/self_improv/{args.exp_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.tag is not None:
            run_name += f"-{args.tag}"
    else:
        run_name = args.exp_name

    copy_run_metadata(run_name, args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device: {device}")
    env_kwargs = load_env_kwargs(args)
    envs, eval_envs, current_env_id = make_envs(args, env_kwargs, run_name, args.evaluate, continual_env_schedule)
    current_env_index = 0
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
                group=f"{args.wandb_group}/{args.env_id}",
            )
        writer = SummaryWriter(f"ckpt/{run_name}/tb")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(writer=writer, log_wandb=args.track)

    agent = build_small_agent(args, device, env_kwargs)
    reward_agent = deepcopy(agent).to(device)
    freeze_reward_model(reward_agent)
    success_threshold = calibrate_success_threshold(reward_agent, eval_envs, args)

    trainable_parameters = configure_trainable_policy(agent, args)
    optimizer = optim.Adam(trainable_parameters, lr=args.learning_rate, eps=1e-5)

    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    best_success_once = 0.0
    best_success_end = 0.0
    start_time = time.time()
    training_start_time = time.monotonic()
    runtime_tracker = ActiveRuntimeTracker.from_env(wall_clock_start_time=training_start_time)
    output_dir = Path(f"ckpt/{run_name}")
    json_metrics = JsonMetricsLogger(output_dir)

    next_obs, _ = envs.reset(seed=args.seed)

    if continual_env_schedule is not None:
        print(
            f"Continual env schedule enabled: envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, current_env_id, current_env_index, success_threshold
        if continual_env_schedule is None:
            return False, False, None

        elapsed_minutes = runtime_tracker.current_minutes()
        scheduled_env_index = bisect.bisect_right(
            continual_env_schedule.change_time_points,
            elapsed_minutes,
        )
        if scheduled_env_index >= len(continual_env_schedule.env_ids):
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes

        previous_env_id = current_env_id
        current_env_index = scheduled_env_index
        current_env_id = continual_env_schedule.env_ids[current_env_index]
        print(
            f"Switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        envs.close()
        eval_envs.close()
        envs, eval_envs = make_envs_for_env_id(
            args,
            current_env_id,
            env_kwargs,
            run_name,
            env_index=current_env_index,
            evaluate_only=args.evaluate,
        )
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        success_threshold = calibrate_success_threshold(reward_agent, eval_envs, args)
        print(f"Recalibrated success_threshold={success_threshold:.6f} for env {current_env_id}")
        return True, False, elapsed_minutes

    initial_metrics = evaluate(agent, eval_envs, args, logger, global_step, reward_agent, success_threshold)
    last_eval_metrics = dict(initial_metrics)
    json_metrics.append(
        build_metric_entry(
            update=0,
            global_step=global_step,
            current_env_id=current_env_id,
            current_env_index=current_env_index,
            elapsed_minutes=0.0,
            eval_metrics=initial_metrics,
            extras={"success_threshold": success_threshold},
        )
    )
    print(f"initial eval: {initial_metrics}")

    if args.evaluate:
        json_metrics.save_final_eval(last_eval_metrics)
        envs.close()
        eval_envs.close()
        if logger is not None:
            logger.close()
        return

    last_iteration = 0
    for iteration in range(1, args.num_iterations + 1):
        last_iteration = iteration

        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/env_index", current_env_index, global_step)
        if switched_env:
            print(f"continuing training on env[{current_env_index}]={current_env_id}")
        if should_stop_for_schedule:
            print(
                f"Reached continual env schedule end at elapsed={elapsed_minutes:.2f} minutes, "
                f"stopping training."
            )
            break

        if args.max_time is not None:
            elapsed_minutes = runtime_tracker.current_minutes()
            if elapsed_minutes >= args.max_time:
                print(f"Reached max_time={args.max_time} minutes, stopping.")
                break

        agent.train()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lr_now

        rollout_start = time.perf_counter()
        mean_progress_reward = 0.0
        predicted_success_hits = 0.0
        predicted_success_total = 0.0
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            with torch.no_grad():
                current_scores = compute_progress_score(reward_agent, next_obs)
                action, logprob, _, _ = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs_after_step, _, terminations, truncations, infos = envs.step(action)
            done_mask_tensor = torch.logical_or(terminations, truncations)
            done_mask, final_obs, final_episode = filter_final_observation(infos)
            reward_next_obs = merge_next_obs_with_final_obs(next_obs_after_step, final_obs, done_mask)

            with torch.no_grad():
                next_scores = compute_progress_score(reward_agent, reward_next_obs)
            progress_reward = (next_scores - current_scores) * args.reward_scale
            rewards[step] = progress_reward
            dones[step] = done_mask_tensor.float()
            mean_progress_reward += progress_reward.mean().item()

            if done_mask is not None:
                final_scores = next_scores[done_mask]
                predicted_success = (final_scores >= success_threshold).float()
                predicted_success_hits += predicted_success.sum().item()
                predicted_success_total += predicted_success.numel()
                if logger is not None:
                    for key, value in final_episode.items():
                        logger.add_scalar(f"train/{key}", value[done_mask].float().mean().item(), global_step)
                    logger.add_scalar("train/predicted_success_at_end", predicted_success.mean().item(), global_step)

            next_obs = next_obs_after_step

        rollout_time = time.perf_counter() - rollout_start
        runtime_tracker.add_active_seconds(rollout_time)
        mean_progress_reward /= max(1, args.num_steps)

        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            running_return = rewards[t] + args.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        b_obs = obs.reshape((-1,))
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_inds = np.arange(args.batch_size)
        np.random.shuffle(b_inds)

        update_start = time.perf_counter()
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)

        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()

            mb_returns = b_returns[mb_inds].detach()
            if args.normalize_returns:
                mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)

            pg_loss = -(args.reinforce_coef * mb_returns * newlogprob).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()

        update_time = time.perf_counter() - update_start
        runtime_tracker.add_active_seconds(update_time)

        metrics = evaluate(agent, eval_envs, args, logger, global_step, reward_agent, success_threshold)
        success_once = metrics.get("success_once", 0.0)
        success_end = metrics.get("success_at_end", 0.0)
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
                    "mean_progress_reward": mean_progress_reward,
                    "predicted_success_rate": predicted_success_hits / max(1.0, predicted_success_total),
                    "success_threshold": success_threshold,
                },
            )
        )

        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/env_index", current_env_index, global_step)
        if switched_env:
            print(f"switched to env[{current_env_index}]={current_env_id} after evaluation")
        if should_stop_for_schedule:
            print(
                f"Reached continual env schedule end at elapsed={elapsed_minutes:.2f} minutes after evaluation, "
                f"stopping training."
            )
            break

        print(
            f"iter={iteration} success_once={success_once:.4f} success_end={success_end:.4f} "
            f"progress_reward={mean_progress_reward:.6f} "
            f"pred_success={predicted_success_hits / max(1.0, predicted_success_total):.4f}"
        )

        if logger is not None:
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("returns/mean_return", b_returns.mean().item(), global_step)
            logger.add_scalar("returns/max_return", b_returns.max().item(), global_step)
            logger.add_scalar("returns/min_return", b_returns.min().item(), global_step)
            logger.add_scalar("rewards/mean_progress_reward", mean_progress_reward, global_step)
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("reward_model/success_threshold", success_threshold, global_step)

        if success_once >= best_success_once:
            best_success_once = success_once
            save_checkpoint(
                run_name,
                "best_success_once.pt",
                agent,
                optimizer,
                iteration,
                success_threshold,
                {"success_once": best_success_once, "success_at_end": success_end},
            )
        if success_end >= best_success_end:
            best_success_end = success_end
            save_checkpoint(
                run_name,
                "best_success_end.pt",
                agent,
                optimizer,
                iteration,
                success_threshold,
                {"success_once": success_once, "success_at_end": best_success_end},
            )

        if args.save_model and iteration % args.eval_freq == 0:
            save_checkpoint(
                run_name,
                "last.pt",
                agent,
                optimizer,
                iteration,
                success_threshold,
                {"success_once": success_once, "success_at_end": success_end},
            )

    if args.save_model:
        save_checkpoint(run_name, "last.pt", agent, optimizer, last_iteration, success_threshold, None)

    json_metrics.save_final_eval(last_eval_metrics)
    envs.close()
    eval_envs.close()
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
