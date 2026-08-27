import argparse
import ast
import bisect
import os
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
for candidate in (THIS_DIR, PARENT_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from train.common.mwe_runtime import ActiveRuntimeTracker
from train.common.mwe_checkpoint import maybe_save_model_checkpoint
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs
from train.common.mwe_eval import use_train_success_only
from train.common.memory_accounting import (
    DEFAULT_EXCLUDED_RUNTIME_PHASE_NAMES,
    MemoryPhaseTracker,
    write_module_memory_exclusion_metadata,
)
from train.common.time_breakdown import (
    empty_module_breakdown,
    snapshot_time_breakdown_to_metric,
    update_combined_search_enhancement_seconds,
    write_time_breakdown,
)
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import gymnasium as gym
import matplotlib
import mani_skill.envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv, torch_clone_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train.vla_adapter_new.model_impl.env as hold_cube_env  # noqa: F401
import workloads.hold_in_hand  # noqa: F401
from train.vla_adapter_new.model_impl import online_rl_hold_cube_in_hand as reference
from train.vla_adapter_new.model_impl.online_rl import (
    mkdir,
    parse_bool,
    plot_metrics_history,
    save_json,
    save_metrics_history,
    save_rollout_progress,
    set_seed,
    strip_module_prefix,
)
from train.vla_adapter_new.ours.generate_static_small_model import (
    feedback_static_small_model_to_large_model,
    generate_static_small_model_with_returning_pruning_info,
    inherit_static_small_model_retained_channels,
)
from train.vla_adapter_new.ours.model_with_fbs_test import convert_to_fbs_model


DEFAULT_MODEL_DIR = "ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_WORKDIR = "train/vla_adapter_new/ours/outputs/online_rl_cl"
DEFAULT_VERIFY_SUMMARY_NAME = "workload_verify_summary.json"
DEFAULT_FBS_CHECKPOINT = "train/vla_adapter_new/ours/pretrained_model_with_fbs.pth"


class HandSafeManiSkillVectorEnv(ManiSkillVectorEnv):
    def step(self, actions):  # type: ignore[override]
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        episode_info: Optional[dict] = None
        if self.record_metrics:
            episode_info = dict()
            self.returns += rew
            if "success" in infos:
                self.success_once = self.success_once | infos["success"]
                episode_info["success_once"] = self.success_once.clone()
                episode_info["success_at_end"] = infos["success"].clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
                episode_info["fail_at_end"] = infos["fail"].clone()
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


@dataclass
class ContinualEnvSchedule:
    env_ids: List[str]
    change_time_points: List[float]


@dataclass
class Args:
    mode: str = "train"
    seed: int = 1
    env_id: str = "HoldCubeInHandObjectScaleDown1p2-v1"
    envs_id: str = "['HoldCubeInHandObjectScaleDown1p2-v1', 'HoldCubeInHandObjectScaleDown1p4-v1']"
    env_change_time_points: str = "[15, 30]"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    output_dir: str = DEFAULT_WORKDIR
    num_envs: int = 256
    num_eval_envs: int = 8
    num_steps: int = 50
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-5
    backbone_learning_rate: float = 3e-5
    head_learning_rate: float = 3e-5
    state_learning_rate: float = 3e-5
    value_head_learning_rate: float = 3e-5
    weight_decay: float = 1e-6
    gamma: float = 0.8
    gae_lambda: float = 0.9
    update_epochs: int = 2
    num_minibatches: int = 16
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    minibatch_target_kl_factor: float = 1.0
    eval_episodes: int = 50
    eval_every_updates: int = 1
    max_episode_steps: Optional[int] = 100
    cuda_device: str = "0"
    save_video: bool = False
    save_train_video_freq: int = 10
    train_video_num_envs: int = 4
    test_video_num_envs: int = 4
    test_video_episodes: int = 4
    run_setup_smoke: bool = False
    max_runtime_hours: float = 8.0
    early_stop_zero_success_minutes: float = 45.0
    rollout_micro_batch_size: int = 256
    eval_micro_batch_size: int = 256
    update_micro_batch_size: int = 32
    rollout_progress_log_interval: int = 10
    freeze_vla_backbone: bool = False
    backbone_warmup_updates: int = 0
    action_dim: int = 16
    state_dim: int = 105
    run_name: Optional[str] = None
    large_agent_checkpoint: str = DEFAULT_FBS_CHECKPOINT
    continue_train_from: Optional[str] = None
    max_sparsity: float = 0.9
    small_model_generation_strategy: str = "target-single-traj"
    small_model_generation_policy: str = "small"
    small_model_feedback_schedule: Optional[str] = None
    small_model_regeneration_schedule: str = "before_per_rollout"
    small_model_feedback_alpha: float = 1.0
    reset_optimizer_after_regeneration: bool = True
    small_model_regeneration_increment_ratio: float = 1.0


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
    args = Args(**vars(parser.parse_args()))
    if os.environ.get("MWE", "0") == "1":
        # Keep the same training path while using a deliberately tiny footprint for
        # verification runs. VLA language-model activations dominate memory during
        # PPO, so MWE prioritizes proving the path is runnable over throughput.
        args.num_envs = 4
        args.num_eval_envs = 1
        args.num_steps = 4
        args.update_epochs = 1
        args.num_minibatches = 2
        args.rollout_micro_batch_size = 4
        args.eval_micro_batch_size = 4
        args.update_micro_batch_size = 2
        # Initialization still exercises the selected scaling method, while the
        # repeated regeneration path is outside this minimal run and can require
        # architecture-specific checkpoint shapes.
        args.small_model_feedback_schedule = "once"
        args.small_model_regeneration_schedule = "once"
        args.total_timesteps = max(args.total_timesteps, 10**12)
        mwe_runtime_minutes = float(os.environ.get("MWE_MAX_RUNTIME_MINUTES", "5.0"))
        if mwe_runtime_minutes <= 0:
            raise ValueError("MWE_MAX_RUNTIME_MINUTES must be positive")
        args.max_runtime_hours = mwe_runtime_minutes / 60.0
        args.early_stop_zero_success_minutes = max(args.early_stop_zero_success_minutes, 5.0)
    return args


def get_maniskill_backend_kwargs(device: torch.device) -> Dict[str, str]:
    if device.type == "cuda":
        return {
            "sim_backend": "physx_cuda:0",
            "render_backend": "cuda:0",
        }
    return {
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
    }


def make_vector_env_for_env_id(
    args: Args,
    device: torch.device,
    env_id: str,
    num_envs: int,
    record_metrics: bool = True,
    video_output_dir: Optional[Path] = None,
    video_max_steps: Optional[int] = None,
) -> ManiSkillVectorEnv:
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        env_id,
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
            f"`envs_id` and `env_change_time_points` must have the same length, got {len(env_ids)} and {len(time_points)}"
        )
    last_time_point = None
    for time_point in time_points:
        if time_point <= 0:
            raise ValueError("All `env_change_time_points` must be positive")
        if last_time_point is not None and time_point <= last_time_point:
            raise ValueError("`env_change_time_points` must be strictly increasing")
        last_time_point = time_point
    return ContinualEnvSchedule(env_ids=env_ids, change_time_points=time_points)


def plot_success_time_curve(output_dir: Path, metrics_history: List[Dict[str, Any]]) -> None:
    if not metrics_history:
        return
    plt.figure(figsize=(10, 6))
    plotted_any = False
    for metric_key in ("train_success_once", "train_success_at_end", "eval_success_once", "eval_success_at_end"):
        xs, ys = [], []
        for metric in metrics_history:
            value = metric.get(metric_key)
            elapsed_hours = metric.get("elapsed_hours")
            if value is None or elapsed_hours is None:
                continue
            xs.append(float(elapsed_hours) * 60.0)
            ys.append(float(value))
        if not xs:
            continue
        plotted_any = True
        plt.plot(xs, ys, marker="o", linewidth=2, label=metric_key)
    if not plotted_any:
        plt.close()
        return
    plots_dir = mkdir(output_dir / "plots")
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Success Rate")
    plt.title("Success Curve vs Time")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "success_time_curve.png", dpi=200)
    plt.close()


def should_early_stop_zero_success(metrics_history: List[Dict[str, Any]], threshold_minutes: float) -> Tuple[bool, float]:
    if not metrics_history:
        return False, 0.0
    latest_elapsed_minutes = float(metrics_history[-1].get("elapsed_hours", 0.0)) * 60.0
    max_success = 0.0
    for metric in metrics_history:
        for metric_name in ("train_success_once", "train_success_at_end", "eval_success_once", "eval_success_at_end"):
            value = metric.get(metric_name)
            if value is not None:
                max_success = max(max_success, float(value))
    return latest_elapsed_minutes >= threshold_minutes and max_success <= 0.0, max_success


def save_workload_verify_summary(output_dir: Path, payload: Dict[str, Any]) -> None:
    save_json(output_dir / DEFAULT_VERIFY_SUMMARY_NAME, payload)


def summarize_success_series(metrics_history: List[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
    values = [float(metric[key]) for metric in metrics_history if metric.get(key) is not None]
    if not values:
        return {"initial": None, "final": None, "average": None, "max": None, "improvement": None}
    return {
        "initial": values[0],
        "final": values[-1],
        "average": float(sum(values) / len(values)),
        "max": max(values),
        "improvement": values[-1] - values[0],
    }


def get_completed_episode_metrics(infos: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    final_info = infos.get("final_info")
    done_mask = infos.get("_final_info")
    if final_info is None or done_mask is None:
        return None, {}
    episode_metrics = final_info.get("episode")
    if not isinstance(episode_metrics, dict):
        return done_mask, {}
    return done_mask, episode_metrics


def summarize_episode_metrics(episode_metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, Tuple[float, int]]:
    summary = {}
    for key, values in episode_metrics.items():
        if not values:
            continue
        cat = torch.cat(values)
        summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
    return summary


def restore_small_policy_dtypes(policy: reference.HandVLAAdapterActorCritic, device: torch.device) -> None:
    policy.to(device=device)
    policy.device = device
    policy.vla.to(device=device, dtype=torch.bfloat16)
    policy.state_projector.to(device=device, dtype=torch.float32)
    policy.context_projector.to(device=device, dtype=torch.float32)
    policy.actor_head.to(device=device, dtype=torch.float32)
    policy.value_head.to(device=device, dtype=torch.float32)
    policy._buffers["action_bin_centers"] = policy.action_bin_centers.to(device=device, dtype=torch.float32)


def load_policy_state_from_checkpoint(checkpoint_path: str, policy: nn.Module) -> Dict[str, Any]:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"checkpoint not found at {checkpoint_path}; keep current initialization")
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
    elif isinstance(checkpoint, dict) and "agent" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["agent"])
    else:
        policy_state = strip_module_prefix(checkpoint)
    policy.load_state_dict(policy_state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def _extract_single_env_step(obs: Dict[str, Any], action_bins: torch.Tensor, env_idx: int) -> Dict[str, Any]:
    rgbs = reference.extract_rgb_batch_from_obs(obs)
    states = reference.extract_hand_state_batch_from_obs(obs)
    return {
        "rgbs": rgbs[env_idx].clone(),
        "states": states[env_idx].copy(),
        "action_bins": action_bins[env_idx].detach().cpu().clone(),
    }


def _stack_trajectory_steps(trajectory_steps: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    return {
        "rgbs": torch.stack([step["rgbs"] for step in trajectory_steps], dim=0).to(dtype=torch.uint8),
        "states": np.stack([step["states"] for step in trajectory_steps], axis=0).astype(np.float32),
        "action_bins": torch.stack([step["action_bins"] for step in trajectory_steps], dim=0).to(device=device, dtype=torch.long),
    }


def collect_best_return_trajectory_sample(
    policy: reference.HandVLAAdapterActorCritic,
    eval_envs: ManiSkillVectorEnv,
    num_steps: int,
    micro_batch_size: int,
    device: torch.device,
    reset_seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    was_training = policy.training
    policy.eval()
    if reset_seed is None:
        obs, _ = eval_envs.reset()
    else:
        obs, _ = eval_envs.reset(seed=reset_seed)
    num_envs = len(reference.extract_rgb_batch_from_obs(obs))
    running_trajectories = [[] for _ in range(num_envs)]
    running_returns = [0.0 for _ in range(num_envs)]
    finished_trajectories = []
    forward_seconds = 0.0

    with torch.no_grad():
        for _ in range(num_steps):
            rgbs = reference.extract_rgb_batch_from_obs(obs)
            states = reference.extract_hand_state_batch_from_obs(obs)
            forward_start_time = time.perf_counter()
            _, _, _, _, action_bins = reference.batched_get_action_and_value_no_grad(
                policy,
                rgbs,
                states,
                micro_batch_size=micro_batch_size,
                deterministic=True,
            )
            actions = policy.bin_indices_to_env_actions(action_bins)
            for env_idx in range(num_envs):
                running_trajectories[env_idx].append(_extract_single_env_step(obs, action_bins, env_idx))
            forward_seconds += time.perf_counter() - forward_start_time
            obs, rewards, terminations, truncations, _ = eval_envs.step(actions)
            reward_values = torch.as_tensor(rewards).detach().cpu().view(-1)
            done_mask = torch.logical_or(torch.as_tensor(terminations), torch.as_tensor(truncations)).cpu().view(-1).bool()
            for env_idx in range(num_envs):
                running_returns[env_idx] += float(reward_values[env_idx].item())
                if done_mask[env_idx]:
                    finished_trajectories.append(
                        {"return": running_returns[env_idx], "steps": running_trajectories[env_idx]}
                    )
                    running_trajectories[env_idx] = []
                    running_returns[env_idx] = 0.0

    for env_idx in range(num_envs):
        if running_trajectories[env_idx]:
            finished_trajectories.append({"return": running_returns[env_idx], "steps": running_trajectories[env_idx]})

    if was_training:
        policy.train()
    if not finished_trajectories:
        raise RuntimeError("Failed to collect any trajectory for small model generation")

    best = max(finished_trajectories, key=lambda item: item["return"])
    return _stack_trajectory_steps(best["steps"], device), float(best["return"]), forward_seconds


def resolve_generation_policy_agent(args: Args, large_agent, small_agent=None):
    if args.small_model_generation_policy == "small":
        return (small_agent if small_agent is not None else large_agent), ("small" if small_agent is not None else "large")
    if args.small_model_generation_policy == "large":
        return large_agent, "large"
    if args.small_model_generation_policy == "better":
        if small_agent is None:
            return large_agent, "large"
        return None, "better"
    raise NotImplementedError(f"Unknown small_model_generation_policy: {args.small_model_generation_policy}")


def collect_sample_for_small_model_generation(
    args: Args,
    large_agent,
    small_agent,
    eval_envs: ManiSkillVectorEnv,
    device: torch.device,
) -> Dict[str, Any]:
    forward_seconds = 0.0
    if args.small_model_generation_strategy in {"target-batch", "target-single"}:
        obs, _ = eval_envs.reset()
        rgbs = reference.extract_rgb_batch_from_obs(obs)
        states = reference.extract_hand_state_batch_from_obs(obs)
        generation_agent, _ = resolve_generation_policy_agent(args, large_agent=large_agent, small_agent=small_agent)
        forward_start_time = time.perf_counter()
        _, _, _, _, action_bins = reference.batched_get_action_and_value_no_grad(
            generation_agent,
            rgbs,
            states,
            micro_batch_size=args.eval_micro_batch_size,
            deterministic=True,
        )
        forward_seconds += time.perf_counter() - forward_start_time
        if args.small_model_generation_strategy == "target-single":
            return {
                "rgbs": rgbs[:1].clone(),
                "states": states[:1].copy(),
                "action_bins": action_bins[:1].to(device=device, dtype=torch.long),
            }, forward_seconds
        return {
            "rgbs": rgbs.clone(),
            "states": states.copy(),
            "action_bins": action_bins.to(device=device, dtype=torch.long),
        }, forward_seconds

    if args.small_model_generation_strategy == "target-single-traj":
        generation_agent, generation_policy = resolve_generation_policy_agent(args, large_agent=large_agent, small_agent=small_agent)
        if generation_policy == "better":
            comparison_seed = np.random.randint(0, 2**31 - 1)
            large_sample, large_return, large_forward_seconds = collect_best_return_trajectory_sample(
                large_agent,
                eval_envs,
                args.num_steps,
                args.eval_micro_batch_size,
                device,
                reset_seed=int(comparison_seed),
            )
            small_sample, small_return, small_forward_seconds = collect_best_return_trajectory_sample(
                small_agent,
                eval_envs,
                args.num_steps,
                args.eval_micro_batch_size,
                device,
                reset_seed=int(comparison_seed),
            )
            forward_seconds += large_forward_seconds + small_forward_seconds
            return (small_sample if small_return >= large_return else large_sample), forward_seconds
        sample, _, forward_seconds = collect_best_return_trajectory_sample(
            generation_agent,
            eval_envs,
            args.num_steps,
            args.eval_micro_batch_size,
            device,
        )
        return sample, forward_seconds

    raise NotImplementedError(f"Unknown small_model_generation_strategy: {args.small_model_generation_strategy}")


def should_regenerate_small_model_before_rollout(
    schedule: str,
    update: int,
    start_update: int,
    current_success_end: Optional[float],
    success_end_at_last_regeneration: Optional[float],
    update_at_last_regeneration: Optional[int],
) -> bool:
    if schedule == "once":
        return False
    if schedule == "before_per_rollout":
        return update > start_update
    threshold_prefix = "before_per_rollout_if_success_improv_is_larger_than_"
    if schedule.startswith(threshold_prefix):
        threshold = float(schedule[len(threshold_prefix) :])
        return (
            update > start_update
            and current_success_end is not None
            and success_end_at_last_regeneration is not None
            and current_success_end - success_end_at_last_regeneration > threshold
        )
    threshold_match = re.fullmatch(
        r"before_per_rollout_if_success_improv_less_than_([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)_for_(\d+)_iters",
        schedule,
    )
    if threshold_match is not None:
        threshold = float(threshold_match.group(1))
        num_iters = int(threshold_match.group(2))
        return (
            update > start_update
            and current_success_end is not None
            and success_end_at_last_regeneration is not None
            and update_at_last_regeneration is not None
            and update - update_at_last_regeneration >= num_iters
            and current_success_end - success_end_at_last_regeneration < threshold
        )
    raise NotImplementedError(f"Unknown small_model_regeneration_schedule: {schedule}")


def should_feedback_small_model_before_rollout(
    schedule: str,
    update: int,
    start_update: int,
    current_success_end: Optional[float],
    success_end_at_last_feedback: Optional[float],
) -> bool:
    if schedule == "once":
        return False
    if schedule == "before_per_rollout":
        return update > start_update
    threshold_prefix = "before_per_rollout_if_success_improv_is_larger_than_"
    if schedule.startswith(threshold_prefix):
        threshold = float(schedule[len(threshold_prefix) :])
        return (
            update > start_update
            and current_success_end is not None
            and success_end_at_last_feedback is not None
            and current_success_end - success_end_at_last_feedback > threshold
        )
    raise NotImplementedError(f"Unknown small_model_feedback_schedule: {schedule}")


def resolve_small_model_feedback_schedule(args: Args) -> str:
    if args.small_model_feedback_schedule is not None:
        return args.small_model_feedback_schedule
    if args.small_model_regeneration_schedule in {"once", "before_per_rollout"}:
        return args.small_model_regeneration_schedule
    if args.small_model_regeneration_schedule.startswith("before_per_rollout_if_success_improv_is_larger_than_"):
        return args.small_model_regeneration_schedule
    return "once"


def reset_optimizer_state_for_model(optimizer: optim.Optimizer, model: nn.Module) -> None:
    optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        optimizer.state.pop(param, None)


def regenerate_small_model_in_place(
    large_agent,
    small_agent,
    current_pruning_info,
    optimizer,
    args: Args,
    eval_envs,
    device: torch.device,
):
    sample, forward_seconds = collect_sample_for_small_model_generation(
        args,
        large_agent=large_agent,
        small_agent=small_agent,
        eval_envs=eval_envs,
        device=device,
    )

    enhancer_start_time = time.perf_counter()
    regenerated_small_agent, new_pruning_info = generate_static_small_model_with_returning_pruning_info(
        large_agent,
        sample_batch=sample,
        device=device,
        dtype=torch.bfloat16,
        previous_pruning_info=current_pruning_info,
        regeneration_increment_ratio=args.small_model_regeneration_increment_ratio,
        verify=True,
    )
    restore_small_policy_dtypes(regenerated_small_agent, device)
    regenerated_small_agent.configure_trainable_modules(train_backbone=not args.freeze_vla_backbone)
    regenerated_small_agent.eval_micro_batch_size = args.eval_micro_batch_size

    if args.small_model_regeneration_increment_ratio < 1.0 and current_pruning_info is not None:
        inherit_static_small_model_retained_channels(
            regenerated_small_agent,
            small_agent,
            new_pruning_info,
            current_pruning_info,
        )

    small_agent.load_state_dict(regenerated_small_agent.state_dict(), strict=True)
    if args.reset_optimizer_after_regeneration:
        reset_optimizer_state_for_model(optimizer, small_agent)

    merge_stats = new_pruning_info.get("merge_stats", {})
    if merge_stats:
        replaced_ratios = []
        for layer_stats in merge_stats.values():
            merged_count = max(int(layer_stats.get("merged_count", 0)), 1)
            replaced_ratios.append(float(layer_stats.get("replaced_count", 0)) / merged_count)
        if replaced_ratios:
            print(
                f"[regen] replaced_ratio avg={sum(replaced_ratios) / len(replaced_ratios):.4f} "
                f"min={min(replaced_ratios):.4f} max={max(replaced_ratios):.4f}"
            )
    enhancer_seconds = time.perf_counter() - enhancer_start_time
    return new_pruning_info, forward_seconds, enhancer_seconds


def maybe_load_training_checkpoint(
    checkpoint_path: Optional[str],
    large_agent: nn.Module,
    small_agent: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[int, int, float, Optional[dict]]:
    if not checkpoint_path:
        return 1, 0, -1.0, None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    large_agent.load_state_dict(strip_module_prefix(checkpoint["large_agent"]), strict=True)
    small_agent.load_state_dict(strip_module_prefix(checkpoint["small_agent"]), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        int(checkpoint.get("update", 0)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_success_once", -1.0)),
        checkpoint.get("pruning_info"),
    )


def save_training_checkpoint(
    output_path: Path,
    large_agent: nn.Module,
    small_agent: nn.Module,
    optimizer: optim.Optimizer,
    pruning_info: Optional[dict],
    update: int,
    global_step: int,
    best_success_once: float,
) -> None:
    maybe_save_model_checkpoint(
        {
            "large_agent": large_agent.state_dict(),
            "small_agent": small_agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "pruning_info": pruning_info,
            "update": update,
            "global_step": global_step,
            "best_success_once": best_success_once,
        },
        output_path,
    )


def train(args: Args) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_device)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    continual_env_schedule = build_continual_env_schedule(args)
    current_env_index = 0
    current_env_id = continual_env_schedule.env_ids[0] if continual_env_schedule is not None else args.env_id

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = mkdir(Path(args.output_dir) / run_name)
    save_json(output_dir / "args.json", asdict(args))
    memory_phase_tracker = MemoryPhaseTracker(output_dir)
    memory_phase_tracker.mark("setup", force=True)

    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] current_env={current_env_id}")

    module_breakdown = empty_module_breakdown()
    cumulative_rollout_seconds = 0.0
    cumulative_training_seconds = 0.0

    large_agent = reference.HandVLAAdapterActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    ).to(device)
    large_agent = convert_to_fbs_model(large_agent, device).to(device)
    load_policy_state_from_checkpoint(args.large_agent_checkpoint, large_agent)
    large_agent.eval_micro_batch_size = args.eval_micro_batch_size
    memory_exclusion_path = write_module_memory_exclusion_metadata(
        output_dir,
        module=large_agent,
        label="large_agent",
        reason="VLASelect large model can be offloaded during small-model online training; exclude its resident parameter/buffer memory from memory-footprint plots.",
        excluded_runtime_phase_names=DEFAULT_EXCLUDED_RUNTIME_PHASE_NAMES,
    )
    print(f"[setup] memory exclusion metadata saved to {memory_exclusion_path}")

    memory_phase_tracker.mark("workload_initialization")
    workload_init_start_time = time.perf_counter()
    envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_envs, record_metrics=True)
    eval_envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_eval_envs, record_metrics=True)
    module_breakdown["workload_initialization_seconds"] += time.perf_counter() - workload_init_start_time
    memory_phase_tracker.mark("large_model_runtime_excluded")
    search_start_time = time.perf_counter()
    initial_sample, forward_seconds = collect_sample_for_small_model_generation(
        args,
        large_agent=large_agent,
        small_agent=None,
        eval_envs=eval_envs,
        device=device,
    )
    module_breakdown["large_model_forward_seconds"] += forward_seconds

    enhancer_start_time = time.perf_counter()
    small_agent, current_pruning_info = generate_static_small_model_with_returning_pruning_info(
        large_agent,
        sample_batch=initial_sample,
        device=device,
        dtype=torch.bfloat16,
        verify=True,
    )
    restore_small_policy_dtypes(small_agent, device)
    small_agent.configure_trainable_modules(train_backbone=not args.freeze_vla_backbone and args.backbone_warmup_updates <= 0)
    small_agent.eval_micro_batch_size = args.eval_micro_batch_size
    module_breakdown["small_model_generation_seconds"] += time.perf_counter() - enhancer_start_time
    update_combined_search_enhancement_seconds(module_breakdown)
    optimizer = reference.build_optimizer(args, small_agent)
    memory_phase_tracker.mark("evaluation")

    start_update, global_step, best_success_once, resumed_pruning_info = maybe_load_training_checkpoint(
        args.continue_train_from,
        large_agent,
        small_agent,
        optimizer,
    )
    if resumed_pruning_info is not None:
        current_pruning_info = resumed_pruning_info

    global_batch_size = args.num_envs * args.num_steps
    local_batch_size = global_batch_size
    local_minibatch_size = max(1, local_batch_size // args.num_minibatches)
    num_updates = max(1, args.total_timesteps // global_batch_size)

    env_actions_buf = torch.zeros((args.num_steps, args.num_envs, args.action_dim), device=device)
    action_bins_buf = torch.zeros((args.num_steps, args.num_envs, args.action_dim), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    metrics_history: List[Dict[str, Any]] = []
    train_start_time = time.time()
    training_start_time = time.monotonic()
    runtime_tracker = ActiveRuntimeTracker.from_env(wall_clock_start_time=training_start_time)
    stop_reason = "completed"
    feedback_schedule = resolve_small_model_feedback_schedule(args)
    success_end_at_last_small_model_feedback = None
    success_end_at_last_small_model_regeneration = None
    update_at_last_small_model_regeneration = None
    current_success_end = None

    if start_update <= 1 and global_step == 0:
        memory_phase_tracker.mark("evaluation")
        initial_eval_metrics = reference.evaluate_policy(small_agent, eval_envs, args.eval_episodes)
        initial_metric = {
            "update": 0,
            "global_step": 0,
            "elapsed_hours": runtime_tracker.current_hours(),
            "env_id": current_env_id,
            "env_index": current_env_index,
        }
        initial_metric.update({f"eval_{k}": v for k, v in initial_eval_metrics.items()})
        module_breakdown["online_rl_completion_seconds"] = cumulative_rollout_seconds + cumulative_training_seconds
        snapshot_time_breakdown_to_metric(
            initial_metric,
            rollout_seconds=0.0,
            training_seconds=0.0,
            cumulative_rollout_seconds=cumulative_rollout_seconds,
            cumulative_training_seconds=cumulative_training_seconds,
            module_breakdown=module_breakdown,
        )
        metrics_history.append(initial_metric)
        current_success_end = float(initial_metric.get("eval_success_at_end", initial_metric.get("eval_success_once", 0.0)))
        success_end_at_last_small_model_feedback = current_success_end
        success_end_at_last_small_model_regeneration = current_success_end
        update_at_last_small_model_regeneration = 0
        if initial_metric.get("eval_success_once", initial_metric.get("eval_success", 0.0)) >= best_success_once:
            best_success_once = float(initial_metric.get("eval_success_once", initial_metric.get("eval_success", 0.0)))
            save_training_checkpoint(
                output_dir / "best_policy.pt",
                large_agent,
                small_agent,
                optimizer,
                current_pruning_info,
                0,
                0,
                best_success_once,
            )
        save_json(output_dir / "latest_metrics.json", initial_metric)
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
        plot_success_time_curve(output_dir, metrics_history)

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_id, current_env_index
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = runtime_tracker.current_minutes()
        scheduled_env_index = bisect.bisect_right(continual_env_schedule.change_time_points, elapsed_minutes)
        if scheduled_env_index >= len(continual_env_schedule.env_ids):
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes
        previous_env_id = current_env_id
        current_env_index = scheduled_env_index
        current_env_id = continual_env_schedule.env_ids[current_env_index]
        print(
            f"[env] switching from {previous_env_id} to {current_env_id} at elapsed={elapsed_minutes:.2f} minutes"
        )
        memory_phase_tracker.mark("workload_initialization")
        workload_init_start_time = time.perf_counter()
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_envs, record_metrics=True)
        eval_envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_eval_envs, record_metrics=True)
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        next_done = torch.zeros(args.num_envs, device=device)
        module_breakdown["workload_initialization_seconds"] += time.perf_counter() - workload_init_start_time
        return True, False, elapsed_minutes

    for update in range(start_update, num_updates + 1):
        switched_env, should_stop_for_schedule, _ = maybe_switch_envs()
        if switched_env:
            current_success_end = None
            success_end_at_last_small_model_feedback = None
            success_end_at_last_small_model_regeneration = None
            update_at_last_small_model_regeneration = None
        if should_stop_for_schedule:
            stop_reason = "continual_schedule_end"
            break

        if not args.freeze_vla_backbone and args.backbone_warmup_updates > 0 and update == args.backbone_warmup_updates + 1:
            small_agent.configure_trainable_modules(train_backbone=True)
            reference.set_optimizer_group_lr(optimizer, "vla", args.backbone_learning_rate)

        if update % args.eval_every_updates == 0 or (update == start_update and not metrics_history):
            memory_phase_tracker.mark("evaluation")
            eval_metrics = reference.evaluate_policy(small_agent, eval_envs, args.eval_episodes)
            current_success_end = float(eval_metrics.get("success_at_end", eval_metrics.get("success_once", 0.0)))
            if success_end_at_last_small_model_feedback is None:
                success_end_at_last_small_model_feedback = current_success_end
            if success_end_at_last_small_model_regeneration is None:
                success_end_at_last_small_model_regeneration = current_success_end
                update_at_last_small_model_regeneration = update
            if eval_metrics.get("success_once", eval_metrics.get("success", 0.0)) >= best_success_once:
                best_success_once = float(eval_metrics.get("success_once", eval_metrics.get("success", 0.0)))
                save_training_checkpoint(
                    output_dir / "best_policy.pt",
                    large_agent,
                    small_agent,
                    optimizer,
                    current_pruning_info,
                    update,
                    global_step,
                    best_success_once,
                )

        switched_env, should_stop_for_schedule, _ = maybe_switch_envs()
        if switched_env:
            current_success_end = None
            success_end_at_last_small_model_feedback = None
            success_end_at_last_small_model_regeneration = None
            update_at_last_small_model_regeneration = None
        if should_stop_for_schedule:
            stop_reason = "continual_schedule_end"
            break

        if should_feedback_small_model_before_rollout(
            feedback_schedule,
            update,
            start_update,
            current_success_end=current_success_end,
            success_end_at_last_feedback=success_end_at_last_small_model_feedback,
        ):
            memory_phase_tracker.mark("large_model_runtime_excluded")
            print("[ours] feedback small model before rollout")
            feedback_start_time = time.perf_counter()
            feedback_static_small_model_to_large_model(
                large_agent,
                small_agent,
                current_pruning_info,
                alpha=args.small_model_feedback_alpha,
            )
            module_breakdown["small_model_feedback_seconds"] += time.perf_counter() - feedback_start_time
            success_end_at_last_small_model_feedback = current_success_end

        if should_regenerate_small_model_before_rollout(
            args.small_model_regeneration_schedule,
            update,
            start_update,
            current_success_end=current_success_end,
            success_end_at_last_regeneration=success_end_at_last_small_model_regeneration,
            update_at_last_regeneration=update_at_last_small_model_regeneration,
        ):
            memory_phase_tracker.mark("large_model_runtime_excluded")
            print("[ours] regenerate small model before rollout")
            current_pruning_info, forward_seconds, enhancer_seconds = regenerate_small_model_in_place(
                large_agent,
                small_agent,
                current_pruning_info,
                optimizer,
                args,
                eval_envs,
                device,
            )
            module_breakdown["large_model_forward_seconds"] += forward_seconds
            module_breakdown["small_model_generation_seconds"] += enhancer_seconds
            update_combined_search_enhancement_seconds(module_breakdown)
            success_end_at_last_small_model_regeneration = current_success_end
            update_at_last_small_model_regeneration = update

        memory_phase_tracker.mark("online_rl_rollout")
        small_agent.eval()
        final_values.zero_()
        rollout_rgbs: List[torch.Tensor] = []
        rollout_states: List[np.ndarray] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []
        rollout_start_time = time.perf_counter()

        for step in range(args.num_steps):
            global_step += args.num_envs
            step_rgbs = reference.extract_rgb_batch_from_obs(next_obs)
            step_states = reference.extract_hand_state_batch_from_obs(next_obs)
            rollout_rgbs.append(step_rgbs.clone())
            rollout_states.append(step_states.copy())
            dones_buf[step] = next_done

            action, logprob, _, value, action_bins = reference.batched_get_action_and_value_no_grad(
                small_agent,
                step_rgbs,
                step_states,
                micro_batch_size=args.rollout_micro_batch_size,
                deterministic=False,
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
                elapsed_hours = runtime_tracker.current_hours(extra_active_seconds=time.perf_counter() - rollout_start_time)
                logged_partial_reward_means.append(float(partial_reward_means[-1]))
                print(
                    f"[rollout] update={update}/{num_updates} step={step + 1}/{args.num_steps} "
                    f"reward_mean_so_far={partial_reward_means[-1]:.4f} elapsed_h={elapsed_hours:.2f}"
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
                if "final_observation" in infos and truncation_mask.any():
                    final_obs = infos["final_observation"]
                    bootstrap_idx = truncation_mask.detach().cpu().numpy().astype(bool)
                    final_rgbs = reference.extract_rgb_batch_from_obs(final_obs)[bootstrap_idx]
                    final_states = reference.extract_hand_state_batch_from_obs(final_obs)[bootstrap_idx]
                    final_values[step, truncation_mask] = reference.batched_get_value_no_grad(
                        small_agent,
                        final_rgbs,
                        final_states,
                        micro_batch_size=args.eval_micro_batch_size,
                    ).view(-1)

        rollout_time = time.perf_counter() - rollout_start_time
        runtime_tracker.add_active_seconds(rollout_time)
        cumulative_rollout_seconds += rollout_time

        with torch.no_grad():
            next_value = reference.batched_get_value_no_grad(
                small_agent,
                reference.extract_rgb_batch_from_obs(next_obs),
                reference.extract_hand_state_batch_from_obs(next_obs),
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
        b_advantages = reference.normalize_advantages(advantages, device).reshape(-1)
        b_returns = returns.reshape(-1)
        ev = reference.explained_variance(values_buf, returns, device)

        inds = np.arange(local_batch_size)
        approx_kl = pg_loss_value = v_loss_value = entropy_value = clipfrac_value = 0.0
        stopped_on_minibatch_kl = False
        skipped_updates_on_kl = 0
        small_agent.eval()
        memory_phase_tracker.mark("online_rl_training")
        update_start_time = time.perf_counter()

        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            epoch_stats = defaultdict(list)
            for start in range(0, local_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = inds[start:end]
                mb_stats = reference.ppo_update_with_micro_batches(
                    args=args,
                    policy=small_agent,
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
                if mb_stats.get("skipped_on_kl", 0.0) > 0.0 or float(mb_stats.get("approx_kl", 0.0)) > args.target_kl:
                    stopped_on_minibatch_kl = True
                    break
            approx_kl = float(np.mean(epoch_stats["approx_kl"])) if epoch_stats["approx_kl"] else 0.0
            clipfrac_value = float(np.mean(epoch_stats["clipfrac"])) if epoch_stats["clipfrac"] else 0.0
            pg_loss_value = float(np.mean(epoch_stats["pg_loss"])) if epoch_stats["pg_loss"] else 0.0
            v_loss_value = float(np.mean(epoch_stats["v_loss"])) if epoch_stats["v_loss"] else 0.0
            entropy_value = float(np.mean(epoch_stats["entropy"])) if epoch_stats["entropy"] else 0.0
            if stopped_on_minibatch_kl or approx_kl > args.target_kl:
                break

        update_time = time.perf_counter() - update_start_time
        runtime_tracker.add_active_seconds(update_time)
        cumulative_training_seconds += update_time

        metric = {
            "update": update,
            "global_step": global_step,
            "reward_mean": float(rewards_buf.mean().item()),
            "gae_return_mean": float(returns.mean().item()),
            "value_mean": float(values_buf.mean().item()),
            "explained_variance": ev,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac_value,
            "pg_loss": pg_loss_value,
            "v_loss": v_loss_value,
            "entropy": entropy_value,
            "stopped_on_minibatch_kl": stopped_on_minibatch_kl,
            "skipped_updates_on_kl": skipped_updates_on_kl,
            "elapsed_hours": runtime_tracker.current_hours(),
            "env_id": current_env_id,
            "env_index": current_env_index,
        }
        metric.update(reference.gather_metric_summary(summarize_episode_metrics(train_episode_metrics)))
        if use_train_success_only():
            for source_key, target_key in (
                ("train_success_once", "eval_success_once"),
                ("train_success_at_end", "eval_success_at_end"),
                ("train_success", "eval_success"),
            ):
                value = metric.get(source_key)
                if value is not None:
                    metric[target_key] = value

        if update % args.eval_every_updates == 0 or update == num_updates:
            eval_metrics = reference.evaluate_policy(small_agent, eval_envs, args.eval_episodes)
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
            current_success_end = float(metric.get("eval_success_at_end", metric.get("eval_success_once", 0.0)))
            if metric.get("eval_success_once", 0.0) >= best_success_once:
                best_success_once = float(metric.get("eval_success_once", 0.0))
                save_training_checkpoint(
                    output_dir / "best_policy.pt",
                    large_agent,
                    small_agent,
                    optimizer,
                    current_pruning_info,
                    update,
                    global_step,
                    best_success_once,
                )

        module_breakdown["online_rl_completion_seconds"] = cumulative_rollout_seconds + cumulative_training_seconds
        snapshot_time_breakdown_to_metric(
            metric,
            rollout_seconds=rollout_time,
            training_seconds=update_time,
            cumulative_rollout_seconds=cumulative_rollout_seconds,
            cumulative_training_seconds=cumulative_training_seconds,
            module_breakdown=module_breakdown,
        )
        metrics_history.append(metric)
        print(
            f"[train] update={update}/{num_updates} env={current_env_id} reward={metric['reward_mean']:.4f} "
            f"gae_return={metric['gae_return_mean']:.4f} value_mean={metric['value_mean']:.4f} "
            f"approx_kl={metric['approx_kl']:.5f} eval_success_once={metric.get('eval_success_once', float('nan')):.4f} "
            f"elapsed_h={metric['elapsed_hours']:.2f}"
        )
        save_json(output_dir / "latest_metrics.json", metric)
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
        plot_success_time_curve(output_dir, metrics_history)
        if update % 10 == 0 or update == num_updates:
            save_training_checkpoint(
                output_dir / "latest_policy.pt",
                large_agent,
                small_agent,
                optimizer,
                current_pruning_info,
                update,
                global_step,
                best_success_once,
            )

        reached_zero_success_stop, max_success_observed = should_early_stop_zero_success(
            metrics_history,
            threshold_minutes=args.early_stop_zero_success_minutes,
        )
        if reached_zero_success_stop:
            print(
                f"[train] reached zero-success early stop after {args.early_stop_zero_success_minutes:.1f} minutes "
                f"with max_success={max_success_observed:.4f}"
            )
            stop_reason = "early_stop_zero_success"
            break
        if metric["elapsed_hours"] >= args.max_runtime_hours:
            print(f"[train] reached time limit: {metric['elapsed_hours']:.2f}h >= {args.max_runtime_hours:.2f}h")
            stop_reason = "time_limit"
            break

    final_eval_metrics = reference.evaluate_policy(small_agent, eval_envs, args.eval_episodes)
    last_metric = metrics_history[-1] if metrics_history else {}
    if not final_eval_metrics and use_train_success_only():
        for source_key, target_key in (
            ("train_success_once", "success_once"),
            ("train_success_at_end", "success_at_end"),
            ("train_success", "success"),
            ("eval_success_once", "success_once"),
            ("eval_success_at_end", "success_at_end"),
            ("eval_success", "success"),
        ):
            value = last_metric.get(source_key)
            if value is not None:
                final_eval_metrics[target_key] = value
    save_json(output_dir / "final_eval_metrics.json", final_eval_metrics)
    save_metrics_history(output_dir, metrics_history)
    plot_metrics_history(output_dir, metrics_history)
    plot_success_time_curve(output_dir, metrics_history)
    module_breakdown["online_rl_completion_seconds"] = cumulative_rollout_seconds + cumulative_training_seconds
    write_time_breakdown(
        output_dir,
        sampling_seconds=cumulative_rollout_seconds,
        training_seconds=cumulative_training_seconds,
        module_breakdown=module_breakdown,
    )
    save_workload_verify_summary(
        output_dir,
        {
            "env_id": current_env_id,
            "envs_id": continual_env_schedule.env_ids if continual_env_schedule is not None else [current_env_id],
            "change_time_points": continual_env_schedule.change_time_points if continual_env_schedule is not None else [],
            "output_dir": str(output_dir),
            "stop_reason": stop_reason,
            "elapsed_hours": float(last_metric.get("elapsed_hours", 0.0)),
            "train_success_once_summary": summarize_success_series(metrics_history, "train_success_once"),
            "train_success_at_end_summary": summarize_success_series(metrics_history, "train_success_at_end"),
            "eval_success_once_summary": summarize_success_series(metrics_history, "eval_success_once"),
            "eval_success_at_end_summary": summarize_success_series(metrics_history, "eval_success_at_end"),
            "final_eval_metrics": final_eval_metrics,
            "num_metric_points": len(metrics_history),
        },
    )
    close_envs(envs, eval_envs)
    envs = None
    eval_envs = None
    clear_torch_cuda_cache()


def main() -> None:
    args = parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")
    train(args)


if __name__ == "__main__":
    main()
