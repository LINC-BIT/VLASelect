from __future__ import annotations

import argparse
import ast
import bisect
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
VENDOR_ROOT = THIS_DIR / "vendor"
for candidate in (THIS_DIR, VENDOR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
from train.common.mwe_eval import SUCCESS_METRIC_WINDOW_EPISODES, append_episode_metric_batch, summarize_episode_metric_tensors, trim_episode_metric_tensors
from train.vla_adapter_new.ours.generate_static_small_model import (
    feedback_static_small_model_to_large_model,
)
from api.small_model_scaling_interface import SmallModelScalingInterface
DEFAULT_MODEL_DIR = "eval/ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_WORKDIR = "train/vla_adapter_new/ours/outputs/online_rl_cl"
DEFAULT_VERIFY_SUMMARY_NAME = "workload_verify_summary.json"
DEFAULT_FBS_CHECKPOINT = "train/vla_adapter_new/ours/pretrained_model_with_fbs.pth"

# The runner is model agnostic.  Concrete adapters install a compatibility object here
# that exposes the reference model helpers while implementing VLAModelInterface.
reference: Any = None
ACTIVE_ADAPTER: Any = None


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> torch.Tensor:
    if ACTIVE_ADAPTER is None:
        raise RuntimeError("no active VLA adapter")
    return ACTIVE_ADAPTER.extract_rgb_batch_from_obs(obs)


def extract_state_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    if ACTIVE_ADAPTER is None:
        raise RuntimeError("no active VLA adapter")
    return ACTIVE_ADAPTER.extract_state_batch_from_obs(obs)


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
    env_action_dim: int = 16
    controlled_action_indices: Optional[Tuple[int, ...]] = None
    run_name: Optional[str] = None
    large_agent_checkpoint: str = DEFAULT_FBS_CHECKPOINT
    continue_train_from: Optional[str] = None
    max_sparsity: float = 0.9
    small_model_scaling_strategy: str = "target-single-traj"
    small_model_scaling_policy: str = "small"
    small_model_feedback_schedule: Optional[str] = None
    small_model_regeneration_schedule: str = "before_per_rollout"
    small_model_feedback_alpha: float = 1.0
    reset_optimizer_after_regeneration: bool = True
    small_model_regeneration_increment_ratio: float = 1.0
    scaling_method: Optional[str] = None
    knowledge_exchange_granularity: Optional[str] = None


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
        # verification runs.  VLA language-model activations dominate memory during
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
    if ACTIVE_ADAPTER is None:
        raise RuntimeError("a VLAModelInterface adapter must be installed before creating environments")
    return ACTIVE_ADAPTER.make_vector_env(
        args,
        device=device,
        env_id=env_id,
        num_envs=num_envs,
        record_metrics=record_metrics,
        video_output_dir=video_output_dir,
        video_max_steps=video_max_steps,
    )


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
    for metric_key in ("train_success_once", "train_success_at_end"):
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
        for metric_name in ("train_success_once", "train_success_at_end"):
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


def load_policy_state_from_checkpoint(checkpoint_path: str, policy: nn.Module) -> Dict[str, Any]:
    if not checkpoint_path:
        raise ValueError("--large-agent-checkpoint is required for API VLA training")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"large-agent checkpoint does not exist: {checkpoint_file}. "
            "API VLA training does not support random or uninitialized fallback models."
        )
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
    elif isinstance(checkpoint, dict) and "agent" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["agent"])
    else:
        policy_state = strip_module_prefix(checkpoint)
    policy.load_state_dict(policy_state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


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
    torch.save(
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


def train(
    args: Args,
    small_model_scaling_interface: Optional[SmallModelScalingInterface] = None,
) -> None:
    if ACTIVE_ADAPTER is None or reference is None:
        raise RuntimeError("use run_training(adapter, args) instead of calling train directly")
    if small_model_scaling_interface is None:
        small_model_scaling_interface = SmallModelScalingInterface()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_device)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    continual_env_schedule = build_continual_env_schedule(args)
    current_env_index = 0
    current_env_id = continual_env_schedule.env_ids[0] if continual_env_schedule is not None else args.env_id
    args._current_env_id = current_env_id
    ACTIVE_ADAPTER.apply_environment_contract(args, env_id=current_env_id, device=device)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = mkdir(Path(args.output_dir) / run_name)
    save_json(output_dir / "args.json", asdict(args))

    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] current_env={current_env_id}")
    print(f"[setup] architecture={ACTIVE_ADAPTER.architecture_spec.architecture_name}")
    print(f"[setup] policy_class={ACTIVE_ADAPTER.architecture_spec.policy_class_name}")

    if not args.large_agent_checkpoint:
        raise ValueError("--large-agent-checkpoint is required for API VLA training")
    if not Path(args.large_agent_checkpoint).is_file():
        raise FileNotFoundError(
            f"large-agent checkpoint does not exist: {args.large_agent_checkpoint}"
        )

    large_agent = ACTIVE_ADAPTER.build_policy(Path(args.model_dir), args=args, device=device).to(device)
    large_agent = ACTIVE_ADAPTER.convert_to_fbs_policy(
        large_agent,
        device=device,
        max_sparsity=args.max_sparsity,
    ).to(device)
    load_policy_state_from_checkpoint(args.large_agent_checkpoint, large_agent)
    large_agent.eval_micro_batch_size = args.eval_micro_batch_size

    envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_envs, record_metrics=True)
    eval_envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_eval_envs, record_metrics=True)
    small_agent, current_pruning_info = small_model_scaling_interface.generate_initial_small_model(
        large_agent=large_agent,
        args=args,
        eval_envs=eval_envs,
        device=device,
        adapter=ACTIVE_ADAPTER,
        reference_api=reference,
    )
    optimizer = reference.build_optimizer(args, small_agent)

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

    env_actions_buf = torch.zeros((args.num_steps, args.num_envs, args.env_action_dim), device=device)
    action_bins_buf = torch.zeros((args.num_steps, args.num_envs, args.action_dim), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    metrics_history: List[Dict[str, Any]] = []
    # Start the budget only after model/FBS/small-policy initialization.  The clock
    # can be paused around environment setup and small-model regeneration so the
    # reported curve measures RL rollout/sampling time rather than preparation.
    active_training_seconds = 0.0
    active_clock_start = time.monotonic()
    active_clock_running = False
    stop_reason = "completed"
    feedback_schedule = resolve_small_model_feedback_schedule(args)
    success_end_at_last_small_model_feedback = None
    success_end_at_last_small_model_regeneration = None
    update_at_last_small_model_regeneration = None
    current_success_end = None
    runtime_limit_seconds = float(args.max_runtime_hours) * 3600.0

    def start_training_clock() -> None:
        nonlocal active_clock_start, active_clock_running
        if not active_clock_running:
            active_clock_start = time.monotonic()
            active_clock_running = True

    def pause_training_clock() -> None:
        nonlocal active_training_seconds, active_clock_running
        if active_clock_running:
            active_training_seconds += time.monotonic() - active_clock_start
            active_clock_running = False

    def elapsed_training_seconds() -> float:
        if active_clock_running:
            return active_training_seconds + (time.monotonic() - active_clock_start)
        return active_training_seconds

    def elapsed_training_hours() -> float:
        return elapsed_training_seconds() / 3600.0

    def collect_initial_training_metric() -> Dict[str, Any]:
        """Measure the initial policy on the training environments before PPO updates."""
        initial_obs, _ = envs.reset(seed=args.seed)
        initial_episode_metrics = defaultdict(list)
        baseline_steps = max(1, int(args.max_episode_steps or 100))
        small_agent.eval()
        with torch.no_grad():
            for _ in range(baseline_steps):
                rgbs = extract_rgb_batch_from_obs(initial_obs)
                states = extract_state_batch_from_obs(initial_obs)
                action, _, _, _, _ = reference.batched_get_action_and_value_no_grad(
                    small_agent,
                    rgbs,
                    states,
                    micro_batch_size=args.rollout_micro_batch_size,
                    deterministic=False,
                )
                initial_obs, _, _, _, infos = envs.step(action)
                done_mask, episode_metrics = get_completed_episode_metrics(infos)
                if done_mask is not None and done_mask.any():
                    append_episode_metric_batch(initial_episode_metrics, episode_metrics, done_mask)

        metric: Dict[str, Any] = {
            "update": 0,
            "global_step": 0,
            "elapsed_hours": 0.0,
            "env_id": current_env_id,
            "env_index": current_env_index,
        }
        metric.update(summarize_episode_metric_tensors(initial_episode_metrics))
        return metric

    success_metric_window_episodes = SUCCESS_METRIC_WINDOW_EPISODES
    initial_metric = collect_initial_training_metric()
    metrics_history.append(initial_metric)
    best_success_once = max(best_success_once, float(initial_metric.get("train_success_once", -1.0)))
    current_success_end = float(initial_metric.get("train_success_at_end", initial_metric.get("train_success_once", 0.0)))
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(
        f"[train-init] env={current_env_id} train_success_once="
        f"{initial_metric.get('train_success_once', float('nan')):.4f} "
        f"train_success_at_end={initial_metric.get('train_success_at_end', float('nan')):.4f}"
    )
    save_json(output_dir / "latest_metrics.json", initial_metric)
    save_metrics_history(output_dir, metrics_history)
    plot_metrics_history(output_dir, metrics_history)
    plot_success_time_curve(output_dir, metrics_history)

    # Everything above this point is setup/scaling work and is excluded from the
    # five-minute RL budget.
    start_training_clock()

    def runtime_limit_reached() -> bool:
        return elapsed_training_seconds() >= runtime_limit_seconds

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_id, current_env_index
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = elapsed_training_seconds() / 60.0
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
        pause_training_clock()
        try:
            envs.close()
            eval_envs.close()
            envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_envs, record_metrics=True)
            eval_envs = make_vector_env_for_env_id(args, device, current_env_id, args.num_eval_envs, record_metrics=True)
            next_obs, _ = envs.reset(seed=args.seed + current_env_index)
            next_done = torch.zeros(args.num_envs, device=device)
            args._current_env_id = current_env_id
        finally:
            start_training_clock()
        return True, False, elapsed_minutes

    for update in range(start_update, num_updates + 1):
        if runtime_limit_reached():
            stop_reason = "time_limit"
            break
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

        if runtime_limit_reached():
            stop_reason = "time_limit"
            break

        switched_env, should_stop_for_schedule, _ = maybe_switch_envs()
        if switched_env:
            current_success_end = None
            success_end_at_last_small_model_feedback = None
            success_end_at_last_small_model_regeneration = None
            update_at_last_small_model_regeneration = None
            pause_training_clock()
            try:
                regenerated, current_pruning_info = small_model_scaling_interface.on_environment_switch(
                    args=args,
                    large_agent=large_agent,
                    small_agent=small_agent,
                    current_pruning_info=current_pruning_info,
                    optimizer=optimizer,
                    eval_envs=eval_envs,
                    device=device,
                    adapter=ACTIVE_ADAPTER,
                    reference_api=reference,
                )
            finally:
                start_training_clock()
            if regenerated:
                print("[scaling] regenerated small model after environment switch")
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
            print("[ours] feedback small model before rollout")
            pause_training_clock()
            try:
                feedback_static_small_model_to_large_model(
                    large_agent,
                    small_agent,
                    current_pruning_info,
                    alpha=args.small_model_feedback_alpha,
                )
            finally:
                start_training_clock()
            success_end_at_last_small_model_feedback = current_success_end

        pause_training_clock()
        try:
            regenerated, current_pruning_info = small_model_scaling_interface.maybe_regenerate_small_model_before_rollout(
                args=args,
                update=update,
                start_update=start_update,
                current_success_end=current_success_end,
                success_end_at_last_regeneration=success_end_at_last_small_model_regeneration,
                update_at_last_regeneration=update_at_last_small_model_regeneration,
                large_agent=large_agent,
                small_agent=small_agent,
                current_pruning_info=current_pruning_info,
                optimizer=optimizer,
                eval_envs=eval_envs,
                device=device,
                adapter=ACTIVE_ADAPTER,
                reference_api=reference,
            )
        finally:
            start_training_clock()
        if regenerated:
            success_end_at_last_small_model_regeneration = current_success_end
            update_at_last_small_model_regeneration = update

        small_agent.eval()
        final_values.zero_()
        rollout_rgbs: List[torch.Tensor] = []
        rollout_states: List[np.ndarray] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []

        rollout_steps = 0
        deadline_reached = False
        for step in range(args.num_steps):
            if runtime_limit_reached():
                deadline_reached = True
                break
            rollout_steps = step + 1
            global_step += args.num_envs
            step_rgbs = extract_rgb_batch_from_obs(next_obs)
            step_states = extract_state_batch_from_obs(next_obs)
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
                elapsed_hours = elapsed_training_hours()
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
            if done_mask is not None and done_mask.any():
                append_episode_metric_batch(train_episode_metrics, episode_metrics, done_mask)
                if "final_observation" in infos and truncation_mask.any():
                    final_obs = infos["final_observation"]
                    bootstrap_idx = truncation_mask.detach().cpu().numpy().astype(bool)
                    final_rgbs = extract_rgb_batch_from_obs(final_obs)[bootstrap_idx]
                    final_states = extract_state_batch_from_obs(final_obs)[bootstrap_idx]
                    final_values[step, truncation_mask] = reference.batched_get_value_no_grad(
                        small_agent,
                        final_rgbs,
                        final_states,
                        micro_batch_size=args.eval_micro_batch_size,
                    ).view(-1)

        if rollout_steps == 0:
            stop_reason = "time_limit"
            break
        with torch.no_grad():
            next_value = reference.batched_get_value_no_grad(
                small_agent,
                extract_rgb_batch_from_obs(next_obs),
                extract_state_batch_from_obs(next_obs),
                micro_batch_size=args.eval_micro_batch_size,
            ).view(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
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
        actual_batch_size = rollout_steps * args.num_envs
        local_batch_size = actual_batch_size
        local_minibatch_size = max(1, local_batch_size // args.num_minibatches)
        b_action_bins = action_bins_buf[:rollout_steps].reshape(-1, args.action_dim)
        b_logprobs = logprobs_buf[:rollout_steps].reshape(-1)
        b_values = values_buf[:rollout_steps].reshape(-1)
        b_advantages = reference.normalize_advantages(advantages[:rollout_steps], device).reshape(-1)
        b_returns = returns[:rollout_steps].reshape(-1)
        ev = reference.explained_variance(values_buf[:rollout_steps], returns[:rollout_steps], device)

        inds = np.arange(local_batch_size)
        approx_kl = pg_loss_value = v_loss_value = entropy_value = clipfrac_value = 0.0
        stopped_on_minibatch_kl = False
        skipped_updates_on_kl = 0
        small_agent.eval()

        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            epoch_stats = defaultdict(list)
            for start in range(0, local_batch_size, local_minibatch_size):
                if runtime_limit_reached():
                    deadline_reached = True
                    break
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

        metric = {
            "update": update,
            "global_step": global_step,
            "reward_mean": float(rewards_buf[:rollout_steps].mean().item()),
            "gae_return_mean": float(returns[:rollout_steps].mean().item()),
            "value_mean": float(values_buf[:rollout_steps].mean().item()),
            "explained_variance": ev,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac_value,
            "pg_loss": pg_loss_value,
            "v_loss": v_loss_value,
            "entropy": entropy_value,
            "stopped_on_minibatch_kl": stopped_on_minibatch_kl,
            "skipped_updates_on_kl": skipped_updates_on_kl,
            "elapsed_hours": elapsed_training_hours(),
            "env_id": current_env_id,
            "env_index": current_env_index,
        }
        metric.update(
            reference.gather_metric_summary(
                summarize_episode_metric_tensors(train_episode_metrics, max_num_values=success_metric_window_episodes)
            )
        )
        trim_episode_metric_tensors(train_episode_metrics, success_metric_window_episodes)

        current_success_end = float(metric.get("train_success_at_end", metric.get("train_success_once", 0.0)))
        if success_end_at_last_small_model_feedback is None:
            success_end_at_last_small_model_feedback = current_success_end
        if success_end_at_last_small_model_regeneration is None:
            success_end_at_last_small_model_regeneration = current_success_end
            update_at_last_small_model_regeneration = update
        current_success_once = float(metric.get("train_success_once", 0.0))
        if current_success_once >= best_success_once:
            best_success_once = current_success_once

        metrics_history.append(metric)
        print(
            f"[train] update={update}/{num_updates} env={current_env_id} reward={metric['reward_mean']:.4f} "
            f"gae_return={metric['gae_return_mean']:.4f} value_mean={metric['value_mean']:.4f} "
            f"approx_kl={metric['approx_kl']:.5f} train_success_once={metric.get('train_success_once', float('nan')):.4f} "
            f"elapsed_h={metric['elapsed_hours']:.2f}"
        )
        save_json(output_dir / "latest_metrics.json", metric)
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
        plot_success_time_curve(output_dir, metrics_history)
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
        if deadline_reached or metric["elapsed_hours"] >= args.max_runtime_hours:
            print(f"[train] reached time limit: {metric['elapsed_hours']:.2f}h >= {args.max_runtime_hours:.2f}h")
            stop_reason = "time_limit"
            break

    save_metrics_history(output_dir, metrics_history)
    plot_metrics_history(output_dir, metrics_history)
    plot_success_time_curve(output_dir, metrics_history)
    last_metric = metrics_history[-1] if metrics_history else {}
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
            "num_metric_points": len(metrics_history),
        },
    )
    envs.close()
    eval_envs.close()


def main() -> None:
    args = parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")
    raise RuntimeError("launch one of api/vla_model_interface_examples/*_impl.py instead")


def run_training(
    adapter: Any,
    args: Args,
    small_model_scaling_interface: Optional[SmallModelScalingInterface] = None,
) -> None:
    """Run the unmodified reference continual-learning loop through one interface adapter.

    Apart from the adapter calls at model/environment boundaries, this is the original
    VLA-Adapter continual-learning loop: real FBS insertion, static-small-model
    generation, feedback, regeneration, GAE/PPO, checkpoints, and continual environment
    scheduling are retained above. Training metrics are collected from rollout episodes;
    the expensive standalone evaluation pass is intentionally disabled.
    """
    global ACTIVE_ADAPTER, reference
    ACTIVE_ADAPTER = adapter
    reference = adapter.reference_api
    adapter.register_workloads()
    try:
        train(args, small_model_scaling_interface=small_model_scaling_interface)
    finally:
        ACTIVE_ADAPTER = None
        reference = None


if __name__ == "__main__":
    main()
