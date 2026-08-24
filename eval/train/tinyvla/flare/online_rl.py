import argparse
import ast
import bisect
import os
import shutil
import sys
import time

from train.common.mwe_runtime import ActiveRuntimeTracker
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

import train.tinyvla.model_impl.online_rl_open_cabinet_drawer as reference
import workloads.mobile_arm as mobile_arm_workload
from workloads.mobile_arm import *  # noqa: F401,F403
from ours.libs.train_with_fbs.lib import set_sparsity
from train.vla_adapter_new.model_impl.online_rl import (
    cleanup_runtime,
    mkdir,
    parse_bool,
    plot_metrics_history,
    save_json,
    save_metrics_history,
    strip_module_prefix,
)
from train.common.checkpoint_noise import maybe_apply_checkpoint_noise_to_state_dict
from train.vla_adapter_new.ours.generate_static_small_model import generate_static_small_model
from train.tinyvla.ours.model_with_fbs import convert_to_fbs_model


DEFAULT_MODEL_DIR = reference.DEFAULT_MODEL_DIR
DEFAULT_OUTPUT_DIR = "train/tinyvla/flare/outputs"
DEFAULT_STATIC_MODEL_CHECKPOINT = (
    "ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt"
)
DEFAULT_ENVS_ID = "['OpenCabinetDrawerEasyLevel0-v1', 'OpenCabinetDrawerEasyLevel0-v1']"
DEFAULT_ENV_CHANGE_TIME_POINTS = "[10, 20]"
DEFAULT_SUMMARY_NAME = "flare_training_summary.json"


@dataclass
class Args:
    mode: str = "train"
    seed: int = 1
    env_id: str = "OpenCabinetDrawerEasyLevel0-v1"
    envs_id: Optional[str] = DEFAULT_ENVS_ID
    env_change_time_points: Optional[str] = DEFAULT_ENV_CHANGE_TIME_POINTS
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    static_model_checkpoint: str = DEFAULT_STATIC_MODEL_CHECKPOINT
    resume_from: Optional[str] = None
    num_envs: int = 128
    num_eval_envs: int = 8
    num_steps: int = 100
    total_timesteps: int = 100_000_000
    num_minibatches: int = 16
    update_epochs: int = 2
    learning_rate: float = 6e-5
    backbone_learning_rate: float = 6e-5
    head_learning_rate: float = 6e-5
    state_learning_rate: float = 6e-5
    value_head_learning_rate: float = 6e-5
    weight_decay: float = 1e-6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    minibatch_target_kl_factor: float = 1.0
    eval_episodes: int = 50
    eval_every_updates: int = 50
    max_episode_steps: Optional[int] = None
    cuda_device: str = "0"
    save_video: bool = True
    save_train_video_freq: int = 10
    train_video_num_envs: int = 4
    test_video_num_envs: int = 4
    test_video_episodes: int = 4
    run_setup_smoke: bool = False
    smoke_steps: int = 32
    max_runtime_hours: float = 400.0
    rollout_micro_batch_size: int = 256
    eval_micro_batch_size: int = 256
    update_micro_batch_size: int = 32
    rollout_progress_log_interval: int = 10
    freeze_vla_backbone: bool = False
    backbone_warmup_updates: int = 0
    action_dim: int = 8
    env_action_dim: int = 13
    state_dim: int = 44
    controlled_action_indices: Optional[Tuple[int, ...]] = None
    run_name: Optional[str] = None
    early_stop_zero_success_minutes: float = 45.0
    static_sparsity: float = 0.8


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
    return Args(**vars(parser.parse_args()))


def configure_env_contract(args: Args, device: torch.device) -> Args:
    env_action_dim, state_dim, controlled_action_indices = reference.inspect_env_contract(args, device)
    if (
        env_action_dim != args.env_action_dim
        or state_dim != args.state_dim
        or len(controlled_action_indices) != args.action_dim
    ):
        print(
            "[setup] overriding env contract from probe: "
            f"env_action_dim={env_action_dim}, action_dim={len(controlled_action_indices)}, state_dim={state_dim}"
        )
    return replace(
        args,
        env_action_dim=env_action_dim,
        action_dim=len(controlled_action_indices),
        state_dim=state_dim,
        controlled_action_indices=controlled_action_indices,
    )


@dataclass
class ContinualEnvSchedule:
    env_ids: List[str]
    change_time_points: List[float]


def _parse_cli_sequence(raw_value: Optional[str], arg_name: str, cast_fn) -> Optional[List[Any]]:
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
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in env_id)


def make_vector_env_for_env_id(
    args: Args,
    device: torch.device,
    env_id: str,
    num_envs: int,
    *,
    record_metrics: bool = True,
    video_output_dir: Optional[Path] = None,
    video_max_steps: Optional[int] = None,
):
    env_args = replace(args, env_id=env_id)
    return reference.make_vector_env(
        env_args,
        device,
        num_envs,
        record_metrics=record_metrics,
        video_output_dir=video_output_dir,
        video_max_steps=video_max_steps,
    )


def build_runtime_envs(
    args: Args,
    device: torch.device,
    output_dir: Path,
    env_id: str,
    env_index: int,
):
    print(f"[setup] building env[{env_index}]={env_id}")
    envs = make_vector_env_for_env_id(args, device, env_id, args.num_envs, record_metrics=True)
    eval_envs = make_vector_env_for_env_id(args, device, env_id, args.num_eval_envs, record_metrics=True)
    test_video_envs = None
    if args.save_video:
        env_dir_name = f"env{env_index:02d}-{_sanitize_env_name_for_path(env_id)}"
        test_video_envs = make_vector_env_for_env_id(
            args,
            device,
            env_id,
            min(args.test_video_num_envs, args.num_eval_envs),
            record_metrics=False,
            video_output_dir=output_dir / "test_videos" / env_dir_name,
        )
    return envs, eval_envs, test_video_envs


def load_policy_state_from_checkpoint(checkpoint_path: str, policy: nn.Module) -> Dict[str, Any]:
    if not checkpoint_path:
        print("[setup] empty static checkpoint path; keeping current policy initialization")
        return {}
    if not Path(checkpoint_path).exists():
        print(f"[setup] missing static checkpoint at {checkpoint_path}; keeping current policy initialization")
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = maybe_apply_checkpoint_noise_to_state_dict(
            strip_module_prefix(checkpoint["policy"]),
            checkpoint_path=checkpoint_path,
            state_label="policy",
        )
    else:
        policy_state = maybe_apply_checkpoint_noise_to_state_dict(
            strip_module_prefix(checkpoint),
            checkpoint_path=checkpoint_path,
            state_label="checkpoint",
        )
    policy.load_state_dict(policy_state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def maybe_load_training_checkpoint(
    checkpoint_path: Optional[str],
    policy: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, int, float, float]:
    if not checkpoint_path:
        return 1, 0, -1.0, -1.0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
        raise ValueError(f"training checkpoint is invalid: {checkpoint_path}")
    policy.load_state_dict(strip_module_prefix(checkpoint["policy"]), strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_update = int(checkpoint.get("update", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_success_once = float(checkpoint.get("best_success_once", -1.0))
    best_success_at_end = float(checkpoint.get("best_success_at_end", -1.0))
    return start_update, global_step, best_success_once, best_success_at_end


def restore_small_policy_dtypes(policy: reference.EdgeVLAActorCritic, device: torch.device) -> None:
    policy.to(device=device)
    policy.device = device
    policy.vla.to(device=device, dtype=torch.bfloat16)
    policy.state_projector.to(device=device, dtype=torch.float32)
    policy.context_projector.to(device=device, dtype=torch.float32)
    policy.actor_head.to(device=device, dtype=torch.float32)
    policy.value_head.to(device=device, dtype=torch.float32)
    policy._buffers["action_bin_centers"] = policy.action_bin_centers.to(device=device, dtype=torch.float32)


class SeparateEdgeVLAActorCritic(nn.Module):
    def __init__(self, base_policy: reference.EdgeVLAActorCritic, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.actor_policy = deepcopy(base_policy)
        self.critic_policy = deepcopy(base_policy)
        self.eval_micro_batch_size = int(getattr(base_policy, "eval_micro_batch_size", 32))
        self.actor_policy.eval_micro_batch_size = self.eval_micro_batch_size
        self.critic_policy.eval_micro_batch_size = self.eval_micro_batch_size
        restore_small_policy_dtypes(self.actor_policy, device)
        restore_small_policy_dtypes(self.critic_policy, device)
        self.configure_trainable_modules(train_backbone=True)

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self.actor_policy.configure_trainable_modules(train_backbone=train_backbone)
        self.critic_policy.configure_trainable_modules(train_backbone=train_backbone)
        for parameter in self.actor_policy.value_head.parameters():
            parameter.requires_grad = False
        for parameter in self.critic_policy.actor_head.parameters():
            parameter.requires_grad = False

    def get_action_and_value(
        self,
        rgbs,
        states,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, logprob, entropy, _, bins = self.actor_policy.get_action_and_value(
            rgbs=rgbs,
            states=states,
            action_bins=action_bins,
            deterministic=deterministic,
        )
        value = self.critic_policy.get_value(rgbs=rgbs, states=states)
        return action, logprob, entropy, value, bins

    def get_value(self, rgbs, states) -> torch.Tensor:
        return self.critic_policy.get_value(rgbs=rgbs, states=states)

    def predict_action(self, rgb, state) -> Any:
        self.eval()
        with torch.no_grad():
            action, _, _, _, _ = self.get_action_and_value(
                rgbs=rgb.unsqueeze(0) if isinstance(rgb, torch.Tensor) else reference.np.expand_dims(rgb, axis=0),
                states=state[None, ...] if getattr(state, "ndim", 1) == 1 else state,
                deterministic=True,
            )
        return action[0].detach().cpu().numpy().astype("float32")

    def forward(
        self,
        rgbs,
        states,
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
        raise ValueError(f"Unsupported forward mode: {mode}")


def build_optimizer(args: Args, policy: SeparateEdgeVLAActorCritic) -> optim.Optimizer:
    param_groups = []

    def append_group(module: nn.Module, lr: float, group_name: str) -> None:
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr, "group_name": group_name})

    append_group(policy.actor_policy.vla, args.backbone_learning_rate, "actor_vla")
    append_group(policy.critic_policy.vla, args.backbone_learning_rate, "critic_vla")
    append_group(policy.actor_policy.state_projector, args.state_learning_rate, "actor_state_projector")
    append_group(policy.critic_policy.state_projector, args.state_learning_rate, "critic_state_projector")
    append_group(policy.actor_policy.context_projector, args.head_learning_rate, "actor_context_projector")
    append_group(policy.critic_policy.context_projector, args.head_learning_rate, "critic_context_projector")
    append_group(policy.actor_policy.actor_head, args.head_learning_rate, "actor_head")
    append_group(policy.critic_policy.value_head, args.value_head_learning_rate, "critic_value_head")

    if not param_groups:
        raise ValueError("no trainable parameters found for optimizer")
    return optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def materialize_fbs_caches(
    args: Args,
    device: torch.device,
    fbs_policy: reference.EdgeVLAActorCritic,
) -> None:
    warmup_envs = reference.make_vector_env(args, device, 1, record_metrics=False)
    try:
        obs, _ = warmup_envs.reset(seed=args.seed)
        rgbs = reference.extract_rgb_batch_from_obs(obs)
        states = reference.extract_cabinet_state_batch_from_obs(obs)
        fbs_policy.eval()
        with torch.no_grad():
            fbs_policy.get_action_and_value(rgbs=rgbs, states=states, deterministic=True)
    finally:
        warmup_envs.close()


def build_static_student_policy(args: Args, device: torch.device) -> reference.EdgeVLAActorCritic:
    base_policy = reference.EdgeVLAActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        env_action_dim=args.env_action_dim,
        controlled_action_indices=args.controlled_action_indices,
    ).to(device)
    fbs_policy = convert_to_fbs_model(base_policy, device).to(device)
    load_policy_state_from_checkpoint(args.static_model_checkpoint, fbs_policy)
    set_sparsity(fbs_policy, args.static_sparsity)
    materialize_fbs_caches(args, device, fbs_policy)
    static_policy = generate_static_small_model(fbs_policy, device=device, dtype=torch.bfloat16)
    restore_small_policy_dtypes(static_policy, device)
    del base_policy
    del fbs_policy
    torch.cuda.empty_cache()
    return static_policy


def build_policy(args: Args, device: torch.device) -> SeparateEdgeVLAActorCritic:
    static_policy = build_static_student_policy(args, device)
    policy = SeparateEdgeVLAActorCritic(static_policy, device)
    policy.configure_trainable_modules(train_backbone=not args.freeze_vla_backbone)
    policy.eval_micro_batch_size = args.eval_micro_batch_size
    policy.actor_policy.eval_micro_batch_size = args.eval_micro_batch_size
    policy.critic_policy.eval_micro_batch_size = args.eval_micro_batch_size
    del static_policy
    torch.cuda.empty_cache()
    return policy


def save_training_checkpoint(
    checkpoint_path: Path,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    update: int,
    global_step: int,
    best_success_once: float,
    best_success_at_end: float,
) -> None:
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "global_step": global_step,
            "best_success_once": best_success_once,
            "best_success_at_end": best_success_at_end,
        },
        checkpoint_path,
    )


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


def copy_run_metadata(output_dir: Path, args: Args) -> None:
    code_dir = mkdir(output_dir / "code")
    shutil.copyfile(__file__, code_dir / "online_rl.py")
    save_json(output_dir / "args.json", asdict(args))


def save_summary(output_dir: Path, payload: Dict[str, Any]) -> None:
    save_json(output_dir / DEFAULT_SUMMARY_NAME, payload)


def plot_success_time_curve(output_dir: Path, metrics_history: List[Dict[str, Any]]) -> None:
    # EdgeVLA's base PPO implementation does not ship the hand-workload helper.
    return


def summarize_episode_metrics(episode_metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, Tuple[float, int]]:
    summary = {}
    for key, values in episode_metrics.items():
        if not values:
            continue
        cat = torch.cat(values)
        summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
    return summary


def should_early_stop_zero_success(
    metrics_history: List[Dict[str, Any]],
    threshold_minutes: float,
) -> Tuple[bool, float]:
    if not metrics_history:
        return False, 0.0
    latest_elapsed_minutes = float(metrics_history[-1].get("elapsed_hours", 0.0)) * 60.0
    max_success = 0.0
    for metric in metrics_history:
        for metric_name in (
            "train_success_once",
            "train_success_at_end",
            "eval_success_once",
            "eval_success_at_end",
        ):
            value = metric.get(metric_name)
            if value is not None:
                max_success = max(max_success, float(value))
    return latest_elapsed_minutes >= threshold_minutes and max_success <= 0.0, max_success


def train(args: Args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    reference.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    continual_env_schedule = build_continual_env_schedule(args)
    if continual_env_schedule is not None:
        args = replace(args, env_id=continual_env_schedule.env_ids[0])
        print(
            "[setup] continual env schedule enabled: "
            f"first_env={args.env_id}, "
            f"envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )
    args = configure_env_contract(args, device)

    global_batch_size = args.num_envs * args.num_steps
    local_minibatch_size = max(1, global_batch_size // args.num_minibatches)
    num_updates = max(1, args.total_timesteps // global_batch_size)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = mkdir(Path(args.output_dir) / run_name)
    copy_run_metadata(output_dir, args)
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] device={device}")
    print(
        "[setup] flare core switches: "
        f"update_epochs={args.update_epochs}, ent_coef={args.ent_coef}, feature_sharing=disabled"
    )

    policy = build_policy(args, device)
    optimizer = build_optimizer(args, policy)
    start_update, global_step, best_success_once, best_success_at_end = maybe_load_training_checkpoint(
        args.resume_from,
        policy,
        optimizer,
    )

    if args.run_setup_smoke:
        reference.run_vla_inference_smoke(args, device, output_dir, policy=policy)

    runtime_args = args
    initial_env_id = runtime_args.env_id
    current_env_index = 0
    current_env_id = runtime_args.env_id
    envs, eval_envs, test_video_envs = build_runtime_envs(
        runtime_args,
        device,
        output_dir,
        current_env_id,
        current_env_index,
    )

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
    stopped_early_zero_success = False
    final_eval_metrics: Dict[str, float] = {}

    def maybe_switch_envs() -> Tuple[bool, bool, Optional[float]]:
        nonlocal envs, eval_envs, test_video_envs, next_obs, next_done
        nonlocal runtime_args, current_env_id, current_env_index
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
        runtime_args = replace(args, env_id=current_env_id)
        print(
            f"[setup] switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        close_envs(envs, eval_envs, test_video_envs)
        envs = None
        eval_envs = None
        test_video_envs = None
        clear_torch_cuda_cache()
        envs, eval_envs, test_video_envs = build_runtime_envs(
            runtime_args,
            device,
            output_dir,
            current_env_id,
            current_env_index,
        )
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        eval_envs.reset(seed=args.seed + current_env_index)
        if test_video_envs is not None:
            test_video_envs.reset(seed=args.seed + current_env_index)
        next_done = torch.zeros(args.num_envs, device=device)
        return True, False, elapsed_minutes

    initial_eval_metrics = reference.evaluate_policy(policy, eval_envs, args.eval_episodes)
    initial_metric = {
        "update": 0,
        "global_step": global_step,
        "current_env_id": current_env_id,
        "current_env_index": current_env_index,
        "elapsed_hours": 0.0,
        "reward_mean": float("nan"),
        "return_mean": float("nan"),
        "gae_return_mean": float("nan"),
        "value_mean": float("nan"),
        "explained_variance": float("nan"),
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "pg_loss": 0.0,
        "v_loss": 0.0,
        "entropy": 0.0,
    }
    initial_metric.update({f"eval_{key}": value for key, value in initial_eval_metrics.items()})
    metrics_history.append(initial_metric)
    save_json(output_dir / "latest_metrics.json", initial_metric)
    save_metrics_history(output_dir, metrics_history)
    plot_metrics_history(output_dir, metrics_history)
    plot_success_time_curve(output_dir, metrics_history)
    initial_success_once = float(initial_eval_metrics.get("success_once", initial_eval_metrics.get("success", 0.0)))
    initial_success_at_end = float(initial_eval_metrics.get("success_at_end", 0.0))
    best_success_once = max(best_success_once, initial_success_once)
    best_success_at_end = max(best_success_at_end, initial_success_at_end)
    save_training_checkpoint(
        output_dir / "best_policy.pt",
        policy,
        optimizer,
        0,
        global_step,
        best_success_once,
        best_success_at_end,
    )
    print(f"[eval] initial_eval={initial_eval_metrics}")

    for update in range(start_update, num_updates + 1):
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if switched_env:
            print(f"[train] switched to env[{current_env_index}]={current_env_id} before update {update}")
            switched_eval_metrics = reference.evaluate_policy(policy, eval_envs, args.eval_episodes)
            print(f"[eval] post_switch_eval={switched_eval_metrics}")
        if should_stop_for_schedule:
            stop_reason = "continual_schedule_end"
            print(
                f"[train] reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes before update {update}"
            )
            break

        policy.eval()
        final_values.zero_()
        rollout_rgbs: List[torch.Tensor] = []
        rollout_states: List[Any] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []
        rollout_start_time = time.perf_counter()
        abort_during_rollout = False
        abort_reason: Optional[str] = None
        rollout_steps_completed = 0

        for step in range(args.num_steps):
            global_step += args.num_envs
            step_rgbs = reference.extract_rgb_batch_from_obs(next_obs)
            step_states = reference.extract_cabinet_state_batch_from_obs(next_obs)
            rollout_rgbs.append(step_rgbs.clone())
            rollout_states.append(step_states.copy())
            dones_buf[step] = next_done

            action, logprob, _, value, action_bins = reference.batched_get_action_and_value_no_grad(
                policy,
                step_rgbs,
                step_states,
                micro_batch_size=args.rollout_micro_batch_size,
            )
            action_bins_buf[step] = action_bins
            logprobs_buf[step] = logprob
            values_buf[step] = value

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            truncation_mask = truncations.to(torch.bool)
            next_done = (terminations | truncations).to(torch.float32)
            rewards_buf[step] = reward.view(-1)
            partial_reward_means.append(float(rewards_buf[: step + 1].mean().item()))
            rollout_steps_completed = step + 1

            if (
                (step + 1) % args.rollout_progress_log_interval == 0
                or step == 0
                or step + 1 == args.num_steps
            ):
                elapsed_hours = runtime_tracker.current_hours(extra_active_seconds=time.perf_counter() - rollout_start_time)
                reward_mean_so_far = float(partial_reward_means[-1])
                logged_partial_reward_means.append(reward_mean_so_far)
                print(
                    f"[rollout] update={update}/{num_updates} step={step + 1}/{args.num_steps} "
                    f"reward_mean_so_far={reward_mean_so_far:.4f} elapsed_h={elapsed_hours:.2f}"
                )
                reference.save_rollout_progress(
                    output_dir=output_dir,
                    update=update,
                    num_updates=num_updates,
                    rollout_step=step + 1,
                    num_steps=args.num_steps,
                    elapsed_hours=elapsed_hours,
                    partial_reward_means=logged_partial_reward_means,
                )
                partial_success_metrics = reference.gather_metric_summary(
                    summarize_episode_metrics(train_episode_metrics)
                )
                partial_max_success = max(
                    float(partial_success_metrics.get("train_success_once", 0.0)),
                    float(partial_success_metrics.get("train_success_at_end", 0.0)),
                )
                if elapsed_hours >= args.max_runtime_hours:
                    abort_during_rollout = True
                    abort_reason = "time_limit"
                    break
                if elapsed_hours * 60.0 >= args.early_stop_zero_success_minutes and partial_max_success <= 0.0:
                    abort_during_rollout = True
                    abort_reason = "early_stop_zero_success"
                    break

            done_mask, episode_metrics = reference.get_completed_episode_metrics(infos)
            if done_mask is not None:
                if done_mask.any():
                    for key, value_tensor in episode_metrics.items():
                        train_episode_metrics[key].append(value_tensor[done_mask].float().detach().cpu())
                if "final_observation" in infos and truncation_mask.any():
                    final_obs = infos["final_observation"]
                    bootstrap_idx = truncation_mask.detach().cpu().numpy().astype(bool)
                    final_rgbs = reference.extract_rgb_batch_from_obs(final_obs)[bootstrap_idx]
                    final_states = reference.extract_cabinet_state_batch_from_obs(final_obs)[bootstrap_idx]
                    final_values[step, truncation_mask] = reference.batched_get_value_no_grad(
                        policy,
                        final_rgbs,
                        final_states,
                        micro_batch_size=args.eval_micro_batch_size,
                    ).view(-1)

        if abort_during_rollout:
            metric = {
                "update": update,
                "global_step": global_step,
                "current_env_id": current_env_id,
                "current_env_index": current_env_index,
                "reward_mean": float(rewards_buf[:rollout_steps_completed].mean().item()),
                "return_mean": float("nan"),
                "gae_return_mean": float("nan"),
                "value_mean": float(values_buf[:rollout_steps_completed].mean().item()),
                "explained_variance": float("nan"),
                "approx_kl": 0.0,
                "clipfrac": 0.0,
                "pg_loss": 0.0,
                "v_loss": 0.0,
                "entropy": 0.0,
                "elapsed_hours": runtime_tracker.current_hours(extra_active_seconds=time.perf_counter() - rollout_start_time),
                "partial_rollout_only": True,
                "rollout_steps_completed": rollout_steps_completed,
            }
            metric.update(reference.gather_metric_summary(summarize_episode_metrics(train_episode_metrics)))
            metrics_history.append(metric)
            save_json(output_dir / "latest_metrics.json", metric)
            save_metrics_history(output_dir, metrics_history)
            plot_metrics_history(output_dir, metrics_history)
            plot_success_time_curve(output_dir, metrics_history)
            print(f"[train] aborting during rollout update={update}/{num_updates} reason={abort_reason}")
            stop_reason = abort_reason or stop_reason
            stopped_early_zero_success = abort_reason == "early_stop_zero_success"
            break

        rollout_time = time.perf_counter() - rollout_start_time
        runtime_tracker.add_active_seconds(rollout_time)

        with torch.no_grad():
            next_value = reference.batched_get_value_no_grad(
                policy,
                reference.extract_rgb_batch_from_obs(next_obs),
                reference.extract_cabinet_state_batch_from_obs(next_obs),
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
        b_states = reference.np.concatenate(rollout_states, axis=0)
        b_action_bins = action_bins_buf.reshape(-1, args.action_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = reference.normalize_advantages(advantages, device).reshape(-1)
        b_returns = returns.reshape(-1)
        ev = reference.explained_variance(values_buf, returns, device)

        inds = reference.np.arange(global_batch_size)
        approx_kl = 0.0
        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_value = 0.0
        clipfrac_value = 0.0
        stopped_on_minibatch_kl = False
        skipped_updates_on_kl = 0

        policy.train()
        update_start_time = time.perf_counter()
        for _ in range(args.update_epochs):
            reference.np.random.shuffle(inds)
            epoch_stats = defaultdict(list)
            for start in range(0, global_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = inds[start:end]
                mb_stats = reference.ppo_update_with_micro_batches(
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
                minibatch_kl = float(mb_stats.get("approx_kl", 0.0))
                if mb_stats.get("skipped_on_kl", 0.0) > 0.0 or minibatch_kl > args.target_kl * args.minibatch_target_kl_factor:
                    stopped_on_minibatch_kl = True
                    break

            approx_kl = float(reference.np.mean(epoch_stats["approx_kl"])) if epoch_stats["approx_kl"] else 0.0
            clipfrac_value = float(reference.np.mean(epoch_stats["clipfrac"])) if epoch_stats["clipfrac"] else 0.0
            pg_loss_value = float(reference.np.mean(epoch_stats["pg_loss"])) if epoch_stats["pg_loss"] else 0.0
            v_loss_value = float(reference.np.mean(epoch_stats["v_loss"])) if epoch_stats["v_loss"] else 0.0
            entropy_value = float(reference.np.mean(epoch_stats["entropy"])) if epoch_stats["entropy"] else 0.0
            if stopped_on_minibatch_kl or approx_kl > args.target_kl:
                break

        update_time = time.perf_counter() - update_start_time
        runtime_tracker.add_active_seconds(update_time)

        metric = {
            "update": update,
            "global_step": global_step,
            "current_env_id": current_env_id,
            "current_env_index": current_env_index,
            "reward_mean": float(rewards_buf.mean().item()),
            "return_mean": float(returns.mean().item()),
            "gae_return_mean": float(returns.mean().item()),
            "value_mean": float(values_buf.mean().item()),
            "explained_variance": float(ev),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac_value,
            "pg_loss": pg_loss_value,
            "v_loss": v_loss_value,
            "entropy": entropy_value,
            "stopped_on_minibatch_kl": stopped_on_minibatch_kl,
            "skipped_updates_on_kl": skipped_updates_on_kl,
            "elapsed_hours": runtime_tracker.current_hours(),
        }
        metric.update(reference.gather_metric_summary(summarize_episode_metrics(train_episode_metrics)))

        if update % args.eval_every_updates == 0 or update == num_updates:
            eval_metrics = reference.evaluate_policy(policy, eval_envs, args.eval_episodes)
            metric.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            if test_video_envs is not None:
                reference.evaluate_policy(
                    policy,
                    test_video_envs,
                    min(args.test_video_episodes, max(1, args.test_video_num_envs)),
                )
            success_once = float(eval_metrics.get("success_once", eval_metrics.get("success", 0.0)))
            success_at_end = float(eval_metrics.get("success_at_end", 0.0))
            if success_once >= best_success_once:
                best_success_once = success_once
                save_training_checkpoint(
                    output_dir / "best_policy.pt",
                    policy,
                    optimizer,
                    update,
                    global_step,
                    best_success_once,
                    max(best_success_at_end, success_at_end),
                )
            if success_at_end >= best_success_at_end:
                best_success_at_end = success_at_end
                save_training_checkpoint(
                    output_dir / "best_success_at_end.pt",
                    policy,
                    optimizer,
                    update,
                    global_step,
                    best_success_once,
                    best_success_at_end,
                )

        metrics_history.append(metric)
        print(
            f"[train] update={update}/{num_updates} step={global_step} "
            f"reward={metric['reward_mean']:.4f} gae_return={metric['gae_return_mean']:.4f} "
            f"approx_kl={metric['approx_kl']:.5f} "
            f"eval_success_once={metric.get('eval_success_once', float('nan')):.4f} "
            f"elapsed_h={metric['elapsed_hours']:.2f}"
        )
        save_json(output_dir / "latest_metrics.json", metric)
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
        plot_success_time_curve(output_dir, metrics_history)
        if update % 10 == 0 or update == num_updates:
            save_training_checkpoint(
                output_dir / "latest_policy.pt",
                policy,
                optimizer,
                update,
                global_step,
                best_success_once,
                best_success_at_end,
            )

        reached_zero_success_stop, max_success_observed = should_early_stop_zero_success(
            metrics_history,
            threshold_minutes=args.early_stop_zero_success_minutes,
        )
        if reached_zero_success_stop:
            stop_reason = "early_stop_zero_success"
            stopped_early_zero_success = True
            print(
                "[train] reached zero-success early stop: "
                f"{args.early_stop_zero_success_minutes:.1f} minutes elapsed with max_success={max_success_observed:.4f}"
            )
            break

        if metric["elapsed_hours"] >= args.max_runtime_hours:
            stop_reason = "time_limit"
            print(f"[train] reached time limit: {metric['elapsed_hours']:.2f}h >= {args.max_runtime_hours:.2f}h")
            break

        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if switched_env:
            print(f"[train] switched to env[{current_env_index}]={current_env_id} after update {update}")
        if should_stop_for_schedule:
            stop_reason = "continual_schedule_end"
            print(
                f"[train] reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes after update {update}"
            )
            break

    final_eval_metrics = reference.evaluate_policy(policy, eval_envs, args.eval_episodes)
    save_json(output_dir / "final_eval_metrics.json", final_eval_metrics)
    save_metrics_history(output_dir, metrics_history)
    plot_metrics_history(output_dir, metrics_history)
    plot_success_time_curve(output_dir, metrics_history)
    save_summary(
        output_dir,
        {
            "env_id": args.env_id,
            "initial_env_id": initial_env_id,
            "final_env_id": current_env_id,
            "continual_env_ids": None if continual_env_schedule is None else list(continual_env_schedule.env_ids),
            "continual_env_change_time_points": (
                None if continual_env_schedule is None else list(continual_env_schedule.change_time_points)
            ),
            "run_name": run_name,
            "stop_reason": stop_reason,
            "stopped_early_zero_success": stopped_early_zero_success,
            "best_success_once": best_success_once,
            "best_success_at_end": best_success_at_end,
            "global_step": global_step,
            "num_updates_completed": metrics_history[-1]["update"] if metrics_history else 0,
            "final_eval_metrics": final_eval_metrics,
            "eval_success_once_summary": summarize_success_series(metrics_history, "eval_success_once"),
            "eval_success_at_end_summary": summarize_success_series(metrics_history, "eval_success_at_end"),
        },
    )

    close_envs(envs, eval_envs, test_video_envs)
    envs = None
    eval_envs = None
    test_video_envs = None
    clear_torch_cuda_cache()


def main() -> None:
    args = parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")
    try:
        train(args)
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
