import argparse
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

sys.path.append(".")

THIS_DIR = Path(__file__).resolve().parent
EDGEVLA_IMPL_DIR = THIS_DIR.parent / "model_impl"
REPO_ROOT = THIS_DIR.parents[2]
for path in (THIS_DIR, EDGEVLA_IMPL_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train.tinyvla.model_impl.online_rl_open_cabinet_drawer as reference
from ours.libs.train_with_fbs.lib import set_sparsity
from train.tinyvla.ours.model_with_fbs import convert_to_fbs_model
from train.vla_adapter_new.model_impl.online_rl import (
    broadcast_object,
    cleanup_runtime,
    distributed_barrier,
    distributed_mean,
    gather_metric_summary,
    init_runtime,
    is_distributed,
    is_main_process,
    iter_slices,
    mkdir,
    parse_bool,
    save_json,
    save_metrics_history,
    set_seed,
    strip_module_prefix,
)


DEFAULT_MODEL_DIR = reference.DEFAULT_MODEL_DIR
DEFAULT_WORKDIR = "train/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs"
DEFAULT_TEACHER_CHECKPOINT = "auto"
DEFAULT_SUMMARY_NAME = "bc_training_summary.json"
TEACHER_SEARCH_DIR = Path("train/tinyvla/model_impl/outputs/ppo_open_cabinet_drawer")
EVAL_SPARSITIES = (0.0, 0.4, 0.8)


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
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT
    student_init_policy: str = "teacher"
    resume_from: Optional[str] = None
    num_envs: int = 128
    num_eval_envs: int = 8
    num_steps: int = 100
    total_timesteps: int = 100_000_000
    num_minibatches: int = 16
    update_epochs: int = 2
    backbone_learning_rate: float = 3e-5
    head_learning_rate: float = 3e-5
    state_learning_rate: float = 3e-5
    value_head_learning_rate: float = 3e-5
    weight_decay: float = 1e-6
    max_grad_norm: float = 0.5
    eval_episodes: int = 50
    eval_every_updates: int = 20
    max_episode_steps: Optional[int] = None
    cuda_device: str = "0"
    save_video: bool = False
    test_video_num_envs: int = 4
    test_video_episodes: int = 4
    run_setup_smoke: bool = False
    max_runtime_hours: float = 50.0
    rollout_micro_batch_size: int = 256
    eval_micro_batch_size: int = 256
    update_micro_batch_size: int = 32
    action_dim: int = 8
    env_action_dim: int = 13
    state_dim: int = 44
    run_name: Optional[str] = None
    freeze_vla_backbone: bool = False
    early_stop_zero_success_minutes: float = 45_000.0


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


def backup_run_sources(output_dir: Path) -> None:
    code_dir = mkdir(output_dir / "code")
    sources = {
        "pretrain_with_fbs_bc.py": Path(__file__).resolve(),
        "model_with_fbs.py": THIS_DIR / "model_with_fbs.py",
        "online_rl_open_cabinet_drawer.py": Path(reference.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {"source": str(source_path), "backup": str(destination)}
    save_json(code_dir / "source_manifest.json", manifest)


def resolve_teacher_checkpoint(path: str) -> str:
    if path != "auto":
        if not Path(path).is_file():
            raise FileNotFoundError(f"teacher checkpoint not found: {path}")
        return path

    candidates = sorted(TEACHER_SEARCH_DIR.glob("*/best_policy.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No teacher checkpoint found under {TEACHER_SEARCH_DIR}; pass --teacher-checkpoint explicitly"
        )
    return str(candidates[-1])


def build_optimizer(args: Args, policy: reference.EdgeVLAActorCritic) -> optim.Optimizer:
    param_groups = [
        {
            "params": list(policy.vla.parameters()),
            "lr": 0.0 if args.freeze_vla_backbone else args.backbone_learning_rate,
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


def load_policy_checkpoint(
    checkpoint_path: str,
    raw_policy: reference.EdgeVLAActorCritic,
    strict: bool = True,
) -> Tuple[int, int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=raw_policy.device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
        start_update = int(checkpoint.get("update", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        best_success = float(checkpoint.get("best_success_once", -1.0))
    else:
        policy_state = strip_module_prefix(checkpoint)
        start_update = 1
        global_step = 0
        optimizer_step = 0
        best_success = -1.0
    raw_policy.load_state_dict(policy_state, strict=strict)
    return start_update, global_step, optimizer_step, best_success


def maybe_load_fbs_checkpoint(
    checkpoint_path: Optional[str],
    raw_policy: reference.EdgeVLAActorCritic,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, int, int, float]:
    if not checkpoint_path:
        return 1, 0, 0, -1.0
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=raw_policy.device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
        start_update = int(checkpoint.get("update", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        best_success = float(checkpoint.get("best_success_once", -1.0))
    else:
        policy_state = strip_module_prefix(checkpoint)
        start_update = 1
        global_step = 0
        optimizer_step = 0
        best_success = -1.0

    raw_policy.load_state_dict(policy_state, strict=True)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return start_update, global_step, optimizer_step, best_success


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def average_trainable_gradients(module: nn.Module) -> None:
    if not is_distributed():
        return
    world_size = dist.get_world_size()
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def sync_trainable_parameters(module: nn.Module) -> None:
    if not is_distributed():
        return
    for parameter in module.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)


def sparsity_tag(sparsity: float) -> str:
    return f"{sparsity:.1f}".replace(".", "_")


def get_current_sparsity(model: nn.Module) -> Optional[float]:
    for module in model.modules():
        if "KTakesAll" in module.__class__.__name__:
            return float(module.k)
    return None


def evaluate_policy_at_sparsities(
    policy: reference.EdgeVLAActorCritic,
    envs,
    target_episodes: int,
    sparsities: Tuple[float, ...] = EVAL_SPARSITIES,
) -> Dict[str, float]:
    previous_sparsity = get_current_sparsity(policy)
    flat_metrics: Dict[str, float] = {}
    success_once_values: List[float] = []
    success_at_end_values: List[float] = []

    try:
        for sparsity in sparsities:
            set_sparsity(policy, sparsity)
            eval_metrics = reference.evaluate_policy(policy, envs, target_episodes)
            prefix = f"eval_sparsity_{sparsity_tag(sparsity)}"
            for key, value in eval_metrics.items():
                flat_metrics[f"{prefix}_{key}"] = float(value)
            success_once = float(eval_metrics.get("success_once", eval_metrics.get("success", 0.0)))
            success_once_values.append(success_once)
            if "success_at_end" in eval_metrics:
                success_at_end_values.append(float(eval_metrics["success_at_end"]))
            print(f"[eval] sparsity={sparsity:.2f} eval_success_once={success_once:.4f}")
    finally:
        if previous_sparsity is not None:
            set_sparsity(policy, previous_sparsity)

    if success_once_values:
        flat_metrics["eval_success_once"] = float(np.mean(success_once_values))
        flat_metrics["eval_success_once_max"] = float(np.max(success_once_values))
    if success_at_end_values:
        flat_metrics["eval_success_at_end"] = float(np.mean(success_at_end_values))
        flat_metrics["eval_success_at_end_max"] = float(np.max(success_at_end_values))
    return flat_metrics


def unflatten_eval_metrics_by_sparsity(
    flat_metrics: Dict[str, float],
    sparsities: Tuple[float, ...] = EVAL_SPARSITIES,
) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for sparsity in sparsities:
        prefix = f"eval_sparsity_{sparsity_tag(sparsity)}_"
        entry = {key[len(prefix) :]: value for key, value in flat_metrics.items() if key.startswith(prefix)}
        if entry:
            nested[f"{sparsity:.1f}"] = entry
    aggregate = {
        key: value
        for key, value in flat_metrics.items()
        if key in {"eval_success_once", "eval_success_once_max", "eval_success_at_end", "eval_success_at_end_max"}
    }
    if aggregate:
        nested["aggregate"] = aggregate
    return nested


def summarize_episode_metrics(train_episode_metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, Tuple[float, int]]:
    summary = {}
    for key, values in train_episode_metrics.items():
        if not values:
            continue
        cat = torch.cat(values)
        summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
    return summary


def collect_teacher_rollout(
    args: Args,
    teacher_policy: reference.EdgeVLAActorCritic,
    envs,
    next_obs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, float], int]:
    rollout_rgbs: List[torch.Tensor] = []
    rollout_states: List[torch.Tensor] = []
    rollout_action_bins: List[torch.Tensor] = []
    rewards = []
    train_episode_metrics = defaultdict(list)

    for _ in range(args.num_steps):
        step_rgbs = reference.extract_rgb_batch_from_obs(next_obs)
        step_states_np = reference.extract_cabinet_state_batch_from_obs(next_obs)
        step_states = torch.from_numpy(step_states_np).to(dtype=torch.float32)
        teacher_action, _, _, _, teacher_action_bins = reference.batched_get_action_and_value_no_grad(
            teacher_policy,
            step_rgbs,
            step_states_np,
            micro_batch_size=args.rollout_micro_batch_size,
            deterministic=True,
        )
        rollout_rgbs.append(step_rgbs.clone())
        rollout_states.append(step_states.clone())
        rollout_action_bins.append(teacher_action_bins.detach().cpu())

        next_obs, reward, _, _, infos = envs.step(teacher_action)
        rewards.append(reward.detach().float())

        done_mask, episode_metrics = reference.get_completed_episode_metrics(infos)
        if done_mask is not None and done_mask.any():
            for key, value_tensor in episode_metrics.items():
                train_episode_metrics[key].append(value_tensor[done_mask].float().detach().cpu())

    reward_tensor = torch.stack(rewards, dim=0)
    rollout_summary = gather_metric_summary(summarize_episode_metrics(train_episode_metrics))
    rollout_summary["reward_mean"] = distributed_mean(float(reward_tensor.mean().item()), teacher_policy.device)
    rollout_summary["teacher_return_mean"] = distributed_mean(
        float(reward_tensor.sum(dim=0).mean().item()),
        teacher_policy.device,
    )
    return (
        torch.cat(rollout_rgbs, dim=0),
        torch.cat(rollout_states, dim=0),
        torch.cat(rollout_action_bins, dim=0),
        next_obs,
        rollout_summary,
        reward_tensor.shape[0],
    )


def run_bc_update(
    args: Args,
    policy: reference.EdgeVLAActorCritic,
    optimizer: optim.Optimizer,
    b_rgbs: torch.Tensor,
    b_states: torch.Tensor,
    b_action_bins: torch.Tensor,
    local_minibatch_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    total = b_action_bins.shape[0]
    indices = np.arange(total)
    epoch_losses: List[float] = []
    minibatch_loss_points: List[Dict[str, float]] = []

    policy.train()
    for epoch_idx in range(args.update_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        minibatch_idx_in_epoch = 0

        for start, end in iter_slices(total, local_minibatch_size):
            if is_main_process():
                if minibatch_idx_in_epoch % 4 == 0:
                    cur_sparsity = 0.0
                elif 1 <= minibatch_idx_in_epoch % 4 <= 2:
                    cur_sparsity = float(np.random.uniform(0.0, 0.8))
                else:
                    cur_sparsity = 0.8
            else:
                cur_sparsity = None
            cur_sparsity = broadcast_object(cur_sparsity)
            set_sparsity(policy, cur_sparsity)

            minibatch_idx_in_epoch += 1
            minibatch_inds = indices[start:end]
            minibatch_total = max(1, len(minibatch_inds))
            minibatch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for micro_start, micro_end in iter_slices(minibatch_total, args.update_micro_batch_size):
                micro_inds = minibatch_inds[micro_start:micro_end]
                micro_weight = (micro_end - micro_start) / minibatch_total
                _, log_prob, _, _, _ = reference.policy_get_action_and_value(
                    policy,
                    b_rgbs[micro_inds],
                    b_states[micro_inds],
                    b_action_bins[micro_inds].to(policy.device, dtype=torch.long),
                )
                bc_loss = (-log_prob.mean()) / policy.policy_action_dim
                (bc_loss * micro_weight).backward()
                minibatch_loss += float(bc_loss.detach().item()) * micro_weight

            average_trainable_gradients(policy)
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_loss += minibatch_loss * (minibatch_total / total)
            minibatch_loss_points.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "minibatch_in_epoch": float(minibatch_idx_in_epoch),
                    "loss": float(minibatch_loss),
                }
            )

        epoch_losses.append(epoch_loss)

    return (
        {
            "bc_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "last_epoch_bc_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "optimizer_steps_in_update": float(len(minibatch_loss_points)),
        },
        minibatch_loss_points,
    )


def save_minibatch_loss_history(output_dir: Path, minibatch_loss_history: List[Dict[str, Any]]) -> None:
    save_json(output_dir / "minibatch_loss_history.json", {"history": minibatch_loss_history})


def plot_bc_metrics_history(
    output_dir: Path,
    metrics_history: List[Dict[str, Any]],
    minibatch_loss_history: List[Dict[str, Any]],
) -> None:
    if not metrics_history and not minibatch_loss_history:
        return
    plots_dir = mkdir(output_dir / "plots")

    def series(metric_key: str) -> Tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []
        for metric in metrics_history:
            value = metric.get(metric_key)
            if value is None:
                continue
            xs.append(int(metric["update"]))
            ys.append(float(value))
        return xs, ys

    loss_xs = [int(point["optimizer_step"]) for point in minibatch_loss_history]
    loss_ys = [float(point["loss"]) for point in minibatch_loss_history]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(2, 2, 1)
    if loss_xs:
        ax.plot(loss_xs, loss_ys, marker="o", linewidth=1.5, markersize=3, label="minibatch_loss")
        ax.legend()
    ax.set_title("BC Loss")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    plot_specs = [
        ("Reward", "Reward", ["reward_mean", "teacher_return_mean"]),
        ("Eval Success Once", "Success Rate", [f"eval_sparsity_{sparsity_tag(s)}_success_once" for s in EVAL_SPARSITIES]),
        ("Eval Success At End", "Success Rate", [f"eval_sparsity_{sparsity_tag(s)}_success_at_end" for s in EVAL_SPARSITIES]),
    ]
    plotted_any = bool(loss_xs)
    for subplot_idx, (title, ylabel, keys) in enumerate(plot_specs, start=2):
        ax = plt.subplot(2, 2, subplot_idx)
        subplot_has_series = False
        for key in keys:
            xs, ys = series(key)
            if not xs:
                continue
            plotted_any = True
            subplot_has_series = True
            label = key
            if key.startswith("eval_sparsity_"):
                parts = key.split("_")
                if len(parts) >= 5:
                    label = f"sparsity={parts[2]}.{parts[3]}"
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if subplot_has_series:
            ax.legend()
    plt.tight_layout()
    if plotted_any:
        plt.savefig(plots_dir / "bc_overview.png", dpi=200)
    plt.close()


def should_early_stop_zero_success(
    metrics_history: List[Dict[str, Any]],
    threshold_minutes: float,
) -> Tuple[bool, float]:
    if threshold_minutes <= 0 or not metrics_history:
        return False, 0.0
    max_success = 0.0
    max_elapsed = 0.0
    for metric in metrics_history:
        max_elapsed = max(max_elapsed, float(metric.get("elapsed_hours", 0.0)))
        for key, value in metric.items():
            if "success" in key and isinstance(value, (int, float)):
                max_success = max(max_success, float(value))
    return max_success <= 0.0 and max_elapsed * 60.0 >= threshold_minutes, max_success


def train(args: Args) -> None:
    device, rank, world_size = init_runtime(args)
    set_seed(args.seed + rank)

    if args.num_envs % world_size != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by world_size={world_size}")
    local_num_envs = args.num_envs // world_size

    teacher_checkpoint = resolve_teacher_checkpoint(args.teacher_checkpoint) if is_main_process() else None
    teacher_checkpoint = broadcast_object(teacher_checkpoint)
    default_run_name = time.strftime("%Y%m%d-%H%M%S") if is_main_process() else None
    run_name = broadcast_object(args.run_name if is_main_process() and args.run_name else default_run_name)
    output_dir = mkdir(Path(args.output_dir) / run_name)

    inferred_env_action_dim, inferred_state_dim, controlled_action_indices = reference.inspect_env_contract(args, device)
    inferred_action_dim = len(controlled_action_indices)
    args.env_action_dim = inferred_env_action_dim
    args.action_dim = inferred_action_dim
    args.state_dim = inferred_state_dim

    if is_main_process():
        backup_run_sources(output_dir)
        args_payload = asdict(args)
        args_payload.update(
            {
                "world_size": world_size,
                "local_num_envs": local_num_envs,
                "resolved_teacher_checkpoint": teacher_checkpoint,
            }
        )
        save_json(output_dir / "args.json", args_payload)
        print(f"[setup] output_dir={output_dir}")
        print(f"[setup] teacher_checkpoint={teacher_checkpoint}")
        print(f"[setup] student_init_policy={args.student_init_policy}")
        print(f"[setup] resume_from={args.resume_from}")

    teacher_policy = reference.EdgeVLAActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        env_action_dim=args.env_action_dim,
        controlled_action_indices=controlled_action_indices,
    ).to(device)
    load_policy_checkpoint(teacher_checkpoint, teacher_policy, strict=True)
    freeze_module(teacher_policy)
    teacher_policy.eval_micro_batch_size = args.rollout_micro_batch_size

    student_policy = reference.EdgeVLAActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        env_action_dim=args.env_action_dim,
        controlled_action_indices=controlled_action_indices,
    ).to(device)
    if args.student_init_policy and args.student_init_policy.lower() != "none":
        init_path = teacher_checkpoint if args.student_init_policy == "teacher" else args.student_init_policy
        load_policy_checkpoint(init_path, student_policy, strict=True)
        if is_main_process():
            print(f"[setup] initialized student before FBS conversion from {init_path}")

    student_policy = convert_to_fbs_model(student_policy, device).to(device)
    student_policy.configure_trainable_modules(train_backbone=not args.freeze_vla_backbone)
    student_policy.eval_micro_batch_size = args.eval_micro_batch_size
    optimizer = build_optimizer(args, student_policy)
    start_update, global_step, optimizer_step, best_success_once = maybe_load_fbs_checkpoint(
        args.resume_from,
        student_policy,
        optimizer,
    )
    student_policy.vla.to(device=device, dtype=torch.bfloat16)

    if is_main_process():
        summary = student_policy.trainable_parameter_summary()
        summary_text = " ".join(f"{name}={trainable}/{total}" for name, (total, trainable) in summary.items())
        print(f"[setup] trainable_params {summary_text}")

    if args.run_setup_smoke and is_main_process() and world_size == 1:
        reference.run_vla_inference_smoke(args, device, output_dir, policy=student_policy)
    elif is_main_process():
        print("[setup] distributed mode: skip VLA smoke")
    distributed_barrier()

    sync_trainable_parameters(student_policy)
    if world_size > 1 and is_main_process():
        print("[setup] distributed mode: using manual gradient all-reduce")

    envs = reference.make_vector_env(args, device, local_num_envs, record_metrics=True)
    eval_envs = reference.make_vector_env(args, device, args.num_eval_envs, record_metrics=True) if is_main_process() else None
    test_video_envs = None
    if args.save_video and is_main_process():
        test_video_envs = reference.make_vector_env(
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
        raise ValueError(f"global minibatch size {global_minibatch_size} must be divisible by world_size={world_size}")
    local_minibatch_size = max(1, global_minibatch_size // world_size)
    num_updates = max(1, args.total_timesteps // global_batch_size)

    next_obs, _ = envs.reset(seed=args.seed + rank)
    metrics_history: List[Dict[str, Any]] = []
    minibatch_loss_history: List[Dict[str, Any]] = []
    train_start_time = time.time()
    stop_reason = "completed"
    final_eval_metrics: Dict[str, Any] = {}

    if is_main_process():
        print(
            f"[setup] world_size={world_size} local_num_envs={local_num_envs} num_updates={num_updates} "
            f"global_batch_size={global_batch_size} local_batch_size={local_batch_size} "
            f"global_minibatch_size={global_minibatch_size} local_minibatch_size={local_minibatch_size}"
        )

    for update in range(start_update, num_updates + 1):
        student_policy.eval()
        rollout_rgbs, rollout_states, rollout_action_bins, next_obs, rollout_stats, rollout_steps_completed = (
            collect_teacher_rollout(args, teacher_policy, envs, next_obs)
        )
        global_step += args.num_envs * rollout_steps_completed

        update_stats, update_minibatch_losses = run_bc_update(
            args,
            student_policy,
            optimizer,
            rollout_rgbs,
            rollout_states,
            rollout_action_bins,
            local_minibatch_size=local_minibatch_size,
        )
        if is_main_process():
            for point in update_minibatch_losses:
                optimizer_step += 1
                minibatch_loss_history.append(
                    {
                        "optimizer_step": optimizer_step,
                        "update": update,
                        "epoch": int(point["epoch"]),
                        "minibatch_in_epoch": int(point["minibatch_in_epoch"]),
                        "loss": float(point["loss"]),
                    }
                )

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        metric = {
            "update": update,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "elapsed_hours": elapsed_hours,
            **rollout_stats,
            **update_stats,
        }

        if is_main_process() and eval_envs is not None and (update % args.eval_every_updates == 0 or update == num_updates):
            print(f"[eval] update={update} evaluating sparsities={list(EVAL_SPARSITIES)}")
            eval_metrics = evaluate_policy_at_sparsities(student_policy, eval_envs, args.eval_episodes)
            metric.update(eval_metrics)
            if test_video_envs is not None:
                evaluate_policy_at_sparsities(
                    student_policy,
                    test_video_envs,
                    min(args.test_video_episodes, max(1, args.test_video_num_envs)),
                )
            success_once = float(eval_metrics.get("eval_success_once", 0.0))
            if success_once >= best_success_once:
                best_success_once = success_once
                torch.save(
                    {
                        "policy": student_policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "best_success_once": best_success_once,
                    },
                    output_dir / "best_policy.pt",
                )

        if is_main_process():
            metrics_history.append(metric)
            print(
                f"[train] update={update}/{num_updates} step={global_step} "
                f"optimizer_step={optimizer_step} bc_loss={metric['bc_loss']:.6f} "
                f"reward={metric['reward_mean']:.4f} "
                f"eval_success_once={metric.get('eval_success_once', float('nan')):.4f} "
                f"elapsed_h={metric['elapsed_hours']:.2f}"
            )
            save_json(output_dir / "latest_metrics.json", metric)
            save_metrics_history(output_dir, metrics_history)
            save_minibatch_loss_history(output_dir, minibatch_loss_history)
            plot_bc_metrics_history(output_dir, metrics_history, minibatch_loss_history)
            if update % 10 == 0 or update == num_updates:
                torch.save(
                    {
                        "policy": student_policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "best_success_once": best_success_once,
                    },
                    output_dir / "latest_policy.pt",
                )

        reached_zero_success_stop = False
        max_success_observed = 0.0
        if is_main_process():
            reached_zero_success_stop, max_success_observed = should_early_stop_zero_success(
                metrics_history,
                threshold_minutes=args.early_stop_zero_success_minutes,
            )
        reached_zero_success_stop = broadcast_object(reached_zero_success_stop)
        max_success_observed = broadcast_object(max_success_observed)
        if reached_zero_success_stop:
            stop_reason = "early_stop_zero_success"
            if is_main_process():
                print(
                    "[train] reached zero-success early stop: "
                    f"{args.early_stop_zero_success_minutes:.1f} minutes elapsed "
                    f"with max_success={max_success_observed:.4f}"
                )
            break

        reached_time_limit = broadcast_object(elapsed_hours >= args.max_runtime_hours)
        if reached_time_limit:
            stop_reason = "time_limit"
            if is_main_process():
                print(f"[train] reached time limit: {elapsed_hours:.2f}h >= {args.max_runtime_hours:.2f}h")
            break

    if is_main_process():
        if eval_envs is not None:
            final_eval_metrics_flat = evaluate_policy_at_sparsities(student_policy, eval_envs, args.eval_episodes)
            final_eval_metrics = unflatten_eval_metrics_by_sparsity(final_eval_metrics_flat)
            save_json(output_dir / "final_eval_metrics.json", final_eval_metrics)
        save_metrics_history(output_dir, metrics_history)
        save_minibatch_loss_history(output_dir, minibatch_loss_history)
        plot_bc_metrics_history(output_dir, metrics_history, minibatch_loss_history)
        save_json(
            output_dir / DEFAULT_SUMMARY_NAME,
            {
                "run_name": run_name,
                "stop_reason": stop_reason,
                "teacher_checkpoint": teacher_checkpoint,
                "student_init_policy": args.student_init_policy,
                "resume_from": args.resume_from,
                "best_success_once": best_success_once,
                "num_updates_completed": metrics_history[-1]["update"] if metrics_history else 0,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "final_eval_metrics": final_eval_metrics,
            },
        )

    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    if test_video_envs is not None:
        test_video_envs.close()


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
