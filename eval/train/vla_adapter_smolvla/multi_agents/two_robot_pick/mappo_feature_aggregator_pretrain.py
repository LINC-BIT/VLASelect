import os
import sys
import time
from datetime import datetime

sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import argparse
import itertools
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import Accelerator
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.two_robot_pick_cube_v2  # noqa: F401
from mani_skill.utils.io_utils import dump_json, load_json
from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy, mappo_update_on_policy_ag
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_stage, get_step_infos
from ours.de_feature_fusion.vla_feature_aggregator import (
    VLAClientForMultiAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import (
    MultiAgentVLAAdapterMAPPOAgent,
    build_batch_from_obs,
)

mp.set_start_method("spawn", force=True)


MODEL_NAME = "vla_adapter_smolvla"


def get_model_name(args) -> str:
    if args.model_backbone == "mixed_tiny_vla_smolvla":
        return "vla_adapter_smolvla_mappo"
    return MODEL_NAME


def resolve_ckpt_dir(args):
    ckpt_task_name = args.ckpt_task_name if args.ckpt_task_name is not None else args.task_name
    task_dir = os.path.join(args.save_dir, f"{ckpt_task_name}/ppo/{args.robot_name}/{get_model_name(args)}")
    os.makedirs(task_dir, exist_ok=True)

    fresh_run_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    resume_dir = args.resume_dir if args.resume_dir is not None else fresh_run_dir
    root_dir = resume_dir
    os.makedirs(root_dir, exist_ok=True)
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
    }


def make_collate_fn(agent_names):
    def collate_fn(obs):
        return build_batch_from_obs(obs, agent_names)

    return collate_fn


def make_sample_fn(agent_names, agent, deterministic=True):
    collate_fn = make_collate_fn(agent_names)

    def sample_fn(obs):
        batch = collate_fn(obs)
        return agent.get_action(batch, deterministic=deterministic)

    return sample_fn


def load_client_aggregators(clients, load_dir, ckpt_prefix):
    if load_dir is None:
        return
    for name in clients.keys():
        ag_path = os.path.join(load_dir, f"{ckpt_prefix}{name}.pt")
        if os.path.exists(ag_path):
            clients[name].load_feature_aggregators(ag_path)
            print(f"[Checkpoint] loaded feature aggregators for {name}: {ag_path}")
        else:
            print(f"[Checkpoint] missing feature aggregators for {name}: {ag_path}")


def infer_aggregator_ckpt_prefix(init_agent_path):
    if init_agent_path is None:
        return "best_ag_"
    ckpt_name = os.path.basename(init_agent_path)
    if ckpt_name.startswith("latest"):
        return "latest_ag_"
    if ckpt_name.startswith("best"):
        return "best_ag_"
    print(
        f"[Checkpoint] unrecognized init agent checkpoint name {ckpt_name}; "
        "default feature aggregators to best_ag_"
    )
    return "best_ag_"


def get_agent_info(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = test_env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": test_env.single_action_space[agent_names[0]].shape[0],
    }
    test_env.close()
    return info


def save_checkpoint(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def infer_best_step_from_metrics(metrics_path, min_step=None):
    if not os.path.exists(metrics_path):
        return None, None
    try:
        metrics_log = load_json(metrics_path)
    except Exception as exc:
        print(f"[Resume] failed to read metrics for best-step inference: {exc}")
        return None, None

    best_score = None
    best_step = None
    for row in metrics_log:
        if "score" not in row or "step" not in row:
            continue
        score = float(row["score"])
        step = int(row["step"])
        if min_step is not None and step < int(min_step):
            continue
        if best_score is None or score >= best_score:
            best_score = score
            best_step = step
    return best_step, best_score


def infer_best_score_from_metrics_log(metrics_log, min_step=None):
    best_score = None
    for row in metrics_log:
        if "score" not in row:
            continue
        score = float(row["score"])
        step = row.get("step")
        if min_step is not None:
            if step is None or int(step) < int(min_step):
                continue
        if best_score is None or score >= best_score:
            best_score = score
    return best_score


def get_best_save_start_step(args) -> int:
    configured = getattr(args, "best_save_start_step", -1)
    if configured is None or int(configured) < 0:
        return int(args.phase1_end_step)
    return int(configured)


def maybe_cuda_sync(device):
    if device is None or not torch.cuda.is_available():
        return
    if getattr(device, "type", None) != "cuda":
        return
    torch.cuda.synchronize(device)


def timed_section_start(args, device):
    if not getattr(args, "log_timing", False):
        return None
    maybe_cuda_sync(device)
    return time.perf_counter()


def timed_section_end(args, device, start_time, bucket, key):
    if start_time is None:
        return
    maybe_cuda_sync(device)
    bucket[key] = bucket.get(key, 0.0) + (time.perf_counter() - start_time)


def maybe_print_timing(args, accelerator, global_steps, rollout_steps, timing_parts):
    if not getattr(args, "log_timing", False):
        return
    if not accelerator.is_main_process:
        return
    rollout_idx = global_steps // max(int(rollout_steps), 1)
    if rollout_idx % max(int(args.timing_log_every), 1) != 0:
        return

    total = sum(float(value) for value in timing_parts.values())
    ordered_keys = [
        "save_eval",
        "rollout",
        "gae",
        "update",
        "ag_post",
        "state_stats",
        "main_log",
    ]
    details = ", ".join(
        f"{key}={timing_parts.get(key, 0.0):.3f}s"
        for key in ordered_keys
    )
    print(f"[Timing] rollout={rollout_idx} total={total:.3f}s | {details}")


def should_update_best_checkpoint(args, global_steps: int) -> bool:
    return int(global_steps) >= get_best_save_start_step(args)


def log_agent_feature_cosine_similarity(
    writer,
    ag_data_infos,
    global_steps,
    metric_prefix="monitoring/communication/agent_feature_cosine",
):
    agent_features = {}
    for agent_name, message in ag_data_infos.items():
        if message is None or message.get("feature") is None:
            continue
        feature = message["feature"]
        if not torch.is_tensor(feature):
            feature = torch.as_tensor(feature)
        feature = feature.detach().float()
        if feature.ndim == 3:
            feature = feature.mean(dim=(0, 1))
        elif feature.ndim == 2:
            feature = feature.mean(dim=0)
        elif feature.ndim != 1:
            continue
        agent_features[agent_name] = feature.cpu()

    if len(agent_features) < 2:
        return

    cosine_values = []
    for agent_a, agent_b in itertools.combinations(sorted(agent_features.keys()), 2):
        cos_sim = F.cosine_similarity(
            agent_features[agent_a].unsqueeze(0),
            agent_features[agent_b].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
        writer.add_scalar(f"{metric_prefix}/{agent_a}_vs_{agent_b}", cos_sim, global_steps)
        cosine_values.append(cos_sim)

    if cosine_values:
        cosine_tensor = torch.tensor(cosine_values, dtype=torch.float32)
        writer.add_scalar(f"{metric_prefix}_mean", cosine_tensor.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_min", cosine_tensor.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_max", cosine_tensor.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_std", cosine_tensor.std(unbiased=False).item(), global_steps)


def log_uploaded_q_task_stats(
    writer,
    ag_data_infos,
    global_steps,
    metric_prefix="monitoring/communication/uploaded_q_task",
    enable_histograms=True,
):
    q_task_debug_strs = []
    all_q_task_values = []

    for agent_name, feature_message in ag_data_infos.items():
        if feature_message is None:
            continue

        meta = feature_message.get("meta")
        if not isinstance(meta, dict):
            continue

        q_task = meta.get("q_task")
        if q_task is None:
            continue

        if not torch.is_tensor(q_task):
            q_task = torch.as_tensor(q_task, dtype=torch.float32)
        q_task = q_task.detach().float().view(-1).cpu()
        if q_task.numel() == 0:
            continue

        if enable_histograms:
            writer.add_histogram(f"{metric_prefix}/{agent_name}", q_task, global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_mean", q_task.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_min", q_task.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_max", q_task.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_std", q_task.std(unbiased=False).item(), global_steps)

        q_task_values = ", ".join(f"{value:.3f}" for value in q_task.tolist())
        q_task_debug_strs.append(
            f"{agent_name}: mean={q_task.mean().item():.3f}, min={q_task.min().item():.3f}, "
            f"max={q_task.max().item():.3f}, values=[{q_task_values}]"
        )
        all_q_task_values.append(q_task)

    if all_q_task_values:
        all_q_task = torch.cat(all_q_task_values, dim=0)
        writer.add_scalar(f"{metric_prefix}_mean", all_q_task.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_min", all_q_task.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_max", all_q_task.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_std", all_q_task.std(unbiased=False).item(), global_steps)

    if q_task_debug_strs:
        print(f'Uploaded q_task: {" | ".join(q_task_debug_strs)}')


def log_effective_transmission_stats(
    writer,
    clients,
    global_steps,
    metric_prefix="monitoring/communication/effective_transmission",
):
    if clients is None:
        return

    total_bytes_sum = 0
    matched_bytes_sum = 0
    total_forwards_sum = 0
    matched_forwards_sum = 0
    debug_strs = []

    for client_name, client in clients.items():
        stats = client.debug_last_selected_payload_stats()
        if not isinstance(stats, dict):
            continue

        total_bytes = int(stats.get("total_bytes", 0))
        matched_bytes = int(stats.get("matched_bytes", 0))
        total_forwards = int(stats.get("total_forwards", 0))
        matched_forwards = int(stats.get("matched_forwards", 0))
        effective_ratio = (
            float(matched_bytes) / float(total_bytes)
            if total_bytes > 0
            else 0.0
        )

        writer.add_scalar(f"{metric_prefix}/{client_name}_total_bytes", total_bytes, global_steps)
        writer.add_scalar(f"{metric_prefix}/{client_name}_matched_bytes", matched_bytes, global_steps)
        writer.add_scalar(f"{metric_prefix}/{client_name}_effective_ratio", effective_ratio, global_steps)
        writer.add_scalar(f"{metric_prefix}/{client_name}_total_forwards", total_forwards, global_steps)
        writer.add_scalar(f"{metric_prefix}/{client_name}_matched_forwards", matched_forwards, global_steps)

        total_bytes_sum += total_bytes
        matched_bytes_sum += matched_bytes
        total_forwards_sum += total_forwards
        matched_forwards_sum += matched_forwards
        debug_strs.append(
            f"{client_name}: matched={matched_bytes}/{total_bytes}B, forwards={matched_forwards}/{total_forwards}"
        )

    if total_bytes_sum > 0:
        writer.add_scalar(f"{metric_prefix}_total_bytes", total_bytes_sum, global_steps)
        writer.add_scalar(f"{metric_prefix}_matched_bytes", matched_bytes_sum, global_steps)
        writer.add_scalar(
            f"{metric_prefix}_effective_ratio",
            float(matched_bytes_sum) / float(total_bytes_sum),
            global_steps,
        )
    if total_forwards_sum > 0:
        writer.add_scalar(f"{metric_prefix}_total_forwards", total_forwards_sum, global_steps)
        writer.add_scalar(f"{metric_prefix}_matched_forwards", matched_forwards_sum, global_steps)
        writer.add_scalar(
            f"{metric_prefix}_matched_forward_ratio",
            float(matched_forwards_sum) / float(total_forwards_sum),
            global_steps,
        )

    if debug_strs:
        print(f'Effective transmission: {" | ".join(debug_strs)}')


def _mean_summary_tensors(summary_tensors):
    valid_tensors = [tensor.float() for tensor in summary_tensors if tensor is not None]
    if not valid_tensors:
        return None
    return torch.stack(valid_tensors, dim=0).mean(dim=0)


def iter_client_feature_aggregators(client):
    for value in client.feature_aggregators.values():
        if isinstance(value, (list, tuple)):
            for aggregator in value:
                yield aggregator
        else:
            yield value


def _log_pairwise_cosine_stats(writer, feature_by_agent, global_steps, metric_prefix):
    cosine_values = []
    skipped_pairs = 0
    for agent_a, agent_b in itertools.combinations(sorted(feature_by_agent.keys()), 2):
        if feature_by_agent[agent_a].shape != feature_by_agent[agent_b].shape:
            skipped_pairs += 1
            continue
        cos_sim = F.cosine_similarity(
            feature_by_agent[agent_a].unsqueeze(0),
            feature_by_agent[agent_b].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
        writer.add_scalar(f"{metric_prefix}/{agent_a}_vs_{agent_b}", cos_sim, global_steps)
        cosine_values.append(cos_sim)

    writer.add_scalar(f"{metric_prefix}_skipped_pairs", skipped_pairs, global_steps)

    if not cosine_values:
        return None

    cosine_tensor = torch.tensor(cosine_values, dtype=torch.float32)
    writer.add_scalar(f"{metric_prefix}_mean", cosine_tensor.mean().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_min", cosine_tensor.min().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_max", cosine_tensor.max().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_std", cosine_tensor.std(unbiased=False).item(), global_steps)
    return cosine_tensor.mean().item()


def log_feature_aggregator_agent_cosine_similarity(
    writer,
    clients,
    global_steps,
    metric_prefix="monitoring/aggregation/agent_cosine",
):
    stream_specs = [
        ("feature", "feature_local", "feature_fused"),
        ("action", "action_local", "action_fused"),
    ]

    for stream_name, local_key, fused_key in stream_specs:
        local_by_agent = {}
        fused_by_agent = {}
        for agent_name, client in clients.items():
            summaries = client.debug_feature_aggregator_feature_summaries()
            local_summary = _mean_summary_tensors(summary.get(local_key) for summary in summaries.values())
            fused_summary = _mean_summary_tensors(summary.get(fused_key) for summary in summaries.values())
            if local_summary is not None:
                local_by_agent[agent_name] = local_summary
            if fused_summary is not None:
                fused_by_agent[agent_name] = fused_summary

        local_mean = _log_pairwise_cosine_stats(
            writer,
            local_by_agent,
            global_steps,
            f"{metric_prefix}/{stream_name}/local",
        )
        fused_mean = _log_pairwise_cosine_stats(
            writer,
            fused_by_agent,
            global_steps,
            f"{metric_prefix}/{stream_name}/fused",
        )
        if local_mean is not None and fused_mean is not None:
            writer.add_scalar(
                f"{metric_prefix}/{stream_name}/fused_minus_local_mean",
                fused_mean - local_mean,
                global_steps,
            )


def set_optimizer_phase_lrs(args, optimizer, critic_only: bool, train_actor_base: bool) -> None:
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        if group_name in {"vla", "vla_actor_vla", "smolvla_actor_vla"}:
            group["lr"] = 0.0 if (critic_only or not train_actor_base) else args.backbone_learning_rate
        elif group_name in {"state_projector", "vla_actor_state_projector", "smolvla_actor_state_projector"}:
            group["lr"] = 0.0 if (critic_only or not train_actor_base) else args.state_learning_rate
        elif group_name in {"context_projector", "actor_head", "vla_actor_heads", "smolvla_actor_heads"}:
            group["lr"] = 0.0 if (critic_only or not train_actor_base) else args.head_learning_rate
        elif group_name == "actor_heads":
            group["lr"] = 0.0 if (critic_only or not train_actor_base) else args.actor_hook_learning_rate
        elif group_name in {"critic_state_encoder", "critic_visual_encoder", "critic"}:
            group["lr"] = args.value_head_learning_rate
        elif group_name == "feature_aggregator":
            group["lr"] = args.feature_aggregator_learning_rate


def set_requires_grad(parameters, requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def set_client_feature_aggregator_requires_grad(clients, requires_grad: bool) -> None:
    if clients is None:
        return
    params = []
    for client in clients.values():
        for aggregator in iter_client_feature_aggregators(client):
            params.extend(list(aggregator.module.parameters()))
    set_requires_grad(params, requires_grad)


def get_training_phase(args, step: int):
    if step < args.phase1_end_step:
        return {
            "index": 1,
            "name": "aggregator_only",
            "train_head": False,
            "train_encoder": False,
        }
    if step < args.phase2_end_step:
        return {
            "index": 2,
            "name": "aggregator_plus_head",
            "train_head": True,
            "train_encoder": False,
        }
    return {
        "index": 3,
        "name": "aggregator_plus_head_lora",
        "train_head": True,
        "train_encoder": True,
    }


def get_phase_ent_coef(args, phase_index: int, default_ent_coef: float) -> float:
    phase_specific = getattr(args, f"ent_coef_phase{phase_index}", None)
    if phase_specific is None:
        return float(default_ent_coef)
    return float(phase_specific)


def maybe_load_initial_agent(args, ckpt, agent):
    if args.resume_dir is not None:
        resume_agent_path = ckpt["best_agent"] if args.resume_use_best_agent else ckpt["latest_agent"]
        if not os.path.exists(resume_agent_path) and args.resume_use_best_agent and os.path.exists(ckpt["latest_agent"]):
            print(
                f"[Resume] requested best-agent resume but {resume_agent_path} is missing; "
                f"fall back to latest agent {ckpt['latest_agent']}"
            )
            resume_agent_path = ckpt["latest_agent"]
        if os.path.exists(resume_agent_path):
            print(f"[Train] resume PPO agent from {resume_agent_path}")
            agent.load_checkpoint_state_dict(torch.load(resume_agent_path, map_location="cpu"))
            best_step = None
            if args.resume_use_best_agent and resume_agent_path == ckpt["best_agent"]:
                best_step, best_score = infer_best_step_from_metrics(
                    ckpt["metrics"], min_step=get_best_save_start_step(args)
                )
                if best_step is not None:
                    print(f"[Resume] inferred best checkpoint step={best_step} score={best_score:.4f}")
            return {
                "resumed": True,
                "used_best_agent": bool(args.resume_use_best_agent and resume_agent_path == ckpt["best_agent"]),
                "best_step": best_step,
            }
    if args.init_agent_path:
        load_agent_checkpoint(agent, args.init_agent_path, map_location="cpu", label="PPO init")
    return {"resumed": False, "used_best_agent": False, "best_step": None}


def build_clients(args, agent, infos, agent_names, device):
    if args.model_backbone == "mixed_tiny_vla_smolvla":
        return {
            agent.vla_agent_name: VLAClientForMultiAgent(
                name=agent.vla_agent_name,
                large_model=agent,
                action_position_layer_prefix="vla_actor_action_position_placeholders",
                action_position_actor_layer_prefix="vla_actor_action_position_actor_placeholders",
                local_feature_dim=agent.vla_actor.hidden_dim * 3,
                num_action_positions=infos["action_dim"],
                device=device,
                local_action_dim=infos["action_dim"],
                max_episode_steps=args.max_episode_steps,
                feature_aggregator_attention_num_heads=args.feature_aggregator_attention_num_heads,
                feature_aggregator_gate_type=args.feature_aggregator_gate_type,
                feature_aggregator_gate_activation=args.feature_aggregator_gate_activation,
                feature_aggregator_norm_type=args.feature_aggregator_norm_type,
                feature_aggregator_feature_gate_open_max=args.feature_aggregator_feature_gate_open_max,
                feature_aggregator_action_gate_open_max=args.feature_aggregator_action_gate_open_max,
                feature_aggregator_q_ret_weight=args.feature_aggregator_q_ret_weight,
                feature_aggregator_q_attn_weight=args.feature_aggregator_q_attn_weight,
                feature_aggregator_remote_dropout_prob=args.feature_aggregator_remote_dropout_prob,
                feature_aggregator_remote_noise_std=args.feature_aggregator_remote_noise_std,
                feature_aggregator_remote_stale_shift_max=args.feature_aggregator_remote_stale_shift_max,
                feature_selector_topk_trajectories=args.feature_selector_topk_trajectories,
                feature_selector_temporal_pool_steps=args.feature_selector_temporal_pool_steps,
                feature_selector_strategy=args.feature_selector_strategy,
                eval_feature_selector_strategy=args.eval_feature_selector_strategy,
                feature_selector_alpha=args.feature_selector_alpha,
            ),
            agent.smolvla_agent_name: VLAClientForMultiAgent(
                name=agent.smolvla_agent_name,
                large_model=agent,
                action_position_layer_prefix="smolvla_actor_action_position_placeholders",
                action_position_actor_layer_prefix="smolvla_actor_action_position_actor_placeholders",
                local_feature_dim=agent.smolvla_actor.hidden_dim,
                num_action_positions=infos["action_dim"],
                device=device,
                local_action_dim=infos["action_dim"],
                max_episode_steps=args.max_episode_steps,
                feature_aggregator_attention_num_heads=args.feature_aggregator_attention_num_heads,
                feature_aggregator_gate_type=args.feature_aggregator_gate_type,
                feature_aggregator_gate_activation=args.feature_aggregator_gate_activation,
                feature_aggregator_norm_type=args.feature_aggregator_norm_type,
                feature_aggregator_feature_gate_open_max=args.feature_aggregator_feature_gate_open_max,
                feature_aggregator_action_gate_open_max=args.feature_aggregator_action_gate_open_max,
                feature_aggregator_q_ret_weight=args.feature_aggregator_q_ret_weight,
                feature_aggregator_q_attn_weight=args.feature_aggregator_q_attn_weight,
                feature_aggregator_remote_dropout_prob=args.feature_aggregator_remote_dropout_prob,
                feature_aggregator_remote_noise_std=args.feature_aggregator_remote_noise_std,
                feature_aggregator_remote_stale_shift_max=args.feature_aggregator_remote_stale_shift_max,
                feature_selector_topk_trajectories=args.feature_selector_topk_trajectories,
                feature_selector_temporal_pool_steps=args.feature_selector_temporal_pool_steps,
                feature_selector_strategy=args.feature_selector_strategy,
                eval_feature_selector_strategy=args.eval_feature_selector_strategy,
                feature_selector_alpha=args.feature_selector_alpha,
            ),
        }

    return {
        name: VLAClientForMultiAgent(
            name=name,
            large_model=agent,
            action_position_layer_prefix=f"actor_action_position_placeholders.{name}",
            action_position_actor_layer_prefix=f"actor_action_position_actor_placeholders.{name}",
            local_feature_dim=agent.actor.hidden_dim * 3,
            num_action_positions=infos["action_dim"],
            device=device,
            local_action_dim=infos["action_dim"],
            max_episode_steps=args.max_episode_steps,
            feature_aggregator_attention_num_heads=args.feature_aggregator_attention_num_heads,
            feature_aggregator_gate_type=args.feature_aggregator_gate_type,
            feature_aggregator_gate_activation=args.feature_aggregator_gate_activation,
            feature_aggregator_norm_type=args.feature_aggregator_norm_type,
            feature_aggregator_feature_gate_open_max=args.feature_aggregator_feature_gate_open_max,
            feature_aggregator_action_gate_open_max=args.feature_aggregator_action_gate_open_max,
            feature_aggregator_q_ret_weight=args.feature_aggregator_q_ret_weight,
            feature_aggregator_q_attn_weight=args.feature_aggregator_q_attn_weight,
            feature_aggregator_remote_dropout_prob=args.feature_aggregator_remote_dropout_prob,
            feature_aggregator_remote_noise_std=args.feature_aggregator_remote_noise_std,
            feature_aggregator_remote_stale_shift_max=args.feature_aggregator_remote_stale_shift_max,
            feature_selector_topk_trajectories=args.feature_selector_topk_trajectories,
            feature_selector_temporal_pool_steps=args.feature_selector_temporal_pool_steps,
            feature_selector_strategy=args.feature_selector_strategy,
            eval_feature_selector_strategy=args.eval_feature_selector_strategy,
            feature_selector_alpha=args.feature_selector_alpha,
        )
        for name in agent_names
    }


def main(args):
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    ckpt = resolve_ckpt_dir(args)
    step_infos = get_step_infos(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    collate_fn = make_collate_fn(agent_names)

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"])
        print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    else:
        writer = None

    if args.model_backbone == "mixed_tiny_vla_smolvla":
        agent = MixedTinyVLAAdapterSmolVLASFTAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        )
        if getattr(args, "full_kl_coef", 0.0) > 0 or getattr(args, "log_full_kl", False):
            if accelerator.is_main_process:
                print("[Config] mixed_tiny_vla_smolvla disables full token-distribution KL; force full_kl_coef=0 and log_full_kl=False")
            args.full_kl_coef = 0.0
            args.log_full_kl = False
    else:
        agent = MultiAgentVLAAdapterMAPPOAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            model_backbone=args.model_backbone,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        )
    if accelerator.is_main_process:
        total_params = sum(parameter.numel() for parameter in agent.parameters())
        trainable_params = sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
        print(
            f"[Model] backbone={args.model_backbone} total_params={total_params / 1e6:.2f}M "
            f"trainable_params={trainable_params / 1e6:.2f}M"
        )
    if args.model_backbone == "mixed_tiny_vla_smolvla":
        base_param_groups = [
            {
                "params": [p for p in agent.vla_actor.vla.parameters() if p.requires_grad],
                "lr": args.backbone_learning_rate,
                "group_name": "vla_actor_vla",
            },
            {
                "params": [p for p in agent.vla_actor.state_projector.parameters() if p.requires_grad],
                "lr": args.state_learning_rate,
                "group_name": "vla_actor_state_projector",
            },
            {
                "params": [p for p in agent.vla_actor.context_projector.parameters() if p.requires_grad]
                + [p for p in agent.vla_actor.actor_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "vla_actor_heads",
            },
            {
                "params": [p for p in agent.smolvla_actor.vla.parameters() if p.requires_grad],
                "lr": args.backbone_learning_rate,
                "group_name": "smolvla_actor_vla",
            },
            {
                "params": [p for p in agent.smolvla_actor.state_projector.parameters() if p.requires_grad],
                "lr": args.state_learning_rate,
                "group_name": "smolvla_actor_state_projector",
            },
            {
                "params": [p for p in agent.smolvla_actor.context_projector.parameters() if p.requires_grad]
                + [p for p in agent.smolvla_actor.action_mean_head.parameters() if p.requires_grad]
                + [p for p in agent.smolvla_actor.log_std_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "smolvla_actor_heads",
            },
            {
                "params": [p for p in agent.critic_state_encoder.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic_state_encoder",
            },
            {
                "params": [p for p in agent.critic_visual_encoder.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic_visual_encoder",
            },
            {
                "params": [p for p in agent.critic.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic",
            },
        ]
    else:
        base_param_groups = [
            {
                "params": [p for p in agent.actor.vla.parameters() if p.requires_grad],
                "lr": args.backbone_learning_rate,
                "group_name": "vla",
            },
            {
                "params": [p for p in agent.actor.state_projector.parameters() if p.requires_grad],
                "lr": args.state_learning_rate,
                "group_name": "state_projector",
            },
            {
                "params": [p for p in agent.actor.context_projector.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "context_projector",
            },
            {
                "params": [p for p in agent.actor.actor_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "actor_head",
            },
            {
                "params": [p for p in agent.actor_heads.parameters() if p.requires_grad],
                "lr": args.actor_hook_learning_rate,
                "group_name": "actor_heads",
            },
            {
                "params": [p for p in agent.critic_state_encoder.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic_state_encoder",
            },
            {
                "params": [p for p in agent.critic_visual_encoder.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic_visual_encoder",
            },
            {
                "params": [p for p in agent.critic.parameters() if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic",
            },
        ]
    base_param_groups = [group for group in base_param_groups if group["params"]]
    head_trainable_parameters = []
    encoder_trainable_parameters = []
    critic_trainable_parameters = []
    for group in base_param_groups:
        group_name = group["group_name"]
        params = list(group["params"])
        if group_name in {"context_projector", "actor_head", "actor_heads", "vla_actor_heads", "smolvla_actor_heads"}:
            head_trainable_parameters.extend(params)
        elif group_name in {"vla", "state_projector", "vla_actor_vla", "vla_actor_state_projector", "smolvla_actor_vla", "smolvla_actor_state_projector"}:
            encoder_trainable_parameters.extend(params)
        elif group_name in {"critic_state_encoder", "critic_visual_encoder", "critic"}:
            critic_trainable_parameters.extend(params)

    resume_info = maybe_load_initial_agent(args, ckpt, agent)
    global_steps = 0
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        if resume_info["resumed"] and resume_info["used_best_agent"]:
            inferred_best_step = resume_info["best_step"]
            if inferred_best_step is not None:
                global_steps = int(inferred_best_step)
            else:
                global_steps = int(latest_opt["step"])
                print(
                    f"[Resume] best-agent step could not be inferred; "
                    f"fall back to latest optimizer step={global_steps}"
                )

    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "none",
        "max_episode_steps": args.max_episode_steps,
        "sim_backend": "physx_cuda",
    }
    env_kwargs_for_eval = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }
    if args.enable_mixed_train_envs:
        env_kwargs.update(
            {
                "enable_mixed_domain_randomization": True,
                "mixed_domain_randomization_clear_ratio": args.mixed_env_clean_ratio,
                "mixed_domain_randomization_mild_ratio": args.mixed_env_mild_ratio,
                "mixed_domain_randomization_hard_ratio": args.mixed_env_hard_ratio,
            }
        )
        print(
            "[MixedEnv] Using internal batched-env domain randomization split: "
            f"clear={args.mixed_env_clean_ratio}, "
            f"mild={args.mixed_env_mild_ratio}, "
            f"hard={args.mixed_env_hard_ratio}"
        )
    if args.evaluate_mode and args.eval_agent_dir is None:
        raise ValueError("Please provide --eval-agent-dir for evaluation mode")

    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs_for_eval,
        video_dir=(
            os.path.join(args.eval_agent_dir, "videos_eval")
            if args.evaluate_mode
            else f"{ckpt['video_dir']}_train"
        ),
    )

    if args.evaluate_mode:
        clients = build_clients(args, agent, infos, agent_names, device)
        load_client_aggregators(clients, args.eval_agent_dir, "best_ag_")
        client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
        for recv_name, client_recv in clients.items():
            for send_name, client_send_info in client_infos.items():
                if recv_name != send_name:
                    client_recv.add_feature_aggregator(send_name, client_send_info)
        for client in clients.values():
            client.use_eval_feature_selector_strategy()

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.load_checkpoint_state_dict(
            torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu")
        )
        unwrapped_agent.eval()
        print("[Evaluate] Start evaluation only mode")
        eval_metrics = evaluate(
            n=args.eval_episodes,
            sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        if accelerator.is_main_process:
            for key, value in eval_metrics.items():
                mean = value.mean()
                print(f"eval_{key}_mean={mean}")
            dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)
        return

    clients = build_clients(args, agent, infos, agent_names, device)
    if args.resume_dir is not None:
        ckpt_prefix = "best_ag_" if args.resume_use_best_agent else "latest_ag_"
        load_client_aggregators(clients, ckpt["root_dir"], ckpt_prefix)
    elif args.init_agent_path:
        load_client_aggregators(
            clients,
            os.path.dirname(args.init_agent_path),
            infer_aggregator_ckpt_prefix(args.init_agent_path),
        )

    client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
    for recv_name, client_recv in clients.items():
        for send_name, client_send_info in client_infos.items():
            if recv_name != send_name:
                client_recv.add_feature_aggregator(send_name, client_send_info)

    aggregator_param_groups = []
    training_feature_aggregator_modules = []
    for client_name, client in clients.items():
        feature_aggregators_parameters = client.get_feature_aggregators_parameters()
        for remote_name, fap in feature_aggregators_parameters.items():
            group_name = f"{client_name}_from_{remote_name}"
            training_feature_aggregator_modules.append(group_name)
            fap_list = list(fap)
            if len(fap_list) == 0:
                continue
            for p in fap_list:
                p.requires_grad = True
            aggregator_param_groups.append(
                {
                    "params": fap_list,
                    "lr": args.feature_aggregator_learning_rate,
                    "eps": 1e-5,
                    "group_name": "feature_aggregator",
                }
            )

    optimizer = optim.AdamW(base_param_groups + aggregator_param_groups, eps=1e-5, weight_decay=args.weight_decay)
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        if not (resume_info["resumed"] and resume_info["used_best_agent"]):
            optimizer.load_state_dict(latest_opt["opt"])
            global_steps = int(latest_opt["step"])

    agent, optimizer = accelerator.prepare(agent, optimizer)

    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=args.ignore_partial_reset,
        record_metrics=True,
    )
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    best_save_start_step = get_best_save_start_step(args)
    best_score = infer_best_score_from_metrics_log(metrics_log, min_step=best_save_start_step)
    if best_score is None:
        best_score = -1.0
    elif accelerator.is_main_process:
        print(
            f"[Resume] restored historical best score={best_score:.4f} "
            f"from metrics log (best_save_start_step={best_save_start_step})"
        )
    start_time = time.time()
    last_save_skip = args.resume_dir is not None
    default_ent_coef = float(args.ent_coef)

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)
    current_phase = None
    set_optimizer_phase_lrs(args, optimizer, critic_only=False, train_actor_base=True)
    set_requires_grad(head_trainable_parameters, False)
    set_requires_grad(encoder_trainable_parameters, False)
    set_requires_grad(critic_trainable_parameters, True)
    agent.freeze_state_stats()

    while global_steps < step_infos["total_steps"]:
        timing_parts = {}
        phase = get_training_phase(args, global_steps)
        if phase["name"] != current_phase:
            current_phase = phase["name"]
            phase_ent_coef = get_phase_ent_coef(args, phase["index"], default_ent_coef)
            print(
                f"[Phase] step={global_steps} -> phase{phase['index']} "
                f"({phase['name']}, train_head={phase['train_head']}, "
                f"train_encoder={phase['train_encoder']}, ent_coef={phase_ent_coef:g})"
            )
        args.ent_coef = get_phase_ent_coef(args, phase["index"], default_ent_coef)
        if accelerator.is_main_process:
            writer.add_scalar("training/ent_coef", args.ent_coef, global_steps)
        set_optimizer_phase_lrs(
            args,
            optimizer,
            critic_only=False,
            train_actor_base=phase["train_encoder"],
        )
        set_requires_grad(head_trainable_parameters, phase["train_head"])
        set_requires_grad(encoder_trainable_parameters, phase["train_encoder"])
        set_requires_grad(critic_trainable_parameters, True)

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.eval()
        for client in clients.values():
            client.eval()

        save_eval_start = timed_section_start(args, device)
        if not last_save_skip and global_steps % step_infos["save_interval_steps"] == 0 and accelerator.is_main_process:
            save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
            save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
            for name in agent_names:
                clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"latest_ag_{name}.pt"))

            set_client_feature_aggregator_requires_grad(clients, False)
            for client in clients.values():
                client.use_eval_feature_selector_strategy()
            eval_metrics = evaluate(
                n=args.eval_episodes,
                sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
                eval_envs=eval_envs,
            )
            # Evaluation freezes aggregators so their cached communication path is
            # deterministic. Restore trainability immediately afterwards so the
            # following PPO updates can continue optimizing them.
            set_client_feature_aggregator_requires_grad(clients, True)
            for client in clients.values():
                client.use_train_feature_selector_strategy()
            for key, value in eval_metrics.items():
                mean = value.mean()
                writer.add_scalar(f"eval/{key}", mean, global_steps)
                print(f"eval_{key}_mean={mean}")

            score = eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean()
            pbar.set_postfix(eval_score=score)
            metrics_log.append({"step": global_steps, "score": float(score)})
            dump_json(ckpt["metrics"], metrics_log)
            if should_update_best_checkpoint(args, global_steps) and score >= best_score:
                best_score = score
                save_checkpoint(ckpt["best_agent"], unwrapped_agent.checkpoint_state_dict())
                for name in agent_names:
                    clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"best_ag_{name}.pt"))
                print(f"[Eval] New best model saved (score={score:.3f})")
            elif not should_update_best_checkpoint(args, global_steps):
                print(
                    f"[Eval] skip best checkpoint update at step={global_steps} "
                    f"until aggregator warmup ends at step={best_save_start_step}"
                )
        timed_section_end(args, device, save_eval_start, timing_parts, "save_eval")

        last_save_skip = False

        for client in clients.values():
            client.train()
        rollout_start = timed_section_start(args, device)
        rollout = collect_rollout(
            args=args,
            agent=agent,
            collate_fn=collate_fn,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            accelerator=accelerator,
            writer=writer,
            global_step=global_steps,
            clients=clients,
        )
        timed_section_end(args, device, rollout_start, timing_parts, "rollout")
        communication_snapshots = None
        if len(rollout) == 11:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                old_token_logits_buf,
                communication_snapshots,
            ) = rollout
        elif len(rollout) == 10:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                old_token_logits_buf,
            ) = rollout
        else:
            (
                obs_buf,
                action_bin_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
            ) = rollout
            old_token_logits_buf = None

        gae_start = timed_section_start(args, device)
        adv_buf, ret_buf = compute_gae(
            rew_buf,
            done_buf,
            val_buf,
            final_val_buf,
            next_obs,
            next_done,
            agent,
            collate_fn,
            args,
            accelerator,
        )
        timed_section_end(args, device, gae_start, timing_parts, "gae")

        data = (
            obs_buf.reshape((-1,)),
            action_bin_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )
        if old_token_logits_buf is not None:
            data = data + (old_token_logits_buf.reshape((-1,)),)
        if communication_snapshots is not None:
            flat_communication_snapshots = []
            num_envs = done_buf.shape[1]
            for step_snapshot in communication_snapshots:
                flat_communication_snapshots.extend([step_snapshot] * num_envs)
            data = data + (flat_communication_snapshots,)

        all_aggregator_params = []
        for client in clients.values():
            for fa in iter_client_feature_aggregators(client):
                for p in fa.module.parameters():
                    all_aggregator_params.append(p)
        has_active_aggregator = any(
            fa.remote_features is not None
            for client in clients.values()
            for fa in iter_client_feature_aggregators(client)
        )
        active_head_trainable = phase["train_head"] and len(head_trainable_parameters) > 0
        active_encoder_trainable = phase["train_encoder"] and len(encoder_trainable_parameters) > 0
        has_active_base_training = active_head_trainable or active_encoder_trainable
        agent.eval()
        # Keep aggregators trainable during pretrain updates. The eval path above
        # temporarily disables gradients for deterministic evaluation/caching.
        set_client_feature_aggregator_requires_grad(clients, True)
        update_start = timed_section_start(args, device)
        if has_active_aggregator and len(all_aggregator_params) > 0:
            stats = mappo_update_on_policy_ag(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                phase["name"],
                clients,
                writer,
                -1,
            )
        elif has_active_base_training:
            print("[Aggregation] no active remote messages, fallback to base PPO update")
            stats = mappo_update_on_policy(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                phase["name"],
                writer,
                -1,
            )
        else:
            print("[Aggregation] no active remote messages and base model frozen, skipping update")
            stats = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "old_approx_kl": 0.0,
                "full_kl": 0.0,
                "argmax_change_frac": 0.0,
            }
        timed_section_end(args, device, update_start, timing_parts, "update")

        ag_post_start = timed_section_start(args, device)
        ag_data_infos = {}
        for client_name, client in clients.items():
            ag_data_infos[client_name] = client.export_feature_and_action()
        log_uploaded_q_task_stats(
            writer,
            ag_data_infos,
            global_steps,
            enable_histograms=not args.disable_ag_debug_histograms,
        )
        log_effective_transmission_stats(
            writer,
            clients,
            global_steps,
        )
        log_agent_feature_cosine_similarity(writer, ag_data_infos, global_steps)
        for recv_name, client_recv in clients.items():
            for send_name, ag_data in ag_data_infos.items():
                if recv_name != send_name:
                    client_recv.receive_feature_and_action(send_name, ag_data)
        for client_name, client in clients.items():
            feature_aggregators_parameters = client.get_feature_aggregators_parameters()
            for remote_name, fap in feature_aggregators_parameters.items():
                group_name = f"{client_name}_from_{remote_name}"
                if group_name not in training_feature_aggregator_modules:
                    training_feature_aggregator_modules.append(group_name)
                    fap_list = list(fap)
                    if len(fap_list) == 0:
                        continue
                    for p in fap_list:
                        p.requires_grad = True
                    optimizer.add_param_group(
                        {
                            "params": fap_list,
                            "lr": args.feature_aggregator_learning_rate,
                            "eps": 1e-5,
                            "group_name": "feature_aggregator",
                        }
                    )
            feature_aggregators_gate_g = client.debug_feature_aggregators()
            gate_g_strs = []
            for remote_name, gate_info in feature_aggregators_gate_g.items():
                for stream_name, gate_g in gate_info.items():
                    if gate_g is None:
                        continue
                    metric_prefix = f"monitoring/aggregation/gate_g/{client_name}/from_{remote_name}/{stream_name}"
                    if not args.disable_ag_debug_histograms:
                        writer.add_histogram(metric_prefix, gate_g, global_steps)
                    writer.add_scalar(f"{metric_prefix}_mean", gate_g.mean(), global_steps)
                    writer.add_scalar(f"{metric_prefix}_std", gate_g.std(), global_steps)
                    gate_g_strs.append(f"{remote_name}.{stream_name}={gate_g.mean().item():.4f}")
            if gate_g_strs:
                print(f'Client {client_name} gate_g_mean: {", ".join(gate_g_strs)}')
        log_feature_aggregator_agent_cosine_similarity(writer, clients, global_steps)
        timed_section_end(args, device, ag_post_start, timing_parts, "ag_post")

        # Keep actor RMS fixed to the pretrained checkpoint statistics so PPO
        # old/new log-probs are always compared under the same actor input
        # normalization. Still refresh critic RMS between iterations.
        state_stats_start = timed_section_start(args, device)
        agent.unfreeze_state_stats()
        agent.update_state_stats(obs_buf.reshape((-1,)), update_actor=False, update_critic=True)
        agent.update_state_stats(next_obs, update_actor=False, update_critic=True)
        agent.freeze_state_stats()
        timed_section_end(args, device, state_stats_start, timing_parts, "state_stats")

        global_steps += step_infos["rollot_steps"]
        pbar.update(step_infos["rollot_steps"])

        if accelerator.is_main_process:
            main_log_start = timed_section_start(args, device)
            y_pred = val_buf.flatten(0, 1).cpu().numpy()
            y_true = ret_buf.flatten(0, 1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = global_steps / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", sps, global_steps)
            writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
            writer.add_scalar("loss/value", stats["value_loss"], global_steps)
            writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
            writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
            writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
            writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
            writer.add_scalar("loss/full_kl", stats["full_kl"], global_steps)
            writer.add_scalar("loss/argmax_change_frac", stats["argmax_change_frac"], global_steps)
            writer.add_scalar("loss/explained_var", explained_var, global_steps)
            timed_section_end(args, device, main_log_start, timing_parts, "main_log")
            maybe_print_timing(args, accelerator, global_steps, step_infos["rollot_steps"], timing_parts)

    if accelerator.is_main_process:
        unwrapped_agent = accelerator.unwrap_model(agent)
        save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
        save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
        for name in agent_names:
            clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"latest_ag_{name}.pt"))
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="TwoRobotPickCube-v2")
    parser.add_argument("--ckpt-task-name", type=str, default="TwoRobotPickCube-v2_ag")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-eval-envs", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--num-minibatch", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--rollout-minibatch-size", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--state-learning-rate", type=float, default=3e-6)
    parser.add_argument("--actor-hook-learning-rate", type=float, default=3e-6)
    parser.add_argument("--feature-aggregator-learning-rate", type=float, default=3e-5)
    parser.add_argument("--value-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument(
        "--clip-vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RL4VLA-style clipped value loss instead of plain MSE value loss.",
    )
    parser.add_argument(
        "--value-huber-delta",
        type=float,
        default=10.0,
        help="Huber delta for clipped value loss.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--ent-coef-phase1", type=float, default=None)
    parser.add_argument("--ent-coef-phase2", type=float, default=None)
    parser.add_argument("--ent-coef-phase3", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.005)
    parser.add_argument(
        "--aggregator-target-kl",
        dest="aggregator_target_kl",
        type=float,
        default=2.0,
        help="KL threshold used by the feature-aggregator update stage for early stopping.",
    )
    parser.add_argument(
        "--full-kl-coef",
        type=float,
        default=0.05,
        help="Extra KL penalty over the full token distribution at each autoregressive action step.",
    )
    parser.add_argument(
        "--log-full-kl",
        action="store_true",
        help="Log rollout-old vs update-new full token-distribution KL and argmax drift even when the penalty is disabled.",
    )
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-use-best-agent", action="store_true")
    parser.add_argument("--save-interval-per-rollout", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--best-save-start-step",
        type=int,
        default=-1,
        help="First global step eligible to update best_agent/best_ag checkpoints. Default -1 means phase1_end_step.",
    )
    parser.add_argument(
        "--phase1-end-step",
        type=int,
        default=500_000,
        help="End step of phase 1: freeze backbone and head, train only feature aggregators.",
    )
    parser.add_argument(
        "--phase2-end-step",
        type=int,
        default=1_000_000,
        help="End step of phase 2: train feature aggregators plus policy heads, keep backbone LoRA frozen.",
    )
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="pandas_pandas")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="mixed_tiny_vla_smolvla",
        choices=["openvla", "tiny", "mixed_tiny_vla_smolvla"],
        help="Choose the original OpenVLA adapter backbone, the local tiny autoregressive VLA backbone, or the mixed tiny VLA + SmolVLA backbone.",
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--init-agent-path", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default="rgb+state_dict")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--vision-token-pool-size",
        type=int,
        default=None,
        help="Optionally downsample visual patch tokens before the language model; should match BC pretrain if warm-starting from BC.",
    )
    parser.add_argument("--tiny-hidden-dim", type=int, default=640)
    parser.add_argument("--tiny-vision-layers", type=int, default=7)
    parser.add_argument("--tiny-decoder-layers", type=int, default=8)
    parser.add_argument("--tiny-attention-heads", type=int, default=10)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--feature-selector-alpha", type=float, default=0.2)
    parser.add_argument("--feature-selector-topk-trajectories", type=int, default=None)
    parser.add_argument("--feature-selector-temporal-pool-steps", type=int, default=None)
    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="topk_return",
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument(
        "--eval-feature-selector-strategy",
        type=str,
        default=None,
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument("--feature-aggregator-attention-num-heads", type=int, default=4)
    parser.add_argument(
        "--feature-aggregator-gate-type",
        type=str,
        default="two-layers",
        choices=["single-layer", "two-layers"],
    )
    parser.add_argument(
        "--feature-aggregator-gate-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh"],
    )
    parser.add_argument(
        "--feature-aggregator-norm-type",
        type=str,
        default="none",
        choices=["none", "layernorm"],
    )
    parser.add_argument("--feature-aggregator-feature-gate-open-max", type=float, default=0.25)
    parser.add_argument("--feature-aggregator-action-gate-open-max", type=float, default=0.10)
    parser.add_argument("--feature-aggregator-q-ret-weight", type=float, default=0.85)
    parser.add_argument("--feature-aggregator-q-attn-weight", type=float, default=0.15)
    parser.add_argument("--feature-aggregator-remote-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-stale-shift-max", type=int, default=0)
    parser.add_argument("--gate-reg-coef", type=float, default=0.0)
    parser.add_argument("--gate-target-mean", type=float, default=0.6)
    parser.add_argument("--gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-reg-coef", type=float, default=None)
    parser.add_argument("--feature-gate-target-mean", type=float, default=None)
    parser.add_argument("--feature-gate-std-coef", type=float, default=None)
    parser.add_argument("--action-gate-reg-coef", type=float, default=None)
    parser.add_argument("--action-gate-target-mean", type=float, default=None)
    parser.add_argument("--action-gate-std-coef", type=float, default=None)
    parser.add_argument("--feature-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--action-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--feature-consistency-coef", type=float, default=0.0)
    parser.add_argument("--action-consistency-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="native",
        choices=["residual", "native"],
        help="PPO default uses native VLA action-token logits instead of the residual actor head path.",
    )
    parser.set_defaults(communication_replay=False)
    parser.add_argument(
        "--communication-replay",
        dest="communication_replay",
        action="store_true",
        help="Replay rollout-time inter-agent communication context during aggregator PPO update.",
    )
    parser.add_argument(
        "--no-communication-replay",
        dest="communication_replay",
        action="store_false",
        help="Do not replay rollout-time inter-agent communication context during aggregator PPO update.",
    )
    parser.add_argument(
        "--enable-mixed-train-envs",
        action="store_true",
        help="enable clear/mild/hard mixing inside one batched train env",
    )
    parser.add_argument(
        "--mixed-env-clean-ratio",
        type=float,
        default=0.5,
        help="ratio of clear sub-envs inside the batched train env when --enable-mixed-train-envs is set",
    )
    parser.add_argument(
        "--mixed-env-mild-ratio",
        type=float,
        default=0.3,
        help="ratio of mild sub-envs inside the batched train env when --enable-mixed-train-envs is set",
    )
    parser.add_argument(
        "--mixed-env-hard-ratio",
        type=float,
        default=0.2,
        help="ratio of hard sub-envs inside the batched train env when --enable-mixed-train-envs is set",
    )
    parser.add_argument(
        "--mixed-env-shard-size",
        type=int,
        default=1,
        help="deprecated; internal batched-env mixing now handles per-subenv randomization directly",
    )
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="print per-rollout timing breakdown for rollout / update / aggregator logging diagnosis",
    )
    parser.add_argument(
        "--timing-log-every",
        type=int,
        default=1,
        help="print timing every N rollouts when --log-timing is enabled",
    )
    parser.add_argument(
        "--disable-ag-debug-histograms",
        action="store_true",
        help="disable TensorBoard histograms for q_task and gate_g while keeping scalar diagnostics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
