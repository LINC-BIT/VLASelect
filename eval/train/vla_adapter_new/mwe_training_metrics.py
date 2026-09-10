"""Shared MWE hyperparameters and train-environment accuracy collection."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch
from train.common.mwe_eval import SUCCESS_METRIC_WINDOW_EPISODES


def use_train_success_only() -> bool:
    return os.environ.get("VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY", "0") == "1"


def apply_mwe_overrides(args: Any) -> None:
    """Match VLASelect's MWE rollout footprint and accuracy horizon."""
    if os.environ.get("MWE", "0") != "1":
        return
    os.environ.setdefault("VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY", "1")
    args.num_envs = 4
    args.num_eval_envs = 1
    args.num_steps = int(os.environ.get("MWE_NUM_STEPS", "4"))
    if args.num_steps < 1:
        raise ValueError("MWE_NUM_STEPS must be positive")
    args.max_episode_steps = 100
    args.update_epochs = 1
    args.num_minibatches = 2
    args.rollout_micro_batch_size = 4
    args.eval_micro_batch_size = 4
    args.update_micro_batch_size = 2
    for name in ("small_model_feedback_schedule", "small_model_regeneration_schedule"):
        if hasattr(args, name):
            setattr(args, name, "once")
    if hasattr(args, "total_timesteps"):
        args.total_timesteps = max(args.total_timesteps, 10**12)
    mwe_runtime_minutes = float(os.environ.get("MWE_MAX_RUNTIME_MINUTES", "5.0"))
    if mwe_runtime_minutes <= 0:
        raise ValueError("MWE_MAX_RUNTIME_MINUTES must be positive")
    if hasattr(args, "max_runtime_hours"):
        args.max_runtime_hours = mwe_runtime_minutes / 60.0
    if hasattr(args, "max_time"):
        args.max_time = mwe_runtime_minutes
    if hasattr(args, "early_stop_zero_success_minutes"):
        args.early_stop_zero_success_minutes = max(args.early_stop_zero_success_minutes, 5.0)


def collect_training_policy_metric(envs: Any, agent: Any, args: Any, reference: Any) -> dict[str, float]:
    """Reset the training env and collect completed-episode success for 100 steps."""
    obs, _ = envs.reset(seed=args.seed)
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    agent.eval()
    with torch.no_grad():
        for _ in range(max(1, int(args.max_episode_steps or 100))):
            rgbs = reference.extract_rgb_batch_from_obs(obs)
            states = reference.extract_hand_state_batch_from_obs(obs)
            action, _, _, _, _ = reference.batched_get_action_and_value_no_grad(
                agent,
                rgbs,
                states,
                micro_batch_size=args.rollout_micro_batch_size,
                deterministic=False,
            )
            obs, _, _, _, infos = envs.step(action)
            done_mask = infos.get("_final_info")
            final_info = infos.get("final_info")
            if done_mask is None or not bool(done_mask.any()) or not isinstance(final_info, dict):
                continue
            done_mask = torch.as_tensor(done_mask, dtype=torch.bool)
            episode = final_info.get("episode")
            if not isinstance(episode, dict):
                continue
            for key, value in episode.items():
                value_tensor = torch.as_tensor(value)
                values[key].append(value_tensor[done_mask].float().detach().cpu())
    return {
        f"train_{key}": float(torch.cat(batch[-SUCCESS_METRIC_WINDOW_EPISODES:]).mean().item())
        for key, batch in values.items()
        if batch
    }


def add_mwe_metric_aliases(metric: dict[str, Any], collected: dict[str, float]) -> None:
    """Expose the train-environment collector under overhead's eval aliases."""
    metric.update(collected)
    for source_key, target_key in (
        ("train_success_once", "eval_success_once"),
        ("train_success_at_end", "eval_success_at_end"),
        ("train_success", "eval_success"),
    ):
        if source_key in collected:
            metric[target_key] = collected[source_key]
