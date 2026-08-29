"""Past-environment evaluation used only by the copied forgetting trainers."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
from typing import Any, Callable

import torch

from train.common.env_cleanup import close_envs


@contextmanager
def _without_video(args):
    original = getattr(args, "capture_video", False)
    original_num_envs = getattr(args, "num_envs", 1)
    args.capture_video = False
    args.num_envs = 1
    try:
        yield
    finally:
        args.capture_video = original
        args.num_envs = original_num_envs


@torch.no_grad()
def _success_once(agent, eval_envs, num_steps: int) -> float | None:
    was_training = agent.training
    agent.eval()
    metrics: list[torch.Tensor] = []
    obs, _ = eval_envs.reset()
    for _ in range(num_steps):
        obs, _, _, _, infos = eval_envs.step(agent.get_action(obs, deterministic=True))
        if "final_info" not in infos:
            continue
        mask = infos.get("_final_info")
        episode = infos["final_info"].get("episode", {})
        value = episode.get("success_once")
        if value is None:
            continue
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if isinstance(mask, torch.Tensor) and tensor.ndim > 0 and tensor.shape[0] == mask.shape[0]:
            tensor = tensor[mask]
        if tensor.numel():
            metrics.append(tensor.float().reshape(-1).cpu())
    agent.train(was_training)
    return torch.cat(metrics).mean().item() if metrics else None


def record_past_env_snapshot(
    *,
    agent,
    args,
    env_ids: list[str],
    completed_env_index: int,
    elapsed_minutes: float,
    global_step: int,
    update: int,
    json_metrics,
    make_env_pair: Callable[[str, int], Any],
) -> None:
    """Evaluate the current model on Env 0..completed_env_index exactly once."""
    snapshot_number = completed_env_index + 1
    for entry in getattr(json_metrics, "history", []):
        if entry.get("forgetting_snapshot_after_env") == snapshot_number:
            return

    historical_envs: dict[str, dict[str, Any]] = {}
    eval_steps = int(os.environ.get("FORGETTING_EVAL_STEPS", str(args.num_eval_steps)))
    if eval_steps < 1:
        raise ValueError("FORGETTING_EVAL_STEPS must be positive")
    with _without_video(args):
        for env_index, env_id in enumerate(env_ids[:snapshot_number]):
            made = make_env_pair(env_id, env_index)
            train_envs, eval_envs = made[0], made[1]
            try:
                value = _success_once(agent, eval_envs, eval_steps)
            finally:
                close_envs(train_envs, eval_envs)
            historical_envs[str(env_index)] = {
                "env_id": env_id,
                "success_once": value,
            }

    values = [item["success_once"] for item in historical_envs.values() if item["success_once"] is not None]
    entry = {
        "update": int(update),
        "global_step": int(global_step),
        "current_env_id": env_ids[completed_env_index],
        "current_env_index": int(completed_env_index),
        "elapsed_hours": float(elapsed_minutes) / 60.0,
        "forgetting_snapshot_after_env": snapshot_number,
        "historical_envs": historical_envs,
        "historical_success_once": sum(values) / len(values) if values else None,
    }
    json_metrics.append(entry)
    print(
        f"[forgetting] after Env {snapshot_number}: "
        f"historical_success_once={entry['historical_success_once']}"
    )
