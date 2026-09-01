from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


def summarize_episode_metrics(episode_metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, Tuple[float, int]]:
    summary: Dict[str, Tuple[float, int]] = {}
    for key, values in episode_metrics.items():
        if not values:
            continue
        cat = torch.cat(values)
        summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
    return summary


def append_live_episode_metrics(live_episode_metrics: Dict[str, List[torch.Tensor]], infos: Dict[str, object]) -> None:
    episode_metrics = infos.get("episode")
    if not isinstance(episode_metrics, dict):
        return
    for key in ("success_once", "success_at_end"):
        value = episode_metrics.get(key)
        if value is None:
            value = episode_metrics.get("success")
        if value is None:
            continue
        live_episode_metrics[key].append(value.detach().float().cpu())


def summarize_training_episode_metrics(
    completed_episode_metrics: Dict[str, List[torch.Tensor]],
    live_episode_metrics: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Dict[str, Tuple[float, int]]:
    summary = summarize_episode_metrics(completed_episode_metrics)
    if not live_episode_metrics:
        return summary
    live_summary = summarize_episode_metrics(live_episode_metrics)
    for key, value in live_summary.items():
        if key in summary:
            summary[f"{key}_completed"] = summary[key]
        summary[key] = value
    return summary
