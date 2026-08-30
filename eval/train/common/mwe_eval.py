from __future__ import annotations

import os
from typing import Any

import torch


_USE_TRAIN_SUCCESS_ONLY_ENV = "VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY"
SUCCESS_METRIC_WINDOW_EPISODES = 16


def use_train_success_only() -> bool:
    return os.environ.get(_USE_TRAIN_SUCCESS_ONLY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def append_episode_metric_batch(
    metric_store: dict[str, list[torch.Tensor]],
    episode_payload: dict[str, Any],
    done_mask: torch.Tensor | None = None,
) -> None:
    cpu_mask = done_mask.detach().to(device="cpu") if isinstance(done_mask, torch.Tensor) else None
    for key, value in episode_payload.items():
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device="cpu", dtype=torch.float32)
        if cpu_mask is not None and tensor.ndim > 0 and tensor.shape[0] == cpu_mask.shape[0]:
            tensor = tensor[cpu_mask]
        if tensor.numel() == 0:
            continue
        metric_store.setdefault(key, []).append(tensor.reshape(-1))


def summarize_episode_metric_tensors(
    metric_store: dict[str, list[torch.Tensor]],
    max_num_values: int | None = None,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in metric_store.items():
        flattened = [value.reshape(-1) for value in values if isinstance(value, torch.Tensor) and value.numel() > 0]
        if not flattened:
            continue
        merged = torch.cat(flattened)
        if max_num_values is not None and max_num_values > 0 and merged.numel() > max_num_values:
            merged = merged[-max_num_values:]
        # Keep the metric naming contract shared by training and plotting.
        summary[f"train_{key}"] = merged.mean().item()
    return summary


def trim_episode_metric_tensors(metric_store: dict[str, list[torch.Tensor]], max_num_values: int) -> None:
    if max_num_values <= 0:
        return
    for key, values in list(metric_store.items()):
        flattened = [value.reshape(-1) for value in values if isinstance(value, torch.Tensor) and value.numel() > 0]
        if not flattened:
            metric_store.pop(key, None)
            continue
        merged = torch.cat(flattened)
        if merged.numel() > max_num_values:
            merged = merged[-max_num_values:]
        metric_store[key] = [merged]
