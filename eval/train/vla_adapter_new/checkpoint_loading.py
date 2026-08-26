from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn


def load_compatible_policy_state(
    policy_state: Dict[str, Any],
    policy: nn.Module,
    checkpoint_path: str,
) -> bool:
    current_state = policy.state_dict()
    compatible_state: Dict[str, Any] = {}
    missing_keys: List[str] = []
    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for key, value in policy_state.items():
        current_value = current_state.get(key)
        if current_value is None:
            missing_keys.append(key)
            continue
        if tuple(value.shape) != tuple(current_value.shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(current_value.shape)))
            continue
        compatible_state[key] = value

    skipped_count = len(missing_keys) + len(shape_mismatches)
    if skipped_count > 0:
        print(
            f"[setup] static checkpoint {checkpoint_path} is only partially compatible with the current model; "
            f"loading {len(compatible_state)} tensors and skipping {skipped_count}"
        )
        for key, checkpoint_shape, current_shape in shape_mismatches[:10]:
            print(
                f"[setup] skipped shape mismatch: {key} "
                f"checkpoint={list(checkpoint_shape)} current={list(current_shape)}"
            )
        if len(shape_mismatches) > 10:
            print(f"[setup] ... and {len(shape_mismatches) - 10} more shape mismatches")
        if missing_keys:
            preview = ", ".join(missing_keys[:10])
            suffix = "" if len(missing_keys) <= 10 else ", ..."
            print(f"[setup] skipped missing keys from checkpoint target model: {preview}{suffix}")

    if not compatible_state:
        print(
            f"[setup] no compatible tensors found in static checkpoint {checkpoint_path}; "
            "keeping current policy initialization"
        )
        return False

    missing_after_load, unexpected_after_load = policy.load_state_dict(compatible_state, strict=False)
    if unexpected_after_load:
        print(f"[setup] unexpected keys ignored while loading static checkpoint: {unexpected_after_load}")
    if missing_after_load:
        print(f"[setup] keeping {len(missing_after_load)} model tensors from the current initialization")
    return True
