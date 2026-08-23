from __future__ import annotations

from typing import Any, Dict

import torch


def load_agent_checkpoint(
    agent: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    label: str = "checkpoint",
) -> Dict[str, Any]:
    state_dict = torch.load(checkpoint_path, map_location=map_location)

    if hasattr(agent, "load_checkpoint_state_dict"):
        try:
            agent.load_checkpoint_state_dict(state_dict)
            print(f"[Checkpoint] loaded {label} with native loader: {checkpoint_path}")
            return {"mode": "native", "matched": len(state_dict), "skipped": 0}
        except Exception as exc:
            print(f"[Checkpoint] native loader failed for {label}: {exc}")

    target_state = agent.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped.append(key)
            continue
        if target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        matched[key] = value

    if not matched:
        raise RuntimeError(f"No compatible parameters found in {label}: {checkpoint_path}")

    merged_state = dict(target_state)
    merged_state.update(matched)
    agent.load_state_dict(merged_state, strict=True)
    print(
        f"[Checkpoint] partially loaded {label}: matched={len(matched)} skipped={len(skipped)} path={checkpoint_path}"
    )
    return {"mode": "partial", "matched": len(matched), "skipped": len(skipped)}
