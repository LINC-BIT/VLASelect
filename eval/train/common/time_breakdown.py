from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODULE_KEYS = (
    "workload_initialization_seconds",
    "optimal_network_search_and_selective_model_enhancement_seconds",
    "selective_knowledge_accumulation_seconds",
    "online_rl_completion_seconds",
)


def _normalized_module_breakdown(module_breakdown: dict[str, Any] | None) -> dict[str, float]:
    payload = {}
    raw = module_breakdown or {}
    for key in MODULE_KEYS:
        try:
            payload[key] = float(raw.get(key, 0.0))
        except (TypeError, ValueError):
            payload[key] = 0.0
    return payload


def build_time_breakdown_payload(
    *,
    sampling_seconds: float,
    training_seconds: float,
    module_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_module_breakdown = _normalized_module_breakdown(module_breakdown)
    time_breakdown = {
        "sampling_seconds": float(sampling_seconds),
        "training_seconds": float(training_seconds),
    }
    if module_breakdown is not None:
        time_breakdown["module_breakdown"] = normalized_module_breakdown
    payload = {
        "sampling_seconds": float(sampling_seconds),
        "training_seconds": float(training_seconds),
        "time_breakdown": time_breakdown,
    }
    if module_breakdown is not None:
        payload["module_breakdown"] = normalized_module_breakdown
        payload["vlaselect_module_breakdown"] = normalized_module_breakdown
    return payload


def augment_summary_payload(
    summary: dict[str, Any],
    *,
    sampling_seconds: float,
    training_seconds: float,
    module_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary.update(
        build_time_breakdown_payload(
            sampling_seconds=sampling_seconds,
            training_seconds=training_seconds,
            module_breakdown=module_breakdown,
        )
    )
    return summary


def write_time_breakdown(
    output_dir: Path,
    *,
    sampling_seconds: float,
    training_seconds: float,
    module_breakdown: dict[str, Any] | None = None,
    filename: str = "time_breakdown.json",
) -> Path:
    payload = build_time_breakdown_payload(
        sampling_seconds=sampling_seconds,
        training_seconds=training_seconds,
        module_breakdown=module_breakdown,
    )
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
