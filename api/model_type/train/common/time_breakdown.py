from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CANONICAL_MODULE_KEYS = (
    "workload_initialization_seconds",
    "large_model_forward_seconds",
    "small_model_generation_seconds",
    "large_model_forward_and_small_model_generation_seconds",
    "small_model_feedback_seconds",
    "online_rl_completion_seconds",
)
LEGACY_MODULE_KEY_TO_CANONICAL = {
    "optimal_network_searcher_seconds": "large_model_forward_seconds",
    "selective_model_enhancer_seconds": "small_model_generation_seconds",
    "optimal_network_search_and_selective_model_enhancement_seconds": "large_model_forward_and_small_model_generation_seconds",
    "selective_knowledge_accumulation_seconds": "small_model_feedback_seconds",
}
CANONICAL_TO_LEGACY_MODULE_KEY = {
    canonical: legacy for legacy, canonical in LEGACY_MODULE_KEY_TO_CANONICAL.items()
}
MODULE_KEYS = CANONICAL_MODULE_KEYS + tuple(LEGACY_MODULE_KEY_TO_CANONICAL.keys())


def empty_module_breakdown() -> dict[str, float]:
    return {key: 0.0 for key in MODULE_KEYS}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _module_value(module_breakdown: dict[str, Any], key: str) -> float:
    value = _safe_float(module_breakdown.get(key))
    if value is not None:
        return value
    legacy_key = CANONICAL_TO_LEGACY_MODULE_KEY.get(key)
    if legacy_key is not None:
        value = _safe_float(module_breakdown.get(legacy_key))
        if value is not None:
            return value
    canonical_key = LEGACY_MODULE_KEY_TO_CANONICAL.get(key)
    if canonical_key is not None:
        value = _safe_float(module_breakdown.get(canonical_key))
        if value is not None:
            return value
    return 0.0


def normalize_module_breakdown(module_breakdown: dict[str, Any] | None) -> dict[str, float]:
    raw = module_breakdown or {}
    payload = empty_module_breakdown()
    payload["workload_initialization_seconds"] = _module_value(raw, "workload_initialization_seconds")
    payload["large_model_forward_seconds"] = _module_value(raw, "large_model_forward_seconds")
    payload["small_model_generation_seconds"] = _module_value(raw, "small_model_generation_seconds")
    payload["large_model_forward_and_small_model_generation_seconds"] = (
        payload["large_model_forward_seconds"] + payload["small_model_generation_seconds"]
    )
    payload["small_model_feedback_seconds"] = _module_value(raw, "small_model_feedback_seconds")
    payload["online_rl_completion_seconds"] = _module_value(raw, "online_rl_completion_seconds")
    for legacy_key, canonical_key in LEGACY_MODULE_KEY_TO_CANONICAL.items():
        payload[legacy_key] = payload[canonical_key]
    return payload


def update_combined_search_enhancement_seconds(module_breakdown: dict[str, Any]) -> dict[str, Any]:
    combined_value = _module_value(module_breakdown, "large_model_forward_seconds") + _module_value(
        module_breakdown, "small_model_generation_seconds"
    )
    module_breakdown["large_model_forward_and_small_model_generation_seconds"] = combined_value
    module_breakdown["optimal_network_search_and_selective_model_enhancement_seconds"] = combined_value
    return module_breakdown


def _normalized_module_breakdown(module_breakdown: dict[str, Any] | None) -> dict[str, float]:
    return normalize_module_breakdown(module_breakdown)


def _metric_module_breakdown(metric: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(metric, dict):
        return None
    candidates = [
        metric.get("module_breakdown"),
        metric.get("vlaselect_module_breakdown"),
    ]
    time_breakdown = metric.get("time_breakdown")
    if isinstance(time_breakdown, dict):
        candidates.append(time_breakdown.get("module_breakdown"))
    for candidate in candidates:
        if isinstance(candidate, dict):
            return _normalized_module_breakdown(candidate)
    return None


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


def snapshot_time_breakdown_to_metric(
    metric: dict[str, Any],
    *,
    rollout_seconds: float,
    training_seconds: float,
    cumulative_rollout_seconds: float | None = None,
    cumulative_training_seconds: float | None = None,
    module_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_value = float(rollout_seconds)
    training_value = float(training_seconds)
    total_rollout = float(cumulative_rollout_seconds) if cumulative_rollout_seconds is not None else rollout_value
    total_training = float(cumulative_training_seconds) if cumulative_training_seconds is not None else training_value
    normalized_module_breakdown = _normalized_module_breakdown(module_breakdown) if module_breakdown is not None else None

    metric["rollout_seconds"] = rollout_value
    metric["training_seconds"] = training_value
    metric["cumulative_rollout_seconds"] = total_rollout
    metric["cumulative_training_seconds"] = total_training
    metric["time_breakdown"] = {
        "sampling_seconds": total_rollout,
        "training_seconds": total_training,
    }
    if normalized_module_breakdown is not None:
        metric["time_breakdown"]["module_breakdown"] = normalized_module_breakdown
        metric["module_breakdown"] = normalized_module_breakdown
        metric["vlaselect_module_breakdown"] = normalized_module_breakdown
    return metric


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


def _metric_rollout_increment(metric: dict[str, Any]) -> float:
    for key in ("rollout_seconds", "sampling_seconds", "rollout_time"):
        value = _safe_float(metric.get(key))
        if value is not None:
            return value
    return 0.0


def _metric_training_increment(metric: dict[str, Any]) -> float:
    value = _safe_float(metric.get("training_seconds"))
    if value is not None:
        return value
    value = _safe_float(metric.get("update_seconds"))
    if value is not None:
        return value
    total = 0.0
    for key in ("update_time", "rl_update_time", "sl_time"):
        value = _safe_float(metric.get(key))
        if value is not None:
            total += value
    return total


def write_time_breakdown_from_metrics_history(
    output_dir: Path,
    metrics_history: list[dict[str, Any]],
    *,
    filename: str = "time_breakdown.json",
) -> Path:
    latest_metric = next(
        (metric for metric in reversed(metrics_history) if isinstance(metric, dict)),
        {},
    )
    total_rollout = _safe_float(latest_metric.get("cumulative_rollout_seconds"))
    if total_rollout is None:
        total_rollout = sum(_metric_rollout_increment(metric) for metric in metrics_history if isinstance(metric, dict))

    total_training = _safe_float(latest_metric.get("cumulative_training_seconds"))
    if total_training is None:
        total_training = sum(_metric_training_increment(metric) for metric in metrics_history if isinstance(metric, dict))

    return write_time_breakdown(
        output_dir,
        sampling_seconds=float(total_rollout),
        training_seconds=float(total_training),
        module_breakdown=_metric_module_breakdown(latest_metric),
        filename=filename,
    )


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
