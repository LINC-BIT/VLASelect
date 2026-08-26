from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from common.figure_compose import compose_grid_figure, render_legend_image
from common.template_pdf_fill import fill_ours_overhead_template
TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_table"
ALL_METHODS_TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_all_methods_table"
MODULES_TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_modules_table"
SAME_ACC_TABLE_ROOT = SCRIPT_DIR / "overhead_same_acc_table"
FIG_ALL_METHODS = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.pdf"
FIG_ALL_METHODS_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.svg"
FIG_ALL_METHODS_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.png"
FIG_MODULES = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.pdf"
FIG_MODULES_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.svg"
FIG_MODULES_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.png"
ALL_METHODS_PANEL_DIR = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS_panels"

FAMILY_ORDER = ["octo", "vla_adapter_new", "tinyvla", "edgevla"]
PANEL_LABELS = {
    "octo": "a",
    "vla_adapter_new": "b",
    "tinyvla": "c",
    "edgevla": "d",
}
WORKLOAD_NAMES = {
    "octo": "Single-arm robot",
    "vla_adapter_new": "Dexterous hand",
    "tinyvla": "Mobile manipulator",
    "edgevla": "Humanoid robot",
}
FAMILY_DISPLAY_NAMES = {
    "octo": "Octo",
    "vla_adapter_new": "VLA-Adapter",
    "tinyvla": "TinyVLA",
    "edgevla": "EdgeVLA",
}
EXPECTED_METHODS = {
    "octo": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours_single_agent", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "vla_adapter_new": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "tinyvla": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "edgevla": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
}
MODULE_SPECS = [
    ("workload_initialization_seconds", "Workload init", "#4C78A8"),
    ("optimal_network_searcher_seconds", "Optimal network searcher", "#3c3c3c"),
    ("selective_model_enhancer_seconds", "Selective model enhancer", "#8f8f8f"),
    (
        "optimal_network_search_and_selective_model_enhancement_seconds",
        "Net search + SME",
        "#F58518",
    ),
    ("selective_knowledge_accumulation_seconds", "SKA", "#54A24B"),
    ("online_rl_completion_seconds", "Online RL", "#E45756"),
]
MODULE_FIGURE_SPECS = [
    ("optimal_network_searcher_seconds", "Optimal network searcher", "#3c3c3c", ""),
    ("selective_model_enhancer_seconds", "Selective model enhancer", "#8f8f8f", ""),
    ("selective_knowledge_accumulation_seconds", "Selective knowledge accumulator", "#d6d6d6", "/"),
]
NO_DATA_TEXT = "No data"
SAME_ACC_FAMILY_CONFIGS = {
    "edgevla": {"metric_key": "eval_success_once", "loader": "history"},
    "octo": {"metric_key": "eval/success_once", "loader": "tensorboard"},
    "tinyvla": {"metric_key": "eval_success_once", "loader": "history"},
    "vla_adapter_new": {"metric_key": "eval_success_once", "loader": "history"},
}
SAME_ACC_VLASELECT_METHODS_BY_FAMILY = {
    "octo": ["ours_single_agent", "ours"],
    "vla_adapter_new": ["ours"],
    "tinyvla": ["ours"],
    "edgevla": ["ours", "ours_single_agent"],
}
SAME_ACC_HISTORY_METRIC_ALIASES_BY_FAMILY = {
    "octo": ("eval_success_once", "eval/success_once", "success_once"),
    "vla_adapter_new": ("eval_success_once", "train_success_once", "success_once"),
    "tinyvla": ("eval_success_once", "train_success_once", "success_once"),
    "edgevla": ("eval_success_once", "success_once"),
}
MODULE_FIGURE_KEYS = tuple(key for key, _, _, _ in MODULE_FIGURE_SPECS)


def available_sans_serif_fonts() -> list[str]:
    candidates = ["Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans", "sans-serif"]
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    matched = [name for name in candidates if name in installed]
    return matched or ["DejaVu Sans", "Liberation Sans", "sans-serif"]


@dataclass
class MethodBreakdown:
    sampling_seconds: float = 0.0
    training_seconds: float = 0.0
    has_data: bool = False
    source: str = ""
    module_breakdown: dict[str, float] | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _finite_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolve_eval_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (EVAL_ROOT / raw_path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(EVAL_ROOT))
    except ValueError:
        return str(path)


def _extract_module_breakdown(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("module_breakdown"),
        payload.get("modules"),
        payload.get("vlaselect_module_breakdown"),
        payload.get("time_breakdown", {}).get("module_breakdown") if isinstance(payload.get("time_breakdown"), dict) else None,
        payload.get("breakdown", {}).get("module_breakdown") if isinstance(payload.get("breakdown"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {key: _safe_float(candidate.get(key, 0.0)) for key, _, _ in MODULE_SPECS}
    if any(key in payload for key, _, _ in MODULE_SPECS):
        return {key: _safe_float(payload.get(key, 0.0)) for key, _, _ in MODULE_SPECS}
    return {key: 0.0 for key, _, _ in MODULE_SPECS}


def _has_module_figure_data(module_breakdown: dict[str, Any] | None) -> bool:
    if not isinstance(module_breakdown, dict):
        return False
    return any(_safe_float(module_breakdown.get(key, 0.0)) > 0.0 for key in MODULE_FIGURE_KEYS)


def _extract_sampling_training(payload: dict[str, Any]) -> tuple[float, float, bool]:
    for container_key in ("time_breakdown", "breakdown", "timing_breakdown"):
        candidate = payload.get(container_key)
        if isinstance(candidate, dict):
            sampling = _safe_float(
                candidate.get("sampling_seconds", candidate.get("rollout_seconds", candidate.get("sampling_time_seconds", 0.0)))
            )
            training = _safe_float(
                candidate.get("training_seconds", candidate.get("update_seconds", candidate.get("training_time_seconds", 0.0)))
            )
            return sampling, training, sampling > 0.0 or training > 0.0
    sampling = _safe_float(payload.get("sampling_seconds", payload.get("rollout_seconds", 0.0)))
    training = _safe_float(payload.get("training_seconds", payload.get("update_seconds", 0.0)))
    return sampling, training, sampling > 0.0 or training > 0.0


def _candidate_breakdown_paths(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / "time_breakdown.json",
        run_dir / "timing_breakdown.json",
        run_dir / "breakdown.json",
    ]
    candidates.extend(sorted(run_dir.glob("*training_summary.json")))
    candidates.extend(sorted((run_dir / "analysis").glob("*breakdown*.json")) if (run_dir / "analysis").exists() else [])
    return [path for path in candidates if path.exists()]


def _load_history_time_breakdown(run_dir: Path) -> MethodBreakdown | None:
    history = _load_history(run_dir)
    if not history:
        return None
    latest_metric = next((metric for metric in reversed(history) if isinstance(metric, dict)), None)
    if latest_metric is None:
        return None
    module_breakdown = _extract_module_breakdown(latest_metric)
    sampling_seconds = _finite_float_or_none(latest_metric.get("cumulative_rollout_seconds"))
    training_seconds = _finite_float_or_none(latest_metric.get("cumulative_training_seconds"))
    if sampling_seconds is None:
        sampling_seconds = sum(_safe_float(metric.get("rollout_seconds")) for metric in history if isinstance(metric, dict))
    if training_seconds is None:
        training_seconds = sum(_safe_float(metric.get("training_seconds")) for metric in history if isinstance(metric, dict))
    has_data = (
        (sampling_seconds or 0.0) > 0.0
        or (training_seconds or 0.0) > 0.0
        or any(_safe_float(value) > 0.0 for value in module_breakdown.values())
    )
    if not has_data:
        return None
    history_path = run_dir / "metrics_history.json"
    return MethodBreakdown(
        sampling_seconds=float(sampling_seconds or 0.0),
        training_seconds=float(training_seconds or 0.0),
        has_data=True,
        source=_display_path(history_path),
        module_breakdown=module_breakdown,
    )


def _load_tensorboard_time_breakdown(run_dir: Path) -> MethodBreakdown | None:
    tb_dir = _find_tb_dir(run_dir)
    if tb_dir is None:
        return None
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
    except Exception:
        return None
    tags = set(accumulator.Tags().get("scalars", []))

    def scalar_total(tag: str) -> float:
        if tag not in tags:
            return 0.0
        try:
            return sum(float(event.value) for event in accumulator.Scalars(tag))
        except Exception:
            return 0.0

    sampling_seconds = scalar_total("time/rollout_time")
    training_seconds = (
        scalar_total("time/update_time")
        + scalar_total("time/rl_update_time")
        + scalar_total("time/sl_time")
    )
    has_data = sampling_seconds > 0.0 or training_seconds > 0.0
    if not has_data:
        return None
    return MethodBreakdown(
        sampling_seconds=sampling_seconds,
        training_seconds=training_seconds,
        has_data=True,
        source=_display_path(tb_dir),
        module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS},
    )


def load_method_breakdown(run_dir: Path) -> MethodBreakdown:
    for candidate in _candidate_breakdown_paths(run_dir):
        payload = _read_json(candidate)
        if not isinstance(payload, dict):
            continue
        sampling, training, has_data = _extract_sampling_training(payload)
        module_breakdown = _extract_module_breakdown(payload)
        if has_data or any(value > 0.0 for value in module_breakdown.values()):
            return MethodBreakdown(
                sampling_seconds=sampling,
                training_seconds=training,
                has_data=True,
                source=_display_path(candidate),
                module_breakdown=module_breakdown,
            )
    history_breakdown = _load_history_time_breakdown(run_dir)
    if history_breakdown is not None:
        return history_breakdown
    tensorboard_breakdown = _load_tensorboard_time_breakdown(run_dir)
    if tensorboard_breakdown is not None:
        return tensorboard_breakdown
    return MethodBreakdown(module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS})


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(run_dir / "metrics_history.json")
    if not isinstance(payload, dict):
        return []
    history = payload.get("history", [])
    return [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []


def _collect_history_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    for index, metric in enumerate(_load_history(run_dir)):
        value = _finite_float_or_none(metric.get(metric_key))
        if value is None:
            continue
        elapsed_hours = _finite_float_or_none(metric.get("elapsed_hours"))
        x_value = elapsed_hours if elapsed_hours is not None else float(index)
        series.append((x_value, value))
    return series


def _find_tb_dir(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "tb", run_dir / "[agent]" / "tb"):
        if candidate.is_dir():
            return candidate
    for search_root in (run_dir, run_dir.parent):
        if not search_root.exists():
            continue
        nested = sorted(path for path in search_root.glob("**/tb") if path.is_dir())
        if nested:
            return nested[0]
    return None


def _collect_tensorboard_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
    tb_dir = _find_tb_dir(run_dir)
    if tb_dir is None:
        return []
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
    except Exception:
        return []
    tags = accumulator.Tags().get("scalars", [])
    if metric_key not in tags:
        return []
    events = accumulator.Scalars(metric_key)
    if not events:
        return []
    base_time = events[0].wall_time
    return [((event.wall_time - base_time) / 3600.0, float(event.value)) for event in events]


def _extract_history_success_value(family: str, metric: dict[str, Any]) -> float | None:
    for key in SAME_ACC_HISTORY_METRIC_ALIASES_BY_FAMILY.get(family, ()): 
        value = _finite_float_or_none(metric.get(key))
        if value is not None:
            return value
    return None


def _collect_same_acc_series(family: str, run_dir: Path) -> list[tuple[float, float]]:
    config = SAME_ACC_FAMILY_CONFIGS[family]
    if config["loader"] == "tensorboard":
        return _collect_tensorboard_series(run_dir, config["metric_key"])
    return _collect_history_series(run_dir, config["metric_key"])


def _pick_same_acc_vlaselect_method(methods: list[dict[str, Any]], family: str) -> dict[str, Any] | None:
    by_name = {method.get("name"): method for method in methods if isinstance(method, dict)}
    for name in SAME_ACC_VLASELECT_METHODS_BY_FAMILY.get(family, []):
        if name in by_name:
            return by_name[name]
    return None


def _first_reach_hours(series: list[tuple[float, float]], target_accuracy: float) -> float | None:
    for elapsed_hours, value in series:
        if value >= target_accuracy:
            return elapsed_hours
    return None


def _metric_active_runtime_seconds(metric: dict[str, Any]) -> float | None:
    rollout = _finite_float_or_none(metric.get("cumulative_rollout_seconds"))
    training = _finite_float_or_none(metric.get("cumulative_training_seconds"))
    if rollout is not None or training is not None:
        return float(rollout or 0.0) + float(training or 0.0)
    time_breakdown = metric.get("time_breakdown")
    if isinstance(time_breakdown, dict):
        sampling = _finite_float_or_none(time_breakdown.get("sampling_seconds"))
        training = _finite_float_or_none(time_breakdown.get("training_seconds"))
        if sampling is not None or training is not None:
            return float(sampling or 0.0) + float(training or 0.0)
    return None


def _same_acc_summary_path(top_manifest: dict[str, Any]) -> Path | None:
    manifest_path = _resolve_same_acc_manifest_path(top_manifest)
    if manifest_path is None:
        return None
    candidate = manifest_path.parent / "overhead_same_acc_summary.json"
    return candidate if candidate.exists() else None


def _load_same_acc_cutoff_hours(top_manifest: dict[str, Any]) -> tuple[dict[tuple[str, str], float], str]:
    summary_path = _same_acc_summary_path(top_manifest)
    if summary_path is None:
        return {}, ""
    payload = _read_json(summary_path)
    if not isinstance(payload, list):
        return {}, _display_path(summary_path)
    cutoff_hours_by_method: dict[tuple[str, str], float] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family", "")).strip()
        method = str(row.get("method", "")).strip()
        cutoff_hours = _finite_float_or_none(row.get("reach_hours"))
        if not family or not method or cutoff_hours is None or cutoff_hours <= 0.0:
            continue
        cutoff_hours_by_method[(family, method)] = float(cutoff_hours)
    return cutoff_hours_by_method, _display_path(summary_path)


def _metric_sampling_training_cumulative(metric: dict[str, Any]) -> tuple[float | None, float | None]:
    sampling = _finite_float_or_none(metric.get("cumulative_rollout_seconds"))
    training = _finite_float_or_none(metric.get("cumulative_training_seconds"))
    if sampling is not None or training is not None:
        return sampling, training
    time_breakdown = metric.get("time_breakdown")
    if isinstance(time_breakdown, dict):
        sampling = _finite_float_or_none(time_breakdown.get("sampling_seconds"))
        training = _finite_float_or_none(time_breakdown.get("training_seconds"))
        if sampling is not None or training is not None:
            return sampling, training
    return None, None


def _interpolate_breakdown_at_cutoff(points: list[tuple[float, float, float]], cutoff_hours: float) -> tuple[float, float]:
    if not points or cutoff_hours <= 0.0:
        return 0.0, 0.0
    points = sorted(points, key=lambda item: item[0])
    if cutoff_hours <= points[0][0]:
        first_hours, first_sampling, first_training = points[0]
        if first_hours <= 0.0:
            return max(0.0, first_sampling), max(0.0, first_training)
        ratio = max(0.0, min(1.0, cutoff_hours / first_hours))
        return max(0.0, first_sampling * ratio), max(0.0, first_training * ratio)
    previous_hours = 0.0
    previous_sampling = 0.0
    previous_training = 0.0
    for current_hours, current_sampling, current_training in points:
        if cutoff_hours <= current_hours:
            span = current_hours - previous_hours
            if span <= 0.0:
                return max(0.0, current_sampling), max(0.0, current_training)
            ratio = max(0.0, min(1.0, (cutoff_hours - previous_hours) / span))
            sampling = previous_sampling + ratio * (current_sampling - previous_sampling)
            training = previous_training + ratio * (current_training - previous_training)
            return max(0.0, sampling), max(0.0, training)
        previous_hours = current_hours
        previous_sampling = current_sampling
        previous_training = current_training
    return max(0.0, points[-1][1]), max(0.0, points[-1][2])


def _load_history_time_breakdown_until_cutoff(run_dir: Path, cutoff_hours: float) -> MethodBreakdown | None:
    history = _load_history(run_dir)
    if not history:
        return None
    points: list[tuple[float, float, float]] = []
    for metric in history:
        if not isinstance(metric, dict):
            continue
        elapsed_hours = _finite_float_or_none(metric.get("elapsed_hours"))
        if elapsed_hours is None:
            continue
        sampling, training = _metric_sampling_training_cumulative(metric)
        if sampling is None and training is None:
            continue
        points.append((elapsed_hours, float(sampling or 0.0), float(training or 0.0)))
    if not points:
        return None
    sampling_seconds, training_seconds = _interpolate_breakdown_at_cutoff(points, cutoff_hours)
    history_path = run_dir / "metrics_history.json"
    return MethodBreakdown(
        sampling_seconds=sampling_seconds,
        training_seconds=training_seconds,
        has_data=(sampling_seconds > 0.0 or training_seconds > 0.0),
        source=_display_path(history_path),
        module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS},
    )


def _load_tensorboard_time_breakdown_until_cutoff(run_dir: Path, cutoff_hours: float) -> MethodBreakdown | None:
    tb_dir = _find_tb_dir(run_dir)
    if tb_dir is None:
        return None
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
    except Exception:
        return None
    tags = set(accumulator.Tags().get("scalars", []))
    if "time/elapsed_minutes" not in tags:
        return None
    try:
        elapsed_events = accumulator.Scalars("time/elapsed_minutes")
    except Exception:
        return None
    if not elapsed_events:
        return None
    elapsed_hours_by_step: dict[int, float] = {}
    for event in elapsed_events:
        try:
            elapsed_hours_by_step[int(event.step)] = float(event.value) / 60.0
        except Exception:
            continue
    if not elapsed_hours_by_step:
        return None
    sampling_seconds = 0.0
    training_seconds = 0.0

    def accumulate(tag: str) -> float:
        if tag not in tags:
            return 0.0
        try:
            events = accumulator.Scalars(tag)
        except Exception:
            return 0.0
        total = 0.0
        for event in events:
            elapsed_hours = elapsed_hours_by_step.get(int(event.step))
            if elapsed_hours is None or elapsed_hours > cutoff_hours:
                continue
            total += float(event.value)
        return total

    sampling_seconds += accumulate("time/rollout_time")
    training_seconds += accumulate("time/update_time")
    training_seconds += accumulate("time/rl_update_time")
    training_seconds += accumulate("time/sl_time")
    has_data = sampling_seconds > 0.0 or training_seconds > 0.0
    if not has_data:
        return None
    return MethodBreakdown(
        sampling_seconds=sampling_seconds,
        training_seconds=training_seconds,
        has_data=True,
        source=_display_path(tb_dir),
        module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS},
    )


def load_method_breakdown_until_cutoff(run_dir: Path, cutoff_hours: float) -> MethodBreakdown:
    history_breakdown = _load_history_time_breakdown_until_cutoff(run_dir, cutoff_hours)
    if history_breakdown is not None:
        return history_breakdown
    tensorboard_breakdown = _load_tensorboard_time_breakdown_until_cutoff(run_dir, cutoff_hours)
    if tensorboard_breakdown is not None:
        return tensorboard_breakdown
    return MethodBreakdown(module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS})


def _load_same_acc_reach_metric(family: str, run_dir: Path) -> dict[str, Any] | None:
    history = _load_history(run_dir)
    if not history:
        return None
    target_accuracy: float | None = None
    for metric in history:
        value = _extract_history_success_value(family, metric)
        if value is None:
            continue
        target_accuracy = value if target_accuracy is None else max(target_accuracy, value)
    if target_accuracy is None:
        return None
    for metric in history:
        value = _extract_history_success_value(family, metric)
        if value is not None and value >= target_accuracy:
            return metric
    return None


def _resolve_same_acc_manifest_path(top_manifest: dict[str, Any]) -> Path | None:
    raw_path = str(top_manifest.get("same_acc_manifest", "")).strip()
    candidates: list[Path] = []
    if raw_path:
        candidates.append(_resolve_eval_path(raw_path))
    latest_path = SAME_ACC_TABLE_ROOT / "latest.txt"
    if latest_path.exists():
        stamp = latest_path.read_text(encoding="utf-8").strip()
        if stamp:
            candidates.append((SAME_ACC_TABLE_ROOT / stamp / "manifest.json").resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_same_acc_vlaselect_times_seconds(top_manifest: dict[str, Any]) -> tuple[dict[str, float], str]:
    manifest_path = _resolve_same_acc_manifest_path(top_manifest)
    if manifest_path is None:
        return {}, ""
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        return {}, _display_path(manifest_path)
    panels = payload.get("families", payload.get("panels", []))
    times_by_family: dict[str, float] = {}
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        family = str(panel.get("family", ""))
        if family not in FAMILY_ORDER:
            continue
        suite_manifest_ref = str(panel.get("suite_manifest", ""))
        if not suite_manifest_ref:
            continue
        suite_manifest = _read_json(_resolve_eval_path(suite_manifest_ref))
        if not isinstance(suite_manifest, dict):
            continue
        methods = [method for method in suite_manifest.get("methods", []) if isinstance(method, dict)]
        vlaselect_method = _pick_same_acc_vlaselect_method(methods, family)
        if vlaselect_method is None:
            continue
        run_dir_ref = str(vlaselect_method.get("run_dir", ""))
        if not run_dir_ref:
            continue
        reach_metric = _load_same_acc_reach_metric(family, _resolve_eval_path(run_dir_ref))
        if not isinstance(reach_metric, dict):
            continue
        active_runtime_seconds = _metric_active_runtime_seconds(reach_metric)
        if active_runtime_seconds is not None:
            times_by_family[family] = active_runtime_seconds
    return times_by_family, _display_path(manifest_path)


def _load_same_acc_vlaselect_module_breakdowns(top_manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_path = _resolve_same_acc_manifest_path(top_manifest)
    if manifest_path is None:
        return {}, ""
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        return {}, _display_path(manifest_path)
    panels = payload.get("families", payload.get("panels", []))
    module_rows: dict[str, dict[str, Any]] = {}
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        family = str(panel.get("family", ""))
        if family not in FAMILY_ORDER:
            continue
        suite_manifest_ref = str(panel.get("suite_manifest", ""))
        if not suite_manifest_ref:
            continue
        suite_manifest = _read_json(_resolve_eval_path(suite_manifest_ref))
        if not isinstance(suite_manifest, dict):
            continue
        methods = [method for method in suite_manifest.get("methods", []) if isinstance(method, dict)]
        vlaselect_method = _pick_same_acc_vlaselect_method(methods, family)
        if vlaselect_method is None:
            continue
        run_dir_ref = str(vlaselect_method.get("run_dir", ""))
        if not run_dir_ref:
            continue
        run_dir = _resolve_eval_path(run_dir_ref)
        reach_metric = _load_same_acc_reach_metric(family, run_dir)
        if not isinstance(reach_metric, dict):
            continue
        module_breakdown = _extract_module_breakdown(reach_metric)
        active_runtime_seconds = _metric_active_runtime_seconds(reach_metric)
        if active_runtime_seconds is not None:
            module_breakdown["online_rl_completion_seconds"] = active_runtime_seconds
        module_rows[family] = {
            "module_breakdown": module_breakdown,
            "active_runtime_seconds": float(active_runtime_seconds or 0.0),
            "source": _display_path(run_dir / "metrics_history.json"),
        }
    return module_rows, _display_path(manifest_path)


def _default_top_manifest() -> dict[str, Any]:
    panels = []
    for family in FAMILY_ORDER:
        panels.append(
            {
                "family": family,
                "suite_manifest": "",
                "suite_root": "",
                "launch_log": "",
                "suite_stamp": "no-data",
                "panel_label": PANEL_LABELS[family],
                "workload_name": WORKLOAD_NAMES[family],
                "display_name": FAMILY_DISPLAY_NAMES[family],
            }
        )
    return {
        "suite_stamp": "no-data",
        "table_root": "overhead/overhead_breakdown_table",
        "figure_all_methods_output": "overhead/FIG_BREAKDOWN_ALL_METHODS.pdf",
        "figure_modules_output": "overhead/FIG_BREAKDOWN_MODULES.pdf",
        "all_methods_csv": "overhead/overhead_breakdown_table/BREAKDOWN_ALL_METHODS.csv",
        "modules_csv": "overhead/overhead_breakdown_table/BREAKDOWN_MODULES.csv",
        "same_acc_manifest": "",
        "panels": panels,
        "families": panels,
    }


def load_top_manifest_from_table_root(
    table_root: Path,
    manifest_path: str | None,
) -> tuple[dict[str, Any], Path | None]:
    if manifest_path:
        path = Path(manifest_path)
        payload = _read_json(path)
        return (payload if isinstance(payload, dict) else _default_top_manifest(), path)
    latest_path = table_root / "latest.txt"
    if latest_path.exists():
        stamp = latest_path.read_text(encoding="utf-8").strip()
        if stamp:
            candidate = table_root / stamp / "manifest.json"
            payload = _read_json(candidate)
            if isinstance(payload, dict):
                return payload, candidate
    return _default_top_manifest(), None


def load_top_manifest(manifest_path: str | None) -> tuple[dict[str, Any], Path | None]:
    return load_top_manifest_from_table_root(TABLE_ROOT, manifest_path)


def _family_panels(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("families", manifest.get("panels", [])):
        family = item.get("family")
        if family:
            result[family] = item
    for family in FAMILY_ORDER:
        result.setdefault(
            family,
            {
                "family": family,
                "panel_label": PANEL_LABELS[family],
                "workload_name": WORKLOAD_NAMES[family],
                "display_name": FAMILY_DISPLAY_NAMES[family],
                "suite_manifest": "",
                "suite_root": "",
                "launch_log": "",
            },
        )
    return result


def _load_suite_manifest(panel: dict[str, Any]) -> dict[str, Any] | None:
    manifest_ref = panel.get("suite_manifest") or ""
    if not manifest_ref:
        return None
    payload = _read_json(_resolve_eval_path(str(manifest_ref)))
    return payload if isinstance(payload, dict) else None


def _expected_method_entries(family: str) -> list[dict[str, Any]]:
    return [
        {"name": name, "display_name": display_name, "run_dir": ""}
        for name, display_name in EXPECTED_METHODS[family]
    ]


def _normalize_display_name(name: str, display_name: str) -> str:
    if name in {"ours", "ours_single_agent"}:
        return "VLASelect"
    return display_name or name


def prepare_breakdown_tables(manifest: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    same_acc_times_seconds, same_acc_source = _load_same_acc_vlaselect_times_seconds(manifest)
    same_acc_module_breakdowns, same_acc_module_source = _load_same_acc_vlaselect_module_breakdowns(manifest)
    same_acc_cutoff_hours_by_method, same_acc_cutoff_source = _load_same_acc_cutoff_hours(manifest)

    for family, panel in _family_panels(manifest).items():
        suite_manifest = _load_suite_manifest(panel)
        methods = suite_manifest.get("methods", []) if isinstance(suite_manifest, dict) else []
        if not methods:
            methods = _expected_method_entries(family)

        suite_map = {method.get("name"): method for method in methods if method.get("name")}
        ordered_methods: list[dict[str, Any]] = []
        for name, display_name in EXPECTED_METHODS[family]:
            method = dict(suite_map.get(name, {}))
            method.setdefault("name", name)
            method.setdefault("display_name", display_name)
            method.setdefault("run_dir", "")
            ordered_methods.append(method)

        family_module_row_added = False
        for method in ordered_methods:
            run_dir_ref = method.get("run_dir") or ""
            run_dir = _resolve_eval_path(str(run_dir_ref)) if run_dir_ref else Path("")
            cutoff_hours = same_acc_cutoff_hours_by_method.get((family, str(method.get("name", ""))))
            if run_dir_ref and cutoff_hours is not None and cutoff_hours > 0.0:
                breakdown = load_method_breakdown_until_cutoff(run_dir, cutoff_hours)
                if not breakdown.has_data:
                    breakdown = load_method_breakdown(run_dir)
            else:
                breakdown = load_method_breakdown(run_dir) if run_dir_ref else MethodBreakdown(module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS})
            display_name = _normalize_display_name(method["name"], method.get("display_name", ""))
            all_rows.append(
                {
                    "family": family,
                    "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                    "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                    "method_name": method["name"],
                    "display_name": display_name,
                    "sampling_seconds": breakdown.sampling_seconds,
                    "training_seconds": breakdown.training_seconds,
                    "total_seconds": breakdown.sampling_seconds + breakdown.training_seconds,
                    "has_breakdown_data": int(breakdown.has_data),
                    "source": breakdown.source,
                }
            )
            if method["name"] in {"ours", "ours_single_agent"}:
                module_breakdown = dict(breakdown.module_breakdown or {key: 0.0 for key, _, _ in MODULE_SPECS})
                same_acc_runtime_seconds = _safe_float(same_acc_times_seconds.get(family))
                runtime_source = breakdown.source
                same_acc_module_row = same_acc_module_breakdowns.get(family)
                module_source = breakdown.source
                if isinstance(same_acc_module_row, dict):
                    candidate_breakdown = same_acc_module_row.get("module_breakdown")
                    if _has_module_figure_data(candidate_breakdown):
                        module_breakdown = dict(candidate_breakdown)
                        module_source = str(same_acc_module_row.get("source") or module_source)
                        runtime_source = str(same_acc_module_row.get("source") or same_acc_module_source or runtime_source)
                elif same_acc_runtime_seconds > 0.0:
                    runtime_source = same_acc_source or runtime_source
                total_seconds = sum(_safe_float(module_breakdown.get(key, 0.0)) for key in MODULE_FIGURE_KEYS)
                module_rows.append(
                    {
                        "family": family,
                        "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                        "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                        "display_name": display_name,
                        **module_breakdown,
                        "total_seconds": total_seconds,
                        "has_module_data": int(_has_module_figure_data(module_breakdown)),
                        "source": module_source,
                        "runtime_source": runtime_source,
                        "same_acc_runtime_seconds": same_acc_runtime_seconds,
                    }
                )
                family_module_row_added = True

        if not family_module_row_added:
            module_breakdown = {key: 0.0 for key, _, _ in MODULE_SPECS}
            module_rows.append(
                {
                    "family": family,
                    "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                    "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                    "display_name": "VLASelect",
                    **module_breakdown,
                    "total_seconds": 0.0,
                    "has_module_data": 0,
                    "source": "",
                    "runtime_source": same_acc_source if same_acc_source else "",
                    "same_acc_runtime_seconds": _safe_float(same_acc_times_seconds.get(family)),
                }
            )

    write_csv(
        output_root / "BREAKDOWN_ALL_METHODS.csv",
        [
            "family",
            "panel_label",
            "workload_name",
            "method_name",
            "display_name",
            "sampling_seconds",
            "training_seconds",
            "total_seconds",
            "has_breakdown_data",
            "source",
        ],
        all_rows,
    )
    write_csv(
        output_root / "BREAKDOWN_MODULES.csv",
        [
            "family",
            "panel_label",
            "workload_name",
            "display_name",
            *[key for key, _, _ in MODULE_SPECS],
            "total_seconds",
            "has_module_data",
            "source",
            "runtime_source",
            "same_acc_runtime_seconds",
        ],
        module_rows,
    )
    (output_root / "breakdown_summary.json").write_text(
        json.dumps(
            {
                "all_methods_rows": all_rows,
                "module_rows": module_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return all_rows, module_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _set_common_axis_style(ax: Any) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_time_axis_locator() -> MaxNLocator:
    return MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)


def _dynamic_time_axis_upper(
    values: np.ndarray | list[float],
    margin_ratio: float = 0.12,
    default_upper: float = 1.0,
) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return default_upper
    max_value = float(np.max(data))
    if max_value <= 0.0:
        return default_upper

    raw_upper = max_value * (1.0 + margin_ratio)
    tick_values = _build_time_axis_locator().tick_values(0.0, raw_upper)
    positive_ticks = [float(tick) for tick in tick_values if tick > 0.0]
    if positive_ticks:
        return positive_ticks[-1]
    return raw_upper


def apply_dynamic_time_axis(
    ax: Any,
    values: np.ndarray | list[float],
    margin_ratio: float = 0.12,
    default_upper: float = 1.0,
) -> float:
    upper = _dynamic_time_axis_upper(
        values,
        margin_ratio=margin_ratio,
        default_upper=default_upper,
    )
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(_build_time_axis_locator())
    return upper


def plot_all_methods(rows: list[dict[str, Any]]) -> None:
    grouped = _group_rows(rows, 'family')
    panel_paths: list[Path] = []
    legend_path = ALL_METHODS_PANEL_DIR / 'legend.png'
    render_legend_image([('Sampling', {'color': '#4C78A8', 'linestyle': '-'}), ('Model training', {'color': '#F58518', 'linestyle': '-'})], legend_path, ncol=2, fontsize=11)

    for family in FAMILY_ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.8), constrained_layout=False)
        family_rows = grouped.get(family, [])
        labels = [row['display_name'] for row in family_rows]
        sampling = np.array([_safe_float(row['sampling_seconds']) for row in family_rows], dtype=float)
        training = np.array([_safe_float(row['training_seconds']) for row in family_rows], dtype=float)
        has_data = [int(row.get('has_breakdown_data', 0)) == 1 for row in family_rows]
        x = np.arange(len(labels))

        ax.bar(x, sampling, color='white', edgecolor='black', linewidth=2.0, width=0.72)
        ax.bar(x, training, bottom=sampling, color='white', edgecolor='black', linewidth=2.0, hatch='/', width=0.72)
        ax.set_title(f"({PANEL_LABELS[family]}) {WORKLOAD_NAMES[family]}", fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=42, ha='right', fontsize=9)
        ax.set_ylabel('Time (s)', fontsize=11)
        _set_common_axis_style(ax)

        apply_dynamic_time_axis(ax, sampling + training, margin_ratio=0.12, default_upper=1.0)
        for x_pos, ok in zip(x, has_data):
            if not ok:
                ax.text(x_pos, ax.get_ylim()[1] * 0.05, NO_DATA_TEXT, rotation=90, ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        ALL_METHODS_PANEL_DIR.mkdir(parents=True, exist_ok=True)
        panel_path = ALL_METHODS_PANEL_DIR / f'{family}.png'
        fig.savefig(panel_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        panel_paths.append(panel_path)

    compose_grid_figure(
        panel_paths,
        output_paths=[FIG_ALL_METHODS, FIG_ALL_METHODS_SVG, FIG_ALL_METHODS_PNG],
        rows=1,
        cols=4,
        figsize=(20.0, 5.0),
        legend_path=legend_path,
        legend_height_ratio=0.11,
        wspace=0.02,
        hspace=0.02,
        dpi=200,
    )

def plot_modules(rows: list[dict[str, Any]]) -> None:
    grouped = {row["family"]: row for row in rows}
    workloads = [f"Workload {idx + 1}:{WORKLOAD_NAMES[family]}" for idx, family in enumerate(FAMILY_ORDER)]
    y_positions = np.arange(len(workloads))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": available_sans_serif_fonts(),
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "svg.fonttype": "none",
        }
    )

    values = np.array(
        [
            [_safe_float(grouped.get(family, {}).get(key, 0.0)) for key, _, _, _ in MODULE_FIGURE_SPECS]
            for family in FAMILY_ORDER
        ],
        dtype=float,
    )

    fig_height = max(3.35, 0.42 * len(workloads) + 1.2)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), constrained_layout=False)

    left = np.zeros(len(FAMILY_ORDER), dtype=float)
    for module_index, (_, label, color, hatch) in enumerate(MODULE_FIGURE_SPECS):
        ax.barh(
            y_positions,
            values[:, module_index],
            left=left,
            height=0.5,
            label=label,
            zorder=10,
            edgecolor='black',
            color=color,
            hatch=hatch,
            lw=1,
        )
        left += values[:, module_index]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(workloads, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.grid(zorder=-10, axis='x', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.tick_params(axis='y', length=0)

    if float(np.max(values)) <= 0.0:
        ax.set_xlim(0.0, 1.0)
        for idx, family in enumerate(FAMILY_ORDER):
            row = grouped.get(family)
            if not row or int(row.get("has_module_data", 0)) != 1:
                ax.text(0.02, idx, NO_DATA_TEXT, ha='left', va='center', fontsize=9)
    else:
        totals = np.sum(values, axis=1)
        upper = _dynamic_time_axis_upper(totals, margin_ratio=0.12, default_upper=1.0)
        ax.set_xlim(0.0, upper)
        for idx, family in enumerate(FAMILY_ORDER):
            row = grouped.get(family)
            if not row or int(row.get("has_module_data", 0)) != 1:
                ax.text(0.02 * upper, idx, NO_DATA_TEXT, ha='left', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_MODULES, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_MODULES_SVG, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_MODULES_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    try:
        fill_ours_overhead_template(FIG_MODULES, FIG_MODULES_PNG)
    except Exception as exc:
        print(f'[template] failed to fill VLASelect-overhead template: {exc}')


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(manifest_path: str | None, prepare_only: bool) -> None:
    manifest, resolved_manifest_path = load_top_manifest(manifest_path)
    if resolved_manifest_path is not None:
        output_root = resolved_manifest_path.parent
    elif manifest.get("suite_stamp") not in {None, "", "no-data"}:
        output_root = TABLE_ROOT / str(manifest["suite_stamp"])
    else:
        output_root = TABLE_ROOT

    all_rows, module_rows = prepare_breakdown_tables(manifest, output_root)
    if prepare_only:
        return
    if not all_rows:
        all_rows = load_csv_rows(output_root / "BREAKDOWN_ALL_METHODS.csv")
    if not module_rows:
        module_rows = load_csv_rows(output_root / "BREAKDOWN_MODULES.csv")
    plot_all_methods(all_rows)
    plot_modules(module_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    run(args.manifest, args.prepare_only)


if __name__ == "__main__":
    main()
