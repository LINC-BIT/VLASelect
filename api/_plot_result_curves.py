"""Shared plotting helper for unified API training-result curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# This directory may contain artifacts from before ``knowledge_distillation`` was
# renamed to ``logit_distillation``.  Do not mix that legacy run into the unified
# comparison chart.
_IGNORED_METHOD_DIRS = {"knowledge_distillation"}
_METHOD_LABELS = {
    "default": "VLASelect",
    "attn_distillation": "Attention Distillation",
    "data_distillation": "Data Distillation",
    "distillm": "DistiLLM",
    "edgeta": "EdgeTA",
    "feature_distillation": "Feature Distillation",
    "llm_in_a_flash": "LLM in a Flash",
    "llm_pruner": "LLM-Pruner",
    "logit_distillation": "Logit Distillation",
    "minillm": "MiniLLM",
    "powerinfer": "PowerInfer",
}
# Keep the VLASelect baseline visually consistent across every comparison plot.
_METHOD_COLORS = {
    "default": "#C00000",
}
_METHOD_ORDER = [
    "logit_distillation",
    "feature_distillation",
    "attn_distillation",
    "data_distillation",
    "minillm",
    "distillm",
    "llm_in_a_flash",
    "powerinfer",
    "llm_pruner",
    "edgeta",
    "default",
]


def _smooth(values: List[float], alpha: float = 0.25) -> List[float]:
    """Apply light exponential smoothing while preserving the first sample."""
    if len(values) < 2:
        return values[:]
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _interpolate_missing(
    xs: List[float], ys: List[float], common_end: float
) -> Tuple[List[float], List[float]]:
    """Fill missing accuracy samples and extend each curve to a common endpoint."""
    valid_indices = [index for index, value in enumerate(ys) if math.isfinite(value)]
    if not valid_indices:
        return [], []

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    # A run that produced only one metric sample would otherwise be rendered as
    # an invisible marker.  Give it a horizontal segment from time zero to the
    # comparison endpoint so it remains visible alongside the other curves.
    if len(valid_indices) == 1:
        value = ys[first_valid]
        return [0.0, common_end], [value, value]

    filled = list(ys)
    for index in range(len(ys)):
        if math.isfinite(ys[index]):
            continue
        if index <= first_valid:
            filled[index] = ys[first_valid]
            continue
        if index >= last_valid:
            # Training can finish several rollout intervals without a completed
            # episode; carry the last observed accuracy through that tail.
            filled[index] = ys[last_valid]
            continue
        previous = max(item for item in valid_indices if item < index)
        following = min(item for item in valid_indices if item > index)
        span = xs[following] - xs[previous]
        ratio = (xs[index] - xs[previous]) / span if span else 0.0
        filled[index] = ys[previous] + ratio * (ys[following] - ys[previous])

    output_xs = list(xs)
    output_ys = filled
    if output_xs[-1] < common_end:
        output_xs.append(common_end)
        output_ys.append(filled[-1])
    return output_xs, output_ys


def _load_history(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    history = payload.get("history") if isinstance(payload, dict) else payload
    if not isinstance(history, list):
        return None
    return [item for item in history if isinstance(item, dict)]


def _latest_history(method_dir: Path) -> Optional[Tuple[Path, List[Dict[str, Any]]]]:
    candidates = list(method_dir.glob("metrics_history.json"))
    candidates.extend(method_dir.glob("*/metrics_history.json"))
    candidates = [path for path in candidates if path.is_file()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        history = _load_history(path)
        if history:
            return path, history
    return None


def _single_run_label(run_dir: Path) -> str:
    """Use the selected method/granularity as the label for a single run plot."""
    args_path = run_dir / "args.json"
    try:
        payload = json.loads(args_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        selector_keys = ("scaling_method", "knowledge_exchange_granularity")
        for key in selector_keys:
            value = payload.get(key)
            if value:
                return str(value)
        # A run with explicit empty selectors is the VLASelect baseline.
        if any(key in payload for key in selector_keys):
            return "default"
    return run_dir.parent.name if run_dir.parent != run_dir else run_dir.name


def _metric_value(metric: Dict[str, Any], requested: str) -> Optional[float]:
    candidates = [requested]
    if requested == "train_success_once":
        candidates.extend(("train_success_at_end", "success_once", "success_at_end"))
    elif requested == "train_success_at_end":
        candidates.extend(("train_success_once", "success_at_end", "success_once"))
    for key in candidates:
        value = metric.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _time_average(xs: List[float], ys: List[float]) -> Optional[float]:
    if not xs or not ys or len(xs) != len(ys):
        return None
    if len(xs) == 1:
        return ys[0] if math.isfinite(ys[0]) else None
    total_span = xs[-1] - xs[0]
    if total_span <= 0.0:
        return ys[-1] if math.isfinite(ys[-1]) else None
    area = 0.0
    for left_x, right_x, left_y, right_y in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if not (math.isfinite(left_y) and math.isfinite(right_y)):
            continue
        width = right_x - left_x
        if width <= 0.0:
            continue
        area += width * (left_y + right_y) * 0.5
    return area / total_span


def _build_accuracy_summary(
    series: List[Tuple[str, List[float], List[float]]], common_end: float
) -> Dict[str, Any]:
    """Summarize VLASelect's time-averaged accuracy against the baseline mean."""
    final_values: Dict[str, float] = {}
    time_averages: Dict[str, float] = {}
    for label, xs, ys in series:
        interpolated_xs, interpolated_ys = _interpolate_missing(xs, ys, common_end)
        if interpolated_ys and math.isfinite(interpolated_ys[-1]):
            final_values[label] = interpolated_ys[-1]
        time_avg = _time_average(interpolated_xs, interpolated_ys)
        if time_avg is not None and math.isfinite(time_avg):
            time_averages[label] = time_avg

    vlaselect_final = final_values.get("default")
    vlaselect_mean = time_averages.get("default")
    other_final_values = [value for label, value in final_values.items() if label != "default"]
    other_time_averages = [value for label, value in time_averages.items() if label != "default"]
    others_final_mean = (sum(other_final_values) / len(other_final_values)) if other_final_values else None
    others_time_mean = (sum(other_time_averages) / len(other_time_averages)) if other_time_averages else None
    absolute_gain_fraction = None
    absolute_gain_points = None
    relative_improvement_percent = None
    if vlaselect_mean is not None and others_time_mean is not None:
        absolute_gain_fraction = vlaselect_mean - others_time_mean
        absolute_gain_points = absolute_gain_fraction * 100.0
        if others_time_mean != 0.0:
            relative_improvement_percent = absolute_gain_fraction / others_time_mean * 100.0

    return {
        "vlaselect_mean_accuracy": vlaselect_mean,
        "other_methods_mean_accuracy": others_time_mean,
        "absolute_gain_fraction": absolute_gain_fraction,
        "absolute_gain_points": absolute_gain_points,
        "relative_improvement_percent": relative_improvement_percent,
        "num_compared_methods": len(other_time_averages),
        "time_averages": time_averages,
        "vlaselect_final_accuracy": vlaselect_final,
        "other_methods_mean_final_accuracy": others_final_mean,
        "final_values": final_values,
    }


def plot_category(
    results_dir: Path,
    output_path: Path,
    *,
    title: str,
    metric: str = "train_success_once",
    label_overrides: Optional[Mapping[str, str]] = None,
) -> int:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {results_dir}")
    series: List[Tuple[str, List[float], List[float]]] = []
    direct_history = results_dir / "metrics_history.json"
    if direct_history.is_file():
        loaded = _load_history(direct_history)
        method_histories = [(_single_run_label(results_dir), loaded)] if loaded else []
    else:
        method_histories = []
        for method_dir in sorted(
            path for path in results_dir.iterdir() if path.is_dir() and path.name not in _IGNORED_METHOD_DIRS
        ):
            loaded = _latest_history(method_dir)
            if loaded is not None:
                method_histories.append((method_dir.name, loaded[1]))

    for method_name, history in method_histories:
        xs: List[float] = []
        ys: List[float] = []
        for item in history:
            elapsed_hours = item.get("elapsed_hours")
            value = _metric_value(item, metric)
            if elapsed_hours is None or value is None:
                continue
            try:
                xs.append(float(elapsed_hours) * 60.0)
                ys.append(value)
            except (TypeError, ValueError):
                continue
        if xs:
            series.append((method_name, xs, ys))
    if not series:
        raise RuntimeError(f"no usable metrics_history.json found under {results_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = {method: index for index, method in enumerate(_METHOD_ORDER)}
    series.sort(key=lambda item: order.get(item[0], len(order)))
    point_counts = [sum(math.isfinite(value) for value in ys) for _, _, ys in series]
    all_single_point = all(count == 1 for count in point_counts)
    common_end = 2.0 if all_single_point else max(xs[-1] for _, xs, _ in series)
    accuracy_summary = _build_accuracy_summary(series, common_end)
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, (label, xs, ys) in enumerate(series):
        curve_end = common_end
        if point_counts[index] == 1 and not all_single_point:
            other_ends = [
                other_xs[-1]
                for other_index, (_, other_xs, _) in enumerate(series)
                if other_index != index
            ]
            curve_end = max(other_ends)
        interpolated_xs, interpolated_ys = _interpolate_missing(xs, ys, curve_end)
        if interpolated_xs:
            axis.plot(
                interpolated_xs,
                _smooth(interpolated_ys),
                linewidth=2,
                label=(label_overrides or {}).get(label, _METHOD_LABELS.get(label, label)),
                color=_METHOD_COLORS.get(label),
            )
    axis.set_xlabel("Time (minutes)", fontsize=15)
    axis.set_ylabel("Accuracy", fontsize=15)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=13)
    figure.subplots_adjust(right=0.76)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary_output = output_path.with_suffix('.summary.json')
    summary_output.write_text(json.dumps({
        'output': str(output_path),
        'title': title,
        'metric': metric,
        **accuracy_summary,
    }, indent=2), encoding='utf-8')

    print(f"[comparison] summary={summary_output}")
    print(
        f"[comparison] vlaselect_mean_accuracy={accuracy_summary['vlaselect_mean_accuracy']:.4f}"
        if accuracy_summary['vlaselect_mean_accuracy'] is not None
        else "[comparison] vlaselect_mean_accuracy=NA"
    )
    print(
        f"[comparison] other_methods_mean_accuracy={accuracy_summary['other_methods_mean_accuracy']:.4f}"
        if accuracy_summary['other_methods_mean_accuracy'] is not None
        else "[comparison] other_methods_mean_accuracy=NA"
    )
    print(
        f"[comparison] vlaselect_final_accuracy={accuracy_summary['vlaselect_final_accuracy']:.4f}"
        if accuracy_summary['vlaselect_final_accuracy'] is not None
        else "[comparison] vlaselect_final_accuracy=NA"
    )
    print(
        f"[comparison] other_methods_mean_final_accuracy={accuracy_summary['other_methods_mean_final_accuracy']:.4f}"
        if accuracy_summary['other_methods_mean_final_accuracy'] is not None
        else "[comparison] other_methods_mean_final_accuracy=NA"
    )
    print(
        f"[comparison] absolute_gain_points={accuracy_summary['absolute_gain_points']:.2f}"
        if accuracy_summary['absolute_gain_points'] is not None
        else "[comparison] absolute_gain_points=NA"
    )
    print(
        f"[comparison] relative_improvement_percent={accuracy_summary['relative_improvement_percent']:.2f}"
        if accuracy_summary['relative_improvement_percent'] is not None
        else "[comparison] relative_improvement_percent=NA"
    )
    print(f"[comparison] num_compared_methods={accuracy_summary['num_compared_methods']}")
    return len(series)


def run_cli(
    default_results_dir: Path,
    default_output: Path,
    title: str,
    *,
    label_overrides: Optional[Mapping[str, str]] = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--metric", default="train_success_once")
    args = parser.parse_args()
    count = plot_category(
        args.results_dir,
        args.output,
        title=title,
        metric=args.metric,
        label_overrides=label_overrides,
    )
    print(f"wrote {args.output} ({count} curves)")
