#!/usr/bin/env python3
"""Plot the ICL comparison curves from the two run metric histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_KEYS = {
    "success_once": ("eval_success_once", "success_once", "eval/success_once"),
    "success_at_end": (
        "eval_success_at_end",
        "eval_success_end",
        "success_at_end",
        "success_end",
        "eval/success_at_end",
        "eval/success_end",
    ),
}


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + value * (1.0 - smoothing))
    return smoothed


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "metrics_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"missing metrics history: {history_path}")

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload, dict) else payload
    if not isinstance(history, list):
        raise ValueError(f"invalid metrics history format: {history_path}")
    return [entry for entry in history if isinstance(entry, dict)]


def collect_series(run_dir: Path, metric: str) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    for index, entry in enumerate(load_history(run_dir)):
        raw_value = next((entry.get(key) for key in METRIC_KEYS[metric] if entry.get(key) is not None), None)
        if raw_value is None:
            continue

        elapsed_hours = entry.get("elapsed_hours")
        if elapsed_hours is None:
            # Older logs may not have elapsed_hours. Use the evaluation index
            # as a monotonic diagnostic fallback rather than inventing a time
            # unit from global_step.
            elapsed_minutes = float(index)
        else:
            try:
                elapsed_minutes = float(elapsed_hours) * 60.0
            except (TypeError, ValueError):
                continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{metric} must be in [0, 1], got {value} in {run_dir}")
        series.append((elapsed_minutes, value * 100.0))
    return series


def build_gain_summary(
    metric: str,
    smoothing: float,
    vlaselect_series: list[tuple[float, float]],
    ricl_series: list[tuple[float, float]],
) -> dict[str, Any]:
    vlaselect_raw = [value for _, value in vlaselect_series]
    ricl_raw = [value for _, value in ricl_series]
    vlaselect_smoothed = smooth_values(vlaselect_raw, smoothing)
    ricl_smoothed = smooth_values(ricl_raw, smoothing)

    def pack_pair(name: str, lhs: list[float], rhs: list[float]) -> dict[str, Any]:
        lhs_final = lhs[-1]
        rhs_final = rhs[-1]
        lhs_mean = mean(lhs)
        rhs_mean = mean(rhs)
        final_gain = lhs_final - rhs_final
        mean_gain = lhs_mean - rhs_mean
        final_relative = (final_gain / rhs_final * 100.0) if rhs_final != 0.0 else None
        mean_relative = (mean_gain / rhs_mean * 100.0) if rhs_mean != 0.0 else None
        return {
            'name': name,
            'vlaselect_final_accuracy': lhs_final,
            'ricl_final_accuracy': rhs_final,
            'final_absolute_gain_points': final_gain,
            'final_relative_gain_percent': final_relative,
            'vlaselect_mean_accuracy': lhs_mean,
            'ricl_mean_accuracy': rhs_mean,
            'mean_absolute_gain_points': mean_gain,
            'mean_relative_gain_percent': mean_relative,
            'compared_points': min(len(lhs), len(rhs)),
        }

    return {
        'metric': metric,
        'smoothing': smoothing,
        'raw': pack_pair('raw', vlaselect_raw, ricl_raw),
        'smoothed': pack_pair('smoothed', vlaselect_smoothed, ricl_smoothed),
    }


def draw_plot(
    vlaselect_dir: Path,
    ricl_dir: Path,
    output_path: Path,
    metric: str,
    smoothing: float = 0.8,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    curves = [
        ("VLASelect", vlaselect_dir, "#2563eb"),
        ("RICL", ricl_dir, "#ea580c"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    missing: list[str] = []
    all_x: list[float] = []
    all_y: list[float] = []
    plotted_series: dict[str, list[tuple[float, float]]] = {}
    for label, run_dir, color in curves:
        series = collect_series(run_dir, metric)
        if not series:
            missing.append(label)
            continue
        plotted_series[label] = series
        xs, ys = zip(*series)
        smoothed_ys = smooth_values(list(ys), smoothing)
        ax.plot(
            xs,
            smoothed_ys,
            label=label,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=3.5,
        )
        all_x.extend(xs)
        all_y.extend(smoothed_ys)

    if missing:
        raise ValueError(f"no {metric} data found for: {', '.join(missing)}")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Accuracy / success rate (%)")
    ax.set_title("ICL comparison")
    ax.set_ylim(min(all_y) - 10.0, 100.0)
    if all_x:
        ax.set_xlim(left=0.0, right=max(max(all_x), 1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    summary = build_gain_summary(metric, smoothing, plotted_series['VLASelect'], plotted_series['RICL'])
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlaselect-run-dir", type=Path, required=True)
    parser.add_argument("--ricl-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--metric", choices=tuple(METRIC_KEYS), default="success_once")
    parser.add_argument("--smoothing", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 <= args.smoothing <= 1.0:
        parser.error("--smoothing must be in [0, 1]")

    try:
        summary = draw_plot(
            args.vlaselect_run_dir,
            args.ricl_run_dir,
            args.output,
            args.metric,
            args.smoothing,
            args.summary_output,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(1, f"[ICL] error: {exc}\n")
    print(f"[ICL] Plot saved to {args.output}")
    if args.summary_output is not None:
        print(f"[ICL] Summary saved to {args.summary_output}")
    raw = summary['raw']
    print(
        "[ICL] Final accuracy: "
        f"VLASelect={raw['vlaselect_final_accuracy']:.2f}% "
        f"RICL={raw['ricl_final_accuracy']:.2f}% "
        f"gain={raw['final_absolute_gain_points']:.2f} points "
        f"relative={raw['final_relative_gain_percent'] if raw['final_relative_gain_percent'] is not None else 'NA'}"
    )
    print(
        "[ICL] Mean accuracy: "
        f"VLASelect={raw['vlaselect_mean_accuracy']:.2f}% "
        f"RICL={raw['ricl_mean_accuracy']:.2f}% "
        f"gain={raw['mean_absolute_gain_points']:.2f} points "
        f"relative={raw['mean_relative_gain_percent'] if raw['mean_relative_gain_percent'] is not None else 'NA'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
