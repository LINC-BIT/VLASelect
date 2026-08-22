from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PLOT_FONT_SIZE = 36
PLOT_FIGURE_SIZE = (9.6, 8)
DEFAULT_PLOT_INTERVAL_SECONDS = 60.0
DEFAULT_SMOOTHING = 0.7

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
    }
)


METHOD_STYLES = {
    "conrft": {"color": "#4C78A8", "linestyle": "-"},
    "flare": {"color": "#59A14F", "linestyle": "-"},
    "improv_vla": {"color": "#4D4D4D", "linestyle": "-"},
    "edgeta": {"color": "#A6A6A6", "linestyle": "--"},
    "convertnet": {"color": "#CEBB6C", "linestyle": "--"},
    "ours": {"color": "#C44E52", "linestyle": "-"},
    "ppo_gen": {"color": "#4C78A8", "linestyle": "--"},
    "self_improv": {"color": "#9A9A9A", "linestyle": "-"},
    "vla_rft": {"color": "#59A14F", "linestyle": "--"},
    "world_env": {"color": "#4D4D4D", "linestyle": "--"},
}

LEGEND_ORDER = [
    "conrft",
    "flare",
    "improv_vla",
    "self_improv",
    "ppo_gen",
    "vla_rft",
    "world_env",
    "edgeta",
    "convertnet",
    "ours",
]

TAG_ALIASES = {
    "eval/success_at_end": ["eval_success_at_end"],
    "eval/success_once": ["eval_success_once"],
}


def process_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "metrics_history.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    history = payload.get("history", payload if isinstance(payload, list) else [])
    return history if isinstance(history, list) else []


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def collect_series(run_dir: Path, tag: str) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    metric_keys = TAG_ALIASES.get(tag, [tag])
    for index, metric in enumerate(load_history(run_dir)):
        y_value = None
        for metric_key in metric_keys:
            y_value = finite_float(metric.get(metric_key))
            if y_value is not None:
                break
        if y_value is None:
            continue
        elapsed_hours = finite_float(metric.get("elapsed_hours"))
        x_value = elapsed_hours * 60.0 if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return series


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + value * (1.0 - smoothing))
    return smoothed


def format_label(display_name: str, raw_values: list[float]) -> str:
    if not raw_values:
        return display_name
    average = sum(raw_values) / len(raw_values)
    return f"{display_name} (avg.={average:.4f})"


def compute_average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compute_ours_average_improvement(method_averages: dict[str, float]) -> float | None:
    ours_average = method_averages.get("ours")
    if ours_average is None:
        return None
    other_averages = [value for name, value in method_averages.items() if name != "ours"]
    if not other_averages:
        return None
    return (ours_average - (sum(other_averages) / len(other_averages))) * 100.0


def compute_method_averages_for_tag(manifest: dict[str, Any], tag: str) -> dict[str, tuple[str, float]]:
    averages: dict[str, tuple[str, float]] = {}
    for method in manifest.get("methods", []):
        method_name = method["name"]
        series = collect_series(Path(method["run_dir"]), tag)
        if not series:
            continue
        raw_values = [point[1] for point in series]
        average = compute_average(raw_values)
        if average is None:
            continue
        averages[method_name] = (method.get("display_name", method_name), average)
    return averages


def resolve_plot_end_minutes(manifest: dict[str, Any], tag: str) -> float:
    change_points = manifest.get("env_change_time_points")
    if isinstance(change_points, str):
        try:
            parsed = ast.literal_eval(change_points)
        except (SyntaxError, ValueError):
            parsed = []
    elif isinstance(change_points, list):
        parsed = change_points
    else:
        parsed = []
    finite_points = [point for point in (finite_float(value) for value in parsed) if point is not None and point > 0.0]
    if finite_points:
        return max(finite_points)

    observed_series_end = 0.0
    for method in manifest.get("methods", []):
        series = collect_series(Path(method["run_dir"]), tag)
        if series:
            observed_series_end = max(observed_series_end, series[-1][0])
    if observed_series_end > 0.0:
        return observed_series_end
    return 1.0


def order_legend_entries(entries: list[tuple[str, str, dict[str, str]]]) -> list[tuple[str, str, dict[str, str]]]:
    order_index = {name: idx for idx, name in enumerate(LEGEND_ORDER)}
    return sorted(entries, key=lambda entry: (order_index.get(entry[0], len(order_index)), entry[1]))


def save_legend_image(output_path: Path, legend_entries: list[tuple[str, str, dict[str, str]]]) -> None:
    if not legend_entries:
        return
    ordered_entries = order_legend_entries(legend_entries)
    handles = [
        Line2D(
            [0],
            [0],
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=3.6,
        )
        for _, _, style in ordered_entries
    ]
    labels = [label for _, label, _ in ordered_entries]
    legend_path = output_path.with_name(f"{output_path.stem}_legend.png")
    legend_fig_height = max(2.2, 0.78 * len(labels))
    fig = plt.figure(figsize=(8, legend_fig_height))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=1,
        handlelength=2.8,
    )
    fig.savefig(legend_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(legend_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_plot(manifest: dict[str, Any], output_path: Path, tag: str, smoothing: float) -> float | None:
    plt.figure(figsize=PLOT_FIGURE_SIZE)
    plotted = 0
    pending = []
    method_averages: dict[str, float] = {}
    legend_entries: list[tuple[str, str, dict]] = []
    plot_end_minutes = resolve_plot_end_minutes(manifest, tag)

    for method in manifest.get("methods", []):
        method_name = method["name"]
        series = collect_series(Path(method["run_dir"]), tag)
        if not series:
            pending.append(method.get("display_name", method_name))
            continue

        xs = [point[0] for point in series]
        raw_values = [point[1] for point in series]
        average = compute_average(raw_values)
        if average is not None:
            method_averages[method_name] = average
        style = METHOD_STYLES.get(method_name, {})
        line_label = format_label(method.get("display_name", method_name), raw_values)
        plt.plot(
            xs,
            smooth_values(raw_values, smoothing),
            linewidth=3.6,
            label=line_label,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
        )
        legend_entries.append((method_name, line_label, style))
        plotted += 1

    if plotted == 0:
        plt.text(
            0.5,
            0.5,
            f"No scalar data yet for {tag}",
            ha="center",
            va="center",
            fontsize=PLOT_FONT_SIZE,
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Success Rate")
    plt.xlim(0.0, plot_end_minutes)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.savefig(output_path.with_suffix(".svg"))
    plt.close()
    save_legend_image(output_path, legend_entries)
    return compute_ours_average_improvement(method_averages)


def write_improvement_summary(
    manifest: dict[str, Any],
    output_dir: Path,
    smoothing: float,
    improvements: dict[str, float | None],
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_path = output_dir / "ours_average_improvement.txt"
    lines = [
        f"saved_at: {timestamp}",
        f"smoothing: {smoothing:.2f}",
    ]
    for stem, label in (
        ("success_at_end", "Success@End"),
        ("success_once", "Success@Once"),
    ):
        improvement = improvements.get(stem)
        if improvement is None:
            lines.append(f"{label}: unavailable")
        else:
            lines.append(f"{label}: {improvement:+.2f}%")

    success_once_averages = compute_method_averages_for_tag(manifest, "eval/success_once")
    ours_entry = success_once_averages.get("ours")
    if ours_entry is not None:
        lines.append(f"Ours Average Success@Once: {ours_entry[1]:.6f}")

    other_entries = [
        (name, display_name, average)
        for name, (display_name, average) in success_once_averages.items()
        if name != "ours"
    ]
    if other_entries:
        others_mean = sum(average for _, _, average in other_entries) / len(other_entries)
        lines.append(f"Others Mean Success@Once: {others_mean:.6f}")
        lines.append("Other Methods Average Success@Once:")
        average_by_method = {name: average for name, _, average in other_entries}
        ordered_other_entries = order_legend_entries([(name, display_name, {}) for name, display_name, _ in other_entries])
        for method_name, display_name, _ in ordered_other_entries:
            lines.append(f"{display_name}: {average_by_method[method_name]:.6f}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_snapshot_pair(manifest: dict[str, Any], output_dir: Path, smoothing: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    improvements: dict[str, float | None] = {}
    targets = [
        ("eval/success_at_end", "success_at_end"),
        ("eval/success_once", "success_once"),
    ]
    for tag, stem in targets:
        latest_path = output_dir / f"{stem}_latest.png"
        history_path = history_dir / f"{timestamp}_{stem}.png"
        improvements[stem] = draw_plot(manifest, latest_path, tag, smoothing)
        draw_plot(manifest, history_path, tag, smoothing)
    write_improvement_summary(manifest, output_dir, smoothing, improvements)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_PLOT_INTERVAL_SECONDS)
    parser.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING)
    args = parser.parse_args()

    if not 0.0 <= args.smoothing <= 1.0:
        raise ValueError(f"--smoothing must be in [0.0, 1.0], got {args.smoothing}")

    while True:
        manifest = load_manifest(args.manifest)
        save_snapshot_pair(manifest, args.output_dir, args.smoothing)
        scheduler_pid = manifest.get("scheduler_pid")
        if scheduler_pid is None and all(not process_is_alive(method.get("pid")) for method in manifest.get("methods", [])):
            break
        if scheduler_pid is not None and not process_is_alive(scheduler_pid):
            break
        time.sleep(args.interval_seconds)

    save_snapshot_pair(load_manifest(args.manifest), args.output_dir, args.smoothing)


if __name__ == "__main__":
    main()
