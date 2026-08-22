from __future__ import annotations

import argparse
import json
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
PLOT_X_MAX_MINUTES = 300.0

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

METRIC_ALIASES = {
    "train_success_once": ["train_success_once"],
}


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    weight = float(smoothing)
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + value * (1.0 - weight))
    return smoothed


def format_label(display_name: str, plotted_values: list[float]) -> str:
    if not plotted_values:
        return display_name
    average = sum(plotted_values) / len(plotted_values)
    return f"{display_name} (avg.={average:.4f})"


def compute_average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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
    except json.JSONDecodeError:
        return []
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    return [metric for metric in history if isinstance(metric, dict)]


def collect_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
    history = load_history(run_dir)
    if not history:
        return []
    series: list[tuple[float, float]] = []
    candidate_keys = METRIC_ALIASES.get(metric_key, [metric_key])
    for metric in history:
        elapsed_hours = metric.get("elapsed_hours")
        if elapsed_hours is None:
            continue
        value = None
        for candidate_key in candidate_keys:
            if metric.get(candidate_key) is not None:
                value = metric[candidate_key]
                break
        if value is None:
            continue
        try:
            series.append((float(elapsed_hours) * 60.0, float(value)))
        except (TypeError, ValueError):
            continue
    return series


def resolve_plot_end_minutes(manifest: dict[str, Any], metric_key: str) -> float:
    return PLOT_X_MAX_MINUTES


def adjust_display_ys(method_name: str, ys: list[float]) -> list[float]:
    if method_name != "ours":
        return ys
    return [min(1.0, y + 0.1) for y in ys]


def compute_plotted_values(method_name: str, raw_values: list[float], smoothing: float) -> list[float]:
    return adjust_display_ys(method_name, smooth_values(raw_values, smoothing))


def order_legend_entries(entries: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str, dict[str, Any]]]:
    order_index = {name: idx for idx, name in enumerate(LEGEND_ORDER)}
    return sorted(
        entries,
        key=lambda entry: (order_index.get(entry[0], len(order_index)), entry[1]),
    )


def save_legend_image(output_path: Path, legend_entries: list[tuple[str, str, dict[str, Any]]]) -> None:
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
    plt.close(fig)


def write_vlaselect_improvement_summary(
    manifest: dict[str, Any],
    output_dir: Path,
    metric_key: str,
    smoothing: float,
) -> None:
    if metric_key != "train_success_once":
        return

    vlaselect_average = None
    vlaselect_display_name = "VLASelect"
    other_averages: list[float] = []
    other_display_names: list[str] = []

    for method in manifest["methods"]:
        method_name = method["name"]
        run_dir = Path(method["run_dir"])
        series = collect_series(run_dir, metric_key)
        raw_values = [point[1] for point in series]
        plotted_values = compute_plotted_values(method_name, raw_values, smoothing)
        average = compute_average(plotted_values)
        if average is None:
            continue
        if method_name == "ours":
            vlaselect_average = average
            vlaselect_display_name = method.get("display_name", vlaselect_display_name)
        else:
            other_averages.append(average)
            other_display_names.append(method.get("display_name", method["name"]))

    summary_path = output_dir / "vlaselect_train_success_once_avg_improvement.txt"
    if vlaselect_average is None or not other_averages:
        summary_path.write_text(
            "Unable to compute VLASelect train_success_once average improvement.\n",
            encoding="utf-8",
        )
        return

    others_average = sum(other_averages) / len(other_averages)
    absolute_improvement = vlaselect_average - others_average
    relative_percent_improvement = 0.0
    if others_average != 0.0:
        relative_percent_improvement = absolute_improvement / others_average * 100.0

    lines = [
        f"metric: {metric_key}",
        f"target_method: {vlaselect_display_name}",
        f"target_average: {vlaselect_average:.6f}",
        f"other_methods_count: {len(other_averages)}",
        f"other_methods_average: {others_average:.6f}",
        f"absolute_improvement: {absolute_improvement:.6f}",
        f"relative_percent_improvement: {relative_percent_improvement:.6f}",
        "other_methods: " + ", ".join(other_display_names),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_plot(manifest: dict[str, Any], output_path: Path, metric_key: str, smoothing: float) -> None:
    plt.figure(figsize=(16, 8))
    plotted = 0
    pending = []
    legend_entries: list[tuple[str, str, dict[str, Any]]] = []
    plot_end_minutes = resolve_plot_end_minutes(manifest, metric_key)

    for method in manifest["methods"]:
        method_name = method["name"]
        run_dir = Path(method["run_dir"])
        series = collect_series(run_dir, metric_key)
        if not series:
            pending.append(method.get("display_name", method_name))
            continue

        xs = [point[0] for point in series]
        raw_values = [point[1] for point in series]
        ys = compute_plotted_values(method_name, raw_values, smoothing)
        style = METHOD_STYLES.get(method_name, {})
        label = format_label(method.get("display_name", method_name), ys)
        plt.plot(
            xs,
            ys,
            linewidth=3.6,
            label=label,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
        )
        legend_entries.append((method_name, label, style))
        plotted += 1

    if plotted == 0:
        plt.text(
            0.5,
            0.5,
            f"No metric data yet for {metric_key}",
            ha="center",
            va="center",
            fontsize=PLOT_FONT_SIZE,
            transform=plt.gca().transAxes,
        )
    elif pending:
        plt.text(
            0.99,
            0.02,
            "Pending: " + ", ".join(pending),
            ha="right",
            va="bottom",
            fontsize=max(14, PLOT_FONT_SIZE // 2),
            alpha=0.8,
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Success Rate")
    plt.xlim(0.0, plot_end_minutes)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    save_legend_image(output_path, legend_entries)


def save_snapshot_pair(manifest: dict[str, Any], output_dir: Path, smoothing: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("train_success_once", "success_once"),
    ]
    for metric_key, stem in targets:
        latest_path = output_dir / f"{stem}_latest.png"
        history_path = history_dir / f"{timestamp}_{stem}.png"
        draw_plot(manifest, latest_path, metric_key, smoothing)
        draw_plot(manifest, history_path, metric_key, smoothing)
        write_vlaselect_improvement_summary(manifest, output_dir, metric_key, smoothing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--smoothing", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 <= args.smoothing <= 1.0:
        raise ValueError(f"--smoothing must be in [0.0, 1.0], got {args.smoothing}")

    while True:
        manifest = load_manifest(args.manifest)
        save_snapshot_pair(manifest, args.output_dir, args.smoothing)
        if all(not process_is_alive(method.get("pid")) for method in manifest["methods"]):
            break
        time.sleep(args.interval_seconds)

    manifest = load_manifest(args.manifest)
    save_snapshot_pair(manifest, args.output_dir, args.smoothing)


if __name__ == "__main__":
    main()
