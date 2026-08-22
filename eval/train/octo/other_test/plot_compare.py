from __future__ import annotations

import argparse
import csv
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

from train.octo.other_test.common import load_gpu_metrics_rows, load_tb_scalars


PLOT_FONT_SIZE = 26

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": max(16, PLOT_FONT_SIZE - 4),
    }
)


METHOD_STYLES = {
    "compressed": {"color": "#C44E52", "linestyle": "-"},
    "original": {"color": "#4C78A8", "linestyle": "--"},
    "original_peft": {"color": "#9A9A9A", "linestyle": "--"},
}


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    weight = float(smoothing)
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + value * (1.0 - weight))
    return smoothed


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


def collect_success_series(run_dir: Path, tag: str = "eval/success_once") -> list[tuple[float, float]]:
    events = load_tb_scalars(run_dir, tag)
    if not events:
        return []
    base_time = events[0].wall_time
    series: list[tuple[float, float]] = []
    for event in events:
        series.append(((event.wall_time - base_time) / 60.0, float(event.value)))
    return series


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_speed(method: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(method["run_dir"])
    sps_events = load_tb_scalars(run_dir, "charts/SPS")
    rollout_events = load_tb_scalars(run_dir, "time/rollout_time")
    update_events = load_tb_scalars(run_dir, "time/update_time")
    success_events = load_tb_scalars(run_dir, "eval/success_once")
    gpu_rows = load_gpu_metrics_rows(run_dir)

    success_latest = float(success_events[-1].value) if success_events else None
    sps_values = [float(event.value) for event in sps_events]
    rollout_values = [float(event.value) for event in rollout_events]
    update_values = [float(event.value) for event in update_events]

    elapsed_minutes = None
    if gpu_rows:
        elapsed_seconds = _safe_float(gpu_rows[-1].get("elapsed_seconds"))
        if elapsed_seconds is not None:
            elapsed_minutes = elapsed_seconds / 60.0
    if elapsed_minutes is None and success_events:
        elapsed_minutes = (success_events[-1].wall_time - success_events[0].wall_time) / 60.0

    gpu_mem_values = [_safe_float(row.get("gpu_memory_used_mb")) for row in gpu_rows]
    gpu_mem_values = [value for value in gpu_mem_values if value is not None]

    return {
        "name": method["name"],
        "display_name": method["display_name"],
        "run_dir": str(run_dir),
        "gpu": method.get("gpu"),
        "success_once_latest": success_latest,
        "elapsed_minutes": elapsed_minutes,
        "sps_latest": sps_values[-1] if sps_values else None,
        "sps_avg": sum(sps_values) / len(sps_values) if sps_values else None,
        "sps_max": max(sps_values) if sps_values else None,
        "rollout_time_avg_sec": sum(rollout_values) / len(rollout_values) if rollout_values else None,
        "update_time_avg_sec": sum(update_values) / len(update_values) if update_values else None,
        "gpu_memory_avg_mb": sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None,
        "gpu_memory_max_mb": max(gpu_mem_values) if gpu_mem_values else None,
    }


def write_speed_csv(manifest: dict[str, Any], output_dir: Path) -> None:
    rows = [summarize_speed(method) for method in manifest["methods"]]
    latest_path = output_dir / "speed_summary_latest.csv"
    history_dir = output_dir / "speed_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_path = history_dir / f"{timestamp}_speed_summary.csv"
    fieldnames = [
        "name",
        "display_name",
        "run_dir",
        "gpu",
        "success_once_latest",
        "elapsed_minutes",
        "sps_latest",
        "sps_avg",
        "sps_max",
        "rollout_time_avg_sec",
        "update_time_avg_sec",
        "gpu_memory_avg_mb",
        "gpu_memory_max_mb",
    ]
    for path in [latest_path, history_path]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def save_legend_image(output_path: Path, entries: list[tuple[str, str]]) -> None:
    if not entries:
        return
    handles = []
    labels = []
    for method_name, label in entries:
        style = METHOD_STYLES.get(method_name, {})
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=3.6,
            )
        )
        labels.append(label)
    fig = plt.figure(figsize=(8, max(2.2, 0.78 * len(labels))))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False, ncol=1, handlelength=2.8)
    fig.savefig(output_path.with_name(f"{output_path.stem}_legend.png"), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_plot(
    manifest: dict[str, Any],
    output_path: Path,
    smoothing: float,
    x_max_minutes: float | None,
) -> None:
    plt.figure(figsize=(16, 8))
    entries: list[tuple[str, str]] = []
    max_minutes = 1.0
    pending: list[str] = []
    plotted = 0

    for method in manifest["methods"]:
        method_name = method["name"]
        run_dir = Path(method["run_dir"])
        series = collect_success_series(run_dir)
        if not series:
            pending.append(method["display_name"])
            continue
        xs = [point[0] for point in series]
        raw_values = [point[1] for point in series]
        ys = smooth_values(raw_values, smoothing)
        max_minutes = max(max_minutes, xs[-1])
        style = METHOD_STYLES.get(method_name, {})
        plt.plot(
            xs,
            ys,
            linewidth=3.6,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            label=method["display_name"],
        )
        entries.append((method_name, method["display_name"]))
        plotted += 1

    if plotted == 0:
        plt.text(0.5, 0.5, "No scalar data yet for eval/success_once", ha="center", va="center", transform=plt.gca().transAxes)
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
    plt.xlim(0.0, x_max_minutes if x_max_minutes is not None else max_minutes)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    save_legend_image(output_path, entries)


def save_snapshot_pair(
    manifest: dict[str, Any],
    output_dir: Path,
    smoothing: float,
    x_max_minutes: float | None,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "success_once_latest.png"
    history_path = history_dir / f"{timestamp}_success_once.png"
    draw_plot(manifest, latest_path, smoothing, x_max_minutes)
    draw_plot(manifest, history_path, smoothing, x_max_minutes)
    write_speed_csv(manifest, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=100.0)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--x-max-minutes", type=float, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.smoothing <= 1.0:
        raise ValueError(f"--smoothing must be in [0.0, 1.0], got {args.smoothing}")
    if args.x_max_minutes is not None and args.x_max_minutes <= 0.0:
        raise ValueError(f"--x-max-minutes must be > 0.0, got {args.x_max_minutes}")

    while True:
        manifest = load_manifest(args.manifest)
        save_snapshot_pair(manifest, args.output_dir, args.smoothing, args.x_max_minutes)
        if all(not process_is_alive(method.get("pid")) for method in manifest["methods"] if method.get("pid") is not None):
            break
        time.sleep(args.interval_seconds)

    manifest = load_manifest(args.manifest)
    save_snapshot_pair(manifest, args.output_dir, args.smoothing, args.x_max_minutes)


if __name__ == "__main__":
    main()
