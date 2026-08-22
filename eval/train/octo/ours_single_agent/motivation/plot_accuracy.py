from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.octo.ours_single_agent.motivation.training_lib import (
    LATEST_RUN_FILES,
    WORKLOAD_ENVS,
)


OUTPUT_PATH = Path("train/octo/ours_single_agent/motivation/res.png")

METRIC = 'success_once'


def load_latest_run_dir(mode: str) -> Optional[Path]:
    manifest_path = LATEST_RUN_FILES[mode]
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        run_dir = Path(payload["run_dir"])
        if run_dir.exists():
            return run_dir

    fallback_root = Path("ckpt") / WORKLOAD_ENVS[0] / "ours" / "octo" / f"{mode}_model"
    if not fallback_root.exists():
        return None
    candidates = sorted(
        [path for path in fallback_root.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_metric_rows(mode: str) -> List[Dict[str, float]]:
    run_dir = load_latest_run_dir(mode)
    if run_dir is None:
        return []
    metrics_path = run_dir / "motivation_eval_metrics.jsonl"
    if not metrics_path.exists():
        return []

    rows: List[Dict[str, float]] = []
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def smooth_values(values: List[float], window_size: int) -> List[float]:
    if window_size <= 1 or len(values) <= 1:
        return list(values)

    smoothed: List[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window_size:
            running_sum -= values[index - window_size]
        current_window_size = min(index + 1, window_size)
        smoothed.append(running_sum / current_window_size)
    return smoothed


def render_plot(smooth_window: int) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_rows = load_metric_rows("original")
    small_rows = load_metric_rows("small")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    has_data = False

    if original_rows:
        has_data = True
        original_x = [row["elapsed_minutes"] for row in original_rows]
        original_y = [row[METRIC] for row in original_rows]
        # if smooth_window > 1:
        #     ax.plot(
        #         original_x,
        #         original_y,
        #         marker="o",
        #         linewidth=1,
        #         linestyle="--",
        #         alpha=0.3,
        #         label="Original Model (raw)",
        #     )
        ax.plot(
            original_x,
            smooth_values(original_y, smooth_window),
            # marker="o",
            linewidth=2,
            label="Large Model",
        )
    if small_rows:
        has_data = True
        small_x = [row["elapsed_minutes"] for row in small_rows]
        small_y = [row[METRIC] for row in small_rows]
        # if smooth_window > 1:
        #     ax.plot(
        #         small_x,
        #         small_y,
        #         marker="o",
        #         linewidth=1,
        #         linestyle="--",
        #         alpha=0.3,
        #         label="Static Small Model (raw)",
        #     )
        ax.plot(
            small_x,
            smooth_values(small_y, smooth_window),
            # marker="o",
            linewidth=2,
            label="Small Model",
        )

    title = f"{METRIC} vs Time"
    # if smooth_window > 1:
    #     title += f" (moving average window={smooth_window})"
    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(METRIC)
    ax.grid(True, alpha=0.3)

    if has_data:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No evaluation data found yet.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Saved accuracy plot to {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be a positive integer")

    while True:
        render_plot(args.smooth_window)
        if not args.watch:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
