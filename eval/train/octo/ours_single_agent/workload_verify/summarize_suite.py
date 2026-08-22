from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


MAX_PLOT_MINUTES = 30.0
ELAPSED_TAG = "time/elapsed_minutes"
SUCCESS_ONCE_TAG = "eval/success_once"
SUCCESS_AT_END_TAG = "eval/success_end"
JSON_SUCCESS_ONCE_KEYS = ("eval_success_once",)
JSON_SUCCESS_AT_END_KEYS = ("eval_success_at_end", "eval_success_end")


def load_manifest(suite_dir: Path) -> dict:
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot find manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "metrics_history.json"
    if history_path.is_file():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            return [entry for entry in payload["history"] if isinstance(entry, dict)]
    nested_history_paths = sorted(path for path in run_dir.glob("**/metrics_history.json") if path.is_file())
    for nested_history_path in nested_history_paths:
        try:
            payload = json.loads(nested_history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            return [entry for entry in payload["history"] if isinstance(entry, dict)]
    return []


def collect_json_points(run_dir: Path, metric_keys: tuple[str, ...]) -> list[tuple[float, float]]:
    points = []
    for index, metric in enumerate(load_history(run_dir)):
        value = None
        for metric_key in metric_keys:
            if metric.get(metric_key) is not None:
                value = metric[metric_key]
                break
        if value is None:
            continue
        elapsed_hours = metric.get("elapsed_hours")
        try:
            minute = float(elapsed_hours) * 60.0 if elapsed_hours is not None else float(index)
            score = float(value)
        except (TypeError, ValueError):
            continue
        points.append((minute, score))
    return points


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def find_tb_dir(run_dir: Path) -> Path | None:
    direct_tb_dir = run_dir / "tb"
    if direct_tb_dir.is_dir():
        return direct_tb_dir

    single_agent_tb_dir = run_dir / "[agent]" / "tb"
    if single_agent_tb_dir.is_dir():
        return single_agent_tb_dir

    nested_tb_dirs = sorted(path for path in run_dir.glob("**/tb") if path.is_dir())
    return nested_tb_dirs[0] if nested_tb_dirs else None


def align_metric_to_elapsed(metric_events, elapsed_events) -> list[tuple[float, float]]:
    if not metric_events or not elapsed_events:
        return []

    elapsed_steps = [event.step for event in elapsed_events]
    elapsed_values = [float(event.value) for event in elapsed_events]
    aligned: list[tuple[float, float]] = []
    for metric_event in metric_events:
        index = bisect_right(elapsed_steps, metric_event.step) - 1
        if index < 0:
            continue
        aligned.append((elapsed_values[index], float(metric_event.value)))
    return aligned


def truncate_points(points: Iterable[tuple[float, float]], max_minutes: float) -> list[tuple[float, float]]:
    return [(minute, value) for minute, value in points if minute <= max_minutes + 1e-6]


def summarize_points(points: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if not points:
        return {
            "num_points": 0,
            "initial": None,
            "final": None,
            "improvement": None,
            "mean": None,
            "last_elapsed_minutes": None,
        }
    values = [value for _, value in points]
    return {
        "num_points": len(points),
        "initial": values[0],
        "final": values[-1],
        "improvement": values[-1] - values[0],
        "mean": sum(values) / len(values),
        "last_elapsed_minutes": points[-1][0],
    }


def render_plot(
    *,
    env_id: str,
    success_once_points: list[tuple[float, float]],
    success_at_end_points: list[tuple[float, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if success_once_points:
        plt.plot(
            [minute for minute, _ in success_once_points],
            [value for _, value in success_once_points],
            linewidth=2,
            label="success_once",
        )
    if success_at_end_points:
        plt.plot(
            [minute for minute, _ in success_at_end_points],
            [value for _, value in success_at_end_points],
            linewidth=2,
            label="success_at_end",
        )

    plt.title(env_id)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Success Rate")
    plt.xlim(0.0, MAX_PLOT_MINUTES)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    if success_once_points or success_at_end_points:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No TensorBoard data found.", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def render_suite_outputs(suite_dir: Path) -> Path:
    manifest = load_manifest(suite_dir)
    plots_dir = suite_dir / "plots"
    summary_csv_path = suite_dir / "summary.csv"

    rows: list[dict[str, object]] = []
    for run in manifest.get("runs", []):
        env_id = run["env_id"]
        run_dir = Path(run["run_dir"])
        tb_dir = find_tb_dir(run_dir)

        success_once_points: list[tuple[float, float]] = collect_json_points(run_dir, JSON_SUCCESS_ONCE_KEYS)
        success_at_end_points: list[tuple[float, float]] = collect_json_points(run_dir, JSON_SUCCESS_AT_END_KEYS)
        success_once_points = truncate_points(success_once_points, MAX_PLOT_MINUTES)
        success_at_end_points = truncate_points(success_at_end_points, MAX_PLOT_MINUTES)
        if (not success_once_points and not success_at_end_points) and tb_dir is not None:
            elapsed_events = load_scalar_events(tb_dir, ELAPSED_TAG)
            success_once_events = load_scalar_events(tb_dir, SUCCESS_ONCE_TAG)
            success_at_end_events = load_scalar_events(tb_dir, SUCCESS_AT_END_TAG)
            success_once_points = truncate_points(
                align_metric_to_elapsed(success_once_events, elapsed_events),
                MAX_PLOT_MINUTES,
            )
            success_at_end_points = truncate_points(
                align_metric_to_elapsed(success_at_end_events, elapsed_events),
                MAX_PLOT_MINUTES,
            )

        plot_path = plots_dir / f"{env_id}.png"
        render_plot(
            env_id=env_id,
            success_once_points=success_once_points,
            success_at_end_points=success_at_end_points,
            output_path=plot_path,
        )

        success_once_summary = summarize_points(success_once_points)
        success_at_end_summary = summarize_points(success_at_end_points)
        rows.append(
            {
                "env_id": env_id,
                "status": run.get("status"),
                "gpu_id": run.get("gpu_id"),
                "returncode": run.get("returncode"),
                "run_dir": str(run_dir),
                "log_file": run.get("log_file"),
                "plot_path": str(plot_path),
                "num_eval_points": max(
                    int(success_once_summary["num_points"]),
                    int(success_at_end_summary["num_points"]),
                ),
                "last_elapsed_minutes": success_at_end_summary["last_elapsed_minutes"]
                if success_at_end_summary["last_elapsed_minutes"] is not None
                else success_once_summary["last_elapsed_minutes"],
                "initial_success_once": success_once_summary["initial"],
                "final_success_once": success_once_summary["final"],
                "improvement_success_once": success_once_summary["improvement"],
                "mean_success_once_30m": success_once_summary["mean"],
                "initial_success_at_end": success_at_end_summary["initial"],
                "final_success_at_end": success_at_end_summary["final"],
                "improvement_success_at_end": success_at_end_summary["improvement"],
                "mean_success_at_end_30m": success_at_end_summary["mean"],
            }
        )

    fieldnames = [
        "env_id",
        "status",
        "gpu_id",
        "returncode",
        "run_dir",
        "log_file",
        "plot_path",
        "num_eval_points",
        "last_elapsed_minutes",
        "initial_success_once",
        "final_success_once",
        "improvement_success_once",
        "mean_success_once_30m",
        "initial_success_at_end",
        "final_success_at_end",
        "improvement_success_at_end",
        "mean_success_at_end_30m",
    ]
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = render_suite_outputs(args.suite_dir.resolve())
    print(summary_path)


if __name__ == "__main__":
    main()
