#!/usr/bin/env python3
"""Render one training-accuracy curve for a model-type verification run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TENSORBOARD_SUCCESS_TAG = "train/success_once"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _smooth(values: list[float], weight: float = 0.7) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append(weight * result[-1] + (1.0 - weight) * value)
    return result


def _load_json_series(run_dir: Path) -> tuple[list[float], list[float]]:
    history_path = run_dir / "metrics_history.json"
    if not history_path.is_file():
        return [], []
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history", payload) if isinstance(payload, dict) else payload
    if not isinstance(history, list):
        return [], []
    points: list[tuple[float, float]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        y_value = next(
            (_number(entry.get(key)) for key in ("eval_success_once", "success_once", "train_success_once")),
            None,
        )
        if y_value is None:
            continue
        minutes = next(
            (_number(entry.get(key)) for key in ("elapsed_minutes", "time_minutes")),
            None,
        )
        if minutes is None:
            hours = _number(entry.get("elapsed_hours"))
            minutes = hours * 60.0 if hours is not None else float(index)
        points.append((max(0.0, minutes), max(0.0, min(1.0, y_value))))
    points.sort(key=lambda item: item[0])
    return [item[0] for item in points], [item[1] for item in points]


def _load_tensorboard_series(run_dir: Path) -> tuple[list[float], list[float]]:
    event_paths = sorted((run_dir / "tb").glob("events.out.tfevents.*"))
    if not event_paths:
        return [], []

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir / "tb"), size_guidance={"scalars": 0})
    accumulator.Reload()
    if TENSORBOARD_SUCCESS_TAG not in accumulator.Tags().get("scalars", []):
        return [], []

    events = accumulator.Scalars(TENSORBOARD_SUCCESS_TAG)
    if not events:
        return [], []
    start_time = min(event.wall_time for event in events)
    return (
        [max(0.0, (event.wall_time - start_time) / 60.0) for event in events],
        [max(0.0, min(1.0, event.value)) for event in events],
    )


def load_series(run_dir: Path) -> tuple[list[float], list[float]]:
    x_values, y_values = _load_json_series(run_dir)
    if x_values:
        return x_values, y_values
    return _load_tensorboard_series(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Training accuracy")
    args = parser.parse_args()

    x_values, y_values = load_series(args.run_dir)
    y_smoothed = _smooth(y_values)
    fig, ax = plt.subplots(figsize=(9.6, 8.0), dpi=200)
    if x_values:
        ax.plot(x_values, y_smoothed, color="#C44E52", linewidth=3.6, label="VLASelect")
        ax.set_xlim(0.0, max(1.0, max(x_values)))
    else:
        ax.text(0.5, 0.5, "No training metrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (minutes)", fontsize=22)
    ax.set_ylabel("Training success rate", fontsize=22)
    ax.set_title(args.title, fontsize=24)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=16)
    if x_values:
        ax.legend(fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[plot] output={args.output}")


if __name__ == "__main__":
    main()
