#!/usr/bin/env python3
"""Aggregate continual-learning accuracy by the number of completed environments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("self_improv", "vla_rft", "world_env", "vlaselect")
DISPLAY = {
    "self_improv": "Self-Improvement",
    "vla_rft": "VLA-RFT",
    "world_env": "WorldEnv",
    "vlaselect": "VLASelect",
}


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [run_dir / "metrics_history.json"]
    candidates.extend(sorted(run_dir.glob("**/metrics_history.json")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = payload.get("history", payload) if isinstance(payload, dict) else payload
        if isinstance(history, list):
            return [item for item in history if isinstance(item, dict)]
    return []


def metric_value(entry: dict[str, Any]) -> float | None:
    for key in ("historical_success_once", "eval_success_once", "success_once", "train_success_once"):
        value = finite(entry.get(key))
        if value is not None:
            return value
    # Optional per-environment payload emitted by future training copies.
    historical = entry.get("historical_envs")
    if isinstance(historical, dict):
        values = [finite(item.get("success_once")) for item in historical.values() if isinstance(item, dict)]
        values = [value for value in values if value is not None]
        if values:
            return sum(values) / len(values)
    return None


def collect_method(run_dir: Path, env_count: int) -> tuple[list[float | None], str]:
    history = load_history(run_dir)
    by_env: dict[int, list[float]] = {}
    quality = "stage-fallback"
    for entry in history:
        index = finite(entry.get("current_env_index"))
        value = metric_value(entry)
        if index is None or value is None:
            continue
        env_index = int(index)
        by_env.setdefault(env_index, []).append(value)
        if "historical_envs" in entry or "historical_success_once" in entry:
            quality = "historical-eval"

    # A true historical snapshot may provide all prior env values in one entry.
    snapshots: dict[int, list[float]] = {}
    for entry in history:
        stage = finite(entry.get("current_env_index"))
        payload = entry.get("historical_envs")
        if stage is None or not isinstance(payload, dict):
            continue
        values = [finite(item.get("success_once")) for item in payload.values() if isinstance(item, dict)]
        values = [value for value in values if value is not None]
        if values:
            snapshots[int(stage)] = values
            quality = "historical-eval"

    series: list[float | None] = []
    latest: dict[int, float] = {}
    for stage in range(env_count):
        if stage in snapshots:
            latest = {index: value for index, value in enumerate(snapshots[stage])}
        elif stage in by_env:
            latest[stage] = by_env[stage][-1]
        values = [latest[index] for index in sorted(latest) if index <= stage]
        series.append(sum(values) / len(values) if values else None)
    return series, quality


def locate_run(root: Path, method: str) -> Path | None:
    direct = root / method
    if direct.exists():
        return direct
    matches = sorted(root.glob(f"**/{method}"))
    return matches[-1] if matches else None


def fmt(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.4f}"


def write_plot(output_dir: Path, series: dict[str, list[float | None]], env_count: int) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "self_improv": "#777777",
        "vla_rft": "#4C78A8",
        "world_env": "#59A14F",
        "vlaselect": "#C44E52",
    }
    markers = {"self_improv": "o", "vla_rft": "s", "world_env": "^", "vlaselect": "D"}
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    x_values = list(range(1, env_count + 1))
    for method in METHODS:
        axis.plot(
            x_values,
            [math.nan if value is None else value for value in series[method]],
            label=DISPLAY[method],
            color=colors[method],
            marker=markers[method],
            linewidth=2.2,
        )
    axis.set_xlabel("Environment/Task")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(x_values)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"forgetting_accuracy.{suffix}", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--env-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    env_count = max(1, args.env_count)
    output_dir = args.output_dir or args.run_root
    output_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, list[float | None]] = {}
    quality: dict[str, str] = {}
    runs: dict[str, str | None] = {}
    for method in METHODS:
        run = locate_run(args.run_root, method)
        runs[method] = str(run) if run else None
        values, source = collect_method(run, env_count) if run else ([None] * env_count, "missing")
        series[method] = values
        quality[method] = source

    rows = []
    for index in range(env_count):
        row = {"env_index": index + 1}
        for method in METHODS:
            row[method] = series[method][index]
        rows.append(row)

    summary_json = {
        "metric": "success_once",
        "improvement": "absolute_percentage_points",
        "env_count": env_count,
        "methods": {method: {"run_dir": runs[method], "quality": quality[method], "series": series[method]} for method in METHODS},
        "rows": rows,
    }
    with (output_dir / "forgetting_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["env_index", *METHODS])
        writer.writeheader()
        writer.writerows(rows)
    write_plot(output_dir, series, env_count)

    table = []
    for baseline in METHODS[:-1]:
        values = []
        for index in range(env_count):
            ours = series["vlaselect"][index]
            other = series[baseline][index]
            values.append((ours - other) * 100.0 if ours is not None and other is not None else None)
        available = [value for value in values if value is not None]
        table.append({"baseline": DISPLAY[baseline], **{f"Env {i + 1}": values[i] for i in range(env_count)}, "All Env": sum(available) / len(available) if available else None})

    # Report the mean uplift across every available baseline at each stage and
    # across all stages. This keeps the final summary useful when baselines have
    # different missing-data patterns.
    average_row: dict[str, float | str | None] = {"baseline": "All baselines average"}
    all_improvements: list[float] = []
    for index in range(env_count):
        stage_values = [
            row[f"Env {index + 1}"]
            for row in table
            if row[f"Env {index + 1}"] is not None
        ]
        average = sum(stage_values) / len(stage_values) if stage_values else None
        average_row[f"Env {index + 1}"] = average
        all_improvements.extend(stage_values)
    average_row["All Env"] = sum(all_improvements) / len(all_improvements) if all_improvements else None
    table.append(average_row)
    summary_json["improvement_table"] = table
    (output_dir / "forgetting_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("Past-Env success_once accuracy (mean over completed Env evaluations)")
    print("env\t" + "\t".join(DISPLAY[method] for method in METHODS))
    for row in rows:
        print(str(row["env_index"]) + "\t" + "\t".join(fmt(row[method]) for method in METHODS))
    print("\nVLASelect improvement over baseline (absolute percentage points)")
    print("baseline\t" + "\t".join([f"Env {i + 1}" for i in range(env_count)] + ["All Env"]))
    for row in table:
        print(row["baseline"] + "\t" + "\t".join(fmt(row[key]) + " pp" if row[key] is not None else "NaN" for key in [f"Env {i + 1}" for i in range(env_count)] + ["All Env"]))
    overall_average = average_row["All Env"]
    if overall_average is None:
        print("VLASelect avg. improvement over all baselines: NaN")
    else:
        print(f"[Result] VLASelect avg. improvement over all baselines: {overall_average:.2f}%")
    print(f"\nsummary_json: {output_dir / 'forgetting_summary.json'}")
    print(f"summary_csv: {output_dir / 'forgetting_summary.csv'}")
    print(f"figure: {output_dir / 'forgetting_accuracy.pdf'}")
    print("data_source: " + ", ".join(f"{DISPLAY[m]}={quality[m]}" for m in METHODS))


if __name__ == "__main__":
    main()
