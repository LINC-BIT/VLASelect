from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator


PLOT_FONT_SIZE = 36
PLOT_FIGURE_HEIGHT = 8
PLOT_FIGURE_WIDTH = 9.6

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
    "ours_single_agent": {"color": "#C44E52", "linestyle": "-"},
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
    "ours_single_agent",
]

TAG_ALIASES = {
    "eval/success_at_end": ["eval_success_at_end", "eval_success_end", "eval/success_at_end", "eval/success_end"],
    "eval/success_once": ["eval_success_once", "eval/success_once"],
}


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    weight = float(smoothing)
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + value * (1.0 - weight))
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
    ours_average = method_averages.get("ours_single_agent")
    if ours_average is None:
        return None
    other_averages = [value for name, value in method_averages.items() if name != "ours_single_agent"]
    if not other_averages:
        return None
    return (ours_average - (sum(other_averages) / len(other_averages))) * 100.0


def compute_average_summary(method_averages: dict[str, float]) -> dict[str, float | None]:
    ours_average = method_averages.get("ours_single_agent")
    other_averages = [value for name, value in method_averages.items() if name != "ours_single_agent"]
    others_average = None
    if other_averages:
        others_average = sum(other_averages) / len(other_averages)
    improvement = None
    if ours_average is not None and others_average is not None:
        improvement = (ours_average - others_average) * 100.0
    return {
        "ours_average": ours_average,
        "others_average": others_average,
        "improvement": improvement,
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


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(run_dir: Path) -> list[dict]:
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
    return [entry for entry in history if isinstance(entry, dict)]


def collect_json_series(run_dir: Path, tag: str):
    history = load_history(run_dir)
    if not history:
        return []
    candidate_tags = TAG_ALIASES.get(tag, [tag])
    series = []
    for index, metric in enumerate(history):
        value = None
        for candidate_tag in candidate_tags:
            if metric.get(candidate_tag) is not None:
                value = metric[candidate_tag]
                break
        if value is None:
            continue
        elapsed_hours = metric.get("elapsed_hours")
        try:
            x_value = float(elapsed_hours) * 60.0 if elapsed_hours is not None else float(index)
            y_value = float(value)
        except (TypeError, ValueError):
            continue
        series.append((x_value, y_value))
    return series


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return accumulator.Scalars(tag)


def collect_series(run_dir: Path, tag: str):
    json_series = collect_json_series(run_dir, tag)
    if json_series:
        return json_series
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return []
    candidate_tags = TAG_ALIASES.get(tag, [tag])
    for candidate_tag in candidate_tags:
        try:
            events = load_scalar_events(tb_dir, candidate_tag)
        except Exception:
            continue
        if not events:
            continue
        base_time = events[0].wall_time
        return [((event.wall_time - base_time) / 60.0, event.value) for event in events]
    return []


def read_max_time_minutes(run_dir: Path) -> float | None:
    candidate_paths = [
        run_dir / "code" / "args.txt",
        run_dir.parent / "code" / "args.txt",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        for line in candidate_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("max_time:"):
                continue
            raw_value = line.split(":", 1)[1].strip()
            if raw_value in {"None", ""}:
                return None
            try:
                return float(raw_value)
            except ValueError:
                return None
    return None


def resolve_plot_end_minutes(manifest: dict, tag: str) -> float:
    max_time_candidates: list[float] = []
    observed_series_end = 0.0
    for method in manifest["methods"]:
        run_dir = Path(method["run_dir"])
        max_time = read_max_time_minutes(run_dir)
        if max_time is not None and max_time > 0.0:
            max_time_candidates.append(max_time)
        series = collect_series(run_dir, tag)
        if series:
            observed_series_end = max(observed_series_end, series[-1][0])
    if max_time_candidates:
        return max(max_time_candidates)
    if observed_series_end > 0.0:
        return observed_series_end
    return 1.0


def order_legend_entries(entries: list[tuple[str, str, dict]]) -> list[tuple[str, str, dict]]:
    order_index = {name: idx for idx, name in enumerate(LEGEND_ORDER)}
    return sorted(
        entries,
        key=lambda entry: (order_index.get(entry[0], len(order_index)), entry[1]),
    )


def save_legend_image(output_path: Path, legend_entries: list[tuple[str, str, dict]]) -> None:
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


def draw_plot(manifest: dict, output_path: Path, tag: str, smoothing: float) -> dict[str, float | None]:
    plt.figure(figsize=(PLOT_FIGURE_WIDTH, PLOT_FIGURE_HEIGHT))
    plotted = 0
    pending = []
    method_averages: dict[str, float] = {}
    legend_entries: list[tuple[str, str, dict]] = []
    plot_end_minutes = resolve_plot_end_minutes(manifest, tag)

    for method in manifest["methods"]:
        method_name = method["name"]
        run_dir = Path(method["run_dir"])
        series = collect_series(run_dir, tag)
        if not series:
            pending.append(method.get("display_name", method_name))
            continue

        xs = [point[0] for point in series]
        raw_values = [point[1] for point in series]
        average = compute_average(raw_values)
        if average is not None:
            method_averages[method_name] = average
        ys = smooth_values(raw_values, smoothing)
        style = METHOD_STYLES.get(method_name, {})
        label = format_label(method.get("display_name", method_name), raw_values)
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
    return compute_average_summary(method_averages)


def write_improvement_summary(
    output_dir: Path,
    smoothing: float,
    summaries: dict[str, dict[str, float | None]],
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_path = output_dir / "ours_single_agent_average_improvement.txt"
    lines = [
        f"saved_at: {timestamp}",
        f"smoothing: {smoothing:.2f}",
    ]
    for stem, label in (
        ("success_at_end", "Success@End"),
        ("success_once", "Success@Once"),
    ):
        summary = summaries.get(stem, {})
        ours_average = summary.get("ours_average")
        others_average = summary.get("others_average")
        improvement = summary.get("improvement")
        if ours_average is None or others_average is None or improvement is None:
            lines.append(f"{label}: unavailable")
            continue
        lines.append(f"{label} ours_avg: {ours_average:.4f}")
        lines.append(f"{label} others_avg: {others_average:.4f}")
        lines.append(f"{label} improvement: {improvement:+.2f}%")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_snapshot_pair(manifest: dict, output_dir: Path, smoothing: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, float | None]] = {}

    targets = [
        ("eval/success_at_end", "success_at_end"),
        ("eval/success_once", "success_once"),
    ]
    for tag, stem in targets:
        latest_path = output_dir / f"{stem}_latest.png"
        history_path = history_dir / f"{timestamp}_{stem}.png"
        summaries[stem] = draw_plot(manifest, latest_path, tag, smoothing)
        draw_plot(manifest, history_path, tag, smoothing)
    write_improvement_summary(output_dir, smoothing, summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--smoothing", type=float, default=0.7)
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
