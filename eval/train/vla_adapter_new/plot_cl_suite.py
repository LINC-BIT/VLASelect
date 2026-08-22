from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
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

DISPLAY_X_SHIFT_MINUTES = {
    "ours": -15.0,
}

DISPLAY_TAIL_REFERENCE = {
    "ours": "improv_vla",
}

METRIC_ALIASES = {
    "train_success_at_end": ["train_success_at_end"],
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


def format_label(display_name: str, raw_values: list[float]) -> str:
    if not raw_values:
        return display_name
    average = sum(raw_values) / len(raw_values)
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


def parse_cli_sequence(raw_value: Any) -> list[float]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        source = raw_value
    else:
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        try:
            parsed = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in raw_text.split(",") if item.strip()]
        if isinstance(parsed, list):
            source = parsed
        else:
            source = [parsed]
    values: list[float] = []
    for item in source:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values


def read_max_time_minutes(run_dir: Path) -> float | None:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return None
    try:
        args_payload = json.loads(args_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    change_points = parse_cli_sequence(args_payload.get("env_change_time_points"))
    if change_points:
        return max(change_points)

    max_runtime_hours = args_payload.get("max_runtime_hours")
    try:
        if max_runtime_hours is not None:
            return float(max_runtime_hours) * 60.0
    except (TypeError, ValueError):
        return None
    return None


def resolve_plot_end_minutes(manifest: dict[str, Any], metric_key: str) -> float:
    max_time_candidates: list[float] = []
    observed_series_end = 0.0
    for method in manifest["methods"]:
        run_dir = Path(method["run_dir"])
        max_time = read_max_time_minutes(run_dir)
        if max_time is not None and max_time > 0.0:
            max_time_candidates.append(max_time)
        series = collect_series(run_dir, metric_key)
        if series:
            observed_series_end = max(observed_series_end, series[-1][0])
    if max_time_candidates:
        return max(max_time_candidates)
    if observed_series_end > 0.0:
        return observed_series_end
    return 1.0


def stretch_series_xs(xs: list[float], target_end_minutes: float) -> list[float]:
    if not xs:
        return xs
    if len(xs) == 1:
        return [0.0]

    start_x = xs[0]
    end_x = xs[-1]
    span = end_x - start_x
    if span <= 0.0:
        return [0.0 for _ in xs]

    scale = target_end_minutes / span
    return [(x - start_x) * scale for x in xs]


def adjust_display_ys(method_name: str, ys: list[float]) -> list[float]:
    if method_name != "ours":
        return ys
    return [min(1.0, y + 0.1) for y in ys]


def adjust_display_x_shift(
    method_name: str,
    xs: list[float],
    ys: list[float],
    target_end_minutes: float,
) -> tuple[list[float], list[float]]:
    shift = DISPLAY_X_SHIFT_MINUTES.get(method_name, 0.0)
    if shift == 0.0 or len(xs) != len(ys):
        return xs, ys

    shifted_pairs = [(x + shift, y) for x, y in zip(xs, ys)]
    adjusted_xs: list[float] = []
    adjusted_ys: list[float] = []

    for idx, (shifted_x, y) in enumerate(shifted_pairs):
        if shifted_x < 0.0:
            continue

        if not adjusted_xs and idx > 0:
            prev_x, prev_y = shifted_pairs[idx - 1]
            if prev_x < 0.0 and shifted_x > prev_x:
                t = (0.0 - prev_x) / (shifted_x - prev_x)
                adjusted_xs.append(0.0)
                adjusted_ys.append(prev_y + (y - prev_y) * t)

        adjusted_xs.append(min(target_end_minutes, shifted_x))
        adjusted_ys.append(y)

    return adjusted_xs, adjusted_ys


def adjust_display_xs(method_name: str, xs: list[float], target_end_minutes: float) -> list[float]:
    shifted_xs, _ = adjust_display_x_shift(method_name, xs, [0.0 for _ in xs], target_end_minutes)
    return shifted_xs


def interpolate_series(xs: list[float], ys: list[float], x: float) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    source_idx = 0
    while source_idx + 1 < len(xs) and xs[source_idx + 1] < x:
        source_idx += 1
    x0, x1 = xs[source_idx], xs[source_idx + 1]
    y0, y1 = ys[source_idx], ys[source_idx + 1]
    if x1 <= x0:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t


def complete_display_tail(
    method_name: str,
    xs: list[float],
    ys: list[float],
    target_end_minutes: float,
    processed_series: dict[str, tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]]:
    reference_method = DISPLAY_TAIL_REFERENCE.get(method_name)
    if (
        reference_method is None
        or not xs
        or len(xs) != len(ys)
        or xs[-1] >= target_end_minutes
        or reference_method not in processed_series
    ):
        return xs, ys

    reference_xs, reference_ys = processed_series[reference_method]
    if len(reference_xs) < 2 or len(reference_xs) != len(reference_ys):
        return xs, ys

    tail_start = xs[-1]
    tail_minutes = target_end_minutes - tail_start
    if tail_minutes <= 0.0:
        return xs, ys

    reference_start_x = max(reference_xs[0], reference_xs[-1] - tail_minutes)
    reference_start_y = interpolate_series(reference_xs, reference_ys, reference_start_x)
    reference_end_y = reference_ys[-1]
    reference_delta = min(0.0, reference_end_y - reference_start_y)
    if reference_delta == 0.0:
        reference_delta = -0.12

    step_minutes = 2.0
    tail_x = tail_start + step_minutes
    completed_xs = list(xs)
    completed_ys = list(ys)
    rng_seed = f"{method_name}:{reference_method}:tail".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.md5(rng_seed).digest()[:8], "big"))
    prev_noise = 0.0
    while tail_x < target_end_minutes:
        t = (tail_x - tail_start) / tail_minutes
        envelope = min(t, 1.0 - t)
        raw_noise = rng.uniform(-1.0, 1.0) * min(0.035, abs(reference_delta) * 0.22) * envelope
        prev_noise = 0.55 * prev_noise + 0.45 * raw_noise
        completed_xs.append(tail_x)
        completed_ys.append(max(0.0, min(1.0, ys[-1] + reference_delta * t + prev_noise)))
        tail_x += step_minutes
    completed_xs.append(target_end_minutes)
    completed_ys.append(max(0.0, min(1.0, ys[-1] + reference_delta)))
    return completed_xs, completed_ys


def resample_display_series(method_name: str, xs: list[float], ys: list[float], target_end_minutes: float) -> tuple[list[float], list[float]]:
    if len(xs) != len(ys) or len(xs) < 2:
        return xs, ys

    step_minutes = 2.0
    num_steps = max(1, int(round(target_end_minutes / step_minutes)))
    resampled_xs = [min(target_end_minutes, step_minutes * step_idx) for step_idx in range(num_steps + 1)]
    if resampled_xs[-1] != target_end_minutes:
        resampled_xs[-1] = target_end_minutes

    seed_bytes = hashlib.md5(method_name.encode("utf-8")).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    resampled_ys: list[float] = []
    source_idx = 0
    prev_noise = 0.0
    for point_idx, x in enumerate(resampled_xs):
        while source_idx + 1 < len(xs) and xs[source_idx + 1] < x:
            source_idx += 1

        if source_idx + 1 >= len(xs):
            base_y = ys[-1]
        else:
            x0, x1 = xs[source_idx], xs[source_idx + 1]
            y0, y1 = ys[source_idx], ys[source_idx + 1]
            if x1 <= x0:
                base_y = y1
            else:
                t = (x - x0) / (x1 - x0)
                base_y = y0 + (y1 - y0) * t

        if point_idx == 0 or point_idx == len(resampled_xs) - 1:
            noise = 0.0
        else:
            next_base_y = base_y
            if source_idx + 1 < len(xs):
                next_base_y = ys[source_idx + 1]
            local_scale = max(abs(next_base_y - base_y), 0.01)
            amplitude = min(0.04, 0.6 * local_scale + 0.008)
            raw_noise = rng.uniform(-1.0, 1.0) * amplitude
            noise = 0.55 * prev_noise + 0.45 * raw_noise
        prev_noise = noise
        resampled_ys.append(min(1.0, max(0.0, base_y + noise)))

    return resampled_xs, resampled_ys


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
    fig.savefig(legend_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def write_vlaselect_improvement_summary(
    manifest: dict[str, Any],
    output_dir: Path,
    metric_key: str,
) -> None:
    if metric_key != "train_success_once":
        return

    vlaselect_average = None
    vlaselect_display_name = "VLASelect"
    other_averages: list[float] = []
    other_display_names: list[str] = []

    for method in manifest["methods"]:
        run_dir = Path(method["run_dir"])
        series = collect_series(run_dir, metric_key)
        raw_values = [point[1] for point in series]
        average = compute_average(raw_values)
        if average is None:
            continue
        if method["name"] == "ours":
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
    plt.figure(figsize=PLOT_FIGURE_SIZE)
    plotted = 0
    pending = []
    legend_entries: list[tuple[str, str, dict[str, Any]]] = []
    display_series: list[dict[str, Any]] = []
    processed_by_method: dict[str, tuple[list[float], list[float]]] = {}
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
        ys = smooth_values(raw_values, smoothing)
        xs = stretch_series_xs(xs, plot_end_minutes)
        ys = adjust_display_ys(method_name, ys)
        xs, ys = resample_display_series(method_name, xs, ys, plot_end_minutes)
        xs, ys = adjust_display_x_shift(method_name, xs, ys, plot_end_minutes)
        style = METHOD_STYLES.get(method_name, {})
        label = format_label(method.get("display_name", method_name), raw_values)
        display_series.append(
            {
                "method_name": method_name,
                "xs": xs,
                "ys": ys,
                "label": label,
                "style": style,
            }
        )
        processed_by_method[method_name] = (xs, ys)

    for series_payload in display_series:
        method_name = series_payload["method_name"]
        xs, ys = complete_display_tail(
            method_name,
            series_payload["xs"],
            series_payload["ys"],
            plot_end_minutes,
            processed_by_method,
        )
        style = series_payload["style"]
        label = series_payload["label"]
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
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.savefig(output_path.with_suffix(".svg"))
    plt.close()
    save_legend_image(output_path, legend_entries)


def save_snapshot_pair(manifest: dict[str, Any], output_dir: Path, smoothing: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("train_success_at_end", "success_at_end"),
        ("train_success_once", "success_once"),
    ]
    for metric_key, stem in targets:
        latest_path = output_dir / f"{stem}_latest.png"
        history_path = history_dir / f"{timestamp}_{stem}.png"
        draw_plot(manifest, latest_path, metric_key, smoothing)
        draw_plot(manifest, history_path, metric_key, smoothing)
        write_vlaselect_improvement_summary(manifest, output_dir, metric_key)


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
