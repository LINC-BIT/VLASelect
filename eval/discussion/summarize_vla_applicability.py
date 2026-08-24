from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_PDF_PATH = SCRIPT_DIR / "FIG_VLA_APPLICABILITY.pdf"
FIGURE_PNG_PATH = SCRIPT_DIR / "FIG_VLA_APPLICABILITY.png"
FIGURE_SVG_PATH = SCRIPT_DIR / "FIG_VLA_APPLICABILITY.svg"
SUMMARY_CSV_PATH = SCRIPT_DIR / "vla_applicability_summary.csv"
SUMMARY_JSON_PATH = SCRIPT_DIR / "vla_applicability_summary.json"

FAMILY_ORDER = ["octo", "vla_adapter_new", "tinyvla", "edgevla"]
FAMILY_DISPLAY_NAMES = {
    "octo": "Octo",
    "vla_adapter_new": "VLA-Adapter",
    "tinyvla": "TinyVLA",
    "edgevla": "EdgeVLA",
}
FAMILY_STYLES = {
    "octo": {"color": "#4C78A8", "linestyle": "-", "linewidth": 3.6},
    "vla_adapter_new": {"color": "#59A14F", "linestyle": "-", "linewidth": 3.6},
    "tinyvla": {"color": "#4D4D4D", "linestyle": "-", "linewidth": 3.6},
    "edgevla": {"color": "#C44E52", "linestyle": "-", "linewidth": 3.6},
}
HISTORY_METRIC_ALIASES_BY_FAMILY = {
    "octo": ("eval_success_once", "eval/success_once", "success_once"),
    "vla_adapter_new": ("train_success_once", "eval_success_once", "success_once"),
    "tinyvla": ("train_success_once", "eval_success_once", "success_once"),
    "edgevla": ("eval_success_once", "success_once"),
}
TB_METRIC_TAGS_BY_FAMILY = {
    "octo": ("eval/success_once",),
    "vla_adapter_new": ("eval/success_once", "train_success_once"),
    "tinyvla": ("eval/success_once", "train_success_once"),
    "edgevla": ("eval/success_once",),
}
TB_ELAPSED_TAGS = ("time/elapsed_minutes",)
SMOOTHING = 0.7
MAX_X_FALLBACK_MINUTES = 5.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 24,
        "svg.fonttype": "none",
    }
)


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_history(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return [item for item in payload["history"] if isinstance(item, dict)]
    return []


def load_run_history(run_dir: Path) -> List[Dict[str, Any]]:
    direct = load_history(run_dir / "metrics_history.json")
    if direct:
        return direct
    for nested_history_path in sorted(path for path in run_dir.glob("**/metrics_history.json") if path.is_file()):
        nested = load_history(nested_history_path)
        if nested:
            return nested
    return []


def process_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.4f}"


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    return _maybe_float(metrics.get(key))


def final_metric(metrics: Optional[Dict[str, Any]], primary_key: str, fallback_key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    return _maybe_float(metrics.get(primary_key, metrics.get(fallback_key)))


def extract_history_success_value(family: str, metric: Dict[str, Any]) -> Optional[float]:
    for key in HISTORY_METRIC_ALIASES_BY_FAMILY.get(family, ()): 
        value = _maybe_float(metric.get(key))
        if value is not None:
            return value
    return None


def extract_elapsed_minutes(metric: Dict[str, Any], default_index: int) -> float:
    elapsed_minutes = _maybe_float(metric.get("elapsed_minutes"))
    if elapsed_minutes is not None:
        return elapsed_minutes
    elapsed_hours = _maybe_float(metric.get("elapsed_hours"))
    if elapsed_hours is not None:
        return elapsed_hours * 60.0
    return float(default_index)


def collect_history_points(run_dir: Path, family: str) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    for index, metric in enumerate(load_run_history(run_dir)):
        value = extract_history_success_value(family, metric)
        if value is None:
            continue
        minute = extract_elapsed_minutes(metric, index)
        points.append((minute, value))
    return points


def find_tb_dir(run_dir: Path) -> Path | None:
    direct_tb_dir = run_dir / "tb"
    if direct_tb_dir.is_dir():
        return direct_tb_dir
    nested_tb_dirs = sorted(path for path in run_dir.glob("**/tb") if path.is_dir())
    return nested_tb_dirs[0] if nested_tb_dirs else None


def load_scalar_events(tb_dir: Path, candidate_tags: tuple[str, ...]):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    for tag in candidate_tags:
        if tag in scalar_tags:
            return accumulator.Scalars(tag)
    return []


def align_metric_to_elapsed(metric_events, elapsed_events) -> List[tuple[float, float]]:
    if not metric_events or not elapsed_events:
        return []
    elapsed_steps = [event.step for event in elapsed_events]
    elapsed_values = [float(event.value) for event in elapsed_events]
    aligned: List[tuple[float, float]] = []
    for metric_event in metric_events:
        index = 0
        lo, hi = 0, len(elapsed_steps)
        while lo < hi:
            mid = (lo + hi) // 2
            if elapsed_steps[mid] <= metric_event.step:
                lo = mid + 1
            else:
                hi = mid
        index = lo - 1
        if index < 0:
            continue
        aligned.append((elapsed_values[index], float(metric_event.value)))
    return aligned


def collect_tensorboard_points(run_dir: Path, family: str) -> List[tuple[float, float]]:
    tb_dir = find_tb_dir(run_dir)
    if tb_dir is None:
        return []
    try:
        elapsed_events = load_scalar_events(tb_dir, TB_ELAPSED_TAGS)
        metric_events = load_scalar_events(tb_dir, TB_METRIC_TAGS_BY_FAMILY.get(family, tuple()))
    except Exception:
        return []
    return align_metric_to_elapsed(metric_events, elapsed_events)


def collect_success_points(run_dir: Path, family: str) -> List[tuple[float, float]]:
    history_points = collect_history_points(run_dir, family)
    if history_points:
        return history_points
    return collect_tensorboard_points(run_dir, family)


def smooth_values(values: List[float], smoothing: float) -> List[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + value * (1.0 - smoothing))
    return smoothed


def summarize_octo_suite(entry: Dict[str, Any]) -> Dict[str, Any]:
    suite_dir = Path(entry["run_dir"])
    summary_rows = load_csv_rows(suite_dir / "summary.csv")
    row = summary_rows[0] if summary_rows else {}
    latest_eval_once = latest_eval_end = final_eval_once = final_eval_end = None
    history_len = 0
    status = "launching" if process_alive(entry.get("pid")) else "failed"
    if row:
        latest_eval_once = _maybe_float(row.get("final_success_once"))
        latest_eval_end = _maybe_float(row.get("final_success_at_end"))
        final_eval_once = latest_eval_once
        final_eval_end = latest_eval_end
        history_len = int(float(row.get("num_eval_points", "0") or "0"))
        status = row.get("status") or status
        if status == "completed":
            status = "completed"
        elif process_alive(entry.get("pid")):
            status = "running"
        else:
            status = "partial"
    return {
        "family": entry["family"],
        "status": status,
        "pid": entry.get("pid"),
        "log_file": entry.get("log_file"),
        "run_dir": str(suite_dir),
        "history_len": history_len,
        "latest_train_once": None,
        "latest_train_end": None,
        "latest_eval_once": latest_eval_once,
        "latest_eval_end": latest_eval_end,
        "final_eval_once": final_eval_once,
        "final_eval_end": final_eval_end,
    }


def resolve_status(entry: Dict[str, Any], run_dir: Path) -> str:
    pid = entry.get("pid")
    final_metrics = load_json(run_dir / "final_eval_metrics.json")
    latest_metrics = load_json(run_dir / "latest_metrics.json")
    if final_metrics is not None:
        return "completed"
    if latest_metrics is not None and process_alive(pid):
        return "running"
    if latest_metrics is not None:
        return "partial"
    if process_alive(pid):
        return "launching"
    return "failed"


def summarize_json_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(entry["run_dir"])
    latest_metrics = load_json(run_dir / "latest_metrics.json")
    final_metrics = load_json(run_dir / "final_eval_metrics.json")
    history = load_run_history(run_dir)
    return {
        "family": entry["family"],
        "status": resolve_status(entry, run_dir),
        "pid": entry.get("pid"),
        "log_file": entry.get("log_file"),
        "run_dir": str(run_dir),
        "history_len": len(history),
        "latest_train_once": latest_metric(latest_metrics, "train_success_once"),
        "latest_train_end": latest_metric(latest_metrics, "train_success_at_end"),
        "latest_eval_once": latest_metric(latest_metrics, "eval_success_once"),
        "latest_eval_end": latest_metric(latest_metrics, "eval_success_at_end"),
        "final_eval_once": final_metric(final_metrics, "success_once", "success"),
        "final_eval_end": final_metric(final_metrics, "success_at_end", "success"),
    }


def summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if entry.get("kind") == "octo_suite":
        return summarize_octo_suite(entry)
    return summarize_json_entry(entry)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("")
    print("VLA applicability summary")
    print("family         status      latest_eval_once  latest_eval_end  final_eval_once  final_eval_end  updates")
    for row in rows:
        print(
            f"{row['family']:<14}"
            f"{row['status']:<12}"
            f"{format_metric(row['latest_eval_once']):<18}"
            f"{format_metric(row['latest_eval_end']):<17}"
            f"{format_metric(row['final_eval_once']):<17}"
            f"{format_metric(row['final_eval_end']):<16}"
            f"{row['history_len']}"
        )
    print("")
    for row in rows:
        print(f"[{row['family']}] run_dir={row['run_dir']}")
        print(f"[{row['family']}] log_file={row['log_file']}")


def all_finished(rows: List[Dict[str, Any]]) -> bool:
    return all(row["status"] in {"completed", "partial", "failed"} for row in rows)


def choose_best_octo_run(suite_dir: Path) -> tuple[Path | None, List[tuple[float, float]]]:
    payload = load_json(suite_dir / "manifest.json")
    if not isinstance(payload, dict):
        return None, []
    best_run_dir: Path | None = None
    best_points: List[tuple[float, float]] = []
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        raw_run_dir = run.get("run_dir")
        if not raw_run_dir:
            continue
        run_dir = Path(raw_run_dir)
        points = collect_success_points(run_dir, "octo")
        if not points:
            continue
        if len(points) > len(best_points):
            best_run_dir = run_dir
            best_points = points
        elif len(points) == len(best_points) and points and best_points and points[-1][0] > best_points[-1][0]:
            best_run_dir = run_dir
            best_points = points
    return best_run_dir, best_points


def collect_plot_series(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    family = str(entry.get("family", ""))
    if not family:
        return None
    if entry.get("kind") == "octo_suite":
        suite_dir = Path(entry["run_dir"])
        source_run_dir, points = choose_best_octo_run(suite_dir)
        if source_run_dir is None:
            source_run_dir = suite_dir
    else:
        source_run_dir = Path(entry["run_dir"])
        points = collect_success_points(source_run_dir, family)
    if not points:
        return {
            "family": family,
            "display_name": FAMILY_DISPLAY_NAMES.get(family, family),
            "source_run_dir": str(source_run_dir),
            "num_points": 0,
            "last_minutes": None,
            "final_accuracy": None,
            "x": [],
            "y": [],
            "style": FAMILY_STYLES.get(family, {"color": "#000000", "linestyle": "-", "linewidth": 3.6}),
        }
    points = sorted(points, key=lambda item: item[0])
    xs = [point[0] for point in points]
    ys_raw = [point[1] for point in points]
    ys = smooth_values(ys_raw, SMOOTHING)
    return {
        "family": family,
        "display_name": FAMILY_DISPLAY_NAMES.get(family, family),
        "source_run_dir": str(source_run_dir),
        "num_points": len(points),
        "last_minutes": xs[-1],
        "final_accuracy": ys_raw[-1],
        "x": xs,
        "y": ys,
        "style": FAMILY_STYLES.get(family, {"color": "#000000", "linestyle": "-", "linewidth": 3.6}),
    }


def resolve_dynamic_xlim(series_items: List[Dict[str, Any]]) -> List[float]:
    max_x = 0.0
    for item in series_items:
        xs = item.get("x", [])
        if xs:
            max_x = max(max_x, max(float(x) for x in xs))
    if max_x <= 0.0:
        return [0.0, MAX_X_FALLBACK_MINUTES]
    padded_right = max_x * 1.03
    right = max(max_x + 0.5, padded_right)
    return [0.0, right]


def render_accuracy_plot(series_items: List[Dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(10.6, 7.2))
    ax = fig.add_subplot(111)

    plotted = 0
    for family in FAMILY_ORDER:
        item = next((candidate for candidate in series_items if candidate["family"] == family), None)
        if item is None or not item.get("x"):
            continue
        style = item["style"]
        ax.plot(
            item["x"],
            item["y"],
            linewidth=style["linewidth"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=item["display_name"],
        )
        plotted += 1

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(resolve_dynamic_xlim(series_items))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    if plotted > 0:
        ax.legend(loc="best", frameon=False)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(FIGURE_PDF_PATH)
    fig.savefig(FIGURE_PNG_PATH, dpi=200)
    fig.savefig(FIGURE_SVG_PATH)
    plt.close(fig)


def write_summary_files(
    *,
    manifest_path: Path,
    rows: List[Dict[str, Any]],
    series_items: List[Dict[str, Any]],
) -> None:
    series_by_family = {item["family"]: item for item in series_items}
    summary_rows: List[Dict[str, Any]] = []
    for row in rows:
        item = series_by_family.get(row["family"], {})
        merged = dict(row)
        merged["display_name"] = FAMILY_DISPLAY_NAMES.get(row["family"], row["family"])
        merged["plot_run_dir"] = item.get("source_run_dir")
        merged["plot_num_points"] = item.get("num_points")
        merged["plot_last_minutes"] = item.get("last_minutes")
        merged["plot_final_accuracy"] = item.get("final_accuracy")
        merged["manifest"] = str(manifest_path)
        summary_rows.append(merged)

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "figure_pdf": str(FIGURE_PDF_PATH),
                "figure_png": str(FIGURE_PNG_PATH),
                "figure_svg": str(FIGURE_SVG_PATH),
                "rows": summary_rows,
                "series": series_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "family",
        "display_name",
        "status",
        "pid",
        "run_dir",
        "plot_run_dir",
        "log_file",
        "history_len",
        "latest_train_once",
        "latest_train_end",
        "latest_eval_once",
        "latest_eval_end",
        "final_eval_once",
        "final_eval_end",
        "plot_num_points",
        "plot_last_minutes",
        "plot_final_accuracy",
        "manifest",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def build_plot_series(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    series_items: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        entry = next((candidate for candidate in entries if candidate.get("family") == family), None)
        if entry is None:
            continue
        item = collect_plot_series(entry)
        if item is not None:
            series_items.append(item)
    return series_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Invalid manifest: {manifest_path}")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"No runs found in manifest: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    try:
        while True:
            rows = [summarize_entry(entry) for entry in entries]
            print_summary(rows)
            if not args.wait or all_finished(rows):
                break
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Interrupted while waiting for runs to finish.")

    series_items = build_plot_series(entries)
    render_accuracy_plot(series_items)
    write_summary_files(manifest_path=manifest_path, rows=rows, series_items=series_items)
    print(f"figure: {FIGURE_PDF_PATH}")
    print(f"summary: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
