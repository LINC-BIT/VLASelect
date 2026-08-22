from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:  # pragma: no cover - optional dependency at runtime
    event_accumulator = None


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from common.figure_compose import compose_grid_figure, render_legend_image
from common.template_pdf_fill import fill_ablation_template
TABLE_ROOT = SCRIPT_DIR / "ablation_table"
LATEST_POINTER = TABLE_ROOT / "latest.txt"
FIGURE_PATH = SCRIPT_DIR / "FIG_ABLATION.pdf"
FIGURE_SVG_PATH = SCRIPT_DIR / "FIG_ABLATION.svg"
FIGURE_PNG_PATH = SCRIPT_DIR / "FIG_ABLATION.png"
SUMMARY_CSV_PATH = SCRIPT_DIR / "ablation_summary.csv"
SUMMARY_JSON_PATH = SCRIPT_DIR / "ablation_summary.json"
PANEL_OUTPUT_DIR = SCRIPT_DIR / 'FIG_ABLATION_panels'
PANEL_FIGURE_SIZE = (5.2, 4.0)
NO_DATA_TEXT = "No data"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

PANEL_SPECS = [
    {
        "panel_label": "a",
        "panel_id": "scaling_law_function",
        "title": "(a) Scaling law function",
        "curves": [
            {
                "curve_id": "without_scaling_law",
                "label": "Without scaling law",
                "color": "#4D4D4D",
                "linestyle": "--",
            },
            {
                "curve_id": "with_scaling_law",
                "label": "With scaling law",
                "color": "#C44E52",
                "linestyle": "-",
            },
        ],
    },
    {
        "panel_label": "b",
        "panel_id": "neuron_grained_scaling_up",
        "title": "(b) Scaling up with neurons",
        "curves": [
            {
                "curve_id": "random",
                "label": "Random",
                "color": "#4D4D4D",
                "linestyle": "--",
            },
            {
                "curve_id": "inverse",
                "label": "Most accuracy-unrelated",
                "color": "#4C78A8",
                "linestyle": ":",
            },
            {
                "curve_id": "neuron_grained",
                "label": "Most accuracy-related",
                "color": "#C44E52",
                "linestyle": "-",
            },
        ],
    },
    {
        "panel_label": "c",
        "panel_id": "scaling_down_freezing_vs_pruning",
        "title": "(c) Scaling down by freezing vs pruning",
        "curves": [
            {
                "curve_id": "freezing",
                "label": "Freezing",
                "color": "#4D4D4D",
                "linestyle": "--",
            },
            {
                "curve_id": "pruning",
                "label": "Pruning",
                "color": "#C44E52",
                "linestyle": "-",
            },
        ],
    },
    {
        "panel_label": "d",
        "panel_id": "neuron_swapping",
        "title": "(d) Neuron swapping",
        "curves": [
            {
                "curve_id": "without_swapping",
                "label": "Without swapping",
                "color": "#4D4D4D",
                "linestyle": "--",
            },
            {
                "curve_id": "with_swapping",
                "label": "With swapping",
                "color": "#C44E52",
                "linestyle": "-",
            },
        ],
    },
    {
        "panel_label": "e",
        "panel_id": "knowledge_accumulation",
        "title": "(e) Knowledge accumulation",
        "curves": [
            {
                "curve_id": "no_accumulation",
                "label": "No accumulation",
                "color": "#4D4D4D",
                "linestyle": "--",
            },
            {
                "curve_id": "accumulate_every_rollout",
                "label": "Every-rollout accumulation",
                "color": "#4C78A8",
                "linestyle": ":",
            },
            {
                "curve_id": "selective_accumulation",
                "label": "Selective accumulation",
                "color": "#C44E52",
                "linestyle": "-",
            },
        ],
    },
]

DEFAULT_METRIC_KEYS = [
    "eval/success_end",
    "eval_success_end",
    "success_at_end",
    "success_end",
    "eval/success_once",
    "eval_success_once",
    "success_once",
]

BLUE = (45.0 / 255.0, 164.0 / 255.0, 205.0 / 255.0)
GREEN = (1.0 / 255.0, 113.0 / 255.0, 0.0 / 255.0)
YELLOW = (205.0 / 255.0, 194.0 / 255.0, 45.0 / 255.0)
PURPLE = (204.0 / 255.0, 46.0 / 255.0, 206.0 / 255.0)
GREY = (146.0 / 255.0, 146.0 / 255.0, 146.0 / 255.0)
BLACK = (60.0 / 255.0, 60.0 / 255.0, 60.0 / 255.0)
RED = (181.0 / 255.0, 23.0 / 255.0, 0.0 / 255.0)


def set_figure_settings(fig_wh_ratio=6.4 / 4.8, std_h=4.8, font_size=24, font_family='Arial'):
    fig = plt.figure(figsize=(std_h * fig_wh_ratio, std_h))
    if font_family is not None:
        plt.rc('font', family=font_family)
    plt.rcParams['font.size'] = str(font_size)
    return fig


GROUP_NAME_MAP = {
    'scaling_law_function': 'scaling law',
    'neuron_grained_scaling_up': 'scaling up',
    'scaling_down_freezing_vs_pruning': 'scaling down',
    'neuron_swapping': 'neuron_swapping',
    'knowledge_accumulation': 'accumulation',
}

VIS_GROUP_SPECS = [
    {
        "group_name": "accumulation",
        "slots": [
            ("no", "knowledge_accumulation", "no_accumulation"),
            ("each", "knowledge_accumulation", "accumulate_every_rollout"),
            ("ours", "knowledge_accumulation", "selective_accumulation"),
        ],
    },
    {
        "group_name": "neuron_swapping",
        "slots": [
            ("random", "neuron_swapping", "without_swapping"),
            ("ours", "neuron_swapping", "with_swapping"),
        ],
    },
    {
        "group_name": "scaling down",
        "slots": [
            ("pruning", "scaling_down_freezing_vs_pruning", "freezing"),
            ("ours", "scaling_down_freezing_vs_pruning", "pruning"),
        ],
    },
    {
        "group_name": "scaling up",
        "slots": [
            ("random", "neuron_grained_scaling_up", "random"),
            ("inverse", "neuron_grained_scaling_up", "inverse"),
            ("ours", "neuron_grained_scaling_up", "neuron_grained"),
        ],
    },
    {
        "group_name": "scaling law",
        "slots": [
            ("traditional", "scaling_law_function", "without_scaling_law"),
            ("ours", "scaling_law_function", "with_scaling_law"),
        ],
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str, base_dir: Path = EVAL_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def default_manifest() -> dict[str, Any]:
    panels = []
    for panel in PANEL_SPECS:
        panels.append(
            {
                "panel_label": panel["panel_label"],
                "panel_id": panel["panel_id"],
                "title": panel["title"],
                "workload_name": "Single-arm robot",
                "curves": [
                    {
                        **curve,
                        "run_dir": "",
                        "metric_source": "auto",
                        "metric_key": "eval/success_end",
                    }
                    for curve in panel["curves"]
                ],
            }
        )
    return {
        "suite_stamp": "no-data",
        "table_root": "ablation/ablation_table",
        "figure_output": "ablation/FIG_ABLATION.pdf",
        "summary_csv": "ablation/ablation_summary.csv",
        "panels": panels,
    }


def find_latest_manifest() -> Path | None:
    if LATEST_POINTER.exists():
        stamp = LATEST_POINTER.read_text(encoding="utf-8").strip()
        if stamp:
            candidate = TABLE_ROOT / stamp / "manifest.json"
            if candidate.exists():
                return candidate
    manifest_paths = sorted(TABLE_ROOT.glob("*/manifest.json"))
    if manifest_paths:
        return manifest_paths[-1]
    return None


def load_manifest(manifest_path: str | None) -> tuple[dict[str, Any], Path | None]:
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            payload = load_json(path)
            if isinstance(payload, dict):
                return payload, path
        return default_manifest(), path
    latest = find_latest_manifest()
    if latest is None:
        return default_manifest(), None
    payload = load_json(latest)
    return (payload if isinstance(payload, dict) else default_manifest()), latest


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except Exception:
        return []
    if isinstance(payload, dict):
        history = payload.get("history", [])
    else:
        history = payload
    return [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []


def metric_candidates(curve: dict[str, Any]) -> list[str]:
    raw = curve.get("metric_key")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    if isinstance(raw, str) and raw:
        return [raw]
    return list(DEFAULT_METRIC_KEYS)


def extract_series_from_rows(rows: list[dict[str, Any]], curve: dict[str, Any]) -> list[tuple[float, float]]:
    x_keys = ["elapsed_minutes", "elapsed_min", "time_minutes", "minutes"]
    y_keys = metric_candidates(curve)
    series = []
    for index, row in enumerate(rows):
        y_value = None
        for key in y_keys:
            y_value = finite_float(row.get(key))
            if y_value is not None:
                break
        if y_value is None:
            continue
        x_value = None
        for key in x_keys:
            x_value = finite_float(row.get(key))
            if x_value is not None:
                break
        if x_value is None:
            elapsed_hours = finite_float(row.get("elapsed_hours"))
            x_value = elapsed_hours * 60.0 if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return series


def find_tb_dir(run_dir: Path) -> Path | None:
    direct_tb_dir = run_dir / "tb"
    if direct_tb_dir.is_dir():
        return direct_tb_dir
    nested_tb_dirs = sorted(path for path in run_dir.glob("**/tb") if path.is_dir())
    return nested_tb_dirs[0] if nested_tb_dirs else None


def extract_tensorboard_series(run_dir: Path, curve: dict[str, Any]) -> list[tuple[float, float]]:
    if event_accumulator is None:
        return []
    tb_dir = find_tb_dir(run_dir)
    if tb_dir is None:
        return []
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
    except Exception:
        return []
    tags = accumulator.Tags().get("scalars", [])
    for metric_key in metric_candidates(curve):
        if metric_key not in tags:
            continue
        events = accumulator.Scalars(metric_key)
        if not events:
            continue
        base_time = events[0].wall_time
        return [((event.wall_time - base_time) / 60.0, float(event.value)) for event in events]
    return []


def collect_series(curve: dict[str, Any]) -> list[tuple[float, float]]:
    run_dir_raw = curve.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        return []
    run_dir = resolve_path(run_dir_raw)
    if not run_dir.exists():
        return []

    metric_source = str(curve.get("metric_source", "auto"))
    if metric_source in {"tensorboard", "auto"}:
        series = extract_tensorboard_series(run_dir, curve)
        if series:
            return series
    if metric_source in {"jsonl", "auto"}:
        for candidate in [
            run_dir / "motivation_eval_metrics.jsonl",
            run_dir / "ablation_eval_metrics.jsonl",
        ]:
            series = extract_series_from_rows(load_jsonl_rows(candidate), curve)
            if series:
                return series
    if metric_source in {"history", "auto"}:
        for candidate in [
            run_dir / "metrics_history.json",
            run_dir / "history.json",
        ]:
            series = extract_series_from_rows(load_history_rows(candidate), curve)
            if series:
                return series
    return []


def normalize_panels(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_panels = manifest.get("panels", [])
    panel_by_id = {
        str(panel.get("panel_id")): panel
        for panel in manifest_panels
        if isinstance(panel, dict) and panel.get("panel_id")
    }
    normalized = []
    for panel_spec in PANEL_SPECS:
        panel = dict(panel_by_id.get(panel_spec["panel_id"], {}))
        panel["panel_label"] = panel_spec["panel_label"]
        panel["panel_id"] = panel_spec["panel_id"]
        panel["title"] = panel_spec["title"]
        panel.setdefault("workload_name", "Single-arm robot")
        curve_by_id = {
            str(curve.get("curve_id")): curve
            for curve in panel.get("curves", [])
            if isinstance(curve, dict) and curve.get("curve_id")
        }
        curves = []
        for curve_spec in panel_spec["curves"]:
            existing_curve = dict(curve_by_id.get(curve_spec["curve_id"], {}))
            merged_curve = dict(curve_spec)
            for key, value in existing_curve.items():
                if key not in {"curve_id", "label", "color", "linestyle"}:
                    merged_curve[key] = value
            merged_curve.setdefault("run_dir", "")
            merged_curve.setdefault("metric_source", "auto")
            merged_curve.setdefault("metric_key", "eval/success_end")
            merged_curve.setdefault("changed_options", [])
            curves.append(merged_curve)
        panel["curves"] = curves
        normalized.append(panel)
    return normalized


def write_summary(rows: list[dict[str, Any]], manifest: dict[str, Any], manifest_path: Path | None) -> None:
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel_label",
                "panel_id",
                "curve_id",
                "curve_label",
                "run_dir",
                "points",
                "last_value",
                "max_value",
                "changed_options",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            {
                "suite_stamp": manifest.get("suite_stamp", "no-data"),
                "manifest_path": str(manifest_path) if manifest_path is not None else "",
                "figure_output": str(FIGURE_PATH.relative_to(EVAL_ROOT)),
                "summary_csv": str(SUMMARY_CSV_PATH.relative_to(EVAL_ROOT)),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_ablation_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in normalize_panels(manifest):
        for curve in panel["curves"]:
            series = collect_series(curve)
            last_value = series[-1][1] if series else 0.0
            max_value = max((item[1] for item in series), default=0.0)
            rows.append(
                {
                    "panel_label": panel["panel_label"],
                    "panel_id": panel["panel_id"],
                    "curve_id": curve["curve_id"],
                    "curve_label": curve.get("label", curve["curve_id"]),
                    "run_dir": curve.get("run_dir", ""),
                    "points": len(series),
                    "last_value": f"{last_value:.6f}",
                    "max_value": f"{max_value:.6f}",
                    "changed_options": curve.get("changed_options", []),
                }
            )
    return rows


def build_vis_data(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, bool]]]:
    row_map = {(row['panel_id'], row['curve_id']): row for row in rows}
    ordered: dict[str, dict[str, tuple[float, bool]]] = {}
    for group_spec in VIS_GROUP_SPECS:
        group_data: dict[str, tuple[float, bool]] = {}
        for slot_name, panel_id, curve_id in group_spec["slots"]:
            row = row_map.get((panel_id, curve_id), {})
            value = float(row.get('last_value', 0.0) or 0.0)
            is_placeholder = int(row.get('points', 0) or 0) == 0
            group_data[slot_name] = (value, is_placeholder)
        ordered[group_spec["group_name"]] = group_data
    return ordered


def _placeholder_bar_value(item_index: int, is_ours: bool) -> float:
    if is_ours:
        return 0.10
    return 0.03 + 0.02 * item_index


def plot_panels(manifest: dict[str, Any], manifest_path: Path | None) -> None:
    rows = build_ablation_rows(manifest)
    data = build_vis_data(manifest, rows)

    original_w = 0.6
    w = original_w * 2 / 3
    set_figure_settings(
        font_family='Arial',
        fig_wh_ratio=w,
        std_h=6.4 / original_w,
        font_size=24,
    )

    group_gap = 1.5
    cur_offset = 0
    all_x, all_y = [], []
    placeholder_text_needed = False

    for _, group_data in data.items():
        x_positions = np.arange(len(group_data)) + cur_offset
        values = []
        for item_index, (key, (value, is_placeholder)) in enumerate(group_data.items()):
            if key == 'ours':
                values.append(0.0)
                continue
            if is_placeholder or value == 0.0:
                values.append(_placeholder_bar_value(item_index, False))
                placeholder_text_needed = True
            else:
                values.append(value)
        values = values[::-1]
        all_x += list(x_positions)
        all_y += list(values)
        cur_offset += len(group_data) + group_gap

    plt.barh(all_x, all_y, color='white', edgecolor=BLACK, lw=2, zorder=10)

    cur_offset = 0
    all_x, all_y = [], []
    raw_x = []
    for _, group_data in data.items():
        x_positions = np.arange(len(group_data)) + cur_offset
        values = []
        for item_index, (key, (value, is_placeholder)) in enumerate(group_data.items()):
            if key != 'ours':
                values.append(0.0)
                continue
            if is_placeholder or value == 0.0:
                values.append(_placeholder_bar_value(item_index, True))
                placeholder_text_needed = True
            else:
                values.append(value)
        values = values[::-1]
        raw_x += list(x_positions)
        all_x += list(x_positions)
        all_y += list(values)
        cur_offset += len(group_data) + group_gap

    plt.barh(all_x, all_y, color='white', edgecolor=RED, lw=2, zorder=10, hatch='/')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.grid(axis='x')
    plt.xlim(right=1.0)
    plt.yticks(raw_x, [''] * len(raw_x))
    plt.savefig(FIGURE_PNG_PATH, dpi=300)
    plt.savefig(FIGURE_SVG_PATH, dpi=300)
    plt.savefig(FIGURE_PATH, dpi=300)
    try:
        fill_ablation_template(FIGURE_PATH, FIGURE_PNG_PATH)
    except Exception as exc:
        print(f'[template] failed to fill ablation template: {exc}')
    plt.clf()

    write_summary(rows, manifest, manifest_path)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args(argv)

    manifest, manifest_path = load_manifest(args.manifest)
    plot_panels(manifest, manifest_path)


if __name__ == "__main__":
    main()
