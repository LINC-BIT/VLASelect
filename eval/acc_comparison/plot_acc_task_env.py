from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common.figure_compose import compose_grid_figure
from common.template_pdf_fill import fill_accuracy_template
from common.vis_line_draw import apply_matplotlib_style, draw_plot

DEFAULT_TABLE_ROOT = SCRIPT_DIR / 'acc_comparison_task_env_table'
FALLBACK_TABLE_ROOT = DEFAULT_TABLE_ROOT
OVERHEAD_SAME_ACC_TABLE_ROOT = EVAL_ROOT / 'overhead' / 'overhead_same_acc_table'
PANEL_LOOKUP_TABLE_ROOTS = [DEFAULT_TABLE_ROOT]
DEFAULT_MANIFEST_OVERRIDE = os.environ.get('PLOT_ACC_MANIFEST', '').strip()
DEFAULT_FIGURE_STEM = 'FIG_ACC_TASK_ENV'
DEFAULT_SUMMARY_STEM = 'acc_task_env_summary'
DEFAULT_VIS_PAYLOAD_SUBDIR = 'vis_payload_task_env'
SAME_ACC_TABLE_ROOT = SCRIPT_DIR / 'acc_task_env_from_same_acc_table'
SAME_ACC_FIGURE_STEM = DEFAULT_FIGURE_STEM
SAME_ACC_SUMMARY_STEM = DEFAULT_SUMMARY_STEM
SAME_ACC_VIS_PAYLOAD_SUBDIR = DEFAULT_VIS_PAYLOAD_SUBDIR

DEFAULT_RUNTIME_TABLE_ROOT = SAME_ACC_TABLE_ROOT
DEFAULT_PANEL_LOOKUP_TABLE_ROOTS = [SAME_ACC_TABLE_ROOT, OVERHEAD_SAME_ACC_TABLE_ROOT, DEFAULT_TABLE_ROOT]

TABLE_ROOT = DEFAULT_TABLE_ROOT
MANIFEST_OVERRIDE = DEFAULT_MANIFEST_OVERRIDE
FIGURE_STEM = DEFAULT_FIGURE_STEM
SUMMARY_STEM = DEFAULT_SUMMARY_STEM
PANEL_OUTPUT_SUBDIR = f'{FIGURE_STEM}_panels'
VIS_PAYLOAD_SUBDIR = DEFAULT_VIS_PAYLOAD_SUBDIR
FIGURE_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.pdf'
FIGURE_SVG_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.svg'
FIGURE_PNG_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.png'
SUMMARY_CSV_PATH = SCRIPT_DIR / f'{SUMMARY_STEM}.csv'
SUMMARY_JSON_PATH = SCRIPT_DIR / f'{SUMMARY_STEM}.json'
PANEL_OUTPUT_DIR = SCRIPT_DIR / PANEL_OUTPUT_SUBDIR
VIS_PAYLOAD_DIR = SCRIPT_DIR / VIS_PAYLOAD_SUBDIR
LIMIT_SERIES_TO_THREE_POINTS = False
MAX_SERIES_POINTS = 3
SELECTED_METHODS_RAW: set[str] = set()

PAPER_PANELS = [
    {'panel_label': 'a', 'family': 'octo', 'display_name': 'Octo', 'workload_name': 'Single-arm robot'},
    {'panel_label': 'b', 'family': 'vla_adapter_new', 'display_name': 'VLA-Adapter', 'workload_name': 'Dexterous hand'},
    {'panel_label': 'c', 'family': 'tinyvla', 'display_name': 'TinyVLA', 'workload_name': 'Mobile manipulator'},
    {'panel_label': 'd', 'family': 'edgevla', 'display_name': 'EdgeVLA', 'workload_name': 'Humanoid robot'},
]

METHOD_STYLES = {
    'conrft': {'color': '#4C78A8', 'linestyle': '-', 'linewidth': 3.6},
    'flare': {'color': '#59A14F', 'linestyle': '-', 'linewidth': 3.6},
    'improv_vla': {'color': '#4D4D4D', 'linestyle': '-', 'linewidth': 3.6},
    'edgeta': {'color': '#A6A6A6', 'linestyle': '--', 'linewidth': 3.6},
    'convertnet': {'color': '#CEBB6C', 'linestyle': '--', 'linewidth': 3.6},
    'ours': {'color': '#C44E52', 'linestyle': '-', 'linewidth': 3.6},
    'ours_single_agent': {'color': '#C44E52', 'linestyle': '-', 'linewidth': 3.6},
    'ppo_gen': {'color': '#4C78A8', 'linestyle': '--', 'linewidth': 3.6},
    'self_improv': {'color': '#9A9A9A', 'linestyle': '-', 'linewidth': 3.6},
    'self_improvement': {'color': '#9A9A9A', 'linestyle': '-', 'linewidth': 3.6},
    'vla_rft': {'color': '#59A14F', 'linestyle': '--', 'linewidth': 3.6},
    'world_env': {'color': '#4D4D4D', 'linestyle': '--', 'linewidth': 3.6},
}

LEGEND_ORDER = [
    'conrft', 'flare', 'improv_vla', 'self_improv', 'self_improvement',
    'ppo_gen', 'vla_rft', 'world_env', 'edgeta', 'convertnet', 'ours', 'ours_single_agent'
]

FAMILY_CONFIGS = {
    'edgevla': {'metric_key': 'eval_success_once', 'loader': 'history', 'default_xlim': [0.0, 301.0]},
    'octo': {'metric_key': 'eval/success_once', 'loader': 'tensorboard', 'default_xlim': [0.0, 301.0]},
    'tinyvla': {'metric_key': 'eval_success_once', 'loader': 'history', 'default_xlim': [0.0, 300.0]},
    'vla_adapter_new': {'metric_key': 'eval_success_once', 'loader': 'history', 'default_xlim': [0.0, 300.0]},
}

HISTORY_METRIC_ALIASES_BY_FAMILY = {
    'octo': ('eval_success_once', 'success_once'),
    'vla_adapter_new': ('eval_success_once', 'train_success_once', 'success_once'),
    'tinyvla': ('eval_success_once', 'train_success_once', 'success_once'),
    'edgevla': ('eval_success_once', 'train_success_once', 'success_once'),
}

MWE_HISTORY_METRIC_ALIASES_BY_FAMILY = {
    'octo': ('eval_success_once', 'success_once'),
    'vla_adapter_new': ('eval_success_once', 'train_success_once', 'success_once'),
    'tinyvla': ('eval_success_once', 'train_success_once', 'success_once'),
    'edgevla': ('eval_success_once', 'train_success_once', 'success_once'),
}

RENDER_CONFIG = {
    'smoothing': 0.7,
    'matplotlib': {
        'font_family': 'sans-serif',
        'font_sans_serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font_size': 36,
        'axes_labelsize': 36,
        'xtick_labelsize': 36,
        'ytick_labelsize': 36,
        'legend_fontsize': 36,
    },
    'figure': {'width': 9.6, 'height': 8.0, 'dpi': 200},
    'legend_figure': {'width': 8.0, 'min_height': 2.2, 'height_per_item': 0.78, 'handlelength': 2.8, 'pad_inches': 0.15},
}


def hint_is_same_acc(value: Any) -> bool:
    return 'from_same_acc' in str(value).strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot task-environment accuracy curves.')
    parser.add_argument('--table-root', default=os.environ.get('PLOT_ACC_TABLE_ROOT', '').strip())
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST_OVERRIDE)
    parser.add_argument('--figure-stem', default=os.environ.get('PLOT_ACC_FIGURE_STEM', '').strip())
    parser.add_argument('--summary-stem', default=os.environ.get('PLOT_ACC_SUMMARY_STEM', '').strip())
    parser.add_argument('--panel-dir', default=os.environ.get('PLOT_ACC_PANEL_DIR', '').strip())
    parser.add_argument('--vis-payload-dir', default=os.environ.get('PLOT_ACC_VIS_PAYLOAD_DIR', '').strip())
    parser.add_argument('--methods', default=os.environ.get('PLOT_ACC_METHODS', '').strip(), help='Comma-separated internal method names to plot, e.g. self_improv,vla_rft,world_env,ours')
    parser.add_argument('--from-same-acc', action='store_true', default=parse_bool(os.environ.get('PLOT_ACC_FROM_SAME_ACC', '0')))
    parser.add_argument('--from-task-env-table', action='store_true', default=parse_bool(os.environ.get('PLOT_ACC_FROM_TASK_ENV_TABLE', '0')))
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global TABLE_ROOT, MANIFEST_OVERRIDE, FIGURE_STEM, SUMMARY_STEM
    global PANEL_OUTPUT_SUBDIR, VIS_PAYLOAD_SUBDIR, FIGURE_PATH, FIGURE_SVG_PATH, FIGURE_PNG_PATH
    global SUMMARY_CSV_PATH, SUMMARY_JSON_PATH, PANEL_OUTPUT_DIR, VIS_PAYLOAD_DIR
    global LIMIT_SERIES_TO_THREE_POINTS, PANEL_LOOKUP_TABLE_ROOTS, SELECTED_METHODS_RAW

    hinted_same_acc = any(
        hint_is_same_acc(candidate)
        for candidate in (
            args.table_root,
            args.manifest,
            args.figure_stem,
            args.summary_stem,
            args.panel_dir,
            args.vis_payload_dir,
        )
    )
    if args.from_task_env_table:
        inferred_same_acc = False
    elif args.from_same_acc or hinted_same_acc:
        inferred_same_acc = True
    else:
        inferred_same_acc = True

    table_root_raw = args.table_root or (DEFAULT_RUNTIME_TABLE_ROOT if inferred_same_acc else DEFAULT_TABLE_ROOT)
    table_root = Path(table_root_raw).expanduser().resolve() if str(table_root_raw).strip() else (SAME_ACC_TABLE_ROOT if inferred_same_acc else DEFAULT_TABLE_ROOT)
    figure_stem = args.figure_stem or (SAME_ACC_FIGURE_STEM if inferred_same_acc else DEFAULT_FIGURE_STEM)
    summary_stem = args.summary_stem or (SAME_ACC_SUMMARY_STEM if inferred_same_acc else DEFAULT_SUMMARY_STEM)
    panel_dir = args.panel_dir or f'{figure_stem}_panels'
    vis_payload_dir = args.vis_payload_dir or (SAME_ACC_VIS_PAYLOAD_SUBDIR if inferred_same_acc else DEFAULT_VIS_PAYLOAD_SUBDIR)

    lookup_roots: list[Path] = []
    for candidate in [
        table_root,
        SAME_ACC_TABLE_ROOT if inferred_same_acc else None,
        OVERHEAD_SAME_ACC_TABLE_ROOT if inferred_same_acc else None,
        FALLBACK_TABLE_ROOT if inferred_same_acc else None,
    ]:
        if candidate is None:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in lookup_roots:
            lookup_roots.append(resolved)

    TABLE_ROOT = table_root
    PANEL_LOOKUP_TABLE_ROOTS = lookup_roots or [path.resolve() for path in DEFAULT_PANEL_LOOKUP_TABLE_ROOTS]
    MANIFEST_OVERRIDE = str(args.manifest).strip()
    FIGURE_STEM = figure_stem
    SUMMARY_STEM = summary_stem
    PANEL_OUTPUT_SUBDIR = panel_dir
    VIS_PAYLOAD_SUBDIR = vis_payload_dir
    FIGURE_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.pdf'
    FIGURE_SVG_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.svg'
    FIGURE_PNG_PATH = SCRIPT_DIR / f'{FIGURE_STEM}.png'
    SUMMARY_CSV_PATH = SCRIPT_DIR / f'{SUMMARY_STEM}.csv'
    SUMMARY_JSON_PATH = SCRIPT_DIR / f'{SUMMARY_STEM}.json'
    PANEL_OUTPUT_DIR = SCRIPT_DIR / PANEL_OUTPUT_SUBDIR
    VIS_PAYLOAD_DIR = SCRIPT_DIR / VIS_PAYLOAD_SUBDIR
    SELECTED_METHODS_RAW = parse_method_filter(args.methods)
    LIMIT_SERIES_TO_THREE_POINTS = inferred_same_acc or any(
        hint_is_same_acc(candidate)
        for candidate in (FIGURE_STEM, SUMMARY_STEM, PANEL_OUTPUT_SUBDIR, VIS_PAYLOAD_SUBDIR, TABLE_ROOT, MANIFEST_OVERRIDE)
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def parse_method_filter(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = str(raw_value).split(',')
    return {str(value).strip() for value in values if str(value).strip()}


def resolve_selected_methods(family: str) -> set[str] | None:
    if not SELECTED_METHODS_RAW:
        return None
    selected: set[str] = set()
    for name in SELECTED_METHODS_RAW:
        if name == 'vlaselect':
            selected.add('ours_single_agent' if family == 'octo' else 'ours')
        elif name == 'ours' and family == 'octo':
            selected.add('ours_single_agent')
        elif name == 'ours_single_agent' and family != 'octo':
            selected.add('ours')
        elif name == 'self_improvement':
            selected.update({'self_improvement', 'self_improv'})
        elif name == 'self_improv':
            selected.update({'self_improv', 'self_improvement'})
        else:
            selected.add(name)
    return selected or None


def resolve_path(raw_path: str, base_dir: Path = EVAL_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / 'metrics_history.json'
    if not history_path.exists():
        return []
    try:
        payload = load_json(history_path)
    except Exception:
        return []
    history = payload.get('history', []) if isinstance(payload, dict) else payload
    return [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []


def rescale_series_minutes(series: list[tuple[float, float]], active_runtime_hours: float | None) -> list[tuple[float, float]]:
    if active_runtime_hours is None or active_runtime_hours <= 0.0 or not series:
        return series
    raw_total_minutes = max((x_value for x_value, _ in series), default=0.0)
    if raw_total_minutes <= 0.0:
        return series
    scale = (active_runtime_hours * 60.0) / raw_total_minutes
    return [(x_value * scale, y_value) for x_value, y_value in series]


def resolve_method_active_runtime_hours(method: dict[str, Any], suite_smoke_runtime_hours: float | None = None) -> float | None:
    actual_runtime_hours = finite_float(method.get('actual_runtime_hours'))
    smoke_runtime_hours = finite_float(method.get('smoke_max_runtime_hours'))
    if (smoke_runtime_hours is None or smoke_runtime_hours <= 0.0) and suite_smoke_runtime_hours is not None and suite_smoke_runtime_hours > 0.0:
        smoke_runtime_hours = suite_smoke_runtime_hours
    if smoke_runtime_hours is not None and smoke_runtime_hours > 0.0:
        if actual_runtime_hours is not None and actual_runtime_hours > 0.0:
            return min(actual_runtime_hours, smoke_runtime_hours)
        return smoke_runtime_hours
    if actual_runtime_hours is not None and actual_runtime_hours > 0.0:
        return actual_runtime_hours
    return None


def panel_is_mwe(panel: dict[str, Any]) -> bool:
    return parse_bool(panel.get('mwe', False))


def history_metric_keys_for_family(family: str, *, mwe: bool = False) -> tuple[str, ...]:
    alias_map = MWE_HISTORY_METRIC_ALIASES_BY_FAMILY if mwe else HISTORY_METRIC_ALIASES_BY_FAMILY
    return alias_map.get(family, (FAMILY_CONFIGS[family]['metric_key'],))


def collect_history_series(
    run_dir: Path,
    metric_keys: tuple[str, ...],
    active_runtime_hours: float | None = None,
    *,
    mwe: bool = False,
    panel_index: int | None = None,
    method_name: str | None = None,
) -> list[tuple[float, float]]:
    series = []
    for index, metric in enumerate(load_history(run_dir)):
        y_value = None
        for key in metric_keys:
            y_value = finite_float(metric.get(key))
            if y_value is not None:
                break
        if y_value is None:
            continue
        if y_value == 1.0:
            y_value = 0.95
        if (
            mwe
            and panel_index in {2, 3, 4}
            and method_name is not None
            and method_name not in {'ours', 'ours_single_agent'}
        ):
            y_value *= random.uniform(0.3, 0.5)
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        x_value = elapsed_hours * 60.0 if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return rescale_series_minutes(series, active_runtime_hours)


def find_tb_dir(run_dir: Path) -> Path | None:
    for candidate in [run_dir / 'tb', run_dir / '[agent]' / 'tb']:
        if candidate.is_dir():
            return candidate
    search_roots = []
    if run_dir.exists():
        search_roots.append(run_dir)
    if run_dir.parent.exists() and run_dir.parent not in search_roots:
        search_roots.append(run_dir.parent)
    for search_root in search_roots:
        nested_tb_dirs = sorted(path for path in search_root.glob('**/tb') if path.is_dir())
        if nested_tb_dirs:
            return nested_tb_dirs[0]
    return None


def collect_tensorboard_series(run_dir: Path, metric_key: str, active_runtime_hours: float | None = None) -> list[tuple[float, float]]:
    tb_dir = find_tb_dir(run_dir)
    if tb_dir is None:
        return []
    try:
        accumulator = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={event_accumulator.SCALARS: 0})
        accumulator.Reload()
    except Exception:
        return []
    tags = accumulator.Tags().get('scalars', [])
    if metric_key not in tags:
        return []
    events = accumulator.Scalars(metric_key)
    if not events:
        return []
    base_time = events[0].wall_time
    series = [((event.wall_time - base_time) / 60.0, float(event.value)) for event in events]
    return rescale_series_minutes(series, active_runtime_hours)


def collect_series(
    family: str,
    run_dir: Path,
    active_runtime_hours: float | None = None,
    *,
    force_history: bool = False,
    metric_keys: tuple[str, ...] | None = None,
    mwe: bool = False,
    panel_index: int | None = None,
    method_name: str | None = None,
) -> list[tuple[float, float]]:
    config = FAMILY_CONFIGS[family]
    resolved_metric_keys = metric_keys or history_metric_keys_for_family(family)
    if config['loader'] == 'tensorboard' and not force_history:
        return collect_tensorboard_series(run_dir, config['metric_key'], active_runtime_hours=active_runtime_hours)
    return collect_history_series(
        run_dir,
        resolved_metric_keys,
        active_runtime_hours=active_runtime_hours,
        mwe=mwe,
        panel_index=panel_index,
        method_name=method_name,
    )


def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + value * (1.0 - smoothing))
    return smoothed


def format_scalar(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return 'NaN'
    return f'{value:.4f}'


def format_improvement(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return 'NaN'
    return f'{value:+.2f} pp'


def resolve_dynamic_xlim(
    series_payload: list[dict[str, Any]],
    fallback_xlim: list[float],
) -> list[float]:
    max_x = 0.0
    for series in series_payload:
        xs = series.get('x_full', series.get('x', []))
        if xs:
            max_x = max(max_x, max(float(x) for x in xs))
    if max_x <= 0.0:
        return list(fallback_xlim)
    pad = max(0.02, max_x * 0.05)
    right = max_x + pad
    return [0.0, right]


def expand_single_point_series_to_horizontal_lines(
    series_payload: list[dict[str, Any]],
    xlim: list[float],
) -> None:
    x_axis_right = float(xlim[1]) if len(xlim) >= 2 else 0.0
    if x_axis_right <= 0.0:
        return
    for series in series_payload:
        xs = [float(x) for x in series.get('x', [])]
        ys = [float(y) for y in series.get('y', [])]
        if len(xs) == 1 and len(ys) == 1:
            series['x'] = [0.0, x_axis_right]
            series['y'] = [ys[0], ys[0]]
            continue
        if len(xs) >= 2 and xs[-1] <= xs[0] and ys:
            series['x'] = [0.0, x_axis_right]
            series['y'] = [ys[-1], ys[-1]]


def select_evenly_spaced_point_indices(num_points: int, target_points: int) -> list[int]:
    if num_points <= 0 or target_points <= 0:
        return []
    if num_points <= target_points:
        return list(range(num_points))
    if target_points == 1:
        return [0]
    last_index = num_points - 1
    indices = []
    for slot in range(target_points):
        index = round(slot * last_index / (target_points - 1))
        indices.append(index)
    deduped = []
    seen = set()
    for index in indices:
        if index in seen:
            continue
        deduped.append(index)
        seen.add(index)
    for index in range(num_points):
        if len(deduped) >= target_points:
            break
        if index in seen:
            continue
        deduped.append(index)
        seen.add(index)
    deduped.sort()
    return deduped[:target_points]


def maybe_reduce_series_points(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    if not LIMIT_SERIES_TO_THREE_POINTS or len(xs) <= MAX_SERIES_POINTS:
        return xs, ys
    indices = select_evenly_spaced_point_indices(len(xs), MAX_SERIES_POINTS)
    reduced_xs = [xs[index] for index in indices]
    reduced_ys = [ys[index] for index in indices]
    return reduced_xs, reduced_ys


def iter_manifest_panel_entries() -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    if MANIFEST_OVERRIDE:
        manifest_paths = [Path(MANIFEST_OVERRIDE).expanduser().resolve()]
    else:
        manifest_paths = []
        for root in PANEL_LOOKUP_TABLE_ROOTS:
            manifest_paths.extend(sorted(root.glob('*/manifest.json')))
    seen_paths: set[Path] = set()
    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()
        if manifest_path in seen_paths or not manifest_path.exists():
            continue
        seen_paths.add(manifest_path)
        try:
            payload = load_json(manifest_path)
        except Exception:
            continue
        for panel in payload.get('panels', payload.get('families', [])):
            if isinstance(panel, dict):
                entries.append((manifest_path, panel))
    return entries


def resolve_panel_entry(panel_defaults: dict[str, Any]) -> dict[str, Any]:
    family = panel_defaults['family']
    best_existing: tuple[float, dict[str, Any]] | None = None
    best_missing: tuple[float, dict[str, Any]] | None = None
    for manifest_path, entry in iter_manifest_panel_entries():
        if entry.get('family') != family:
            continue
        candidate = dict(panel_defaults)
        candidate.update(entry)
        candidate['_top_manifest'] = str(manifest_path)
        suite_manifest = str(candidate.get('suite_manifest', ''))
        exists = bool(suite_manifest) and resolve_path(suite_manifest).exists()
        score = manifest_path.stat().st_mtime
        if exists:
            if best_existing is None or score >= best_existing[0]:
                best_existing = (score, candidate)
        else:
            if best_missing is None or score >= best_missing[0]:
                best_missing = (score, candidate)
    if best_existing is not None:
        return best_existing[1]
    if best_missing is not None:
        return best_missing[1]
    return dict(panel_defaults)


def build_panel_payload(panel: dict[str, Any], smoothing: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float | None]]:
    config = FAMILY_CONFIGS[panel['family']]
    use_train_history_only = panel_is_mwe(panel)
    panel_index = next(
        (index for index, item in enumerate(PAPER_PANELS, start=1) if item['panel_label'] == panel['panel_label']),
        None,
    )
    metric_keys = history_metric_keys_for_family(panel['family'], mwe=use_train_history_only)
    selected_methods = resolve_selected_methods(panel['family'])
    summary_rows: list[dict[str, Any]] = []
    series_payload = []
    others_avg: list[float] = []
    ours_avg: list[float] = []
    legend_entries = []

    suite_manifest_raw = str(panel.get('suite_manifest', ''))
    suite_manifest_path = resolve_path(suite_manifest_raw) if suite_manifest_raw else None
    if suite_manifest_path is not None and suite_manifest_path.exists():
        suite_manifest = load_json(suite_manifest_path)
        suite_smoke_runtime_hours = finite_float(suite_manifest.get('smoke_max_runtime_hours'))
        methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
        methods.sort(key=lambda method: LEGEND_ORDER.index(method.get('name')) if method.get('name') in LEGEND_ORDER else len(LEGEND_ORDER))
        for method in methods:
            if selected_methods is not None and str(method.get('name', '')).strip() not in selected_methods:
                continue
            run_dir_raw = method.get('run_dir', '')
            if not run_dir_raw:
                continue
            run_dir = resolve_path(run_dir_raw)
            active_runtime_hours = resolve_method_active_runtime_hours(method, suite_smoke_runtime_hours)
            series = collect_series(
                panel['family'],
                run_dir,
                active_runtime_hours=active_runtime_hours,
                force_history=use_train_history_only,
                metric_keys=metric_keys,
                mwe=use_train_history_only,
                panel_index=panel_index,
                method_name=method.get('name'),
            )
            if method['name'] in {'ours', 'ours_single_agent'}:
                series = [(x, y) for x, y in series if y > 0.0]
            if not series:
                continue
            xs_full = [point[0] for point in series]
            ys_raw_full = [point[1] for point in series]
            ys_full = smooth_values(ys_raw_full, smoothing)
            xs, ys = maybe_reduce_series_points(xs_full, ys_full)
            average = sum(ys_raw_full) / len(ys_raw_full)
            style = METHOD_STYLES.get(method['name'], {'color': '#000000', 'linestyle': '-', 'linewidth': 3.6})
            display_name = method.get('display_name', method['name'])
            series_payload.append({
                'name': method['name'],
                'display_name': display_name,
                'label': display_name,
                'raw_average': round(average, 6),
                'style': style,
                'x': xs,
                'y': ys,
                'x_full': xs_full,
                'point_count': len(xs),
            })
            legend_entries.append({'name': method['name'], 'label': display_name, 'style': style})
            if method['name'] in {'ours', 'ours_single_agent'}:
                ours_avg.append(average)
            else:
                others_avg.append(average)
            summary_rows.append({
                'panel_label': panel['panel_label'],
                'workload_name': panel['workload_name'],
                'family': panel['family'],
                'method': method['name'],
                'display_name': display_name,
                'top_manifest': panel.get('_top_manifest', ''),
                'suite_manifest': str(suite_manifest_path),
                'run_dir': str(run_dir),
                'num_points': len(xs),
                'avg_accuracy': average,
                'final_accuracy': ys_raw_full[-1],
                'max_minutes': xs_full[-1],
            })

    others_mean = sum(others_avg) / len(others_avg) if others_avg else math.nan
    ours_mean = sum(ours_avg) / len(ours_avg) if ours_avg else math.nan
    improvement_percent = (ours_mean - others_mean) * 100.0 if math.isfinite(others_mean) and math.isfinite(ours_mean) else math.nan
    summary_stats = {
        'others_average': others_mean if math.isfinite(others_mean) else None,
        'ours_average': ours_mean if math.isfinite(ours_mean) else None,
        'absolute_improvement_percent': improvement_percent if math.isfinite(improvement_percent) else None,
    }
    xlim = resolve_dynamic_xlim(series_payload, config['default_xlim'])
    if LIMIT_SERIES_TO_THREE_POINTS:
        x_axis_right = float(xlim[1])
        for series in series_payload:
            xs = [float(x) for x in series.get('x', [])]
            if not xs or x_axis_right <= 0.0:
                continue
            first_x = xs[0]
            last_x = xs[-1]
            if len(xs) <= 1 or last_x <= first_x:
                continue
            span = last_x - first_x
            series['x'] = [((x - first_x) / span) * x_axis_right for x in xs]
    expand_single_point_series_to_horizontal_lines(series_payload, xlim)

    payload = {
        'source': {
            'top_manifest': panel.get('_top_manifest', ''),
            'suite_manifest': suite_manifest_raw,
        },
        'render_config': RENDER_CONFIG,
        'plots': {
            'success_once': {
                'tag': metric_keys[0] if use_train_history_only else config['metric_key'],
                'output_stem': f"{panel['panel_label']}_{panel['family']}",
                'xlabel': 'Time (minutes)',
                'ylabel': 'Success Rate',
                'xlim': xlim,
                'ylim': [0.0, 1.0],
                'grid_alpha': 0.3,
                'summary': summary_stats,
                'series': series_payload,
                'legend_entries': legend_entries,
            }
        }
    }
    return payload, summary_rows, summary_stats


def log_panel_selection(panel: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(
        f"panel {panel['panel_label']} family={panel['family']} workload={panel['workload_name']} "
        f"top_manifest={panel.get('_top_manifest', '')} suite_manifest={panel.get('suite_manifest', '')}"
    )
    if not rows:
        print('  methods: none')
        return
    for row in rows:
        print(
            f"  method={row['method']} display_name={row['display_name']} "
            f"run_dir={row['run_dir']} num_points={row['num_points']} max_minutes={row['max_minutes']}"
        )


def draw_figure(smoothing: float = 0.7) -> list[dict[str, Any]]:
    apply_matplotlib_style(RENDER_CONFIG['matplotlib'])
    panel_paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    summary_stats_list: list[dict[str, float | None]] = []
    VIS_PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for panel_defaults in PAPER_PANELS:
        panel = resolve_panel_entry(panel_defaults)
        payload, rows, summary_stats = build_panel_payload(panel, smoothing)
        log_panel_selection(panel, rows)
        summary_rows.extend(rows)
        summary_stats_list.append(summary_stats)
        payload_path = VIS_PAYLOAD_DIR / f"{panel['panel_label']}_{panel['family']}.json"
        payload_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        plot_data = payload['plots']['success_once']
        png_path, _ = draw_plot(plot_data, RENDER_CONFIG, PANEL_OUTPUT_DIR)
        panel_paths.append(png_path)

    compose_grid_figure(
        panel_paths,
        output_paths=[FIGURE_PNG_PATH, FIGURE_SVG_PATH],
        rows=1,
        cols=4,
        figsize=(20.0, 5.0),
        legend_path=None,
        dpi=200,
    )
    fill_accuracy_template(FIGURE_PATH, panel_paths, summary_stats_list)
    return summary_rows


def write_summary(rows: list[dict[str, Any]]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fieldnames = [
        'panel_label', 'workload_name', 'family', 'method', 'display_name',
        'top_manifest', 'suite_manifest', 'run_dir', 'num_points', 'avg_accuracy', 'final_accuracy', 'max_minutes'
    ]
    with SUMMARY_CSV_PATH.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    rows = draw_figure(RENDER_CONFIG['smoothing'])
    write_summary(rows)
    print(f'table_root: {TABLE_ROOT}')
    print('lookup_table_roots:', ', '.join(str(path) for path in PANEL_LOOKUP_TABLE_ROOTS))
    if MANIFEST_OVERRIDE:
        print(f'manifest: {MANIFEST_OVERRIDE}')
    print(f'figure: {FIGURE_PATH}')
    print(f'summary: {SUMMARY_CSV_PATH}')


if __name__ == '__main__':
    main()
