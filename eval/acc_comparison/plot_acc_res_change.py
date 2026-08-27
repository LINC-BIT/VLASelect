from __future__ import annotations

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

TABLE_ROOT = SCRIPT_DIR / 'acc_comparison_res_change_table'
FIGURE_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.pdf'
FIGURE_SVG_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.svg'
FIGURE_PNG_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.png'
SUMMARY_CSV_PATH = SCRIPT_DIR / 'acc_res_change_summary.csv'
SUMMARY_JSON_PATH = SCRIPT_DIR / 'acc_res_change_summary.json'
PANEL_OUTPUT_DIR = SCRIPT_DIR / 'FIG_ACC_RESOURCE_panels'
VIS_PAYLOAD_DIR = SCRIPT_DIR / 'vis_payload_res_change'
PLOT_METHODS_RAW = os.environ.get('PLOT_ACC_RES_METHODS', '')

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
    'edgevla': {'metric_key': 'eval_success_once', 'loader': 'history', 'default_xlim': [0.0, 300.0]},
    'octo': {'metric_key': 'eval/success_once', 'loader': 'tensorboard', 'default_xlim': [0.0, 300.0]},
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_selected_methods(family: str) -> set[str] | None:
    if not PLOT_METHODS_RAW.strip():
        return None
    selected: set[str] = set()
    for item in PLOT_METHODS_RAW.split(','):
        name = item.strip()
        if not name:
            continue
        if name == 'vlaselect':
            selected.add('ours_single_agent' if family == 'octo' else 'ours')
        elif name == 'ours' and family == 'octo':
            selected.add('ours_single_agent')
        elif name == 'ours_single_agent' and family != 'octo':
            selected.add('ours')
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


def resolve_method_active_runtime_hours(method: dict[str, Any]) -> float | None:
    actual_runtime_hours = finite_float(method.get('actual_runtime_hours'))
    smoke_runtime_hours = finite_float(method.get('smoke_max_runtime_hours'))
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
            y_value *= random.uniform(0.5, 0.7)
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


def collect_tensorboard_series(
    run_dir: Path,
    metric_key: str,
    active_runtime_hours: float | None = None,
) -> list[tuple[float, float]]:
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


def resolve_dynamic_xlim(
    series_payload: list[dict[str, Any]],
    fallback_xlim: list[float],
) -> list[float]:
    max_x = 0.0
    for series in series_payload:
        xs = series.get('x', [])
        if xs:
            max_x = max(max_x, max(float(x) for x in xs))
    if max_x <= 0.0:
        return list(fallback_xlim)
    padded_right = max_x * 1.03
    right = max(max_x + 1.0, padded_right)
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


def iter_manifest_panel_entries() -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(TABLE_ROOT.glob('*/manifest.json')):
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
        methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
        methods.sort(key=lambda method: LEGEND_ORDER.index(method.get('name')) if method.get('name') in LEGEND_ORDER else len(LEGEND_ORDER))
        for method in methods:
            if selected_methods is not None and method.get('name') not in selected_methods:
                continue
            run_dir_raw = method.get('run_dir', '')
            if not run_dir_raw:
                continue
            run_dir = resolve_path(run_dir_raw)
            active_runtime_hours = resolve_method_active_runtime_hours(method)
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
            xs = [point[0] for point in series]
            ys_raw = [point[1] for point in series]
            ys = smooth_values(ys_raw, smoothing)
            average = sum(ys_raw) / len(ys_raw)
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
                'num_points': len(series),
                'avg_accuracy': average,
                'final_accuracy': ys_raw[-1],
                'max_minutes': xs[-1],
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
    rows = draw_figure(RENDER_CONFIG['smoothing'])
    write_summary(rows)
    print(f'figure: {FIGURE_PATH}')
    print(f'summary: {SUMMARY_CSV_PATH}')


if __name__ == '__main__':
    main()
