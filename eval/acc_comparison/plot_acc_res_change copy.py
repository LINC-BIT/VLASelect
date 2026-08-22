from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common.figure_compose import compose_grid_figure
from common.template_pdf_fill import fill_resource_template
from common.vis_line_draw import apply_matplotlib_style, draw_plot

TABLE_ROOT = SCRIPT_DIR / 'acc_comparison_res_change_table'
FIGURE_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.pdf'
FIGURE_SVG_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.svg'
FIGURE_PNG_PATH = SCRIPT_DIR / 'FIG_ACC_RESOURCE.png'
SUMMARY_CSV_PATH = SCRIPT_DIR / 'acc_res_change_summary.csv'
SUMMARY_JSON_PATH = SCRIPT_DIR / 'acc_res_change_summary.json'
PANEL_OUTPUT_DIR = SCRIPT_DIR / 'FIG_ACC_RESOURCE_panels'
VIS_PAYLOAD_DIR = SCRIPT_DIR / 'vis_payload_res_change'

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
    'tinyvla': {'metric_key': 'train_success_once', 'loader': 'history', 'default_xlim': [0.0, 300.0]},
    'vla_adapter_new': {'metric_key': 'train_success_once', 'loader': 'history', 'default_xlim': [0.0, 300.0]},
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


def resolve_path(raw_path: str, base_dir: Path = EVAL_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def parse_sequence(raw_value: Any) -> list[Any]:
    if raw_value in (None, ''):
        return []
    if isinstance(raw_value, list):
        return raw_value
    try:
        parsed = ast.literal_eval(str(raw_value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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


def collect_history_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
    series = []
    for index, metric in enumerate(load_history(run_dir)):
        y_value = finite_float(metric.get(metric_key))
        if y_value is None:
            continue
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        x_value = elapsed_hours * 60.0 if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return series


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


def collect_tensorboard_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
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
    return [((event.wall_time - base_time) / 60.0, float(event.value)) for event in events]


def collect_series(family: str, run_dir: Path) -> list[tuple[float, float]]:
    config = FAMILY_CONFIGS[family]
    if config['loader'] == 'tensorboard':
        return collect_tensorboard_series(run_dir, config['metric_key'])
    return collect_history_series(run_dir, config['metric_key'])


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
        xs = series.get('x', [])
        if xs:
            max_x = max(max_x, max(float(x) for x in xs))
    if max_x <= 0.0:
        return list(fallback_xlim)
    padded_right = max_x * 1.03
    right = max(max_x + 1.0, padded_right)
    return [0.0, right]


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


def get_resource_events(panel: dict[str, Any]) -> list[dict[str, Any]]:
    change_points = parse_sequence(panel.get('resource_change_time_points'))
    directions = parse_sequence(panel.get('resource_change_directions'))
    factors = parse_sequence(panel.get('resource_change_factors'))
    events = []
    for index, raw_x in enumerate(change_points):
        x_value = finite_float(raw_x)
        if x_value is None:
            continue
        direction = str(directions[index]).strip().lower() if index < len(directions) else ''
        factor = finite_float(factors[index]) if index < len(factors) else None
        events.append({
            'x': x_value,
            'direction': direction,
            'factor': factor,
        })
    return events


def choose_focus_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    for event in reversed(events):
        if event['direction'].startswith('dec'):
            return event
    return events[-1]


def average_window(series: list[tuple[float, float]], left: float, right: float) -> float:
    values = [y for x, y in series if left <= x <= right]
    if not values:
        return math.nan
    return sum(values) / len(values)


def build_panel_series(panel: dict[str, Any], smoothing: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float | None], list[float]]:
    series_payload = []
    summary_rows: list[dict[str, Any]] = []
    overall_baselines: list[float] = []
    overall_ours: list[float] = []
    zoom_baselines: list[float] = []
    zoom_ours: list[float] = []
    raw_series_by_method: dict[str, list[tuple[float, float]]] = {}

    suite_manifest_raw = str(panel.get('suite_manifest', ''))
    suite_manifest_path = resolve_path(suite_manifest_raw) if suite_manifest_raw else None
    if suite_manifest_path is not None and suite_manifest_path.exists():
        suite_manifest = load_json(suite_manifest_path)
        methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
        methods.sort(key=lambda method: LEGEND_ORDER.index(method.get('name')) if method.get('name') in LEGEND_ORDER else len(LEGEND_ORDER))
        for method in methods:
            run_dir_raw = method.get('run_dir', '')
            if not run_dir_raw:
                continue
            run_dir = resolve_path(run_dir_raw)
            series = collect_series(panel['family'], run_dir)
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
                'style': style,
                'x': xs,
                'y': ys,
            })
            raw_series_by_method[method['name']] = series
            if method['name'] in {'ours', 'ours_single_agent'}:
                overall_ours.append(average)
            else:
                overall_baselines.append(average)
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

    events = get_resource_events(panel)
    focus_event = choose_focus_event(events)
    if focus_event is not None:
        left = max(0.0, focus_event['x'] - 45.0)
        right = focus_event['x'] + 45.0
        for method_name, series in raw_series_by_method.items():
            window_avg = average_window(series, left, right)
            if method_name in {'ours', 'ours_single_agent'}:
                if math.isfinite(window_avg):
                    zoom_ours.append(window_avg)
            else:
                if math.isfinite(window_avg):
                    zoom_baselines.append(window_avg)
    else:
        left = 0.0
        right = resolve_dynamic_xlim(series_payload, FAMILY_CONFIGS[panel['family']]['default_xlim'])[1]

    overall_baseline_mean = sum(overall_baselines) / len(overall_baselines) if overall_baselines else math.nan
    overall_ours_mean = sum(overall_ours) / len(overall_ours) if overall_ours else math.nan
    overall_improvement = (overall_ours_mean - overall_baseline_mean) * 100.0 if math.isfinite(overall_baseline_mean) and math.isfinite(overall_ours_mean) else math.nan
    zoom_baseline_mean = sum(zoom_baselines) / len(zoom_baselines) if zoom_baselines else math.nan
    zoom_ours_mean = sum(zoom_ours) / len(zoom_ours) if zoom_ours else math.nan
    zoom_improvement = (zoom_ours_mean - zoom_baseline_mean) * 100.0 if math.isfinite(zoom_baseline_mean) and math.isfinite(zoom_ours_mean) else math.nan

    summary_stats = {
        'overall_others_average': overall_baseline_mean if math.isfinite(overall_baseline_mean) else None,
        'overall_ours_average': overall_ours_mean if math.isfinite(overall_ours_mean) else None,
        'overall_improvement_percent': overall_improvement if math.isfinite(overall_improvement) else None,
        'zoom_others_average': zoom_baseline_mean if math.isfinite(zoom_baseline_mean) else None,
        'zoom_ours_average': zoom_ours_mean if math.isfinite(zoom_ours_mean) else None,
        'zoom_improvement_percent': zoom_improvement if math.isfinite(zoom_improvement) else None,
    }
    return series_payload, summary_rows, summary_stats, [left, right]


def _aggregate_group_stats(panel_stats_list: list[dict[str, float | None]], prefix: str) -> dict[str, float | None]:
    others_key = f'{prefix}_others_average'
    ours_key = f'{prefix}_ours_average'
    others_values = [stats[others_key] for stats in panel_stats_list if stats.get(others_key) is not None]
    ours_values = [stats[ours_key] for stats in panel_stats_list if stats.get(ours_key) is not None]
    others_mean = sum(others_values) / len(others_values) if others_values else None
    ours_mean = sum(ours_values) / len(ours_values) if ours_values else None
    if others_mean is None or ours_mean is None:
        improvement = None
    else:
        improvement = (ours_mean - others_mean) * 100.0
    return {
        'others_average': others_mean,
        'ours_average': ours_mean,
        'absolute_improvement_percent': improvement,
    }


def draw_figure(smoothing: float = 0.7) -> list[dict[str, Any]]:
    apply_matplotlib_style(RENDER_CONFIG['matplotlib'])
    VIS_PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    main_paths: list[Path] = []
    zoom_paths: list[Path] = []
    panel_stats_list: list[dict[str, float | None]] = []

    for panel_defaults in PAPER_PANELS:
        panel = resolve_panel_entry(panel_defaults)
        series_payload, rows, panel_stats, zoom_xlim = build_panel_series(panel, smoothing)
        summary_rows.extend(rows)
        panel_stats_list.append(panel_stats)
        config = FAMILY_CONFIGS[panel['family']]
        main_xlim = resolve_dynamic_xlim(series_payload, config['default_xlim'])
        plot_base = {
            'xlabel': 'Time (minutes)',
            'ylabel': 'Success Rate',
            'ylim': [0.0, 1.0],
            'grid_alpha': 0.3,
            'series': series_payload,
            'legend_entries': [],
        }
        payload = {
            'source': {
                'top_manifest': panel.get('_top_manifest', ''),
                'suite_manifest': panel.get('suite_manifest', ''),
            },
            'render_config': RENDER_CONFIG,
            'plots': {},
        }
        main_plot = dict(plot_base)
        main_plot.update({'output_stem': f"{panel['panel_label']}_{panel['family']}_main", 'xlim': main_xlim})
        payload['plots']['main'] = main_plot
        (VIS_PAYLOAD_DIR / f"{panel['panel_label']}_{panel['family']}.json").write_text(json.dumps(payload, indent=2), encoding='utf-8')
        main_png, _ = draw_plot(main_plot, RENDER_CONFIG, PANEL_OUTPUT_DIR)
        main_paths.append(main_png)

        zoom_plot = dict(plot_base)
        zoom_plot.update({'output_stem': f"{panel['panel_label']}_{panel['family']}_zoom", 'xlim': zoom_xlim})
        zoom_png, _ = draw_plot(zoom_plot, RENDER_CONFIG, PANEL_OUTPUT_DIR)
        zoom_paths.append(zoom_png)

    compose_grid_figure(
        main_paths + zoom_paths,
        output_paths=[FIGURE_PNG_PATH, FIGURE_SVG_PATH],
        rows=2,
        cols=4,
        figsize=(20.0, 10.0),
        legend_path=None,
        dpi=200,
    )
    summary_groups = [
        _aggregate_group_stats(panel_stats_list[:2], 'overall'),
        _aggregate_group_stats(panel_stats_list[2:], 'overall'),
        _aggregate_group_stats(panel_stats_list[:2], 'zoom'),
        _aggregate_group_stats(panel_stats_list[2:], 'zoom'),
    ]
    fill_resource_template(FIGURE_PATH, main_paths + zoom_paths, summary_groups)
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
