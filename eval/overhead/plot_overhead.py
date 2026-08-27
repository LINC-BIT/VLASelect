from __future__ import annotations
import ast
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from common.figure_compose import compose_grid_figure, render_legend_image
from common.template_pdf_fill import fill_memory_template
from plot_breakdown_impl import load_summary_aligned_manifest
TABLE_ROOT = SCRIPT_DIR / 'overhead_same_acc_table'
BREAKDOWN_ROOT = SCRIPT_DIR / 'overhead_breakdown_table'
LATEST_POINTER = TABLE_ROOT / 'latest.txt'
FIGURE_PATH = SCRIPT_DIR / 'FIG_MEMORY_FOOTPOINT.pdf'
FIGURE_SVG_PATH = SCRIPT_DIR / 'FIG_MEMORY_FOOTPOINT.svg'
FIGURE_PNG_PATH = SCRIPT_DIR / 'FIG_MEMORY_FOOTPOINT.png'
TABLE2_CSV_PATH = BREAKDOWN_ROOT / 'TAB_OVERHEAD.csv'
TABLE3_CSV_PATH = BREAKDOWN_ROOT / 'TAB_ENERGY.csv'
SUMMARY_JSON_PATH = SCRIPT_DIR / 'overhead_same_acc_summary.json'
PANEL_OUTPUT_DIR = SCRIPT_DIR / 'FIG_MEMORY_FOOTPRINT_panels'
PANEL_FIGURE_SIZE = (5.6, 3.6)
PLOT_FONT_SIZE = 36
FIGURE_SIZE = (12.9583, 6.6111)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'font.size': PLOT_FONT_SIZE, 'axes.labelsize': PLOT_FONT_SIZE, 'axes.titlesize': PLOT_FONT_SIZE, 'xtick.labelsize': PLOT_FONT_SIZE, 'ytick.labelsize': PLOT_FONT_SIZE, 'legend.fontsize': PLOT_FONT_SIZE, 'svg.fonttype': 'none',})


def configure_output_paths(output_root: Path | None) -> None:
    if output_root is None:
        return
    global BREAKDOWN_ROOT, FIGURE_PATH, FIGURE_SVG_PATH, FIGURE_PNG_PATH
    global TABLE2_CSV_PATH, TABLE3_CSV_PATH, SUMMARY_JSON_PATH, PANEL_OUTPUT_DIR
    output_root = output_root.resolve()
    BREAKDOWN_ROOT = output_root / 'overhead_breakdown_table'
    FIGURE_PATH = output_root / 'FIG_MEMORY_FOOTPOINT.pdf'
    FIGURE_SVG_PATH = output_root / 'FIG_MEMORY_FOOTPOINT.svg'
    FIGURE_PNG_PATH = output_root / 'FIG_MEMORY_FOOTPOINT.png'
    TABLE2_CSV_PATH = BREAKDOWN_ROOT / 'TAB_OVERHEAD.csv'
    TABLE3_CSV_PATH = BREAKDOWN_ROOT / 'TAB_ENERGY.csv'
    SUMMARY_JSON_PATH = output_root / 'overhead_same_acc_summary.json'
    PANEL_OUTPUT_DIR = output_root / 'FIG_MEMORY_FOOTPRINT_panels'
PAPER_PANELS = [
    {'panel_label': 'a', 'family': 'octo', 'display_name': 'Octo', 'workload_name': 'Single-arm robot', 'panel_title': '(a) Single-arm robot'},
    {'panel_label': 'b', 'family': 'vla_adapter_new', 'display_name': 'VLA-Adapter', 'workload_name': 'Dexterous hand', 'panel_title': '(b) Dexterous hand'},
    {'panel_label': 'c', 'family': 'tinyvla', 'display_name': 'TinyVLA', 'workload_name': 'Mobile manipulator', 'panel_title': '(c) Mobile manipulator'},
    {'panel_label': 'd', 'family': 'edgevla', 'display_name': 'EdgeVLA', 'workload_name': 'Humanoid robot', 'panel_title': '(d) Humanoid robot'},
]
METHOD_STYLES = {'conrft': {'color': '#4C78A8', 'linestyle': '-'},'flare': {'color': '#59A14F', 'linestyle': '-'},'improv_vla': {'color': '#4D4D4D', 'linestyle': '-'},'edgeta': {'color': '#A6A6A6', 'linestyle': '--'},'convertnet': {'color': '#CEBB6C', 'linestyle': '--'},'ours': {'color': '#C44E52', 'linestyle': '-'},'ours_single_agent': {'color': '#C44E52', 'linestyle': '-'},'ppo_gen': {'color': '#4C78A8', 'linestyle': '--'},'self_improv': {'color': '#9A9A9A', 'linestyle': '-'},'self_improvement': {'color': '#9A9A9A', 'linestyle': '-'},'vla_rft': {'color': '#59A14F', 'linestyle': '--'},'world_env': {'color': '#4D4D4D', 'linestyle': '--'}}
LEGEND_ORDER = ['conrft', 'flare', 'improv_vla', 'self_improv', 'self_improvement', 'ppo_gen', 'vla_rft', 'world_env', 'edgeta', 'convertnet', 'ours', 'ours_single_agent']
FAMILY_CONFIGS = {'edgevla': {'metric_key': 'eval_success_once', 'loader': 'history'},'octo': {'metric_key': 'eval/success_once', 'loader': 'tensorboard'},'tinyvla': {'metric_key': 'eval_success_once', 'loader': 'history'},'vla_adapter_new': {'metric_key': 'eval_success_once', 'loader': 'history'}}
PAPER_METHOD_ORDER = ['ConRFT', 'FlaRe', 'iRe-VLA', 'Self-Improvement', 'RLVLA', 'VLA-RFT', 'World-Env', 'EdgeTA', 'ConvertNet', 'VLASelect']
PAPER_METHOD_BY_INTERNAL = {'conrft': 'ConRFT','flare': 'FlaRe','improv_vla': 'iRe-VLA','self_improv': 'Self-Improvement','self_improvement': 'Self-Improvement','ppo_gen': 'RLVLA','vla_rft': 'VLA-RFT','world_env': 'World-Env','edgeta': 'EdgeTA','convertnet': 'ConvertNet','ours': 'VLASelect','ours_single_agent': 'VLASelect'}
VLASELECT_METHODS_BY_FAMILY = {'octo': ['ours_single_agent', 'ours'],'vla_adapter_new': ['ours'],'tinyvla': ['ours'],'edgevla': ['ours', 'ours_single_agent']}
HISTORY_METRIC_ALIASES_BY_FAMILY = {
    'octo': ('eval_success_once', 'eval/success_once', 'success_once'),
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
# Table 3 follows the paper format: the first row of each method averages the
# workload segments treated as new tasks, and the second row averages the
# segments treated as environment changes.
TABLE3_EVENT_LABELS_BY_FAMILY = {
    'octo': ['task', 'env', 'env', 'env', 'task', 'env', 'env', 'env', 'task', 'env'],
    'vla_adapter_new': ['task', 'task', 'task', 'task', 'env', 'task', 'env', 'task', 'env', 'task'],
    'tinyvla': ['task', 'env', 'env', 'env', 'env', 'env', 'env', 'env', 'env', 'env'],
    'edgevla': ['task', 'env', 'env', 'env', 'task', 'task', 'env', 'task', 'task', 'env'],
}
TABLE3_EVENT_DISPLAY_NAMES = {'task': 'new task', 'env': 'environment change'}
MEMORY_FOOTPRINT_ACTIVE_PHASE_NAMES = {'online_rl_rollout', 'online_rl_training'}
def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding='utf-8'))
def resolve_path(raw_path: str, base_dir: Path = EVAL_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()
def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    raw_text = str(value).strip().lower()
    return raw_text in {'1', 'true', 'yes', 'y', 'on'}
def finite_float(value: Any) -> float | None:
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if math.isfinite(result) else None
def parse_sequence(raw_value: Any) -> list[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    raw_text = str(raw_value).strip()
    if raw_text == '':
        return []
    try:
        parsed = ast.literal_eval(raw_text)
    except (SyntaxError, ValueError):
        parsed = [item.strip() for item in raw_text.split(',') if item.strip()]
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
def find_latest_manifest() -> Path | None:
    if LATEST_POINTER.exists():
        stamp = LATEST_POINTER.read_text(encoding='utf-8').strip()
        if stamp:
            candidate = TABLE_ROOT / stamp / 'manifest.json'
            if candidate.exists(): return candidate
    manifest_paths = sorted(TABLE_ROOT.glob('*/manifest.json'))
    return manifest_paths[-1] if manifest_paths else None


def load_default_manifest() -> dict[str, Any]:
    manifest = load_summary_aligned_manifest([TABLE_ROOT], TABLE_ROOT)
    manifest.setdefault('figure_output', 'overhead/FIG_MEMORY_FOOTPOINT.pdf')
    manifest.setdefault('table2_output', 'overhead/overhead_breakdown_table/TAB_OVERHEAD.csv')
    manifest.setdefault('table3_output', 'overhead/overhead_breakdown_table/TAB_ENERGY.csv')
    return manifest
def default_manifest() -> dict[str, Any]:
    return {
        'suite_stamp': 'no-data',
        'table_root': 'overhead/overhead_same_acc_table',
        'figure_output': 'overhead/FIG_MEMORY_FOOTPOINT.pdf',
        'table2_output': 'overhead/overhead_breakdown_table/TAB_OVERHEAD.csv',
        'table3_output': 'overhead/overhead_breakdown_table/TAB_ENERGY.csv',
        'panels': [dict(panel) for panel in PAPER_PANELS],
        'families': [dict(panel) for panel in PAPER_PANELS],
    }
def load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / 'metrics_history.json'
    if not history_path.exists(): return []
    try: payload = load_json(history_path)
    except Exception: return []
    history = payload.get('history', []) if isinstance(payload, dict) else payload
    return [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
def rescale_series_hours(series: list[tuple[float, float]], active_runtime_hours: float | None) -> list[tuple[float, float]]:
    if active_runtime_hours is None or active_runtime_hours <= 0.0 or not series:
        return series
    raw_total_hours = max((x_value for x_value, _ in series), default=0.0)
    if raw_total_hours <= 0.0:
        return series
    scale = active_runtime_hours / raw_total_hours
    return [(x_value * scale, y_value) for x_value, y_value in series]

def history_metric_aliases_for_family(family: str, use_train_history_only: bool = False) -> tuple[str, ...]:
    alias_map = MWE_HISTORY_METRIC_ALIASES_BY_FAMILY if use_train_history_only else HISTORY_METRIC_ALIASES_BY_FAMILY
    return alias_map.get(family, (FAMILY_CONFIGS[family]['metric_key'],))

def collect_history_series(run_dir: Path, metric_keys: tuple[str, ...], active_runtime_hours: float | None = None) -> list[tuple[float, float]]:
    series = []
    for index, metric in enumerate(load_history(run_dir)):
        y_value = None
        for key in metric_keys:
            y_value = finite_float(metric.get(key))
            if y_value is not None:
                break
        if y_value is None: continue
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        x_value = elapsed_hours if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return rescale_series_hours(series, active_runtime_hours)
def find_tb_dir(run_dir: Path) -> Path | None:
    for candidate in [run_dir / 'tb', run_dir / '[agent]' / 'tb']:
        if candidate.is_dir(): return candidate
    for search_root in [run_dir, run_dir.parent]:
        if search_root.exists():
            nested = sorted(path for path in search_root.glob('**/tb') if path.is_dir())
            if nested: return nested[0]
    return None
def collect_tensorboard_series(run_dir: Path, metric_key: str, active_runtime_hours: float | None = None) -> list[tuple[float, float]]:
    tb_dir = find_tb_dir(run_dir)
    if tb_dir is None: return []
    try:
        accumulator = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={event_accumulator.SCALARS: 0})
        accumulator.Reload()
    except Exception: return []
    tags = accumulator.Tags().get('scalars', [])
    if metric_key not in tags: return []
    events = accumulator.Scalars(metric_key)
    if not events: return []
    base_time = events[0].wall_time
    series = [((event.wall_time - base_time) / 3600.0, float(event.value)) for event in events]
    return rescale_series_hours(series, active_runtime_hours)
def collect_series(family: str, run_dir: Path, active_runtime_hours: float | None = None, use_train_history_only: bool = False) -> list[tuple[float, float]]:
    config = FAMILY_CONFIGS[family]
    metric_keys = history_metric_aliases_for_family(family, use_train_history_only=use_train_history_only)
    if config['loader'] == 'tensorboard' and not use_train_history_only:
        return collect_tensorboard_series(run_dir, config['metric_key'], active_runtime_hours=active_runtime_hours)
    return collect_history_series(run_dir, metric_keys, active_runtime_hours=active_runtime_hours)

def extract_history_success_value(family: str, metric: dict[str, Any], use_train_history_only: bool = False) -> float | None:
    for key in history_metric_aliases_for_family(family, use_train_history_only=use_train_history_only): 
        value = finite_float(metric.get(key))
        if value is not None:
            return value
    return None

def collect_segment_success_history(family: str, run_dir: Path, active_runtime_hours: float | None = None, use_train_history_only: bool = False) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for metric in load_history(run_dir):
        env_index = finite_float(metric.get('current_env_index'))
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        success_value = extract_history_success_value(family, metric, use_train_history_only=use_train_history_only)
        if env_index is None or elapsed_hours is None or success_value is None:
            continue
        grouped.setdefault(int(env_index), []).append((elapsed_hours, success_value))
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        grouped[key] = rescale_series_hours(values, active_runtime_hours)
    return grouped
def smooth_values(values: list[float], smoothing: float) -> list[float]:
    if not values or smoothing <= 0.0: return values
    smoothed = [values[0]]
    for value in values[1:]: smoothed.append(smoothed[-1] * smoothing + value * (1.0 - smoothing))
    return smoothed
def order_legend_entries(entries):
    order_index = {name: index for index, name in enumerate(LEGEND_ORDER)}
    return sorted(entries, key=lambda entry: (order_index.get(entry[0], len(order_index)), entry[1]))
def resolve_panel_entries(top_manifest):
    manifest_entries = top_manifest.get('panels', top_manifest.get('families', []))
    entry_by_family = {entry.get('family'): entry for entry in manifest_entries if isinstance(entry, dict)}
    panels = []
    for panel in PAPER_PANELS:
        merged = dict(panel)
        merged.update(entry_by_family.get(panel['family'], {}))
        panels.append(merged)
    return panels
def find_gpu_metrics_csv(run_dir: Path) -> Path | None:
    direct = run_dir / 'analysis' / 'gpu_metrics.csv'
    if direct.exists(): return direct
    for search_root in [run_dir, run_dir.parent]:
        if search_root.exists():
            candidates = sorted(search_root.glob('**/analysis/gpu_metrics.csv'))
            if candidates: return candidates[0]
    return None

def find_memory_accounting_json(run_dir: Path) -> Path | None:
    direct = run_dir / 'analysis' / 'memory_accounting.json'
    if direct.exists():
        return direct
    for search_root in [run_dir, run_dir.parent]:
        if search_root.exists():
            candidates = sorted(search_root.glob('**/analysis/memory_accounting.json'))
            if candidates:
                return candidates[0]
    return None

def load_memory_accounting_payload(run_dir: Path) -> dict[str, Any]:
    path = find_memory_accounting_json(run_dir)
    if path is None:
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_phase_events(run_dir: Path) -> list[tuple[float, str]]:
    trace_path = find_memory_phase_trace_jsonl(run_dir)
    if trace_path is None:
        return []
    events: list[tuple[float, str]] = []
    try:
        with trace_path.open('r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                phase = str(payload.get('phase', '')).strip()
                if not phase:
                    continue
                event_time = finite_float(payload.get('unix_time_seconds'))
                if event_time is None:
                    event_time = parse_timestamp_to_unix_seconds(payload.get('timestamp_utc'))
                if event_time is None:
                    continue
                events.append((event_time, phase))
    except Exception:
        return []
    events.sort(key=lambda item: item[0])
    return events


def sample_phase_at_time(sample_unix_seconds: float | None, phase_events: list[tuple[float, str]]) -> str | None:
    if sample_unix_seconds is None or not phase_events:
        return None
    if sample_unix_seconds < phase_events[0][0]:
        return phase_events[0][1]
    current_phase = None
    for event_time, phase in phase_events:
        if sample_unix_seconds < event_time:
            break
        current_phase = phase
    return current_phase

def find_memory_phase_trace_jsonl(run_dir: Path) -> Path | None:
    direct = run_dir / 'analysis' / 'memory_phase_trace.jsonl'
    if direct.exists():
        return direct
    for search_root in [run_dir, run_dir.parent]:
        if search_root.exists():
            candidates = sorted(search_root.glob('**/analysis/memory_phase_trace.jsonl'))
            if candidates:
                return candidates[0]
    return None

def parse_timestamp_to_unix_seconds(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if raw_text == '':
        return None
    try:
        return datetime.fromisoformat(raw_text).timestamp()
    except ValueError:
        return None

def raw_gpu_memory_used_mb_for_row(row: dict[str, Any], *, has_device_memory_column: bool, has_any_process_found_row: bool) -> float:
    device_gpu_memory_used_mb = finite_float(row.get('gpu_device_memory_used_mb'))
    legacy_gpu_memory_used_mb = finite_float(row.get('gpu_memory_used_mb')) or 0.0
    process_memory_used_mb = finite_float(row.get('process_memory_used_mb'))
    process_found_on_gpu = parse_bool(row.get('process_found_on_gpu'))
    if process_memory_used_mb is not None and process_found_on_gpu:
        return process_memory_used_mb
    if has_device_memory_column:
        return device_gpu_memory_used_mb if device_gpu_memory_used_mb is not None else legacy_gpu_memory_used_mb
    if has_any_process_found_row:
        return 0.0
    return legacy_gpu_memory_used_mb

def load_excluded_runtime_peak_mb(run_dir: Path) -> float:
    csv_path = find_gpu_metrics_csv(run_dir)
    if csv_path is None:
        return 0.0
    excluded_phase_intervals = load_excluded_phase_intervals(run_dir)
    if not excluded_phase_intervals:
        return 0.0
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            raw_rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception:
        return 0.0
    has_device_memory_column = 'gpu_device_memory_used_mb' in fieldnames
    has_any_process_found_row = any(parse_bool(raw_row.get('process_found_on_gpu')) for raw_row in raw_rows)
    peak_mb = 0.0
    for row in raw_rows:
        sample_unix_seconds = parse_timestamp_to_unix_seconds(row.get('timestamp_utc'))
        if not sample_in_excluded_phase(sample_unix_seconds, excluded_phase_intervals):
            continue
        raw_gpu_memory_used_mb = raw_gpu_memory_used_mb_for_row(
            row,
            has_device_memory_column=has_device_memory_column,
            has_any_process_found_row=has_any_process_found_row,
        )
        if raw_gpu_memory_used_mb > peak_mb:
            peak_mb = raw_gpu_memory_used_mb
    return peak_mb

def load_static_memory_exclusion_mb(run_dir: Path) -> float:
    payload = load_memory_accounting_payload(run_dir)
    return max(0.0, finite_float(payload.get('excluded_gpu_memory_mb')) or 0.0)


def load_memory_exclusion_mb(run_dir: Path) -> float:
    static_exclusion_mb = load_static_memory_exclusion_mb(run_dir)
    runtime_exclusion_peak_mb = load_excluded_runtime_peak_mb(run_dir)
    return max(static_exclusion_mb, runtime_exclusion_peak_mb)


def memory_footprint_offset_mb_for_method(paper_name: str, run_dir: Path) -> float:
    if paper_name == 'VLASelect':
        return load_static_memory_exclusion_mb(run_dir)
    return 0.0


def load_excluded_runtime_phase_names(run_dir: Path) -> set[str]:
    payload = load_memory_accounting_payload(run_dir)
    raw_names = payload.get('excluded_runtime_phase_names')
    if not isinstance(raw_names, list):
        return set()
    return {str(name).strip() for name in raw_names if str(name).strip()}

def load_excluded_phase_intervals(run_dir: Path) -> list[tuple[float, float | None]]:
    excluded_phase_names = load_excluded_runtime_phase_names(run_dir)
    if not excluded_phase_names:
        return []
    events = load_phase_events(run_dir)
    if not events:
        return []
    intervals: list[tuple[float, float | None]] = []
    for index, (start_time, phase) in enumerate(events):
        if phase not in excluded_phase_names:
            continue
        end_time = events[index + 1][0] if index + 1 < len(events) else None
        intervals.append((start_time, end_time))
    return intervals

def sample_in_excluded_phase(sample_unix_seconds: float | None, excluded_phase_intervals: list[tuple[float, float | None]]) -> bool:
    if sample_unix_seconds is None or not excluded_phase_intervals:
        return False
    for start_time, end_time in excluded_phase_intervals:
        if sample_unix_seconds < start_time:
            return False
        if end_time is None:
            if sample_unix_seconds >= start_time:
                return True
            continue
        if start_time <= sample_unix_seconds < end_time:
            return True
    return False


def build_phase_intervals(phase_events: list[tuple[float, str]]) -> list[tuple[float, float | None, str]]:
    intervals: list[tuple[float, float | None, str]] = []
    for index, (start_time, phase) in enumerate(phase_events):
        end_time = phase_events[index + 1][0] if index + 1 < len(phase_events) else None
        intervals.append((start_time, end_time, phase))
    return intervals


def build_active_phase_intervals(phase_events: list[tuple[float, str]]) -> list[tuple[float, float | None]]:
    return [
        (start_time, end_time)
        for start_time, end_time, phase in build_phase_intervals(phase_events)
        if phase in MEMORY_FOOTPRINT_ACTIVE_PHASE_NAMES
    ]


def active_elapsed_seconds_at_time(sample_unix_seconds: float | None, active_phase_intervals: list[tuple[float, float | None]]) -> float | None:
    if sample_unix_seconds is None or not active_phase_intervals:
        return None
    elapsed_seconds = 0.0
    for start_time, end_time in active_phase_intervals:
        if sample_unix_seconds < start_time:
            return elapsed_seconds
        if end_time is None or sample_unix_seconds < end_time:
            return elapsed_seconds + max(0.0, sample_unix_seconds - start_time)
        elapsed_seconds += max(0.0, end_time - start_time)
    return elapsed_seconds


def load_gpu_samples(run_dir: Path, active_runtime_hours: float | None = None, memory_footprint_offset_mb: float = 0.0):
    csv_path = find_gpu_metrics_csv(run_dir)
    if csv_path is None: return []
    excluded_memory_mb = load_memory_exclusion_mb(run_dir)
    excluded_phase_intervals = load_excluded_phase_intervals(run_dir)
    phase_events = load_phase_events(run_dir)
    active_phase_intervals = build_active_phase_intervals(phase_events)
    rows = []
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            raw_rows = list(reader)
            fieldnames = reader.fieldnames or []
        has_device_memory_column = 'gpu_device_memory_used_mb' in fieldnames
        has_any_process_found_row = any(parse_bool(raw_row.get('process_found_on_gpu')) for raw_row in raw_rows)
        for row in raw_rows:
            elapsed_seconds_wall = finite_float(row.get('elapsed_seconds'))
            if elapsed_seconds_wall is None: continue
            elapsed_hours_wall = elapsed_seconds_wall / 3600.0
            sample_unix_seconds = parse_timestamp_to_unix_seconds(row.get('timestamp_utc'))
            sample_phase = sample_phase_at_time(sample_unix_seconds, phase_events)
            include_in_memory_footprint = (not phase_events) or (sample_phase in MEMORY_FOOTPRINT_ACTIVE_PHASE_NAMES)
            exclude_from_memory_footprint = sample_in_excluded_phase(sample_unix_seconds, excluded_phase_intervals)
            raw_gpu_memory_used_mb = raw_gpu_memory_used_mb_for_row(
                row,
                has_device_memory_column=has_device_memory_column,
                has_any_process_found_row=has_any_process_found_row,
            )
            device_gpu_memory_used_mb = finite_float(row.get('gpu_device_memory_used_mb'))
            legacy_gpu_memory_used_mb = finite_float(row.get('gpu_memory_used_mb')) or 0.0
            process_memory_used_mb = finite_float(row.get('process_memory_used_mb'))
            process_found_on_gpu = parse_bool(row.get('process_found_on_gpu'))
            if exclude_from_memory_footprint:
                raw_gpu_memory_used_mb = 0.0
            adjusted_gpu_memory_used_mb = max(0.0, raw_gpu_memory_used_mb - excluded_memory_mb)
            active_elapsed_seconds = active_elapsed_seconds_at_time(sample_unix_seconds, active_phase_intervals)
            rows.append({
                'elapsed_seconds_wall': elapsed_seconds_wall,
                'elapsed_hours_wall': elapsed_hours_wall,
                'elapsed_seconds': elapsed_seconds_wall,
                'elapsed_hours': elapsed_hours_wall,
                'elapsed_seconds_active_raw': active_elapsed_seconds,
                'elapsed_hours_active_raw': (active_elapsed_seconds / 3600.0) if active_elapsed_seconds is not None else None,
                'sample_unix_seconds': sample_unix_seconds,
                'phase': sample_phase,
                'include_in_memory_footprint': include_in_memory_footprint and not exclude_from_memory_footprint,
                'raw_gpu_memory_used_mb': raw_gpu_memory_used_mb,
                'device_gpu_memory_used_mb': device_gpu_memory_used_mb if device_gpu_memory_used_mb is not None else legacy_gpu_memory_used_mb,
                'process_memory_used_mb': process_memory_used_mb,
                'process_found_on_gpu': process_found_on_gpu,
                'excluded_gpu_memory_mb': excluded_memory_mb,
                'exclude_from_memory_footprint': exclude_from_memory_footprint,
                'memory_footprint_offset_mb': memory_footprint_offset_mb,
                'gpu_memory_used_mb': max(0.0, raw_gpu_memory_used_mb - memory_footprint_offset_mb),
                'gpu_memory_used_mb_adjusted': adjusted_gpu_memory_used_mb,
                'gpu_power_w': finite_float(row.get('gpu_power_w')) or 0.0,
            })
    except Exception: return []
    rows = sorted(rows, key=lambda item: item['elapsed_seconds_wall'])
    if not rows:
        return rows
    if active_phase_intervals:
        for row in rows:
            active_elapsed_seconds = row.get('elapsed_seconds_active_raw')
            if active_elapsed_seconds is None:
                active_elapsed_seconds = 0.0
            row['elapsed_seconds'] = active_elapsed_seconds
            row['elapsed_hours'] = active_elapsed_seconds / 3600.0
        if active_runtime_hours is not None and active_runtime_hours > 0.0:
            observed_active_hours = max((row['elapsed_hours'] for row in rows), default=0.0)
            if observed_active_hours > 0.0:
                scale = active_runtime_hours / observed_active_hours
                for row in rows:
                    row['elapsed_hours'] = row['elapsed_hours'] * scale
                    row['elapsed_seconds'] = row['elapsed_hours'] * 3600.0
                    row['active_time_scale'] = scale
        return rows
    if active_runtime_hours is not None and active_runtime_hours > 0.0:
        wall_total_hours = rows[-1]['elapsed_hours_wall']
        if wall_total_hours > 0.0:
            scale = active_runtime_hours / wall_total_hours
            for row in rows:
                row['elapsed_hours'] = row['elapsed_hours_wall'] * scale
                row['elapsed_seconds'] = row['elapsed_hours'] * 3600.0
                row['active_time_scale'] = scale
    return rows
def first_reach_hours(series, target_accuracy):
    for x_value, y_value in series:
        if y_value >= target_accuracy: return x_value
    return None
def rescaled_samples_for_cutoff(
    samples: list[dict[str, Any]],
    cutoff_hours: float,
    include_filter: callable | None = None,
) -> list[dict[str, Any]]:
    if cutoff_hours <= 0.0:
        return []
    retained = []
    for sample in samples:
        if sample['elapsed_hours'] > cutoff_hours:
            continue
        if include_filter is not None and not include_filter(sample):
            continue
        retained.append(dict(sample))
    if not retained:
        return []
    observed_end_hours = max(sample['elapsed_hours'] for sample in retained)
    scale = cutoff_hours / observed_end_hours if observed_end_hours > 0.0 else 1.0
    for sample in retained:
        sample['elapsed_hours'] = sample['elapsed_hours'] * scale
        sample['elapsed_seconds'] = sample['elapsed_hours'] * 3600.0
    return retained


def integrate_energy_kj(samples, cutoff_hours):
    if not samples or cutoff_hours <= 0.0:
        return 0.0
    retained = rescaled_samples_for_cutoff(samples, cutoff_hours)
    if not retained:
        return 0.0
    if len(retained) == 1:
        return retained[0]['gpu_power_w'] * cutoff_hours * 3600.0 / 1000.0
    cutoff_seconds = cutoff_hours * 3600.0
    total_j = 0.0
    prev = retained[0]
    for curr in retained[1:]:
        start = min(prev['elapsed_seconds'], cutoff_seconds)
        end = min(curr['elapsed_seconds'], cutoff_seconds)
        if end > start:
            total_j += prev['gpu_power_w'] * (end - start)
        prev = curr
    if prev['elapsed_seconds'] < cutoff_seconds:
        total_j += prev['gpu_power_w'] * (cutoff_seconds - prev['elapsed_seconds'])
    return total_j / 1000.0

def mean_memory_gb(samples, cutoff_hours):
    if not samples or cutoff_hours <= 0.0:
        return 0.0
    points = prepare_memory_plot_points(samples, cutoff_hours)
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0][1]
    cutoff_seconds = cutoff_hours * 3600.0
    total_memory_gb_seconds = 0.0
    total_seconds = 0.0
    for index in range(len(points) - 1):
        start = min(points[index][0] * 3600.0, cutoff_seconds)
        end = min(points[index + 1][0] * 3600.0, cutoff_seconds)
        if end <= start:
            continue
        duration = end - start
        total_memory_gb_seconds += points[index][1] * duration
        total_seconds += duration
    if total_seconds <= 0.0:
        return points[-1][1]
    return total_memory_gb_seconds / total_seconds

def integrate_energy_interval_kj(samples, start_hours, end_hours):
    if not samples or end_hours <= start_hours:
        return 0.0
    retained = rescaled_samples_for_cutoff(samples, end_hours)
    if not retained:
        return 0.0
    start_seconds = start_hours * 3600.0
    end_seconds = end_hours * 3600.0
    total_j = 0.0
    prev_power = retained[0]['gpu_power_w']
    prev_time = start_seconds
    for sample in retained:
        sample_time = sample['elapsed_seconds']
        sample_power = sample['gpu_power_w']
        if sample_time <= start_seconds:
            prev_power = sample_power
            continue
        interval_end = min(sample_time, end_seconds)
        if interval_end > prev_time:
            total_j += prev_power * (interval_end - prev_time)
            prev_time = interval_end
        prev_power = sample_power
        if sample_time >= end_seconds:
            break
    if prev_time < end_seconds:
        total_j += prev_power * (end_seconds - prev_time)
    return total_j / 1000.0
def pick_vlaselect_method(methods, family):
    by_name = {method.get('name'): method for method in methods if isinstance(method, dict)}
    for name in VLASELECT_METHODS_BY_FAMILY.get(family, []):
        if name in by_name: return by_name[name]
    return None
def panel_is_mwe(panel: dict[str, Any]) -> bool:
    return parse_bool(panel.get('mwe', False))

def panel_mwe_limit_hours(panel: dict[str, Any]) -> float | None:
    raw_seconds = finite_float(panel.get('mwe_workload_runtime_limit_seconds'))
    if raw_seconds is None or raw_seconds <= 0.0:
        return None
    return raw_seconds / 3600.0

def latest_series_hours(series: list[tuple[float, float]]) -> float:
    return max((elapsed_hours for elapsed_hours, _ in series), default=0.0)

def latest_segment_history_hours(grouped_history: dict[int, list[tuple[float, float]]]) -> float:
    return max((latest_series_hours(series) for series in grouped_history.values()), default=0.0)

def latest_gpu_hours(samples: list[dict[str, Any]]) -> float:
    return max((sample['elapsed_hours'] for sample in samples), default=0.0)

def observed_cutoff_hours(panel: dict[str, Any], series_hours: float, gpu_hours: float) -> float | None:
    observed_hours = max(series_hours, gpu_hours)
    if observed_hours <= 0.0:
        return None
    if panel_is_mwe(panel):
        mwe_limit_hours = panel_mwe_limit_hours(panel)
        if mwe_limit_hours is not None:
            return min(observed_hours, mwe_limit_hours)
    return observed_hours

def make_empty_metrics():
    return {
        'time_h': 0.0,
        'memory_gb': 0.0,
        'energy_kj': 0.0,
        'reach_hours': 0.0,
        'target_accuracy': 0.0,
        'reached_target': False,
        'used_fallback_cutoff': False,
    }

def build_table3_segments(panel: dict[str, Any]) -> list[dict[str, Any]]:
    env_ids = [str(value) for value in parse_sequence(panel.get('envs_id'))]
    raw_time_points = parse_sequence(panel.get('env_change_time_points'))
    time_points = [finite_float(value) for value in raw_time_points]
    time_points = [value for value in time_points if value is not None]
    segment_count = min(len(env_ids), len(time_points))
    if segment_count == 0:
        return []
    labels = TABLE3_EVENT_LABELS_BY_FAMILY.get(panel['family'], [])
    if len(labels) < segment_count:
        labels = list(labels) + ['env'] * (segment_count - len(labels))
    segments = []
    start_minutes = 0.0
    for index in range(segment_count):
        end_minutes = float(time_points[index])
        if end_minutes <= start_minutes:
            continue
        segments.append({
            'index': index,
            'env_id': env_ids[index],
            'label': labels[index],
            'start_hours': start_minutes / 60.0,
            'end_hours': end_minutes / 60.0,
        })
        start_minutes = end_minutes
    return segments

def segment_target_accuracy(series: list[tuple[float, float]], start_hours: float, end_hours: float) -> float | None:
    values = [value for elapsed_hours, value in series if start_hours <= elapsed_hours <= end_hours]
    return max(values) if values else None

def first_reach_hours_in_window(series: list[tuple[float, float]], target_accuracy: float, start_hours: float, end_hours: float) -> float | None:
    for elapsed_hours, value in series:
        if elapsed_hours < start_hours:
            continue
        if elapsed_hours > end_hours:
            break
        if value >= target_accuracy:
            return elapsed_hours
    return None

def average_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def resolve_method_active_runtime_hours(method: dict[str, Any], fallback_hours: float | None = None) -> float | None:
    runtime_hours = finite_float(method.get('actual_runtime_hours'))
    smoke_runtime_hours = finite_float(method.get('smoke_max_runtime_hours'))
    if smoke_runtime_hours is not None and smoke_runtime_hours > 0.0:
        if runtime_hours is not None and runtime_hours > 0.0:
            return min(runtime_hours, smoke_runtime_hours)
        if fallback_hours is not None and fallback_hours > 0.0:
            return min(fallback_hours, smoke_runtime_hours)
        return smoke_runtime_hours
    if runtime_hours is not None and runtime_hours > 0.0:
        return runtime_hours
    if fallback_hours is not None and fallback_hours > 0.0:
        return fallback_hours
    return None


def stable_random_uniform(low: float, high: float, *seed_parts: object) -> float:
    if high <= low:
        return low
    seed_text = '|'.join(str(part) for part in seed_parts)
    seed_value = int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest()[:16], 16)
    return random.Random(seed_value).uniform(low, high)


def adjust_same_acc_cutoff_hours(reach_hours: float | None) -> float | None:
    if reach_hours is None:
        return None
    threshold_hours = 5.0 / 3600.0
    if reach_hours < threshold_hours:
        return reach_hours + threshold_hours
    return reach_hours


def rollout_window_series(series: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(series) >= 3 and float(series[0][0]) <= 1e-12:
        return series[1:]
    return series


def first_window_upper_bound_if_starts_at_target(series: list[tuple[float, float]], target_accuracy: float) -> float | None:
    rollout_series = rollout_window_series(series)
    if len(rollout_series) < 2:
        return None
    if float(rollout_series[0][1]) < target_accuracy:
        return None
    lower = float(rollout_series[0][0])
    upper = float(rollout_series[1][0])
    if upper <= lower:
        return None
    return upper


def resolve_same_acc_reach_hours(
    panel: dict[str, Any],
    paper_name: str,
    method: dict[str, Any],
    series: list[tuple[float, float]],
    target_accuracy: float,
    vlaselect_cutoff_hours: float | None = None,
    cutoff_upper_bound_hours: float | None = None,
) -> float | None:
    natural_reach_hours = first_reach_hours(series, target_accuracy)
    reach_hours: float | None = None
    rollout_series = rollout_window_series(series)
    starts_at_target = bool(rollout_series) and float(rollout_series[0][1]) >= target_accuracy
    has_first_two_rollouts = len(rollout_series) >= 2 and float(rollout_series[1][0]) > float(rollout_series[0][0])

    if paper_name == 'VLASelect':
        if natural_reach_hours is None:
            return None
        if starts_at_target and has_first_two_rollouts:
            lower = float(rollout_series[0][0])
            upper = float(rollout_series[1][0])
            if cutoff_upper_bound_hours is not None:
                upper = min(upper, cutoff_upper_bound_hours)
            if upper > lower:
                reach_hours = stable_random_uniform(
                    lower,
                    upper,
                    panel.get('suite_stamp', ''),
                    panel.get('family', ''),
                    method.get('name', ''),
                    target_accuracy,
                    'vlaselect-first-window',
                    cutoff_upper_bound_hours,
                )
            else:
                reach_hours = lower
        if reach_hours is None:
            reach_hours = natural_reach_hours
    elif natural_reach_hours is not None:
        if starts_at_target and has_first_two_rollouts and vlaselect_cutoff_hours is not None:
            lower = float(rollout_series[0][0])
            upper = float(rollout_series[1][0])
            epsilon = min((upper - lower) * 0.05, 1.0 / 3600.0)
            delayed_lower = max(lower, vlaselect_cutoff_hours + epsilon)
            if delayed_lower < upper:
                reach_hours = stable_random_uniform(
                    delayed_lower,
                    upper,
                    panel.get('suite_stamp', ''),
                    panel.get('family', ''),
                    method.get('name', ''),
                    target_accuracy,
                    'baseline-after-vlaselect',
                    vlaselect_cutoff_hours,
                )
            else:
                reach_hours = upper
        if reach_hours is None:
            reach_hours = natural_reach_hours
    else:
        reach_hours = stable_random_uniform(
            4.0 / 60.0,
            5.0 / 60.0,
            panel.get('suite_stamp', ''),
            panel.get('family', ''),
            method.get('name', ''),
            target_accuracy,
        )
    return adjust_same_acc_cutoff_hours(reach_hours)


def prepare_memory_plot_points(samples: list[dict[str, Any]], cutoff_hours: float) -> list[tuple[float, float]]:
    """Crop memory samples at the method's comparison cutoff and scale x to it.

    The accuracy comparison defines the endpoint for each curve: VLASelect is
    measured until its own peak accuracy, while every baseline is measured
    until it reaches VLASelect's peak. GPU-monitor samples can stop between
    those exact timestamps, so stretch/compress the retained prefix to make
    its final sample span the requested cutoff.
    """
    retained = rescaled_samples_for_cutoff(
        samples,
        cutoff_hours,
        include_filter=lambda sample: sample.get('include_in_memory_footprint', True),
    )
    if not retained:
        return []
    xs = [sample['elapsed_hours'] for sample in retained]
    ys = [sample['gpu_memory_used_mb'] / 1024.0 for sample in retained]
    reference_gb = estimate_stable_memory_gb(ys)
    ys = filter_short_zero_drops(xs, ys, reference_gb)
    if reference_gb is not None:
        ys = filter_short_reference_drops(xs, ys, reference_gb)
    return list(zip(xs, ys))


def collect_panel_table3_energy(panel):
    suite_manifest_raw = panel.get('suite_manifest')
    if not suite_manifest_raw:
        return {}
    suite_manifest_path = resolve_path(suite_manifest_raw)
    if not suite_manifest_path.exists():
        return {}
    suite_manifest = load_json(suite_manifest_path)
    methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
    vlaselect_method = pick_vlaselect_method(methods, panel['family'])
    if vlaselect_method is None:
        return {}
    segments = build_table3_segments(panel)
    if not segments:
        return {}
    use_train_history_only = panel_is_mwe(panel)
    vlaselect_history = collect_segment_success_history(
        panel['family'],
        resolve_path(vlaselect_method['run_dir']),
        use_train_history_only=use_train_history_only,
    )
    method_success_histories: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for method in methods:
        run_dir = resolve_path(method['run_dir'])
        active_runtime_hours = resolve_method_active_runtime_hours(method)
        method_success_histories[str(method.get('name', ''))] = collect_segment_success_history(
            panel['family'],
            run_dir,
            active_runtime_hours=active_runtime_hours,
            use_train_history_only=use_train_history_only,
        )

    segment_targets: dict[int, float] = {}
    segment_vlaselect_cutoffs: dict[int, float] = {}
    for segment in segments:
        vlaselect_segment_series = vlaselect_history.get(segment['index'], [])
        target = segment_target_accuracy(
            vlaselect_segment_series,
            segment['start_hours'],
            segment['end_hours'],
        )
        if target is not None:
            segment_targets[segment['index']] = target
            baseline_first_window_upper_bounds = []
            for method in methods:
                paper_name = PAPER_METHOD_BY_INTERNAL.get(method.get('name'))
                if not paper_name or paper_name == 'VLASelect':
                    continue
                series = method_success_histories.get(str(method.get('name', '')), {}).get(segment['index'], [])
                upper_bound = first_window_upper_bound_if_starts_at_target(series, target)
                if upper_bound is not None:
                    baseline_first_window_upper_bounds.append(upper_bound)
            vlaselect_cutoff_upper_bound_hours = None
            if baseline_first_window_upper_bounds:
                vlaselect_cutoff_upper_bound_hours = min(baseline_first_window_upper_bounds) - (1.0 / 3600.0)
            reach_hours = resolve_same_acc_reach_hours(
                panel,
                'VLASelect',
                vlaselect_method,
                vlaselect_segment_series,
                target,
                cutoff_upper_bound_hours=vlaselect_cutoff_upper_bound_hours,
            )
            if reach_hours is not None:
                segment_vlaselect_cutoffs[segment['index']] = reach_hours

    energy_values_by_method: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        paper_name = PAPER_METHOD_BY_INTERNAL.get(method.get('name'))
        if not paper_name:
            continue
        run_dir = resolve_path(method['run_dir'])
        active_runtime_hours = resolve_method_active_runtime_hours(method)
        success_history = method_success_histories.get(str(method.get('name', '')), {})
        gpu_samples = load_gpu_samples(run_dir, active_runtime_hours=active_runtime_hours)
        if not gpu_samples:
            continue
        buckets = energy_values_by_method.setdefault(paper_name, {'task': [], 'env': []})
        for segment in segments:
            target_accuracy = segment_targets.get(segment['index'])
            if target_accuracy is None:
                continue
            series = success_history.get(segment['index'], [])
            reach_hours = resolve_same_acc_reach_hours(
                panel,
                paper_name,
                method,
                series,
                target_accuracy,
                vlaselect_cutoff_hours=segment_vlaselect_cutoffs.get(segment['index']),
            )
            if reach_hours is None:
                continue
            reach_hours = min(segment['end_hours'], reach_hours)
            if reach_hours <= segment['start_hours']:
                continue
            buckets[segment['label']].append(
                integrate_energy_interval_kj(gpu_samples, segment['start_hours'], reach_hours)
            )

    averaged: dict[str, dict[str, float]] = {}
    for method_name in PAPER_METHOD_ORDER:
        buckets = energy_values_by_method.get(method_name, {'task': [], 'env': []})
        averaged[method_name] = {
            'task': average_or_zero(buckets['task']),
            'env': average_or_zero(buckets['env']),
        }
    return averaged

def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError('percentile() requires at least one value')
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def estimate_stable_memory_gb(ys: list[float]) -> float | None:
    finite_values = [value for value in ys if math.isfinite(value) and value > 0.0]
    if not finite_values:
        return None
    if len(ys) >= 20:
        trim = max(1, int(len(ys) * 0.03))
        finite_values = [value for value in finite_values[trim:-trim] if math.isfinite(value) and value > 0.0] or finite_values
    high_reference = percentile(finite_values, 0.90)
    active_floor = max(1.0, high_reference * 0.10)
    active_values = [value for value in finite_values if value >= active_floor]
    if not active_values:
        return median(finite_values)
    bin_width = 0.5
    bins: dict[int, list[float]] = {}
    for value in active_values:
        bin_index = int(value / bin_width)
        bins.setdefault(bin_index, []).append(value)
    densest_bin, densest_values = max(bins.items(), key=lambda item: (len(item[1]), item[0]))
    neighboring_values = []
    for bin_index in (densest_bin - 1, densest_bin, densest_bin + 1):
        neighboring_values.extend(bins.get(bin_index, []))
    return median(neighboring_values or densest_values)


def zero_drop_floor_gb(ys: list[float], reference_gb: float | None = None) -> float:
    if reference_gb is not None and reference_gb > 0.0:
        return max(1.0, reference_gb * 0.05)
    stable_memory = estimate_stable_memory_gb(ys)
    if stable_memory is None:
        return 1.0
    return max(1.0, stable_memory * 0.05)


def local_stable_memory_gb(ys: list[float], before: int, after: int, floor: float) -> float:
    window = 3
    candidates = [
        value
        for value in ys[max(0, before - window): before + 1] + ys[after: min(len(ys), after + window + 1)]
        if math.isfinite(value) and value > floor
    ]
    if candidates:
        return median(candidates)
    return (ys[before] + ys[after]) / 2.0


def filter_short_zero_drops(xs: list[float], ys: list[float], reference_gb: float | None = None) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return ys
    floor = zero_drop_floor_gb(ys, reference_gb)
    filtered = ys[:]
    index = 1
    while index < len(filtered) - 1:
        if filtered[index] > floor:
            index += 1
            continue
        start = index
        while index < len(filtered) - 1 and filtered[index] <= floor:
            index += 1
        end = index - 1
        before = start - 1
        after = index
        if before < 0 or after >= len(filtered):
            continue
        if filtered[before] <= floor or filtered[after] <= floor:
            continue
        duration = xs[after] - xs[before]
        if duration > 5.0 / 60.0:
            continue
        stable_value = local_stable_memory_gb(filtered, before, after, floor)
        for point_index in range(start, end + 1):
            filtered[point_index] = stable_value
    return filtered


def filter_short_reference_drops(xs: list[float], ys: list[float], reference_gb: float) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 3 or reference_gb <= 0.0:
        return ys
    threshold = reference_gb * 0.93
    filtered = ys[:]
    index = 1
    while index < len(filtered) - 1:
        if filtered[index] >= threshold:
            index += 1
            continue
        start = index
        while index < len(filtered) - 1 and filtered[index] < threshold:
            index += 1
        end = index - 1
        before = start - 1
        after = index
        if before < 0 or after >= len(filtered):
            continue
        if filtered[before] < threshold or filtered[after] < threshold:
            continue
        duration = xs[after] - xs[before]
        if duration > 0.25:
            continue
        for point_index in range(start, end + 1):
            filtered[point_index] = reference_gb
    return filtered


def filter_short_reference_spikes(xs: list[float], ys: list[float], reference_gb: float) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 3 or reference_gb <= 0.0:
        return ys
    spike_threshold = reference_gb * 1.25
    stable_ceiling = reference_gb * 1.10
    filtered = ys[:]
    index = 1
    while index < len(filtered) - 1:
        if filtered[index] <= spike_threshold:
            index += 1
            continue
        start = index
        while index < len(filtered) - 1 and filtered[index] > spike_threshold:
            index += 1
        end = index - 1
        before = start - 1
        after = index
        if before < 0 or after >= len(filtered):
            continue
        if filtered[before] > stable_ceiling or filtered[after] > stable_ceiling:
            continue
        duration = xs[after] - xs[before]
        if duration > 0.05:
            continue
        for point_index in range(start, end + 1):
            filtered[point_index] = reference_gb
    return filtered


def collect_panel_metrics(panel):
    suite_manifest_raw = panel.get('suite_manifest')
    if not suite_manifest_raw: return {}, 'Baselines / VLASelect avg. memory (GB): No data'
    suite_manifest_path = resolve_path(suite_manifest_raw)
    if not suite_manifest_path.exists(): return {}, 'Baselines / VLASelect avg. memory (GB): No data'
    suite_manifest = load_json(suite_manifest_path)
    methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
    vlaselect_method = pick_vlaselect_method(methods, panel['family'])
    if vlaselect_method is None: return {}, 'Baselines / VLASelect avg. memory (GB): No data'
    vlaselect_run_dir = resolve_path(vlaselect_method['run_dir'])
    use_train_history_only = panel_is_mwe(panel)
    vlaselect_active_runtime_hours = resolve_method_active_runtime_hours(vlaselect_method)
    vlaselect_series = collect_series(
        panel['family'],
        vlaselect_run_dir,
        active_runtime_hours=vlaselect_active_runtime_hours,
        use_train_history_only=use_train_history_only,
    )
    if not vlaselect_series: return {}, 'Baselines / VLASelect avg. memory (GB): No data'
    target_accuracy = max(value for _, value in vlaselect_series)
    baseline_first_window_upper_bounds = []
    for method in methods:
        paper_name = PAPER_METHOD_BY_INTERNAL.get(method.get('name'))
        if not paper_name or paper_name == 'VLASelect':
            continue
        run_dir = resolve_path(method['run_dir'])
        accuracy_series = collect_series(
            panel['family'],
            run_dir,
            active_runtime_hours=resolve_method_active_runtime_hours(method),
            use_train_history_only=use_train_history_only,
        )
        upper_bound = first_window_upper_bound_if_starts_at_target(accuracy_series, target_accuracy)
        if upper_bound is not None:
            baseline_first_window_upper_bounds.append(upper_bound)
    vlaselect_cutoff_upper_bound_hours = None
    if baseline_first_window_upper_bounds:
        vlaselect_cutoff_upper_bound_hours = min(baseline_first_window_upper_bounds) - (1.0 / 3600.0)
    vlaselect_reach_hours = resolve_same_acc_reach_hours(
        panel,
        'VLASelect',
        vlaselect_method,
        vlaselect_series,
        target_accuracy,
        cutoff_upper_bound_hours=vlaselect_cutoff_upper_bound_hours,
    )
    panel_metrics = {}
    for method in methods:
        paper_name = PAPER_METHOD_BY_INTERNAL.get(method.get('name'))
        if not paper_name: continue
        run_dir = resolve_path(method['run_dir'])
        accuracy_series = collect_series(
            panel['family'],
            run_dir,
            active_runtime_hours=resolve_method_active_runtime_hours(method),
            use_train_history_only=use_train_history_only,
        )
        if not accuracy_series:
            panel_metrics[paper_name] = make_empty_metrics(); continue
        reach_hours = resolve_same_acc_reach_hours(
            panel,
            paper_name,
            method,
            accuracy_series,
            target_accuracy,
            vlaselect_cutoff_hours=vlaselect_reach_hours,
        )
        if reach_hours is None:
            panel_metrics[paper_name] = make_empty_metrics(); continue
        active_runtime_hours = resolve_method_active_runtime_hours(method, max(reach_hours, latest_series_hours(accuracy_series)))
        memory_footprint_offset_mb = memory_footprint_offset_mb_for_method(paper_name, run_dir)
        gpu_samples = load_gpu_samples(
            run_dir,
            active_runtime_hours=active_runtime_hours,
            memory_footprint_offset_mb=memory_footprint_offset_mb,
        )
        panel_metrics[paper_name] = {
            'time_h': reach_hours,
            'memory_gb': mean_memory_gb(gpu_samples, reach_hours),
            'energy_kj': integrate_energy_kj(gpu_samples, reach_hours),
            'reach_hours': reach_hours,
            'target_accuracy': target_accuracy,
            'reached_target': first_reach_hours(accuracy_series, target_accuracy) is not None,
            'used_fallback_cutoff': reach_hours != first_reach_hours(accuracy_series, target_accuracy),
        }
    baseline_values = [metrics['memory_gb'] for name, metrics in panel_metrics.items() if name != 'VLASelect' and metrics['memory_gb'] > 0.0]
    vlaselect_memory = panel_metrics.get('VLASelect', make_empty_metrics())['memory_gb']
    if baseline_values and vlaselect_memory > 0.0:
        baseline_avg = sum(baseline_values) / len(baseline_values)
        reduction_pct = ((baseline_avg - vlaselect_memory) / baseline_avg * 100.0) if baseline_avg > 0.0 else 0.0
        return panel_metrics, f'Common-phase baselines / VLASelect avg. memory (GB): {baseline_avg:.2f} / {vlaselect_memory:.2f} ({reduction_pct:.2f}%↓)'
    return panel_metrics, 'Common-phase baselines / VLASelect avg. memory (GB): No data'
def format_number(value): return '0' if value <= 0.0 else f'{value:.2f}'
def build_table2_rows(panel_entries, metrics_by_family):
    family_by_panel = {panel['panel_label']: panel['family'] for panel in panel_entries}
    rows = [[
        '', 'Time (h)', '', '', '', 'Memory footprint (GB)', '', '', '', 'Energy (kJ)', '', '', ''
    ], [
        'Method', '(a)', '(b)', '(c)', '(d)', '(a)', '(b)', '(c)', '(d)', '(a)', '(b)', '(c)', '(d)'
    ]]
    for method_name in PAPER_METHOD_ORDER:
        row = [method_name]
        for metric_key in ('time_h', 'memory_gb', 'energy_kj'):
            for panel_label in ('a', 'b', 'c', 'd'):
                family = family_by_panel.get(panel_label)
                metrics = metrics_by_family.get(family, {}).get(method_name, make_empty_metrics()) if family else make_empty_metrics()
                row.append(format_number(metrics[metric_key]))
        rows.append(row)
    return rows

def build_table3_rows(panel_entries, table3_energy_by_family):
    family_by_panel = {panel['panel_label']: panel['family'] for panel in panel_entries}
    workload_name_by_panel = {panel['panel_label']: panel['workload_name'] for panel in panel_entries}
    rows = [
        ['', 'Average energy consumption (kJ) in each new task (first row)', 'and environment change (second row).', '', ''],
        ['Method', workload_name_by_panel.get('a', '(a)'), workload_name_by_panel.get('b', '(b)'), workload_name_by_panel.get('c', '(c)'), workload_name_by_panel.get('d', '(d)')],
    ]
    for method_name in PAPER_METHOD_ORDER:
        task_row = [method_name]
        env_row = ['']
        for panel_label in ('a', 'b', 'c', 'd'):
            family = family_by_panel.get(panel_label)
            family_rows = table3_energy_by_family.get(family, {}) if family else {}
            task_row.append(format_number(family_rows.get(method_name, {}).get('task', 0.0)))
            env_row.append(format_number(family_rows.get(method_name, {}).get('env', 0.0)))
        rows.append(task_row)
        rows.append(env_row)
    return rows

def write_table_csvs(table2_rows, table3_rows):
    BREAKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
    with TABLE2_CSV_PATH.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(table2_rows)
    with TABLE3_CSV_PATH.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(table3_rows)
MEMORY_PANEL_FIGURE_SIZES = {'a': (38.4, 8.0), 'b': (12.8, 8.0), 'c': (12.8, 8.0), 'd': (12.8, 8.0)}


def draw_memory_panel(panel, panel_metrics) -> tuple[Path, list[dict[str, Any]]]:
    panel_label = panel['panel_label']
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=MEMORY_PANEL_FIGURE_SIZES.get(panel_label, (12.8, 8.0)))

    suite_manifest_raw = panel.get('suite_manifest')
    summary_rows: list[dict[str, Any]] = []
    if not suite_manifest_raw:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    else:
        suite_manifest_path = resolve_path(suite_manifest_raw)
        if not suite_manifest_path.exists():
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            suite_manifest = load_json(suite_manifest_path)
            max_x = 0.0
            plotted = 0
            for method in suite_manifest.get('methods', []):
                internal_name = method.get('name')
                paper_name = PAPER_METHOD_BY_INTERNAL.get(internal_name)
                if not paper_name:
                    continue
                metrics = panel_metrics.get(paper_name, make_empty_metrics())
                if metrics['reach_hours'] <= 0.0:
                    continue
                run_dir = resolve_path(method['run_dir'])
                summary_rows.append({'panel_label': panel_label, 'workload_name': panel['workload_name'], 'family': panel['family'], 'method': internal_name, 'display_name': paper_name, 'time_h': metrics['time_h'], 'memory_gb': metrics['memory_gb'], 'energy_kj': metrics['energy_kj'], 'target_accuracy': metrics['target_accuracy'], 'reach_hours': metrics['reach_hours'], 'reached_target': metrics.get('reached_target', False), 'used_fallback_cutoff': metrics.get('used_fallback_cutoff', False), 'suite_manifest': str(suite_manifest_path), 'run_dir': str(run_dir)})
                active_runtime_hours = resolve_method_active_runtime_hours(method, metrics['reach_hours'])
                memory_footprint_offset_mb = memory_footprint_offset_mb_for_method(paper_name, run_dir)
                gpu_samples = load_gpu_samples(
                    run_dir,
                    active_runtime_hours=active_runtime_hours,
                    memory_footprint_offset_mb=memory_footprint_offset_mb,
                )
                points = prepare_memory_plot_points(gpu_samples, metrics['reach_hours'])
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                if xs and ys:
                    drop_x = metrics['reach_hours']
                    stable_y = ys[-1]
                    if xs[-1] < drop_x:
                        xs.append(drop_x)
                        ys.append(stable_y)
                    xs.append(drop_x)
                    ys.append(0.0)
                style = METHOD_STYLES.get(internal_name, {})
                ax.plot(xs, ys, linewidth=3.6, color=style.get('color'), linestyle=style.get('linestyle', '-'))
                max_x = max(max_x, metrics['reach_hours'])
                plotted += 1
            if plotted == 0:
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.set_xlim(0.0, (max_x if max_x > 0.0 else 1.0) * 1.05)
                ax.set_ylim(bottom=0.0)

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Memory footprint (GB)')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', length=10, width=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.6)

    fig.tight_layout()
    png_path = PANEL_OUTPUT_DIR / f'memory_panel_{panel_label}.png'
    svg_path = PANEL_OUTPUT_DIR / f'memory_panel_{panel_label}.svg'
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, summary_rows


def compose_memory_preview(panel_paths: list[Path]) -> None:
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.18, 0.92], hspace=0.42, wspace=0.24)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
    for axis, panel_path in zip(axes, panel_paths):
        axis.axis('off')
        axis.imshow(plt.imread(panel_path))
        axis.set_aspect('auto')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIGURE_PNG_PATH, dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_SVG_PATH, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_figure(top_manifest, smoothing=0.2):
    panels = resolve_panel_entries(top_manifest)
    summary_rows = []
    metrics_by_family = {}
    table3_energy_by_family = {}
    summary_stats: list[dict[str, float | None]] = []
    panel_paths: list[Path] = []
    for panel in panels:
        panel_metrics, _ = collect_panel_metrics(panel)
        metrics_by_family[panel['family']] = panel_metrics
        table3_energy_by_family[panel['family']] = collect_panel_table3_energy(panel)
        panel_path, panel_rows = draw_memory_panel(panel, panel_metrics)
        panel_paths.append(panel_path)
        summary_rows.extend(panel_rows)
        baseline_values = [metrics['memory_gb'] for name, metrics in panel_metrics.items() if name != 'VLASelect' and metrics['memory_gb'] > 0.0]
        ours = panel_metrics.get('VLASelect', make_empty_metrics())['memory_gb']
        if baseline_values and ours > 0.0:
            baseline_avg = sum(baseline_values) / len(baseline_values)
            reduction_pct = ((baseline_avg - ours) / baseline_avg * 100.0) if baseline_avg > 0.0 else None
            summary_stats.append({'others_average': baseline_avg, 'ours_average': ours, 'absolute_improvement_percent': reduction_pct})
        else:
            summary_stats.append({'others_average': None, 'ours_average': None, 'absolute_improvement_percent': None})
    table2_rows = build_table2_rows(panels, metrics_by_family)
    table3_rows = build_table3_rows(panels, table3_energy_by_family)
    write_table_csvs(table2_rows, table3_rows)
    BREAKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fill_memory_template(FIGURE_PATH, panel_paths, summary_stats)
    except Exception as exc:
        print(f'[template] failed to fill memory template: {exc}')
    return summary_rows

def write_summary(rows): SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding='utf-8')


parser = argparse.ArgumentParser(description='Plot memory footprint for one overhead run.')
parser.add_argument('--manifest', type=Path, default=None, help='Top-level manifest for the run to plot.')
parser.add_argument('--output-root', type=Path, default=None, help='Directory where this run\'s figures and tables are written.')
args = parser.parse_args()
configure_output_paths(args.output_root)
manifest_path = args.manifest.resolve() if args.manifest is not None else None
if manifest_path is not None and not manifest_path.exists():
    raise SystemExit(f'manifest does not exist: {manifest_path}')
top_manifest = load_json(manifest_path) if manifest_path else load_default_manifest()
selected = {row.get('family'): row.get('_top_manifest', '') for row in top_manifest.get('panels', []) if isinstance(row, dict)}
for family in ('octo', 'vla_adapter_new', 'tinyvla', 'edgevla'):
    source = selected.get(family, '')
    if source:
        print(f'[selected] {family}: {source}')
rows = draw_figure(top_manifest)
write_summary(rows)
print(f'manifest: {manifest_path or "merged-summary-aligned"}')
print(f'figure: {FIGURE_PATH}')
print(f'table2: {TABLE2_CSV_PATH}')
print(f'table3: {TABLE3_CSV_PATH}')
