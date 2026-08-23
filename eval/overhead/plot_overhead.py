from __future__ import annotations
import csv
import json
import math
import sys
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
PAPER_PANELS = [
    {'panel_label': 'a', 'family': 'octo', 'display_name': 'Octo', 'workload_name': 'Single-arm robot', 'panel_title': '(a) Single-arm robot'},
    {'panel_label': 'b', 'family': 'vla_adapter_new', 'display_name': 'VLA-Adapter', 'workload_name': 'Dexterous hand', 'panel_title': '(b) Dexterous hand'},
    {'panel_label': 'c', 'family': 'tinyvla', 'display_name': 'TinyVLA', 'workload_name': 'Mobile manipulator', 'panel_title': '(c) Mobile manipulator'},
    {'panel_label': 'd', 'family': 'edgevla', 'display_name': 'EdgeVLA', 'workload_name': 'Humanoid robot', 'panel_title': '(d) Humanoid robot'},
]
METHOD_STYLES = {'conrft': {'color': '#4C78A8', 'linestyle': '-'},'flare': {'color': '#59A14F', 'linestyle': '-'},'improv_vla': {'color': '#4D4D4D', 'linestyle': '-'},'edgeta': {'color': '#A6A6A6', 'linestyle': '--'},'convertnet': {'color': '#CEBB6C', 'linestyle': '--'},'ours': {'color': '#C44E52', 'linestyle': '-'},'ours_single_agent': {'color': '#C44E52', 'linestyle': '-'},'ppo_gen': {'color': '#4C78A8', 'linestyle': '--'},'self_improv': {'color': '#9A9A9A', 'linestyle': '-'},'self_improvement': {'color': '#9A9A9A', 'linestyle': '-'},'vla_rft': {'color': '#59A14F', 'linestyle': '--'},'world_env': {'color': '#4D4D4D', 'linestyle': '--'}}
LEGEND_ORDER = ['conrft', 'flare', 'improv_vla', 'self_improv', 'self_improvement', 'ppo_gen', 'vla_rft', 'world_env', 'edgeta', 'convertnet', 'ours', 'ours_single_agent']
FAMILY_CONFIGS = {'edgevla': {'metric_key': 'eval_success_once', 'loader': 'history'},'octo': {'metric_key': 'eval/success_once', 'loader': 'tensorboard'},'tinyvla': {'metric_key': 'train_success_once', 'loader': 'history'},'vla_adapter_new': {'metric_key': 'train_success_once', 'loader': 'history'}}
PAPER_METHOD_ORDER = ['ConRFT', 'FlaRe', 'iRe-VLA', 'Self-Improvement', 'RLVLA', 'VLA-RFT', 'World-Env', 'EdgeTA', 'ConvertNet', 'VLASelect']
PAPER_METHOD_BY_INTERNAL = {'conrft': 'ConRFT','flare': 'FlaRe','improv_vla': 'iRe-VLA','self_improv': 'Self-Improvement','self_improvement': 'Self-Improvement','ppo_gen': 'RLVLA','vla_rft': 'VLA-RFT','world_env': 'World-Env','edgeta': 'EdgeTA','convertnet': 'ConvertNet','ours': 'VLASelect','ours_single_agent': 'VLASelect'}
VLASELECT_METHODS_BY_FAMILY = {'octo': ['ours_single_agent', 'ours'],'vla_adapter_new': ['ours'],'tinyvla': ['ours'],'edgevla': ['ours', 'ours_single_agent']}
def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding='utf-8'))
def resolve_path(raw_path: str, base_dir: Path = EVAL_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()
def finite_float(value: Any) -> float | None:
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if math.isfinite(result) else None
def find_latest_manifest() -> Path | None:
    if LATEST_POINTER.exists():
        stamp = LATEST_POINTER.read_text(encoding='utf-8').strip()
        if stamp:
            candidate = TABLE_ROOT / stamp / 'manifest.json'
            if candidate.exists(): return candidate
    manifest_paths = sorted(TABLE_ROOT.glob('*/manifest.json'))
    return manifest_paths[-1] if manifest_paths else None
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
def collect_history_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
    series = []
    for index, metric in enumerate(load_history(run_dir)):
        y_value = finite_float(metric.get(metric_key))
        if y_value is None: continue
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        x_value = elapsed_hours if elapsed_hours is not None else float(index)
        series.append((x_value, y_value))
    return series
def find_tb_dir(run_dir: Path) -> Path | None:
    for candidate in [run_dir / 'tb', run_dir / '[agent]' / 'tb']:
        if candidate.is_dir(): return candidate
    for search_root in [run_dir, run_dir.parent]:
        if search_root.exists():
            nested = sorted(path for path in search_root.glob('**/tb') if path.is_dir())
            if nested: return nested[0]
    return None
def collect_tensorboard_series(run_dir: Path, metric_key: str) -> list[tuple[float, float]]:
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
    return [((event.wall_time - base_time) / 3600.0, float(event.value)) for event in events]
def collect_series(family: str, run_dir: Path) -> list[tuple[float, float]]:
    config = FAMILY_CONFIGS[family]
    return collect_tensorboard_series(run_dir, config['metric_key']) if config['loader'] == 'tensorboard' else collect_history_series(run_dir, config['metric_key'])
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
def load_gpu_samples(run_dir: Path):
    csv_path = find_gpu_metrics_csv(run_dir)
    if csv_path is None: return []
    rows = []
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                elapsed_seconds = finite_float(row.get('elapsed_seconds'))
                if elapsed_seconds is None: continue
                rows.append({'elapsed_seconds': elapsed_seconds,'elapsed_hours': elapsed_seconds / 3600.0,'gpu_memory_used_mb': finite_float(row.get('gpu_memory_used_mb')) or 0.0,'gpu_power_w': finite_float(row.get('gpu_power_w')) or 0.0})
    except Exception: return []
    return sorted(rows, key=lambda item: item['elapsed_seconds'])
def first_reach_hours(series, target_accuracy):
    for x_value, y_value in series:
        if y_value >= target_accuracy: return x_value
    return None
def integrate_energy_kj(samples, cutoff_hours):
    if not samples: return 0.0
    cutoff_seconds = cutoff_hours * 3600.0
    total_j = 0.0
    prev = samples[0]
    for curr in samples[1:]:
        start = min(prev['elapsed_seconds'], cutoff_seconds)
        end = min(curr['elapsed_seconds'], cutoff_seconds)
        if end > start: total_j += prev['gpu_power_w'] * (end - start)
        if curr['elapsed_seconds'] >= cutoff_seconds: break
        prev = curr
    else:
        if prev['elapsed_seconds'] < cutoff_seconds: total_j += prev['gpu_power_w'] * (cutoff_seconds - prev['elapsed_seconds'])
    return total_j / 1000.0
def mean_memory_gb(samples, cutoff_hours):
    values = [sample['gpu_memory_used_mb'] / 1024.0 for sample in samples if sample['elapsed_hours'] <= cutoff_hours]
    return sum(values) / len(values) if values else 0.0
def pick_vlaselect_method(methods, family):
    by_name = {method.get('name'): method for method in methods if isinstance(method, dict)}
    for name in VLASELECT_METHODS_BY_FAMILY.get(family, []):
        if name in by_name: return by_name[name]
    return None
def make_empty_metrics(): return {'time_h': 0.0, 'memory_gb': 0.0, 'energy_kj': 0.0, 'reach_hours': 0.0, 'target_accuracy': 0.0}

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
    vlaselect_series = collect_series(panel['family'], vlaselect_run_dir)
    if not vlaselect_series: return {}, 'Baselines / VLASelect avg. memory (GB): No data'
    target_accuracy = max(value for _, value in vlaselect_series)
    panel_metrics = {}
    for method in methods:
        paper_name = PAPER_METHOD_BY_INTERNAL.get(method.get('name'))
        if not paper_name: continue
        run_dir = resolve_path(method['run_dir'])
        accuracy_series = collect_series(panel['family'], run_dir)
        if not accuracy_series:
            panel_metrics[paper_name] = make_empty_metrics(); continue
        reach_hours = first_reach_hours(accuracy_series, target_accuracy)
        if reach_hours is None:
            panel_metrics[paper_name] = make_empty_metrics(); continue
        gpu_samples = load_gpu_samples(run_dir)
        panel_metrics[paper_name] = {'time_h': reach_hours,'memory_gb': mean_memory_gb(gpu_samples, reach_hours),'energy_kj': integrate_energy_kj(gpu_samples, reach_hours),'reach_hours': reach_hours,'target_accuracy': target_accuracy}
    baseline_values = [metrics['memory_gb'] for name, metrics in panel_metrics.items() if name != 'VLASelect' and metrics['memory_gb'] > 0.0]
    vlaselect_memory = panel_metrics.get('VLASelect', make_empty_metrics())['memory_gb']
    if baseline_values and vlaselect_memory > 0.0:
        baseline_avg = sum(baseline_values) / len(baseline_values)
        reduction_pct = ((baseline_avg - vlaselect_memory) / baseline_avg * 100.0) if baseline_avg > 0.0 else 0.0
        return panel_metrics, f'Baselines / VLASelect avg. memory (GB): {baseline_avg:.2f} / {vlaselect_memory:.2f} ({reduction_pct:.2f}%↓)'
    return panel_metrics, 'Baselines / VLASelect avg. memory (GB): No data'
def format_number(value): return '0' if value <= 0.0 else f'{value:.2f}'
def build_table2_rows(panel_entries, metrics_by_family):
    family_by_panel = {panel['panel_label']: panel['family'] for panel in panel_entries}
    rows = [['', 'Time (h)', '', '', '', 'Memory footprint (GB)', '', '', ''], ['Method', '(a)', '(b)', '(c)', '(d)', '(a)', '(b)', '(c)', '(d)']]
    for method_name in PAPER_METHOD_ORDER:
        row = [method_name]
        for metric_key in ('time_h', 'memory_gb'):
            for panel_label in ('a', 'b', 'c', 'd'):
                family = family_by_panel.get(panel_label)
                metrics = metrics_by_family.get(family, {}).get(method_name, make_empty_metrics()) if family else make_empty_metrics()
                row.append(format_number(metrics[metric_key]))
        rows.append(row)
    return rows

def build_table3_rows(panel_entries, metrics_by_family):
    family_by_panel = {panel['panel_label']: panel['family'] for panel in panel_entries}
    rows = [['', 'Energy consumption (kJ)', '', '', ''], ['Method', '(a)', '(b)', '(c)', '(d)']]
    for method_name in PAPER_METHOD_ORDER:
        row = [method_name]
        for panel_label in ('a', 'b', 'c', 'd'):
            family = family_by_panel.get(panel_label)
            metrics = metrics_by_family.get(family, {}).get(method_name, make_empty_metrics()) if family else make_empty_metrics()
            row.append(format_number(metrics['energy_kj']))
        rows.append(row)
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
                gpu_samples = load_gpu_samples(run_dir)
                points = [
                    (sample['elapsed_hours'], sample['gpu_memory_used_mb'] / 1024.0)
                    for sample in gpu_samples
                    if sample['elapsed_hours'] <= metrics['reach_hours']
                ]
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                reference_gb = metrics['memory_gb'] if metrics['memory_gb'] > 0.0 else estimate_stable_memory_gb(ys)
                ys = filter_short_zero_drops(xs, ys, reference_gb)
                if reference_gb is not None:
                    ys = filter_short_reference_drops(xs, ys, reference_gb)
                    if panel['family'] == 'edgevla' and internal_name == 'flare':
                        ys = filter_short_reference_spikes(xs, ys, reference_gb)
                style = METHOD_STYLES.get(internal_name, {})
                ax.plot(xs, ys, linewidth=3.6, color=style.get('color'), linestyle=style.get('linestyle', '-'))
                max_x = max(max_x, xs[-1])
                plotted += 1
                summary_rows.append({'panel_label': panel_label, 'workload_name': panel['workload_name'], 'family': panel['family'], 'method': internal_name, 'display_name': paper_name, 'time_h': metrics['time_h'], 'memory_gb': metrics['memory_gb'], 'energy_kj': metrics['energy_kj'], 'target_accuracy': metrics['target_accuracy'], 'reach_hours': metrics['reach_hours'], 'suite_manifest': str(suite_manifest_path), 'run_dir': str(run_dir)})
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
    summary_stats: list[dict[str, float | None]] = []
    panel_paths: list[Path] = []
    for panel in panels:
        panel_metrics, _ = collect_panel_metrics(panel)
        metrics_by_family[panel['family']] = panel_metrics
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
    table3_rows = build_table3_rows(panels, metrics_by_family)
    write_table_csvs(table2_rows, table3_rows)
    BREAKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fill_memory_template(FIGURE_PATH, panel_paths, summary_stats)
    except Exception as exc:
        print(f'[template] failed to fill memory template: {exc}')
    return summary_rows

def write_summary(rows): SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding='utf-8')
manifest_path = find_latest_manifest()
top_manifest = load_json(manifest_path) if manifest_path else default_manifest()
rows = draw_figure(top_manifest)
write_summary(rows)
print(f'manifest: {manifest_path or "no-data"}')
print(f'figure: {FIGURE_PATH}')
print(f'table2: {TABLE2_CSV_PATH}')
print(f'table3: {TABLE3_CSV_PATH}')
