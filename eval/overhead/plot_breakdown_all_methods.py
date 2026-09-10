from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from common.figure_compose import compose_grid_figure
from common.template_pdf_fill import fill_sampling_training_template
from plot_breakdown_impl import (
    ALL_METHODS_TABLE_ROOT,
    SAME_ACC_TABLE_ROOT,
    apply_dynamic_time_axis,
    load_csv_rows,
    load_top_manifest_from_table_root,
    prepare_breakdown_tables,
)

PANEL_DIR = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS_panels"
FIG_ALL_METHODS = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.pdf"
FIG_ALL_METHODS_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.svg"
FIG_ALL_METHODS_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.png"
JSON_OUTPUT = SCRIPT_DIR / "training_time_breakdown.json"
PLOT_INPUTS_FILENAME = "breakdown_plot_inputs.json"

DATASET_ORDER = ["octo", "vla_adapter_new", "tinyvla", "edgevla"]
METHOD_ORDER = ["conrft", "flare", "improv_vla", "self_improv", "ppo_gen", "vla_rft", "world_env", "edgeta", "convertnet", "ours"]
LABEL_MAP = {
    "conrft": "ConRFT",
    "flare": "FLaRe",
    "improv_vla": "iRe-VLA",
    "self_improv": "Self-Improvement",
    "ppo_gen": "RLVLA",
    "vla_rft": "VLA-RFT",
    "world_env": "World-Env",
    "edgeta": "EdgeTA",
    "convertnet": "ConvertNet",
    "ours": "VLASelect",
    "ours_single_agent": "VLASelect",
}


def load_latest_manifest() -> tuple[dict, Path | None]:
    latest_path = ALL_METHODS_TABLE_ROOT / 'latest.txt'
    if latest_path.exists():
        stamp = latest_path.read_text(encoding='utf-8').strip()
        candidate = ALL_METHODS_TABLE_ROOT / stamp / 'manifest.json'
        if stamp and candidate.is_file():
            return load_top_manifest_from_table_root(ALL_METHODS_TABLE_ROOT, str(candidate))
        warnings.warn(
            f'Latest breakdown pointer is invalid: {latest_path} -> {stamp or "<empty>"}. '
            'Falling back to the most recently modified manifest.',
            RuntimeWarning,
            stacklevel=2,
        )

    candidates = sorted(
        ALL_METHODS_TABLE_ROOT.glob('*/manifest.json'),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return load_top_manifest_from_table_root(ALL_METHODS_TABLE_ROOT, str(candidates[-1]))

    warnings.warn('No all-methods breakdown manifest was found.', RuntimeWarning, stacklevel=2)
    return {}, None


def load_manifest(manifest_path: str | None) -> tuple[dict, Path | None]:
    if manifest_path:
        return load_top_manifest_from_table_root(ALL_METHODS_TABLE_ROOT, manifest_path)
    return load_latest_manifest()


def load_same_acc_manifest() -> tuple[dict, Path | None]:
    """Load the per-workload merged suite manifest used by the target-acc plot."""
    return load_top_manifest_from_table_root(SAME_ACC_TABLE_ROOT, None)


def warn_if_workloads_incomplete(manifest: dict) -> None:
    panels = manifest.get('panels', manifest.get('families', []))
    by_family = {
        str(panel.get('family')): panel
        for panel in panels
        if isinstance(panel, dict) and str(panel.get('family')) in DATASET_ORDER
    }
    issues: list[str] = []
    for family in DATASET_ORDER:
        panel = by_family.get(family)
        if panel is None:
            issues.append(f'{family}: missing from manifest')
            continue
        suite_manifest_ref = str(panel.get('suite_manifest') or '').strip()
        if not suite_manifest_ref:
            issues.append(f'{family}: suite manifest missing')
            continue
        suite_manifest_path = Path(suite_manifest_ref)
        if not suite_manifest_path.is_absolute():
            suite_manifest_path = EVAL_ROOT / suite_manifest_path
        if not suite_manifest_path.is_file():
            issues.append(f'{family}: suite manifest not found ({suite_manifest_ref})')
            continue
        try:
            suite_manifest = json.loads(suite_manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            issues.append(f'{family}: unreadable suite manifest ({suite_manifest_ref})')
            continue
        if suite_manifest.get('suite_state') != 'finished':
            state = suite_manifest.get('suite_state', 'unknown')
            issues.append(f'{family}: suite_state={state!r}')
    if issues:
        warnings.warn(
            'Latest all-methods breakdown is incomplete; plotting only its available data. '
            + '; '.join(issues),
            RuntimeWarning,
            stacklevel=2,
        )


def resolve_output_root(manifest: dict, resolved_manifest_path: Path | None) -> Path:
    if resolved_manifest_path is not None:
        return resolved_manifest_path.parent
    suite_stamp = str(manifest.get("suite_stamp") or "").strip()
    if suite_stamp and suite_stamp not in {"no-data", "merged-latest"}:
        return ALL_METHODS_TABLE_ROOT / suite_stamp
    return ALL_METHODS_TABLE_ROOT / "merged-summary-aligned"


def resolve_eval_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else EVAL_ROOT / path


def same_acc_summary_path(panel: dict, manifest: dict) -> Path | None:
    """Return the target-accuracy summary that defines this panel's cutoff."""
    explicit_summary = str(panel.get("_same_acc_summary_path") or "").strip()
    if explicit_summary:
        summary_path = resolve_eval_path(explicit_summary)
        if summary_path.is_file():
            return summary_path
    candidates = [
        str(panel.get("same_acc_manifest") or "").strip(),
        str(manifest.get("same_acc_manifest") or "").strip(),
    ]
    for manifest_ref in candidates:
        if not manifest_ref:
            continue
        summary_path = resolve_eval_path(manifest_ref).parent / "overhead_same_acc_summary.json"
        if summary_path.is_file():
            return summary_path

    # This is the same fallback used when a breakdown manifest has no explicit
    # same-accuracy reference.
    latest_path = SCRIPT_DIR / "overhead_same_acc_table" / "latest.txt"
    if latest_path.is_file():
        stamp = latest_path.read_text(encoding="utf-8").strip()
        summary_path = latest_path.parent / stamp / "overhead_same_acc_summary.json"
        if stamp and summary_path.is_file():
            return summary_path
    return None


def apply_target_accuracy_summary(manifest: dict, summary_path: Path) -> dict:
    """Make all workload panels use one plot_overhead_target_acc summary."""
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read --target-accuracy-summary {summary_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit("--target-accuracy-summary must contain the JSON row list written by plot_overhead_target_acc.py")
    result = dict(manifest)
    for key in ("panels", "families"):
        entries = result.get(key, [])
        if not isinstance(entries, list):
            continue
        result[key] = [
            ({**entry, "_same_acc_summary_path": str(summary_path)} if isinstance(entry, dict) else entry)
            for entry in entries
        ]
    return result


def collect_plot_inputs(manifest: dict, rows: list[dict[str, str]]) -> list[dict]:
    """Describe the exact timing and target-accuracy inputs used per panel."""
    panels = manifest.get("panels", manifest.get("families", []))
    panels_by_family = {
        str(panel.get("family")): panel
        for panel in panels
        if isinstance(panel, dict)
    }
    inputs: list[dict] = []
    for family in DATASET_ORDER:
        panel = panels_by_family.get(family, {})
        panel_rows = [row for row in rows if row.get("family") == family]
        summary_path = same_acc_summary_path(panel, manifest)
        target_by_method: dict[str, float] = {}
        reach_hours_by_method: dict[str, float] = {}
        if summary_path is not None:
            try:
                summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                warnings.warn(f"Cannot read target-accuracy summary {summary_path}: {exc}", RuntimeWarning)
                summary_rows = []
            if isinstance(summary_rows, list):
                for row in summary_rows:
                    if not isinstance(row, dict) or str(row.get("family", "")) != family:
                        continue
                    method = str(row.get("method", "")).strip()
                    target = row.get("target_accuracy")
                    reach_hours = row.get("reach_hours")
                    if method and isinstance(target, (int, float)):
                        target_by_method[method] = float(target)
                    if method and isinstance(reach_hours, (int, float)):
                        reach_hours_by_method[method] = float(reach_hours)
        unique_targets = sorted(set(target_by_method.values()))
        inputs.append(
            {
                "family": family,
                "panel_label": panel.get("panel_label", ""),
                "workload_name": panel.get("workload_name", family),
                "target_accuracy_mode": (
                    "per-workload" if len(unique_targets) == 1 and unique_targets else
                    "mixed-per-method" if len(unique_targets) > 1 else "unavailable"
                ),
                "target_accuracy": unique_targets[0] if len(unique_targets) == 1 else None,
                "target_accuracy_by_method": target_by_method,
                "reach_hours_by_method": reach_hours_by_method,
                "target_accuracy_source": str(summary_path) if summary_path is not None else "",
                "timing_inputs": [
                    {
                        "method": row.get("method_name", ""),
                        "source": row.get("source", ""),
                        "sampling_seconds": float(row.get("sampling_seconds", 0.0)),
                        "training_seconds": float(row.get("training_seconds", 0.0)),
                    }
                    for row in panel_rows
                ],
            }
        )
    return inputs


def print_plot_inputs(plot_inputs: list[dict]) -> None:
    for item in plot_inputs:
        target = item["target_accuracy"]
        target_text = f"{target:.4f}" if target is not None else "unavailable"
        print(
            f"[plot-input] {item['family']} ({item['workload_name']}): "
            f"target_acc={target_text}, mode={item['target_accuracy_mode']}, "
            f"target_source={item['target_accuracy_source'] or '<none>'}"
        )
        for timing in item["timing_inputs"]:
            print(
                f"[plot-input]   method={timing['method']} "
                f"sampling_s={timing['sampling_seconds']:.6f} "
                f"training_s={timing['training_seconds']:.6f} "
                f"source={timing['source'] or '<none>'}"
            )


def build_payload(rows: list[dict[str, str]]) -> dict:
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHOD_ORDER:
        breakdown[method] = {'by_dataset': {}}
        for dataset in DATASET_ORDER:
            breakdown[method]['by_dataset'][dataset] = {'rollout_hours': 0.0, 'model_update_hours': 0.0}

    for row in rows:
        family = row.get('family', '')
        method = row.get('method_name', '')
        if family not in DATASET_ORDER:
            continue
        key = 'ours' if method == 'ours_single_agent' else method
        if key not in breakdown:
            continue
        breakdown[key]['by_dataset'][family] = {
            'rollout_hours': float(row.get('sampling_seconds', 0.0)) / 3600.0,
            'model_update_hours': float(row.get('training_seconds', 0.0)) / 3600.0,
        }

    return {
        'dataset_order': DATASET_ORDER,
        'method_order': METHOD_ORDER,
        'breakdown': breakdown,
    }


def draw_panels(payload: dict) -> list[Path]:
    labels = [LABEL_MAP.get(method, method) for method in METHOD_ORDER]
    label_size = 26
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 18,
        'axes.labelsize': label_size,
        'xtick.labelsize': label_size,
        'ytick.labelsize': label_size,
        'hatch.linewidth': 1.8,
        'svg.fonttype': 'none',
    })
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel_paths = []

    for dataset in DATASET_ORDER:
        sample = []
        training = []
        for method in METHOD_ORDER:
            item = payload['breakdown'][method]['by_dataset'][dataset]
            sample.append(float(item['rollout_hours']))
            training.append(float(item['model_update_hours']))

        x_values = np.arange(len(METHOD_ORDER))
        totals = np.array(sample) + np.array(training)
        fig, ax = plt.subplots(figsize=(7.2, 8.0))
        ax.set_ylabel('Time (hours)')
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
        ax.tick_params(axis='x', labelsize=label_size)
        ax.bar(
            x_values,
            sample,
            color='white',
            edgecolor='black',
            linewidth=2.2,
            width=0.72,
        )
        ax.bar(
            x_values,
            training,
            bottom=sample,
            color='white',
            edgecolor='black',
            linewidth=2.2,
            hatch='/',
            width=0.72,
        )
        apply_dynamic_time_axis(ax, totals, margin_ratio=0.12, default_upper=(1.0 / 60.0))
        ax.grid(axis='y', color='#9A9A9A', alpha=0.55, linewidth=1.0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.6)
        fig.tight_layout()
        png_path = PANEL_DIR / f'training_time_breakdown_{dataset}.png'
        svg_path = PANEL_DIR / f'training_time_breakdown_{dataset}.svg'
        fig.savefig(png_path, dpi=220)
        fig.savefig(svg_path, dpi=220)
        plt.close(fig)
        panel_paths.append(png_path)
    return panel_paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument(
        '--target-accuracy-summary',
        type=Path,
        default=None,
        help='JSON summary emitted by plot_overhead_target_acc.py; use its reach_hours as each method cutoff.',
    )
    parser.add_argument(
        '--same-acc-table',
        action='store_true',
        help='Use the merged suites from overhead_same_acc_table (the same data source as plot_overhead_target_acc.py).',
    )
    args = parser.parse_args(argv)
    if args.manifest is not None and args.same_acc_table:
        raise SystemExit('--manifest and --same-acc-table cannot be used together')

    manifest, resolved_manifest_path = (
        load_same_acc_manifest() if args.same_acc_table else load_manifest(args.manifest)
    )
    if args.target_accuracy_summary is not None:
        summary_path = args.target_accuracy_summary.resolve()
        if not summary_path.is_file():
            raise SystemExit(f'target-accuracy summary does not exist: {summary_path}')
        manifest = apply_target_accuracy_summary(manifest, summary_path)
    warn_if_workloads_incomplete(manifest)
    output_root = resolve_output_root(manifest, resolved_manifest_path)
    selected = {row.get('family'): row.get('_top_manifest', '') for row in manifest.get('panels', []) if isinstance(row, dict)}
    for family in DATASET_ORDER:
        source = selected.get(family, '')
        if source:
            print(f'[selected] {family}: {source}')
    all_rows, _ = prepare_breakdown_tables(manifest, output_root)
    if not all_rows:
        all_rows = load_csv_rows(output_root / 'BREAKDOWN_ALL_METHODS.csv')

    payload = build_payload(all_rows)
    plot_inputs = collect_plot_inputs(manifest, all_rows)
    payload['plot_inputs'] = plot_inputs
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot_inputs_output = output_root / PLOT_INPUTS_FILENAME
    plot_inputs_output.write_text(json.dumps(plot_inputs, indent=2), encoding='utf-8')
    print_plot_inputs(plot_inputs)
    panel_paths = draw_panels(payload)
    compose_grid_figure(panel_paths, output_paths=[FIG_ALL_METHODS_PNG, FIG_ALL_METHODS_SVG], rows=1, cols=4, figsize=(20.0, 5.0), legend_path=None, dpi=200)
    fill_sampling_training_template(FIG_ALL_METHODS, panel_paths)
    print(f"Saved JSON: {JSON_OUTPUT}")
    print(f"Saved plot inputs: {plot_inputs_output}")
    print(f"Saved PDF: {FIG_ALL_METHODS}")
    print(f"Saved PNG: {FIG_ALL_METHODS_PNG}")
    print(f"Saved SVG: {FIG_ALL_METHODS_SVG}")


if __name__ == '__main__':
    main()
