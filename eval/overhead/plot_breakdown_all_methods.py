from __future__ import annotations

import argparse
import json
import sys
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
from plot_breakdown_impl import ALL_METHODS_TABLE_ROOT, load_csv_rows, load_top_manifest_from_table_root, prepare_breakdown_tables

PANEL_DIR = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS_panels"
FIG_ALL_METHODS = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.pdf"
FIG_ALL_METHODS_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.svg"
FIG_ALL_METHODS_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.png"
JSON_OUTPUT = SCRIPT_DIR / "training_time_breakdown.json"

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


def resolve_output_root(manifest_path: str | None) -> Path:
    manifest, resolved_manifest_path = load_top_manifest_from_table_root(ALL_METHODS_TABLE_ROOT, manifest_path)
    if resolved_manifest_path is not None:
        return resolved_manifest_path.parent
    if manifest.get("suite_stamp") not in {None, "", "no-data"}:
        return ALL_METHODS_TABLE_ROOT / str(manifest["suite_stamp"])
    return ALL_METHODS_TABLE_ROOT


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
        max_ylim = float(totals.max()) * 1.12 if len(totals) else 1.0
        ax.set_ylim(0, max_ylim if max_ylim > 0.0 else 1.0)
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
    args = parser.parse_args(argv)

    manifest, _ = load_top_manifest_from_table_root(ALL_METHODS_TABLE_ROOT, args.manifest)
    output_root = resolve_output_root(args.manifest)
    all_rows, _ = prepare_breakdown_tables(manifest, output_root)
    if not all_rows:
        all_rows = load_csv_rows(output_root / 'BREAKDOWN_ALL_METHODS.csv')

    payload = build_payload(all_rows)
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    panel_paths = draw_panels(payload)
    compose_grid_figure(panel_paths, output_paths=[FIG_ALL_METHODS_PNG, FIG_ALL_METHODS_SVG], rows=1, cols=4, figsize=(20.0, 5.0), legend_path=None, dpi=200)
    fill_sampling_training_template(FIG_ALL_METHODS, panel_paths)
    print(f"Saved JSON: {JSON_OUTPUT}")
    print(f"Saved PDF: {FIG_ALL_METHODS}")
    print(f"Saved PNG: {FIG_ALL_METHODS_PNG}")
    print(f"Saved SVG: {FIG_ALL_METHODS_SVG}")


if __name__ == '__main__':
    main()
