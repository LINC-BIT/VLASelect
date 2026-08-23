from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
from common.figure_compose import compose_grid_figure, render_legend_image
from common.template_pdf_fill import fill_ours_overhead_template
TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_table"
ALL_METHODS_TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_all_methods_table"
MODULES_TABLE_ROOT = SCRIPT_DIR / "overhead_breakdown_modules_table"
FIG_ALL_METHODS = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.pdf"
FIG_ALL_METHODS_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.svg"
FIG_ALL_METHODS_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS.png"
FIG_MODULES = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.pdf"
FIG_MODULES_SVG = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.svg"
FIG_MODULES_PNG = SCRIPT_DIR / "FIG_BREAKDOWN_MODULES.png"
ALL_METHODS_PANEL_DIR = SCRIPT_DIR / "FIG_BREAKDOWN_ALL_METHODS_panels"

FAMILY_ORDER = ["octo", "vla_adapter_new", "tinyvla", "edgevla"]
PANEL_LABELS = {
    "octo": "a",
    "vla_adapter_new": "b",
    "tinyvla": "c",
    "edgevla": "d",
}
WORKLOAD_NAMES = {
    "octo": "Single-arm robot",
    "vla_adapter_new": "Dexterous hand",
    "tinyvla": "Mobile manipulator",
    "edgevla": "Humanoid robot",
}
FAMILY_DISPLAY_NAMES = {
    "octo": "Octo",
    "vla_adapter_new": "VLA-Adapter",
    "tinyvla": "TinyVLA",
    "edgevla": "EdgeVLA",
}
EXPECTED_METHODS = {
    "octo": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours_single_agent", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "vla_adapter_new": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("edgeta", "EdgeTA"),
        ("convertnet", "ConvertNet"),
        ("ours", "VLASelect"),
        ("ppo_gen", "PPO-Gen"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "tinyvla": [
        ("ppo_gen", "PPO-Gen"),
        ("ours", "VLASelect"),
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("self_improv", "Self-Improv"),
        ("vla_rft", "VLA-RFT"),
        ("world_env", "WorldEnv"),
    ],
    "edgevla": [
        ("conrft", "ConRFT"),
        ("flare", "FLaRe"),
        ("improv_vla", "Improv-VLA"),
        ("ppo_gen", "PPO-Gen"),
    ],
}
MODULE_SPECS = [
    ("workload_initialization_seconds", "Workload init", "#4C78A8"),
    (
        "optimal_network_search_and_selective_model_enhancement_seconds",
        "Net search + SME",
        "#F58518",
    ),
    ("selective_knowledge_accumulation_seconds", "SKA", "#54A24B"),
    ("online_rl_completion_seconds", "Online RL", "#E45756"),
]
NO_DATA_TEXT = "No data"


def available_sans_serif_fonts() -> list[str]:
    candidates = ["Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans", "sans-serif"]
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    matched = [name for name in candidates if name in installed]
    return matched or ["DejaVu Sans", "Liberation Sans", "sans-serif"]


@dataclass
class MethodBreakdown:
    sampling_seconds: float = 0.0
    training_seconds: float = 0.0
    has_data: bool = False
    source: str = ""
    module_breakdown: dict[str, float] | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_module_breakdown(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("module_breakdown"),
        payload.get("modules"),
        payload.get("vlaselect_module_breakdown"),
        payload.get("time_breakdown", {}).get("module_breakdown") if isinstance(payload.get("time_breakdown"), dict) else None,
        payload.get("breakdown", {}).get("module_breakdown") if isinstance(payload.get("breakdown"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {key: _safe_float(candidate.get(key, 0.0)) for key, _, _ in MODULE_SPECS}
    if any(key in payload for key, _, _ in MODULE_SPECS):
        return {key: _safe_float(payload.get(key, 0.0)) for key, _, _ in MODULE_SPECS}
    return {key: 0.0 for key, _, _ in MODULE_SPECS}


def _extract_sampling_training(payload: dict[str, Any]) -> tuple[float, float, bool]:
    for container_key in ("time_breakdown", "breakdown", "timing_breakdown"):
        candidate = payload.get(container_key)
        if isinstance(candidate, dict):
            sampling = _safe_float(
                candidate.get("sampling_seconds", candidate.get("rollout_seconds", candidate.get("sampling_time_seconds", 0.0)))
            )
            training = _safe_float(
                candidate.get("training_seconds", candidate.get("update_seconds", candidate.get("training_time_seconds", 0.0)))
            )
            return sampling, training, sampling > 0.0 or training > 0.0
    sampling = _safe_float(payload.get("sampling_seconds", payload.get("rollout_seconds", 0.0)))
    training = _safe_float(payload.get("training_seconds", payload.get("update_seconds", 0.0)))
    return sampling, training, sampling > 0.0 or training > 0.0


def _candidate_breakdown_paths(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / "time_breakdown.json",
        run_dir / "timing_breakdown.json",
        run_dir / "breakdown.json",
    ]
    candidates.extend(sorted(run_dir.glob("*training_summary.json")))
    candidates.extend(sorted((run_dir / "analysis").glob("*breakdown*.json")) if (run_dir / "analysis").exists() else [])
    return [path for path in candidates if path.exists()]


def load_method_breakdown(run_dir: Path) -> MethodBreakdown:
    for candidate in _candidate_breakdown_paths(run_dir):
        payload = _read_json(candidate)
        if not isinstance(payload, dict):
            continue
        sampling, training, has_data = _extract_sampling_training(payload)
        module_breakdown = _extract_module_breakdown(payload)
        if has_data or any(value > 0.0 for value in module_breakdown.values()):
            return MethodBreakdown(
                sampling_seconds=sampling,
                training_seconds=training,
                has_data=True,
                source=str(candidate.relative_to(EVAL_ROOT)),
                module_breakdown=module_breakdown,
            )
    return MethodBreakdown(module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS})


def _default_top_manifest() -> dict[str, Any]:
    panels = []
    for family in FAMILY_ORDER:
        panels.append(
            {
                "family": family,
                "suite_manifest": "",
                "suite_root": "",
                "launch_log": "",
                "suite_stamp": "no-data",
                "panel_label": PANEL_LABELS[family],
                "workload_name": WORKLOAD_NAMES[family],
                "display_name": FAMILY_DISPLAY_NAMES[family],
            }
        )
    return {
        "suite_stamp": "no-data",
        "table_root": "overhead/overhead_breakdown_table",
        "figure_all_methods_output": "overhead/FIG_BREAKDOWN_ALL_METHODS.pdf",
        "figure_modules_output": "overhead/FIG_BREAKDOWN_MODULES.pdf",
        "all_methods_csv": "overhead/overhead_breakdown_table/BREAKDOWN_ALL_METHODS.csv",
        "modules_csv": "overhead/overhead_breakdown_table/BREAKDOWN_MODULES.csv",
        "panels": panels,
        "families": panels,
    }


def load_top_manifest_from_table_root(
    table_root: Path,
    manifest_path: str | None,
) -> tuple[dict[str, Any], Path | None]:
    if manifest_path:
        path = Path(manifest_path)
        payload = _read_json(path)
        return (payload if isinstance(payload, dict) else _default_top_manifest(), path)
    latest_path = table_root / "latest.txt"
    if latest_path.exists():
        stamp = latest_path.read_text(encoding="utf-8").strip()
        if stamp:
            candidate = table_root / stamp / "manifest.json"
            payload = _read_json(candidate)
            if isinstance(payload, dict):
                return payload, candidate
    return _default_top_manifest(), None


def load_top_manifest(manifest_path: str | None) -> tuple[dict[str, Any], Path | None]:
    return load_top_manifest_from_table_root(TABLE_ROOT, manifest_path)


def _family_panels(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("families", manifest.get("panels", [])):
        family = item.get("family")
        if family:
            result[family] = item
    for family in FAMILY_ORDER:
        result.setdefault(
            family,
            {
                "family": family,
                "panel_label": PANEL_LABELS[family],
                "workload_name": WORKLOAD_NAMES[family],
                "display_name": FAMILY_DISPLAY_NAMES[family],
                "suite_manifest": "",
                "suite_root": "",
                "launch_log": "",
            },
        )
    return result


def _load_suite_manifest(panel: dict[str, Any]) -> dict[str, Any] | None:
    manifest_ref = panel.get("suite_manifest") or ""
    if not manifest_ref:
        return None
    manifest_path = EVAL_ROOT / manifest_ref
    payload = _read_json(manifest_path)
    return payload if isinstance(payload, dict) else None


def _expected_method_entries(family: str) -> list[dict[str, Any]]:
    return [
        {"name": name, "display_name": display_name, "run_dir": ""}
        for name, display_name in EXPECTED_METHODS[family]
    ]


def _normalize_display_name(name: str, display_name: str) -> str:
    if name in {"ours", "ours_single_agent"}:
        return "VLASelect"
    return display_name or name


def prepare_breakdown_tables(manifest: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []

    for family, panel in _family_panels(manifest).items():
        suite_manifest = _load_suite_manifest(panel)
        methods = suite_manifest.get("methods", []) if isinstance(suite_manifest, dict) else []
        if not methods:
            methods = _expected_method_entries(family)

        suite_map = {method.get("name"): method for method in methods if method.get("name")}
        ordered_methods: list[dict[str, Any]] = []
        for name, display_name in EXPECTED_METHODS[family]:
            method = dict(suite_map.get(name, {}))
            method.setdefault("name", name)
            method.setdefault("display_name", display_name)
            method.setdefault("run_dir", "")
            ordered_methods.append(method)

        family_module_row_added = False
        for method in ordered_methods:
            run_dir_ref = method.get("run_dir") or ""
            run_dir = EVAL_ROOT / run_dir_ref if run_dir_ref else Path("")
            breakdown = load_method_breakdown(run_dir) if run_dir_ref else MethodBreakdown(module_breakdown={key: 0.0 for key, _, _ in MODULE_SPECS})
            display_name = _normalize_display_name(method["name"], method.get("display_name", ""))
            all_rows.append(
                {
                    "family": family,
                    "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                    "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                    "method_name": method["name"],
                    "display_name": display_name,
                    "sampling_seconds": breakdown.sampling_seconds,
                    "training_seconds": breakdown.training_seconds,
                    "total_seconds": breakdown.sampling_seconds + breakdown.training_seconds,
                    "has_breakdown_data": int(breakdown.has_data),
                    "source": breakdown.source,
                }
            )
            if method["name"] in {"ours", "ours_single_agent"}:
                module_breakdown = breakdown.module_breakdown or {key: 0.0 for key, _, _ in MODULE_SPECS}
                module_rows.append(
                    {
                        "family": family,
                        "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                        "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                        "display_name": display_name,
                        **module_breakdown,
                        "total_seconds": sum(module_breakdown.values()),
                        "has_module_data": int(any(value > 0.0 for value in module_breakdown.values())),
                        "source": breakdown.source,
                    }
                )
                family_module_row_added = True

        if not family_module_row_added:
            module_breakdown = {key: 0.0 for key, _, _ in MODULE_SPECS}
            module_rows.append(
                {
                    "family": family,
                    "panel_label": panel.get("panel_label", PANEL_LABELS[family]),
                    "workload_name": panel.get("workload_name", WORKLOAD_NAMES[family]),
                    "display_name": "VLASelect",
                    **module_breakdown,
                    "total_seconds": 0.0,
                    "has_module_data": 0,
                    "source": "",
                }
            )

    write_csv(
        output_root / "BREAKDOWN_ALL_METHODS.csv",
        [
            "family",
            "panel_label",
            "workload_name",
            "method_name",
            "display_name",
            "sampling_seconds",
            "training_seconds",
            "total_seconds",
            "has_breakdown_data",
            "source",
        ],
        all_rows,
    )
    write_csv(
        output_root / "BREAKDOWN_MODULES.csv",
        [
            "family",
            "panel_label",
            "workload_name",
            "display_name",
            *[key for key, _, _ in MODULE_SPECS],
            "total_seconds",
            "has_module_data",
            "source",
        ],
        module_rows,
    )
    (output_root / "breakdown_summary.json").write_text(
        json.dumps(
            {
                "all_methods_rows": all_rows,
                "module_rows": module_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return all_rows, module_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _set_common_axis_style(ax: Any) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_all_methods(rows: list[dict[str, Any]]) -> None:
    grouped = _group_rows(rows, 'family')
    panel_paths: list[Path] = []
    legend_path = ALL_METHODS_PANEL_DIR / 'legend.png'
    render_legend_image([('Sampling', {'color': '#4C78A8', 'linestyle': '-'}), ('Model training', {'color': '#F58518', 'linestyle': '-'})], legend_path, ncol=2, fontsize=11)

    for family in FAMILY_ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.8), constrained_layout=False)
        family_rows = grouped.get(family, [])
        labels = [row['display_name'] for row in family_rows]
        sampling = np.array([_safe_float(row['sampling_seconds']) for row in family_rows], dtype=float)
        training = np.array([_safe_float(row['training_seconds']) for row in family_rows], dtype=float)
        has_data = [int(row.get('has_breakdown_data', 0)) == 1 for row in family_rows]
        x = np.arange(len(labels))

        ax.bar(x, sampling, color='white', edgecolor='black', linewidth=2.0, width=0.72)
        ax.bar(x, training, bottom=sampling, color='white', edgecolor='black', linewidth=2.0, hatch='/', width=0.72)
        ax.set_title(f"({PANEL_LABELS[family]}) {WORKLOAD_NAMES[family]}", fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=42, ha='right', fontsize=9)
        ax.set_ylabel('Time (s)', fontsize=11)
        _set_common_axis_style(ax)

        max_total = float(np.max(sampling + training)) if len(labels) else 0.0
        ax.set_ylim(0.0, max_total * 1.22 if max_total > 0.0 else 1.0)
        for x_pos, ok in zip(x, has_data):
            if not ok:
                ax.text(x_pos, ax.get_ylim()[1] * 0.05, NO_DATA_TEXT, rotation=90, ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        ALL_METHODS_PANEL_DIR.mkdir(parents=True, exist_ok=True)
        panel_path = ALL_METHODS_PANEL_DIR / f'{family}.png'
        fig.savefig(panel_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        panel_paths.append(panel_path)

    compose_grid_figure(
        panel_paths,
        output_paths=[FIG_ALL_METHODS, FIG_ALL_METHODS_SVG, FIG_ALL_METHODS_PNG],
        rows=1,
        cols=4,
        figsize=(20.0, 5.0),
        legend_path=legend_path,
        legend_height_ratio=0.11,
        wspace=0.02,
        hspace=0.02,
        dpi=200,
    )

def plot_modules(rows: list[dict[str, Any]]) -> None:
    grouped = {row["family"]: row for row in rows}
    workloads = [f"Workload {idx + 1}:{WORKLOAD_NAMES[family]}" for idx, family in enumerate(FAMILY_ORDER)]
    y_positions = np.arange(len(workloads))

    module_labels = (
        "Optimal network searcher",
        "Selective model enhancer",
        "Selective knowledge accumulator",
    )
    module_colors = ("#3c3c3c", "#8f8f8f", "#d6d6d6")
    module_hatches = ("", "", "/")
    module_keys = (
        "workload_initialization_seconds",
        "optimal_network_search_and_selective_model_enhancement_seconds",
        "selective_knowledge_accumulation_seconds",
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": available_sans_serif_fonts(),
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "svg.fonttype": "none",
        }
    )

    values = np.array(
        [[_safe_float(grouped.get(family, {}).get(key, 0.0)) for key in module_keys] for family in FAMILY_ORDER],
        dtype=float,
    )
    training_times = np.array(
        [_safe_float(grouped.get(family, {}).get("online_rl_completion_seconds", 0.0)) for family in FAMILY_ORDER],
        dtype=float,
    )

    fig_height = max(3.35, 0.42 * len(workloads) + 1.2)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), constrained_layout=False)

    left = np.zeros(len(FAMILY_ORDER), dtype=float)
    for module_index, (label, color, hatch) in enumerate(zip(module_labels, module_colors, module_hatches)):
        ax.barh(
            y_positions,
            values[:, module_index],
            left=left,
            height=0.5,
            label=label,
            zorder=10,
            edgecolor='black',
            color=color,
            hatch=hatch,
            lw=1,
        )
        left += values[:, module_index]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(workloads, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.grid(zorder=-10, axis='x', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.tick_params(axis='y', length=0)

    if float(np.max(values)) <= 0.0:
        ax.set_xlim(0.0, 1.0)
        for idx, family in enumerate(FAMILY_ORDER):
            row = grouped.get(family)
            if not row or int(row.get("has_module_data", 0)) != 1:
                ax.text(0.02, idx, NO_DATA_TEXT, ha='left', va='center', fontsize=9)
    else:
        totals = np.sum(values, axis=1)
        max_total = float(np.max(totals))
        max_training = float(np.max(training_times)) if len(training_times) else 0.0
        right_padding = max(0.15 * max_total, 0.06 * max_training, 6.0)
        ax.set_xlim(0.0, max_total + right_padding)
        text_offset = max(0.02 * max_total, 1.0)
        for idx, family in enumerate(FAMILY_ORDER):
            row = grouped.get(family)
            if not row or int(row.get("has_module_data", 0)) != 1:
                continue
            total = float(totals[idx])
            training_time = float(training_times[idx])
            ax.text(
                total + text_offset,
                idx,
                f"training time: {training_time:.1f}s",
                ha='left',
                va='center',
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(FIG_MODULES, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_MODULES_SVG, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_MODULES_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    try:
        fill_ours_overhead_template(FIG_MODULES, FIG_MODULES_PNG)
    except Exception as exc:
        print(f'[template] failed to fill VLASelect-overhead template: {exc}')


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(manifest_path: str | None, prepare_only: bool) -> None:
    manifest, resolved_manifest_path = load_top_manifest(manifest_path)
    if resolved_manifest_path is not None:
        output_root = resolved_manifest_path.parent
    elif manifest.get("suite_stamp") not in {None, "", "no-data"}:
        output_root = TABLE_ROOT / str(manifest["suite_stamp"])
    else:
        output_root = TABLE_ROOT

    all_rows, module_rows = prepare_breakdown_tables(manifest, output_root)
    if prepare_only:
        return
    if not all_rows:
        all_rows = load_csv_rows(output_root / "BREAKDOWN_ALL_METHODS.csv")
    if not module_rows:
        module_rows = load_csv_rows(output_root / "BREAKDOWN_MODULES.csv")
    plot_all_methods(all_rows)
    plot_modules(module_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    run(args.manifest, args.prepare_only)


if __name__ == "__main__":
    main()
