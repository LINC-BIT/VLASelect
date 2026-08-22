from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from train.octo.ours_single_agent.motivation.training_lib import (
    LATEST_RUN_FILES,
    WORKLOAD_CHANGE_TIME_POINTS,
    WORKLOAD_ENVS,
)


BASE_OUTPUT_DIR = Path("train/octo/ours_single_agent/motivation/res_images")


def load_run_args(run_dir: Path) -> Dict[str, str]:
    args_path = run_dir / "code" / "args.txt"
    if not args_path.exists():
        return {}

    result: Dict[str, str] = {}
    with open(args_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ": " not in line:
                continue
            key, value = line.rstrip("\n").split(": ", 1)
            result[key] = value
    return result


def load_latest_run_dir(mode: str) -> Optional[Path]:
    manifest_path = LATEST_RUN_FILES[mode]
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        run_dir = Path(payload["run_dir"])
        if run_dir.exists():
            return run_dir

    fallback_root = Path("ckpt") / WORKLOAD_ENVS[0] / "ours" / "octo" / f"{mode}_model"
    if not fallback_root.exists():
        return None
    candidates = sorted(
        [path for path in fallback_root.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def snapshot_paths_for_mode(mode: str) -> Dict[int, Path]:
    run_dir = load_latest_run_dir(mode)
    if run_dir is None:
        return {}
    importance_dir = run_dir / "motivation_neuron_importance"
    if not importance_dir.exists():
        return {}

    result: Dict[int, Path] = {}
    for path in importance_dir.glob("iter_*.pt"):
        try:
            iteration = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        result[iteration] = path
    return result


def load_env_schedule_for_mode(mode: str) -> tuple[List[str], List[float]]:
    run_dir = load_latest_run_dir(mode)
    if run_dir is None:
        return WORKLOAD_ENVS, [float(value) for value in WORKLOAD_CHANGE_TIME_POINTS]

    run_args = load_run_args(run_dir)
    env_ids = WORKLOAD_ENVS
    change_time_points: List[float] = [float(value) for value in WORKLOAD_CHANGE_TIME_POINTS]

    if "envs_id" in run_args:
        try:
            env_ids = list(ast.literal_eval(run_args["envs_id"]))
        except (SyntaxError, ValueError):
            env_ids = WORKLOAD_ENVS

    if "env_change_time_points" in run_args:
        try:
            change_time_points = [float(value) for value in ast.literal_eval(run_args["env_change_time_points"])]
        except (SyntaxError, ValueError, TypeError):
            change_time_points = [float(value) for value in WORKLOAD_CHANGE_TIME_POINTS]

    return env_ids, change_time_points


def snapshot_to_matrix(payload: Dict[str, object]) -> np.ndarray:
    matrix = np.full((8, 8), np.nan, dtype=np.float32)
    values = payload.get("values", [])
    for layer_index, neurons in enumerate(values[1:8]):
        for neuron_index, value in enumerate(neurons[:8]):
            matrix[layer_index, neuron_index] = float(value)
    return matrix


def resolve_env_id(
    env_ids: List[str],
    change_time_points: List[float],
    elapsed_minutes: float,
) -> str:
    if not env_ids:
        return WORKLOAD_ENVS[0]
    for env_index, switch_time in enumerate(change_time_points):
        if elapsed_minutes <= switch_time:
            return env_ids[min(env_index, len(env_ids) - 1)]
    return env_ids[-1]


def resolve_shared_normalize(*matrices: np.ndarray) -> colors.Normalize:
    finite_slices = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    if not finite_slices:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    finite_values = np.concatenate(finite_slices)

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.05, 1e-6)
        vmin -= padding
        vmax += padding
    return colors.Normalize(vmin=vmin, vmax=vmax)


def render_comparison_heatmap(
    iteration: int,
    original_env_id: str,
    original_payload: Dict[str, object],
    small_env_id: str,
    small_payload: Dict[str, object],
) -> None:
    original_matrix = snapshot_to_matrix(original_payload)
    small_matrix = snapshot_to_matrix(small_payload)
    norm = resolve_shared_normalize(original_matrix, small_matrix)

    output_dir = BASE_OUTPUT_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_env_id = original_env_id if original_env_id == small_env_id else f"{original_env_id}__{small_env_id}"
    output_path = output_dir / f"{output_env_id}-{iteration}.png"

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#d9d9d9")
    original_masked = np.ma.masked_invalid(original_matrix)
    small_masked = np.ma.masked_invalid(small_matrix)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    plot_specs = (
        ("small_model", original_env_id, original_masked),
        ("original_model", small_env_id, small_masked),
    )

    image = None
    for ax, (mode, env_id, masked) in zip(axes, plot_specs):
        image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{mode}\n{env_id}")
        ax.set_xlabel("Neuron Index")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

    axes[0].set_ylabel("Layer Index")
    fig.suptitle(f"Neuron importance comparison iteration={iteration}")
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved comparison heatmap to {output_path}")


def render_all_heatmaps() -> None:
    original_snapshot_paths = snapshot_paths_for_mode("original")
    small_snapshot_paths = snapshot_paths_for_mode("small")
    if not original_snapshot_paths or not small_snapshot_paths:
        return

    original_env_ids, original_change_time_points = load_env_schedule_for_mode("original")
    small_env_ids, small_change_time_points = load_env_schedule_for_mode("small")

    shared_iterations = sorted(set(original_snapshot_paths).intersection(small_snapshot_paths))
    for iteration in shared_iterations:
        if iteration % 5 != 0:
            continue

        original_payload = torch.load(original_snapshot_paths[iteration], map_location="cpu")
        small_payload = torch.load(small_snapshot_paths[iteration], map_location="cpu")

        original_elapsed_minutes = float(original_payload.get("elapsed_minutes", 0.0))
        small_elapsed_minutes = float(small_payload.get("elapsed_minutes", 0.0))
        original_env_id = resolve_env_id(
            original_env_ids,
            original_change_time_points,
            original_elapsed_minutes,
        )
        small_env_id = resolve_env_id(
            small_env_ids,
            small_change_time_points,
            small_elapsed_minutes,
        )
        render_comparison_heatmap(
            iteration,
            original_env_id,
            original_payload,
            small_env_id,
            small_payload,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()

    while True:
        render_all_heatmaps()
        if not args.watch:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
