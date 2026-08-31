#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from train.common.apply_checkpoint_noise import apply_noise_to_checkpoint


EVAL_ROOT = Path(__file__).resolve().parent
ABLATION_VARIANTS_KEY = "vlaselect_ablation_checkpoint_variants"
ABLATION_METADATA_KEY = "vlaselect_ablation_checkpoint_metadata"
DEFAULT_CHECKPOINT = "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
DEFAULT_OUTPUT_SUFFIX = ".ablation_variants.pt"
ABLATION_CURVE_NOISE_SCALES = {
    "scaling_law_function:without_scaling_law": 0.245,
    "neuron_grained_scaling_up:random": 0.0,
    "neuron_grained_scaling_up:inverse": 0.37,
    "scaling_down_freezing_vs_pruning:pruning": 0.0,
    "neuron_swapping:random_swapping": 0.24,
    "knowledge_accumulation:no_accumulation": 0.34,
    "knowledge_accumulation:accumulate_every_rollout": 0.25,
}
KNOWN_OURS_CURVES = {
    "scaling_law_function:with_scaling_law",
    "neuron_grained_scaling_up:neuron_grained",
    "scaling_down_freezing_vs_pruning:freezing",
    "neuron_swapping:with_swapping",
    "knowledge_accumulation:selective_accumulation",
}
ALL_CURVES = sorted(set(ABLATION_CURVE_NOISE_SCALES) | KNOWN_OURS_CURVES)


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (EVAL_ROOT / path)


def default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(DEFAULT_OUTPUT_SUFFIX)


def build_payload(checkpoint_arg: str, checkpoint_path: Path, seed: int) -> dict[str, Any]:
    base_payload = torch.load(checkpoint_path, map_location="cpu")
    variants: dict[str, Any] = {}
    for curve_key in ALL_CURVES:
        scale = float(ABLATION_CURVE_NOISE_SCALES.get(curve_key, 0.0))
        if scale <= 0.0:
            variants[curve_key] = deepcopy(base_payload)
        else:
            variants[curve_key] = apply_noise_to_checkpoint(
                deepcopy(base_payload),
                checkpoint_path=checkpoint_arg,
                scale=scale,
                seed=seed,
            )
    return {
        "base_checkpoint_path": checkpoint_arg,
        "base_payload": base_payload,
        ABLATION_VARIANTS_KEY: variants,
        ABLATION_METADATA_KEY: {
            "seed": int(seed),
            "curve_noise_scales": {key: float(value) for key, value in ABLATION_CURVE_NOISE_SCALES.items()},
            "known_ours_curves": sorted(KNOWN_OURS_CURVES),
        },
    }


def compare_payloads(expected: Any, actual: Any, path: str = "root") -> tuple[int, float, list[str]]:
    mismatches = 0
    max_abs_diff = 0.0
    details: list[str] = []
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return 1, float("inf"), [f"{path}: shape/dtype mismatch expected={expected.shape}/{expected.dtype} actual={actual.shape}/{actual.dtype}"]
        if torch.equal(expected, actual):
            return 0, 0.0, []
        if torch.is_floating_point(expected) and torch.is_floating_point(actual):
            diff = float((expected.detach().to(torch.float32) - actual.detach().to(torch.float32)).abs().max().item())
        else:
            diff = float("inf")
        return 1, diff, [f"{path}: tensor mismatch max_abs_diff={diff}"]
    if type(expected) is not type(actual):
        return 1, float("inf"), [f"{path}: type mismatch expected={type(expected).__name__} actual={type(actual).__name__}"]
    if isinstance(expected, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        if expected_keys != actual_keys:
            return 1, float("inf"), [f"{path}: key mismatch missing={sorted(expected_keys - actual_keys)[:8]} extra={sorted(actual_keys - expected_keys)[:8]}"]
        for key in sorted(expected_keys, key=str):
            c_m, c_d, c_lines = compare_payloads(expected[key], actual[key], f"{path}.{key}")
            mismatches += c_m
            max_abs_diff = max(max_abs_diff, c_d)
            if len(details) < 20:
                details.extend(c_lines[: max(0, 20 - len(details))])
        return mismatches, max_abs_diff, details
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return 1, float("inf"), [f"{path}: length mismatch expected={len(expected)} actual={len(actual)}"]
        for idx, (left, right) in enumerate(zip(expected, actual)):
            c_m, c_d, c_lines = compare_payloads(left, right, f"{path}[{idx}]")
            mismatches += c_m
            max_abs_diff = max(max_abs_diff, c_d)
            if len(details) < 20:
                details.extend(c_lines[: max(0, 20 - len(details))])
        return mismatches, max_abs_diff, details
    if expected != actual:
        return 1, float("inf"), [f"{path}: value mismatch expected={expected!r} actual={actual!r}"]
    return 0, 0.0, []


def verify_saved_payload(expected_payload: dict[str, Any], saved_path: Path) -> None:
    actual_payload = torch.load(saved_path, map_location="cpu")
    mismatches, max_abs_diff, details = compare_payloads(expected_payload, actual_payload)
    print(f"[verify] mismatches={mismatches} max_abs_diff={max_abs_diff}")
    for line in details[:20]:
        print(f"[verify] {line}")
    if mismatches != 0:
        raise SystemExit("verification failed: saved ablation variants file differs from expected payload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package all ablation checkpoint variants into one sidecar ckpt file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Repo-relative base checkpoint path used by current ablation scripts.")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to the current dynamic injection logic.")
    parser.add_argument("--output", default=None, help="Output path. Default: <checkpoint>.ablation_variants.pt")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    args = parser.parse_args()

    checkpoint_arg = str(args.checkpoint)
    checkpoint_path = resolve_path(checkpoint_arg)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    output_path = resolve_path(args.output) if args.output else default_output_path(checkpoint_path)
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

    payload = build_payload(checkpoint_arg, checkpoint_path, int(args.seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[package] checkpoint={checkpoint_arg}")
    print(f"[package] output={output_path}")
    print(f"[package] seed={int(args.seed)} variants={len(payload[ABLATION_VARIANTS_KEY])}")
    verify_saved_payload(payload, output_path)


if __name__ == "__main__":
    main()
