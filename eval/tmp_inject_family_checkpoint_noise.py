#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from train.common.apply_checkpoint_noise import apply_noise_to_checkpoint


FAMILY_CONFIG = {
    "octo": {
        "checkpoint": "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt",
        "scale": 0.12,
    },
    "vla_adapter_new": {
        "checkpoint": "ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt",
        "scale": 0.35,
    },
    "tinyvla": {
        "checkpoint": "ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt",
        "scale": 0.12,
    },
    "edgevla": {
        "checkpoint": "ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt",
        "scale": 0.32,
    },
}


EVAL_ROOT = Path(__file__).resolve().parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (EVAL_ROOT / path)


def default_output_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix:
        return checkpoint_path.with_suffix(f".permanent-noise{checkpoint_path.suffix}")
    return checkpoint_path.with_name(checkpoint_path.name + ".permanent-noise")


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
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            details.append(f"{path}: key mismatch missing={missing[:8]} extra={extra[:8]}")
            mismatches += 1
        for key in sorted(expected_keys & actual_keys, key=str):
            child_mismatches, child_max, child_details = compare_payloads(expected[key], actual[key], f"{path}.{key}")
            mismatches += child_mismatches
            max_abs_diff = max(max_abs_diff, child_max)
            if len(details) < 20:
                details.extend(child_details[: max(0, 20 - len(details))])
        return mismatches, max_abs_diff, details

    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return 1, float("inf"), [f"{path}: length mismatch expected={len(expected)} actual={len(actual)}"]
        for idx, (left, right) in enumerate(zip(expected, actual)):
            child_mismatches, child_max, child_details = compare_payloads(left, right, f"{path}[{idx}]")
            mismatches += child_mismatches
            max_abs_diff = max(max_abs_diff, child_max)
            if len(details) < 20:
                details.extend(child_details[: max(0, 20 - len(details))])
        return mismatches, max_abs_diff, details

    if expected != actual:
        return 1, float("inf"), [f"{path}: value mismatch expected={expected!r} actual={actual!r}"]
    return 0, 0.0, []


def save_payload(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def inject_checkpoint(
    *,
    family: str,
    checkpoint_arg: str,
    checkpoint_path: Path,
    output_path: Path,
    scale: float,
    seed: int,
) -> None:
    original_payload = torch.load(checkpoint_path, map_location="cpu")
    noisy_payload = apply_noise_to_checkpoint(
        deepcopy(original_payload),
        checkpoint_path=checkpoint_arg,
        scale=scale,
        seed=seed,
    )
    save_payload(noisy_payload, output_path)
    reloaded_payload = torch.load(output_path, map_location="cpu")
    mismatches, max_abs_diff, details = compare_payloads(noisy_payload, reloaded_payload)
    print(f"[inject] family={family} checkpoint={checkpoint_arg}")
    print(f"[inject] scale={scale:g} seed={seed}")
    print(f"[inject] output={output_path}")
    print(f"[verify] mismatches={mismatches} max_abs_diff={max_abs_diff}")
    if details:
        for line in details[:20]:
            print(f"[verify] {line}")
    if mismatches != 0:
        raise SystemExit("verification failed: saved checkpoint differs from dynamic injection result")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Permanently inject the same pretrained-checkpoint noise used by overhead_same_acc. "
            "Use the same repo-relative checkpoint path string as the current scripts, because it participates in seed derivation. "
            "If --family is omitted, all configured families are processed in place."
        )
    )
    parser.add_argument("--family", choices=sorted(FAMILY_CONFIG.keys()), default=None)
    parser.add_argument("--checkpoint", default=None, help="Repo-relative checkpoint path. Defaults to the family's shared pretrained checkpoint.")
    parser.add_argument("--scale", type=float, default=None, help="Override noise scale. Defaults to the family's configured scale.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed passed to the current code-injection logic. Default: 0.")
    parser.add_argument("--output", default=None, help="Output path. Default: sibling *.permanent-noise.* file.")
    parser.add_argument("--no-inplace", action="store_true", help="Write sibling permanent-noise files instead of replacing the original checkpoints.")
    parser.add_argument("--backup-suffix", default=".before_permanent_noise", help="Backup suffix used only with --inplace.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the output path if it already exists.")
    args = parser.parse_args()

    selected_families = [args.family] if args.family else list(FAMILY_CONFIG.keys())
    if len(selected_families) > 1 and args.checkpoint is not None:
        raise SystemExit("--checkpoint can only be used when a single --family is selected")
    if len(selected_families) > 1 and args.output is not None:
        raise SystemExit("--output can only be used when a single --family is selected")

    inplace = not args.no_inplace

    for family in selected_families:
        config = FAMILY_CONFIG[family]
        checkpoint_arg = args.checkpoint or str(config["checkpoint"])
        scale = float(config["scale"] if args.scale is None else args.scale)
        checkpoint_path = resolve_path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")

        if inplace:
            fd, temp_name = tempfile.mkstemp(prefix=checkpoint_path.stem + ".permanent-noise.", suffix=checkpoint_path.suffix, dir=str(checkpoint_path.parent))
            Path(temp_name).unlink(missing_ok=True)
            output_path = Path(temp_name)
        else:
            output_arg = args.output
            output_path = resolve_path(output_arg) if output_arg else default_output_path(checkpoint_path)
            if output_path.exists() and not args.force:
                raise SystemExit(f"output already exists: {output_path} (use --force to overwrite)")

        inject_checkpoint(
            family=family,
            checkpoint_arg=checkpoint_arg,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            scale=scale,
            seed=int(args.seed),
        )

        if inplace:
            backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + args.backup_suffix)
            if backup_path.exists():
                raise SystemExit(f"backup already exists: {backup_path}")
            shutil.move(str(checkpoint_path), str(backup_path))
            shutil.move(str(output_path), str(checkpoint_path))
            print(f"[inplace] backup={backup_path}")
            print(f"[inplace] replaced={checkpoint_path}")


if __name__ == "__main__":
    main()
