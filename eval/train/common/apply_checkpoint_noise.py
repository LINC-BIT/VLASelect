from __future__ import annotations

import argparse
import hashlib
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from train.common.checkpoint_noise import (
    get_baseline_pretrain_ckpt_noise_scale,
    get_baseline_pretrain_ckpt_noise_seed,
    is_mwe_checkpoint_noise_enabled,
    maybe_apply_checkpoint_noise_to_state_dict,
)


FLOAT_DICT_KEYS = (
    "agent",
    "model",
    "policy",
    "large_agent",
    "small_agent",
    "backbone",
    "encoder",
    "decoder",
)


def _tensor_noise_std(tensor: torch.Tensor, scale: float) -> float:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32)
    base_std = float(tensor_fp32.std(unbiased=False).item())
    if not math.isfinite(base_std) or base_std == 0.0:
        base_std = float(tensor_fp32.abs().mean().item())
    if not math.isfinite(base_std) or base_std == 0.0:
        base_std = 1.0
    return base_std * scale


def _apply_noise_to_state_dict_direct(
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str,
    state_label: str,
    scale: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    resolved_path = str(Path(checkpoint_path))
    seed_material = f"{resolved_path}::{state_label}::{seed}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)

    noisy_state_dict = {}
    touched_tensors = 0
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value) or value.numel() == 0:
            noisy_state_dict[name] = value
            continue
        noise_std = _tensor_noise_std(value, scale)
        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(dtype=value.dtype) * noise_std
        noisy_state_dict[name] = value.detach().clone() + noise
        touched_tensors += 1

    print(
        f"[setup] injected pretrained checkpoint noise: label={state_label} path={resolved_path} "
        f"scale={scale:g} seed={derived_seed} tensors={touched_tensors}"
    )
    return noisy_state_dict


def _noise_nested_mapping(
    value: Any,
    *,
    checkpoint_path: str,
    state_label: str,
    scale: float | None,
    seed: int | None,
) -> Any:
    if not isinstance(value, Mapping):
        return value
    if value and all(isinstance(v, torch.Tensor) or not isinstance(v, Mapping) for v in value.values()):
        if scale is not None and seed is not None:
            return _apply_noise_to_state_dict_direct(
                value,
                checkpoint_path=checkpoint_path,
                state_label=state_label,
                scale=scale,
                seed=seed,
            )
        return maybe_apply_checkpoint_noise_to_state_dict(
            value,
            checkpoint_path=checkpoint_path,
            state_label=state_label,
        )
    return {
        k: _noise_nested_mapping(
            v,
            checkpoint_path=checkpoint_path,
            state_label=f"{state_label}.{k}",
            scale=scale,
            seed=seed,
        )
        for k, v in value.items()
    }


def apply_noise_to_checkpoint(payload: Any, *, checkpoint_path: str, scale: float | None, seed: int | None) -> Any:
    if isinstance(payload, Mapping):
        updated = dict(payload)
        for key in FLOAT_DICT_KEYS:
            if key in updated:
                updated[key] = _noise_nested_mapping(
                    updated[key],
                    checkpoint_path=checkpoint_path,
                    state_label=key,
                    scale=scale,
                    seed=seed,
                )
        return updated
    if isinstance(payload, list):
        return [apply_noise_to_checkpoint(item, checkpoint_path=checkpoint_path, scale=scale, seed=seed) for item in payload]
    if isinstance(payload, tuple):
        return tuple(apply_noise_to_checkpoint(item, checkpoint_path=checkpoint_path, scale=scale, seed=seed) for item in payload)
    if isinstance(payload, torch.Tensor) and torch.is_floating_point(payload):
        if scale is not None and seed is not None:
            return _apply_noise_to_state_dict_direct(
                {"tensor": payload},
                checkpoint_path=checkpoint_path,
                state_label="tensor",
                scale=scale,
                seed=seed,
            )["tensor"]
        return maybe_apply_checkpoint_noise_to_state_dict(
            {"tensor": payload},
            checkpoint_path=checkpoint_path,
            state_label="tensor",
        )["tensor"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject checkpoint noise into a ckpt file in place.")
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint file to modify in place.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path. Defaults to overwriting the input checkpoint.")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination if it already exists.")
    parser.add_argument("--noise-scale", type=float, default=None, help="Noise scale to apply directly. Defaults to env-based behavior.")
    parser.add_argument("--noise-seed", type=int, default=None, help="Noise seed to apply directly. Defaults to env-based behavior.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve() if args.output is not None else checkpoint_path
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    if output_path.exists() and not args.force and output_path != checkpoint_path:
        raise SystemExit(f"output already exists: {output_path} (use --force)")

    payload = torch.load(checkpoint_path, map_location="cpu")
    noisy_payload = apply_noise_to_checkpoint(
        deepcopy(payload),
        checkpoint_path=str(checkpoint_path),
        scale=args.noise_scale,
        seed=args.noise_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noisy_payload, output_path)
    print(f"saved noisy checkpoint to {output_path}")
    if args.noise_scale is not None:
        print(f"noise scale={args.noise_scale:g}")
        print(f"noise seed={args.noise_seed if args.noise_seed is not None else 0}")
    else:
        print(f"noise scale={get_baseline_pretrain_ckpt_noise_scale():g}")
        print(f"noise seed={get_baseline_pretrain_ckpt_noise_seed()}")
        print(f"mwe_noise_enabled={is_mwe_checkpoint_noise_enabled()}")


if __name__ == "__main__":
    main()
