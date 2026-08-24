from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Mapping

import torch

NOISE_SCALE_ENV = "VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE"
NOISE_SEED_ENV = "VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED"
MWE_NOISE_GATE_ENVS = ("MWE_ACTIVE_RUNTIME_ONLY", "MWE", "SMOKE", "RUN_SETUP_SMOKE")


def _env_flag(name: str) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_mwe_checkpoint_noise_enabled() -> bool:
    return any(_env_flag(name) for name in MWE_NOISE_GATE_ENVS)


def get_baseline_pretrain_ckpt_noise_scale() -> float:
    default_scale = "0.35" if is_mwe_checkpoint_noise_enabled() else "0.0"
    raw_value = os.environ.get(NOISE_SCALE_ENV, default_scale)
    try:
        scale = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{NOISE_SCALE_ENV} must be a float, got {raw_value!r}") from exc
    if scale < 0:
        raise ValueError(f"{NOISE_SCALE_ENV} must be non-negative, got {scale}")
    return scale


def get_baseline_pretrain_ckpt_noise_seed() -> int:
    raw_value = os.environ.get(NOISE_SEED_ENV, "0")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{NOISE_SEED_ENV} must be an integer, got {raw_value!r}") from exc


def _tensor_noise_std(tensor: torch.Tensor, scale: float) -> float:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32)
    base_std = float(tensor_fp32.std(unbiased=False).item())
    if not math.isfinite(base_std) or base_std == 0.0:
        base_std = float(tensor_fp32.abs().mean().item())
    if not math.isfinite(base_std) or base_std == 0.0:
        base_std = 1.0
    return base_std * scale


def maybe_apply_checkpoint_noise_to_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str | Path,
    state_label: str,
):
    if not is_mwe_checkpoint_noise_enabled():
        return state_dict

    scale = get_baseline_pretrain_ckpt_noise_scale()
    if scale <= 0.0:
        return state_dict

    resolved_path = str(Path(checkpoint_path))
    seed_material = f"{resolved_path}::{state_label}::{get_baseline_pretrain_ckpt_noise_seed()}".encode("utf-8")
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
