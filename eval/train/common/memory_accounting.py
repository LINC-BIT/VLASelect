from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn


def module_parameter_buffer_memory_bytes(module: nn.Module) -> int:
    total_bytes = 0
    for tensor in module.parameters():
        total_bytes += tensor.numel() * tensor.element_size()
    for tensor in module.buffers():
        total_bytes += tensor.numel() * tensor.element_size()
    return int(total_bytes)


def write_memory_exclusion_metadata(
    output_dir: Path,
    *,
    excluded_gpu_memory_mb: float,
    label: str,
    reason: str,
) -> Path:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "reason": reason,
        "excluded_gpu_memory_mb": round(float(excluded_gpu_memory_mb), 6),
    }
    path = analysis_dir / "memory_accounting.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_module_memory_exclusion_metadata(
    output_dir: Path,
    *,
    module: nn.Module,
    label: str,
    reason: str,
) -> Path:
    excluded_mb = module_parameter_buffer_memory_bytes(module) / 1024.0 / 1024.0
    return write_memory_exclusion_metadata(
        output_dir,
        excluded_gpu_memory_mb=excluded_mb,
        label=label,
        reason=reason,
    )
