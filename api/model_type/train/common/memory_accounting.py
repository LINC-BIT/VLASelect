from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch.nn as nn


DEFAULT_EXCLUDED_RUNTIME_PHASE_NAMES = ("large_model_runtime_excluded",)
PHASE_TRACE_FILENAME = "memory_phase_trace.jsonl"


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
    excluded_runtime_phase_names: Iterable[str] | None = None,
) -> Path:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    phase_names = [
        str(phase_name).strip()
        for phase_name in (excluded_runtime_phase_names or ())
        if str(phase_name).strip()
    ]
    payload = {
        "label": label,
        "reason": reason,
        "excluded_gpu_memory_mb": round(float(excluded_gpu_memory_mb), 6),
        "excluded_runtime_phase_names": phase_names,
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
    excluded_runtime_phase_names: Iterable[str] | None = None,
) -> Path:
    excluded_mb = module_parameter_buffer_memory_bytes(module) / 1024.0 / 1024.0
    return write_memory_exclusion_metadata(
        output_dir,
        excluded_gpu_memory_mb=excluded_mb,
        label=label,
        reason=reason,
        excluded_runtime_phase_names=excluded_runtime_phase_names,
    )


class MemoryPhaseTracker:
    def __init__(self, output_dir: Path):
        self.analysis_dir = output_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.analysis_dir / PHASE_TRACE_FILENAME
        self.current_phase: str | None = None

    def mark(self, phase: str, *, note: str | None = None, force: bool = False) -> None:
        normalized_phase = str(phase).strip()
        if not normalized_phase:
            return
        if not force and normalized_phase == self.current_phase:
            return
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "unix_time_seconds": time.time(),
            "phase": normalized_phase,
        }
        if note:
            payload["note"] = str(note)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.current_phase = normalized_phase
