from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Union

import torch

_MWE_GATE_ENVS = ("MWE_ACTIVE_RUNTIME_ONLY", "MWE", "SMOKE", "RUN_SETUP_SMOKE")
_LOGGED_SKIP_PATHS: set[str] = set()


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def should_skip_model_checkpoint_save() -> bool:
    if _env_truthy("VLASELECT_FORCE_SAVE_MODEL_CHECKPOINTS"):
        return False
    return any(_env_truthy(name) for name in _MWE_GATE_ENVS)


def maybe_save_model_checkpoint(payload: Any, output_path: Union[str, Path]) -> bool:
    path = Path(output_path)
    if should_skip_model_checkpoint_save():
        key = str(path)
        if key not in _LOGGED_SKIP_PATHS:
            print(f"[mwe] skip model checkpoint save: {path}")
            _LOGGED_SKIP_PATHS.add(key)
        return False
    torch.save(payload, path)
    return True
