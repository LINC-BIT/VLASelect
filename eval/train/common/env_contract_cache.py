from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

DEFAULT_CACHE_PATH = Path(os.environ.get("VLASELECT_ENV_CONTRACT_CACHE_PATH", "/tmp/vlaselect_env_contract_cache.json"))
CACHE_VERSION = 1


def _env_flag(name: str) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_env_contract_cache_disabled() -> bool:
    return _env_flag("VLASELECT_DISABLE_ENV_CONTRACT_CACHE")


def build_env_contract_cache_key(
    *,
    env_id: str,
    obs_mode: str,
    control_mode: str,
    reward_mode: str,
    device_type: str,
    device_index: int | None,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "env_id": env_id,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "reward_mode": reward_mode,
        "device_type": device_type,
        "device_index": device_index,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache_payload(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_env_contract_from_cache(cache_key: str, cache_path: Path | None = None) -> Tuple[int, int, Tuple[int, ...]] | None:
    if is_env_contract_cache_disabled():
        return None
    cache_path = cache_path or DEFAULT_CACHE_PATH
    payload = _load_cache_payload(cache_path)
    entry = payload.get(cache_key)
    if not isinstance(entry, dict):
        return None
    try:
        env_action_dim = int(entry["env_action_dim"])
        state_dim = int(entry["state_dim"])
        controlled_action_indices = tuple(int(value) for value in entry["controlled_action_indices"])
    except Exception:
        return None
    return env_action_dim, state_dim, controlled_action_indices


def save_env_contract_to_cache(
    cache_key: str,
    env_action_dim: int,
    state_dim: int,
    controlled_action_indices: Iterable[int],
    cache_path: Path | None = None,
) -> None:
    if is_env_contract_cache_disabled():
        return
    cache_path = cache_path or DEFAULT_CACHE_PATH
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _load_cache_payload(cache_path)
        payload[cache_key] = {
            "env_action_dim": int(env_action_dim),
            "state_dim": int(state_dim),
            "controlled_action_indices": [int(value) for value in controlled_action_indices],
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception:
        return
