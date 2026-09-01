from __future__ import annotations

import copy
import io
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import dill
import torch
import torch.nn as nn

MATERIALIZED_FBS_POLICY_BYTES_KEY = "materialized_fbs_policy_bytes"
MATERIALIZED_FBS_METADATA_KEY = "materialized_fbs_policy_metadata"
MATERIALIZED_FBS_FORMAT_VERSION = 1
MATERIALIZED_FBS_PICKLE_PROTOCOL = max(4, pickle.HIGHEST_PROTOCOL)


def build_materialized_fbs_metadata(
    checkpoint_path: str | Path,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "version": MATERIALIZED_FBS_FORMAT_VERSION,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return metadata


def _deserialize_materialized_fbs_policy(
    payload: bytes | bytearray,
    checkpoint_path: str | Path,
) -> Optional[nn.Module]:
    try:
        policy = torch.load(io.BytesIO(payload), map_location="cpu", pickle_module=dill)
    except Exception as exc:
        print(f"[setup] ignoring unreadable materialized FBS policy cache in {checkpoint_path} ({exc})")
        return None
    if not isinstance(policy, nn.Module):
        print(f"[setup] ignoring invalid materialized FBS policy cache in {checkpoint_path}")
        return None
    return policy


def maybe_load_materialized_fbs_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[nn.Module]:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[setup] failed to read checkpoint for materialized FBS cache: {path} ({exc})")
        return None
    if not isinstance(checkpoint, dict):
        return None
    metadata = checkpoint.get(MATERIALIZED_FBS_METADATA_KEY)
    payload = checkpoint.get(MATERIALIZED_FBS_POLICY_BYTES_KEY)
    if expected_metadata is None:
        expected_metadata = build_materialized_fbs_metadata(path)
    if metadata != expected_metadata or not isinstance(payload, (bytes, bytearray)):
        return None
    policy = _deserialize_materialized_fbs_policy(payload, path)
    if policy is None:
        return None
    policy = policy.to(device)
    if hasattr(policy, "device"):
        policy.device = device
    print(f"[setup] loaded cached materialized FBS policy from {path}")
    return policy


def maybe_persist_materialized_fbs_policy_to_checkpoint(
    checkpoint_path: str | Path,
    policy: nn.Module,
    *,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        return
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[setup] failed to read checkpoint for materialized FBS cache: {path} ({exc})")
        return
    if isinstance(checkpoint, dict):
        payload: Dict[str, Any] = dict(checkpoint)
    elif isinstance(checkpoint, Mapping):
        payload = {"policy": dict(checkpoint)}
    else:
        print(f"[setup] skipping materialized FBS cache because checkpoint payload is not dict-like: {path}")
        return
    if expected_metadata is None:
        expected_metadata = build_materialized_fbs_metadata(path)
    existing_payload = payload.get(MATERIALIZED_FBS_POLICY_BYTES_KEY)
    if payload.get(MATERIALIZED_FBS_METADATA_KEY) == expected_metadata and isinstance(existing_payload, (bytes, bytearray)):
        if _deserialize_materialized_fbs_policy(existing_payload, path) is not None:
            return
        print(f"[setup] overwriting unreadable materialized FBS policy cache in {path}")

    policy_cpu = copy.deepcopy(policy).to(device=torch.device("cpu"))
    if hasattr(policy_cpu, "device"):
        policy_cpu.device = torch.device("cpu")
    buffer = io.BytesIO()
    torch.save(
        policy_cpu,
        buffer,
        pickle_module=dill,
        pickle_protocol=MATERIALIZED_FBS_PICKLE_PROTOCOL,
    )
    serialized_policy = buffer.getvalue()
    if _deserialize_materialized_fbs_policy(serialized_policy, path) is None:
        print(f"[setup] skipping materialized FBS cache persist because serialized policy is unreadable: {path}")
        return
    payload[MATERIALIZED_FBS_POLICY_BYTES_KEY] = serialized_policy
    payload[MATERIALIZED_FBS_METADATA_KEY] = expected_metadata

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            torch.save(payload, handle, pickle_protocol=MATERIALIZED_FBS_PICKLE_PROTOCOL)
            handle.flush()
        tmp_path.replace(path)
        print(f"[setup] cached materialized FBS policy into checkpoint: {path}")
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[setup] failed to persist materialized FBS policy cache: {path} ({exc})")
