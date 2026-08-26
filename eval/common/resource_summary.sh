#!/usr/bin/env bash

if [[ -n "${VLASELECT_RESOURCE_SUMMARY_LOADED:-}" ]]; then
    return 0
fi
VLASELECT_RESOURCE_SUMMARY_LOADED=1

VLASELECT_RESOURCE_SUMMARY_ACTIVE=0
VLASELECT_RESOURCE_SUMMARY_PRINTED=0
VLASELECT_RESOURCE_SUMMARY_LABEL=""
VLASELECT_RESOURCE_SUMMARY_ROOT_PID=""
VLASELECT_RESOURCE_SUMMARY_MONITOR_PID=""
VLASELECT_RESOURCE_SUMMARY_STATE_FILE=""
VLASELECT_RESOURCE_SUMMARY_MANIFEST_LIST_FILE=""
VLASELECT_RESOURCE_SUMMARY_START_EPOCH=""
VLASELECT_RESOURCE_SUMMARY_INTERVAL_SECONDS="${VLASELECT_RESOURCE_MONITOR_INTERVAL_SECONDS:-2}"

VLASELECT_RESOURCE_SUMMARY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLASELECT_RESOURCE_SUMMARY_MONITOR_SCRIPT="${VLASELECT_RESOURCE_SUMMARY_DIR}/monitor_process_tree_resources.py"

vlaselect_resource_summary_register_manifest() {
    local manifest_path="$1"
    [[ "$VLASELECT_RESOURCE_SUMMARY_ACTIVE" == "1" ]] || return 0
    [[ -n "$manifest_path" && -n "$VLASELECT_RESOURCE_SUMMARY_MANIFEST_LIST_FILE" ]] || return 0
    printf '%s\n' "$manifest_path" >> "$VLASELECT_RESOURCE_SUMMARY_MANIFEST_LIST_FILE"
}

vlaselect_resource_summary_start() {
    local label="${1:-$(basename "$0")}"
    local root_pid="${2:-$$}"

    if [[ "$VLASELECT_RESOURCE_SUMMARY_ACTIVE" == "1" ]]; then
        return 0
    fi

    VLASELECT_RESOURCE_SUMMARY_LABEL="$label"
    VLASELECT_RESOURCE_SUMMARY_ROOT_PID="$root_pid"
    VLASELECT_RESOURCE_SUMMARY_START_EPOCH="$(date +%s)"

    local tmp_root="${TMPDIR:-/tmp}/vlaselect_resource_summary"
    mkdir -p "$tmp_root"
    local safe_label
    safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
    VLASELECT_RESOURCE_SUMMARY_STATE_FILE="$(mktemp "$tmp_root/${safe_label}.XXXXXX.json")"
    VLASELECT_RESOURCE_SUMMARY_MANIFEST_LIST_FILE="$(mktemp "$tmp_root/${safe_label}.XXXXXX.manifests")"

    python3 -u "$VLASELECT_RESOURCE_SUMMARY_MONITOR_SCRIPT" \
        --root-pid "$VLASELECT_RESOURCE_SUMMARY_ROOT_PID" \
        --output-json "$VLASELECT_RESOURCE_SUMMARY_STATE_FILE" \
        --manifest-list-path "$VLASELECT_RESOURCE_SUMMARY_MANIFEST_LIST_FILE" \
        --poll-seconds "$VLASELECT_RESOURCE_SUMMARY_INTERVAL_SECONDS" &
    VLASELECT_RESOURCE_SUMMARY_MONITOR_PID="$!"
    VLASELECT_RESOURCE_SUMMARY_ACTIVE=1

    if declare -F vlaselect_register_cleanup_pid >/dev/null 2>&1; then
        vlaselect_register_cleanup_pid "$VLASELECT_RESOURCE_SUMMARY_MONITOR_PID"
    fi
    if declare -F vlaselect_register_exit_callback >/dev/null 2>&1; then
        vlaselect_register_exit_callback vlaselect_resource_summary_finalize
    fi
}

vlaselect_resource_summary_finalize() {
    local exit_code="${1:-0}"
    if [[ "$VLASELECT_RESOURCE_SUMMARY_ACTIVE" != "1" || "$VLASELECT_RESOURCE_SUMMARY_PRINTED" == "1" ]]; then
        return 0
    fi
    VLASELECT_RESOURCE_SUMMARY_PRINTED=1

    local monitor_pid="$VLASELECT_RESOURCE_SUMMARY_MONITOR_PID"
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        set +e
        wait "$monitor_pid" 2>/dev/null
        set -e
    fi

    local end_epoch
    end_epoch="$(date +%s)"
    python3 - "$VLASELECT_RESOURCE_SUMMARY_LABEL" "$VLASELECT_RESOURCE_SUMMARY_STATE_FILE" "$VLASELECT_RESOURCE_SUMMARY_START_EPOCH" "$end_epoch" "$exit_code" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def format_seconds(total_seconds: int) -> str:
    hours, rem = divmod(max(total_seconds, 0), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_gib_from_kib(kib: float) -> str:
    gib = kib / 1024.0 / 1024.0
    return f"{gib:.2f} GiB"


def format_gib_from_mib(mib: float) -> str:
    gib = mib / 1024.0
    return f"{gib:.2f} GiB"


label = sys.argv[1]
state_path = Path(sys.argv[2])
start_epoch = int(sys.argv[3])
end_epoch = int(sys.argv[4])
exit_code = int(sys.argv[5])
elapsed_seconds = max(end_epoch - start_epoch, 0)

payload = {}
if state_path.is_file():
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}

peak_cpu_ram_kib = float(payload.get("peak_cpu_ram_kib", 0.0) or 0.0)
peak_vram_mib = float(payload.get("peak_vram_mib", 0.0) or 0.0)
nvidia_smi_available = bool(payload.get("nvidia_smi_available", False))

print(f"[resource-summary] {label}")
print(f"[resource-summary] exit_code={exit_code} wall_time={format_seconds(elapsed_seconds)} ({elapsed_seconds}s)")
print(f"[resource-summary] peak_cpu_ram={format_gib_from_kib(peak_cpu_ram_kib)}")
if nvidia_smi_available:
    print(f"[resource-summary] peak_vram={format_gib_from_mib(peak_vram_mib)}")
else:
    print("[resource-summary] peak_vram=N/A (nvidia-smi unavailable)")
PY

    VLASELECT_RESOURCE_SUMMARY_ACTIVE=0
}
