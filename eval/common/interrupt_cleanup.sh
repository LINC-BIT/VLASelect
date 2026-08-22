#!/usr/bin/env bash

if [[ -n "${VLASELECT_INTERRUPT_CLEANUP_LOADED:-}" ]]; then
    return 0
fi
VLASELECT_INTERRUPT_CLEANUP_LOADED=1

declare -ag VLASELECT_CLEANUP_MANIFESTS=()
declare -ag VLASELECT_CLEANUP_PIDS=()
VLASELECT_CLEANUP_TRAP_INSTALLED=0

vlaselect_register_cleanup_manifest() {
    local manifest_path="$1"
    if [[ -n "$manifest_path" ]]; then
        VLASELECT_CLEANUP_MANIFESTS+=("$manifest_path")
    fi
}

vlaselect_register_cleanup_pid() {
    local pid="$1"
    if [[ -n "$pid" ]]; then
        VLASELECT_CLEANUP_PIDS+=("$pid")
    fi
}

vlaselect_collect_manifest_pids() {
    python - "$1" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(0)

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)

target_keys = {"pid", "monitor_pid", "scheduler_pid", "plotter_pid", "watcher_pid", "train_pid"}
pids = set()

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in target_keys and isinstance(child, int) and child > 0:
                pids.add(child)
            walk(child)
    elif isinstance(value, list):
        for item in value:
            walk(item)

walk(manifest)
for pid in sorted(pids):
    print(pid)
PY
}

vlaselect_start_file_log_tail() {
    local log_file="$1"
    local label="${2:-log}"

    [[ -n "$log_file" ]] || return 0

    python -u - "$log_file" "$label" <<'PY' &
from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path

log_path = Path(sys.argv[1])
label = sys.argv[2]

while not log_path.exists():
    time.sleep(0.5)

with log_path.open("r", encoding="utf-8", errors="replace") as handle:
    recent = deque(handle.readlines(), maxlen=15)
for raw in recent:
    print(f"[{label}] {raw.rstrip()}", flush=True)

with log_path.open("r", encoding="utf-8", errors="replace") as handle:
    handle.seek(0, os.SEEK_END)
    while True:
        line = handle.readline()
        if line:
            print(f"[{label}] {line.rstrip()}", flush=True)
        else:
            time.sleep(0.5)
PY
    vlaselect_register_cleanup_pid "$!"
}

vlaselect_start_manifest_log_tail() {
    local manifest_path="$1"
    local label="${2:-suite}"
    local interval_seconds="${3:-5}"

    [[ -n "$manifest_path" ]] || return 0

    python -u common/watch_manifest_status.py             --manifest "$manifest_path"             --label "$label"             --interval-seconds "$interval_seconds" &
    vlaselect_register_cleanup_pid "$!"
}

vlaselect_cleanup() {
    local rc=$?
    trap - INT TERM EXIT

    local -A seen_pids=()
    local pid
    local manifest_path

    for pid in "${VLASELECT_CLEANUP_PIDS[@]}"; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        seen_pids["$pid"]=1
    done

    for manifest_path in "${VLASELECT_CLEANUP_MANIFESTS[@]}"; do
        [[ -f "$manifest_path" ]] || continue
        while IFS= read -r pid; do
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            seen_pids["$pid"]=1
        done < <(vlaselect_collect_manifest_pids "$manifest_path")
    done

    for pid in "${!seen_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${!seen_pids[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done

    exit "$rc"
}

vlaselect_install_cleanup_trap() {
    if [[ "$VLASELECT_CLEANUP_TRAP_INSTALLED" == "0" ]]; then
        trap vlaselect_cleanup INT TERM EXIT
        VLASELECT_CLEANUP_TRAP_INSTALLED=1
    fi
}
