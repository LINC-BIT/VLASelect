#!/usr/bin/env bash

if [[ -n "${VLASELECT_INTERRUPT_CLEANUP_LOADED:-}" ]]; then
    return 0
fi
VLASELECT_INTERRUPT_CLEANUP_LOADED=1

declare -ag VLASELECT_CLEANUP_MANIFESTS=()
declare -ag VLASELECT_CLEANUP_PIDS=()
declare -ag VLASELECT_EXIT_CALLBACKS=()
VLASELECT_CLEANUP_TRAP_INSTALLED=0
VLASELECT_CLEANUP_RUNNING=0

vlaselect_register_cleanup_manifest() {
    local manifest_path="$1"
    if [[ -n "$manifest_path" ]]; then
        VLASELECT_CLEANUP_MANIFESTS+=("$manifest_path")
        if declare -F vlaselect_resource_summary_register_manifest >/dev/null 2>&1; then
            vlaselect_resource_summary_register_manifest "$manifest_path" || true
        fi
    fi
}

vlaselect_register_cleanup_pid() {
    local pid="$1"
    if [[ -n "$pid" ]]; then
        VLASELECT_CLEANUP_PIDS+=("$pid")
    fi
}

vlaselect_register_exit_callback() {
    local callback_name="$1"
    [[ -n "$callback_name" ]] || return 0

    local existing
    for existing in "${VLASELECT_EXIT_CALLBACKS[@]}"; do
        if [[ "$existing" == "$callback_name" ]]; then
            return 0
        fi
    done
    VLASELECT_EXIT_CALLBACKS+=("$callback_name")
}

vlaselect_run_exit_callbacks() {
    local exit_status="$1"
    local callback_name
    for callback_name in "${VLASELECT_EXIT_CALLBACKS[@]}"; do
        if declare -F "$callback_name" >/dev/null 2>&1; then
            "$callback_name" "$exit_status" || true
        fi
    done
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

vlaselect_print_suite_training_logs() {
    local manifest_path="$1"
    local prefix="${2:-suite-log}"
    local suite_label="${3:-suite}"

    [[ -f "$manifest_path" ]] || return 0

    python - "$manifest_path" "$prefix" "$suite_label" <<'__VLASELECT_LOGS__'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
prefix = sys.argv[2]
suite_label = sys.argv[3]

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)

methods = manifest.get("methods", [])
if not isinstance(methods, list):
    raise SystemExit(0)

for method in methods:
    if not isinstance(method, dict):
        continue
    name = str(method.get("name") or method.get("display_name") or "unknown")
    log_file = str(method.get("log_file") or "").strip()
    run_dir = str(method.get("run_dir") or "").strip()
    status = str(method.get("status") or "").strip()
    if log_file:
        print(f"[{prefix}] training log ({suite_label}/{name}): {log_file}")
    elif run_dir:
        suffix = f" status={status}" if status else ""
        print(f"[{prefix}] training log ({suite_label}/{name}): unavailable; run_dir={run_dir}{suffix}")
__VLASELECT_LOGS__
}

vlaselect_print_log_excerpt() {
    local log_file="$1"
    local lines="${2:-20}"
    local prefix="${3:-log}"

    if [[ -f "$log_file" ]]; then
        echo "[${prefix}] last ${lines} lines from ${log_file}:" >&2
        tail -n "$lines" "$log_file" >&2 || true
    fi
}

vlaselect_report_command_failure() {
    local prefix="$1"
    local label="$2"
    local launch_log="${3:-}"
    local training_log="${4:-}"
    local exit_code="${5:-}"

    local message="[${prefix}] error: ${label}"
    if [[ -n "$exit_code" ]]; then
        message+=" (exit_code=${exit_code})"
    fi
    echo "$message" >&2

    if [[ -n "$launch_log" ]]; then
        echo "[${prefix}] launch log: ${launch_log}" >&2
    fi
    if [[ -n "$training_log" ]]; then
        echo "[${prefix}] training log: ${training_log}" >&2
    fi
}

vlaselect_report_manifest_failures() {
    local manifest_path="$1"
    local prefix="$2"
    local suite_label="${3:-suite}"
    local launch_log="${4:-}"

    local output=""
    local rc=0
    set +e
    output=$(python - "$manifest_path" "$prefix" "$suite_label" "$launch_log" <<'PY_HELPER'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
prefix = sys.argv[2]
suite_label = sys.argv[3]
launch_log = sys.argv[4]

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except OSError as exc:
    print(f"[{prefix}] error ({suite_label}): unable to read manifest {manifest_path}: {exc}")
    if launch_log:
        print(f"[{prefix}] launch log ({suite_label}): {launch_log}")
    raise SystemExit(4)
except json.JSONDecodeError as exc:
    print(f"[{prefix}] error ({suite_label}): invalid manifest {manifest_path}: {exc}")
    if launch_log:
        print(f"[{prefix}] launch log ({suite_label}): {launch_log}")
    raise SystemExit(4)

def iter_entries(payload):
    methods = payload.get("methods")
    if isinstance(methods, list):
        for entry in methods:
            if isinstance(entry, dict):
                yield entry
    runs = payload.get("runs")
    if isinstance(runs, list):
        for entry in runs:
            if isinstance(entry, dict):
                yield entry

failed = []
for entry in iter_entries(manifest):
    status = str(entry.get("status") or "").strip().lower()
    exit_code = entry.get("exit_code")
    has_failure_status = status in {"failed", "error", "errored", "crashed", "aborted", "timeout", "timed_out"}
    has_failure_exit_code = isinstance(exit_code, int) and exit_code != 0
    if has_failure_status or has_failure_exit_code:
        failed.append(entry)

if not failed:
    raise SystemExit(0)

if launch_log:
    print(f"[{prefix}] launch log ({suite_label}): {launch_log}")

for entry in failed:
    name = str(entry.get("name") or entry.get("method") or entry.get("family") or entry.get("display_name") or "unknown")
    status = str(entry.get("status") or "failed").strip()
    exit_code = entry.get("exit_code")
    log_file = str(entry.get("log_file") or "").strip()
    run_dir = str(entry.get("run_dir") or "").strip()
    message = f"[{prefix}] error ({suite_label}/{name}): status={status}"
    if isinstance(exit_code, int):
        message += f" exit_code={exit_code}"
    print(message)
    if log_file:
        print(f"[{prefix}] training log ({suite_label}/{name}): {log_file}")
    elif run_dir:
        print(f"[{prefix}] training log ({suite_label}/{name}): unavailable; run_dir={run_dir}")

raise SystemExit(3)
PY_HELPER
)
    rc=$?
    set -e

    if [[ -n "$output" ]]; then
        printf '%s\n' "$output" >&2
    fi

    if [[ "$rc" -eq 3 ]]; then
        return 1
    fi
    return "$rc"
}

vlaselect_cleanup() {
    local rc=$?
    trap - INT TERM EXIT

    if [[ "$VLASELECT_CLEANUP_RUNNING" == "1" ]]; then
        exit "$rc"
    fi
    VLASELECT_CLEANUP_RUNNING=1

    vlaselect_run_exit_callbacks "$rc"

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
