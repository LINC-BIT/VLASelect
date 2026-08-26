from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from pynvml import (
    NVMLError,
    nvmlDeviceGetComputeRunningProcesses_v3,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


UNKNOWN_USED_GPU_MEMORY = 2**64 - 1
ROLLOUT_PHASE_NAME = "online_rl_rollout"
PHASE_TRACE_FILENAME = "memory_phase_trace.jsonl"


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def to_mb(value: int | None) -> float | None:
    if value is None or value < 0:
        return None
    return value / 1024.0 / 1024.0


def load_process_snapshot() -> dict[int, int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot: dict[int, int] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        snapshot[pid] = ppid
    return snapshot


def collect_descendants(root_pid: int, snapshot: dict[int, int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, ppid in snapshot.items():
        children.setdefault(ppid, []).append(pid)

    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def build_gpu_sample_row(
    *,
    pid: int,
    label: str | None,
    gpu_index: int,
    gpu_name: str,
    start_time: float,
    handle,
) -> dict[str, object]:
    now = time.time()
    snapshot = load_process_snapshot()
    tracked_pids = sorted(collect_descendants(pid, snapshot)) if snapshot else [pid]
    tracked_pid_set = set(tracked_pids)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(now - start_time, 3),
        "pid": pid,
        "label": label or "",
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "gpu_power_w": None,
        "gpu_memory_used_mb": 0.0,
        "gpu_device_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_utilization_pct": None,
        "memory_utilization_pct": None,
        "process_memory_used_mb": 0.0,
        "process_found_on_gpu": False,
        "tracked_pid_count": len(tracked_pids),
        "tracked_gpu_pid_count": 0,
        "tracked_gpu_pids": json.dumps([], separators=(",", ":")),
    }

    try:
        memory = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        row["gpu_power_w"] = round(nvmlDeviceGetPowerUsage(handle) / 1000.0, 3)
        row["gpu_device_memory_used_mb"] = round(to_mb(memory.used) or 0.0, 3)
        row["gpu_memory_total_mb"] = round(to_mb(memory.total) or 0.0, 3)
        row["gpu_utilization_pct"] = float(util.gpu)
        row["memory_utilization_pct"] = float(util.memory)

        tracked_gpu_pids: list[int] = []
        tracked_gpu_memory_mb = 0.0
        for proc in nvmlDeviceGetComputeRunningProcesses_v3(handle):
            proc_pid = getattr(proc, "pid", None)
            if proc_pid not in tracked_pid_set:
                continue
            tracked_gpu_pids.append(int(proc_pid))
            used_gpu_memory = getattr(proc, "usedGpuMemory", None)
            if used_gpu_memory is None or used_gpu_memory == UNKNOWN_USED_GPU_MEMORY:
                continue
            tracked_gpu_memory_mb += to_mb(used_gpu_memory) or 0.0

        tracked_gpu_pids = sorted(set(tracked_gpu_pids))
        row["process_found_on_gpu"] = bool(tracked_gpu_pids)
        row["tracked_gpu_pid_count"] = len(tracked_gpu_pids)
        row["tracked_gpu_pids"] = json.dumps(tracked_gpu_pids, separators=(",", ":"))
        if tracked_gpu_pids:
            tracked_gpu_memory_mb = round(tracked_gpu_memory_mb, 3)
            row["gpu_memory_used_mb"] = tracked_gpu_memory_mb
            row["process_memory_used_mb"] = tracked_gpu_memory_mb
    except NVMLError:
        pass
    return row


def consume_rollout_phase_trigger(phase_trace_path: Path, state: dict[str, object]) -> bool:
    if not phase_trace_path.exists():
        return False
    offset = int(state.get("offset", 0))
    carry = str(state.get("carry", ""))
    try:
        file_size = phase_trace_path.stat().st_size
        if file_size < offset:
            offset = 0
            carry = ""
        with phase_trace_path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except OSError:
        return False

    data = carry + chunk
    lines = data.splitlines(keepends=True)
    next_carry = ""
    if lines and not lines[-1].endswith("\n"):
        next_carry = lines.pop()
    state["offset"] = offset
    state["carry"] = next_carry

    rollout_triggered = False
    last_rollout_event_key = state.get("last_rollout_event_key")
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = str(payload.get("phase", "")).strip()
        if phase != ROLLOUT_PHASE_NAME:
            continue
        event_key = (
            str(payload.get("timestamp_utc", "")),
            str(payload.get("unix_time_seconds", "")),
            phase,
            str(payload.get("note", "")),
        )
        if event_key == last_rollout_event_key:
            continue
        last_rollout_event_key = event_key
        rollout_triggered = True
    state["last_rollout_event_key"] = last_rollout_event_key
    return rollout_triggered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.run_dir / "analysis"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "gpu_metrics.csv"
    meta_path = metrics_dir / "gpu_metrics_meta.json"
    phase_trace_path = metrics_dir / PHASE_TRACE_FILENAME

    start_time = time.time()
    phase_poll_seconds = min(1.0, max(0.1, args.interval_seconds / 20.0))
    metadata = {
        "label": args.label,
        "pid": args.pid,
        "gpu_index": args.gpu_index,
        "interval_seconds": args.interval_seconds,
        "phase_poll_seconds": phase_poll_seconds,
        "phase_trace_path": str(phase_trace_path),
        "rollout_phase_name": ROLLOUT_PHASE_NAME,
        "run_dir": str(args.run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_used_mb_semantics": "tracked process tree GPU memory on the selected GPU",
        "gpu_device_memory_used_mb_semantics": "total GPU memory used on the selected GPU",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(args.gpu_index)
        gpu_name = nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")

        fieldnames = [
            "timestamp_utc",
            "elapsed_seconds",
            "pid",
            "label",
            "gpu_index",
            "gpu_name",
            "gpu_power_w",
            "gpu_memory_used_mb",
            "gpu_device_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "process_memory_used_mb",
            "process_found_on_gpu",
            "tracked_pid_count",
            "tracked_gpu_pid_count",
            "tracked_gpu_pids",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            phase_state: dict[str, object] = {"offset": 0, "carry": ""}
            next_sample_time = start_time

            while True:
                now = time.time()
                rollout_triggered = consume_rollout_phase_trigger(phase_trace_path, phase_state)
                should_sample = rollout_triggered or now >= next_sample_time
                alive = process_is_alive(args.pid)
                if not should_sample and not alive:
                    should_sample = True
                if should_sample:
                    row = build_gpu_sample_row(
                        pid=args.pid,
                        label=args.label,
                        gpu_index=args.gpu_index,
                        gpu_name=gpu_name,
                        start_time=start_time,
                        handle=handle,
                    )
                    writer.writerow(row)
                    f.flush()
                    next_sample_time = time.time() + args.interval_seconds
                if not alive:
                    break
                sleep_seconds = max(0.0, next_sample_time - time.time())
                if sleep_seconds > 0.0:
                    time.sleep(min(phase_poll_seconds, sleep_seconds))
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(0)))
    main()
