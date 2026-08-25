from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


STOP_REQUESTED = False


def _handle_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class ProcInfo:
    ppid: int
    rss_kib: int


def load_process_snapshot() -> dict[int, ProcInfo]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,rss="],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot: dict[int, ProcInfo] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kib = int(parts[2])
        except ValueError:
            continue
        snapshot[pid] = ProcInfo(ppid=ppid, rss_kib=rss_kib)
    return snapshot


def collect_descendants(root_pid: int, snapshot: dict[int, ProcInfo]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, info in snapshot.items():
        children.setdefault(info.ppid, []).append(pid)

    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def load_manifest_paths(list_path: Path | None) -> list[Path]:
    if list_path is None or not list_path.exists():
        return []

    result: list[Path] = []
    seen: set[str] = set()
    try:
        raw_text = list_path.read_text(encoding="utf-8")
    except OSError:
        return []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        path = Path(line)
        resolved = str(path.resolve()) if path.is_absolute() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def collect_manifest_pids(manifest_paths: list[Path]) -> set[int]:
    target_keys = {"pid", "monitor_pid", "scheduler_pid", "plotter_pid", "watcher_pid", "train_pid"}
    pids: set[int] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in target_keys and isinstance(child, int) and child > 0:
                    pids.add(child)
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        walk(payload)
    return pids


def load_gpu_process_memory_mib() -> dict[int, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}

    by_pid: dict[int, float] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        by_pid[pid] = by_pid.get(pid, 0.0) + used_mib
    return by_pid


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-list-path", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    start_time = time.time()
    self_pid = os.getpid()
    nvidia_smi_available = subprocess.run(
        ["nvidia-smi", "--help"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0

    payload: dict[str, object] = {
        "root_pid": args.root_pid,
        "monitor_pid": self_pid,
        "poll_seconds": args.poll_seconds,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "peak_cpu_ram_kib": 0,
        "peak_cpu_pid_count": 0,
        "peak_vram_mib": 0.0,
        "peak_vram_pid_count": 0,
        "current_cpu_ram_kib": 0,
        "current_cpu_pid_count": 0,
        "current_vram_mib": 0.0,
        "current_vram_pid_count": 0,
        "samples": 0,
        "nvidia_smi_available": nvidia_smi_available,
        "finished": False,
    }

    while True:
        snapshot = load_process_snapshot()
        tracked_pids = collect_descendants(args.root_pid, snapshot)

        manifest_paths = load_manifest_paths(args.manifest_list_path)
        manifest_pids = collect_manifest_pids(manifest_paths)
        tracked_pids.update(manifest_pids)
        for pid in manifest_pids:
            tracked_pids.update(collect_descendants(pid, snapshot))

        tracked_pids.discard(self_pid)

        cpu_ram_kib = 0
        cpu_pid_count = 0
        for pid in tracked_pids:
            info = snapshot.get(pid)
            if info is None:
                continue
            cpu_ram_kib += info.rss_kib
            cpu_pid_count += 1

        if nvidia_smi_available:
            gpu_memory_by_pid = load_gpu_process_memory_mib()
            current_vram_mib = sum(gpu_memory_by_pid.get(pid, 0.0) for pid in tracked_pids)
            current_vram_pid_count = sum(1 for pid in tracked_pids if gpu_memory_by_pid.get(pid, 0.0) > 0.0)
        else:
            current_vram_mib = 0.0
            current_vram_pid_count = 0

        payload["samples"] = int(payload.get("samples", 0)) + 1
        payload["last_sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["elapsed_seconds"] = round(time.time() - start_time, 3)
        payload["current_cpu_ram_kib"] = cpu_ram_kib
        payload["current_cpu_pid_count"] = cpu_pid_count
        payload["current_vram_mib"] = round(current_vram_mib, 3)
        payload["current_vram_pid_count"] = current_vram_pid_count

        if cpu_ram_kib >= int(payload.get("peak_cpu_ram_kib", 0)):
            payload["peak_cpu_ram_kib"] = cpu_ram_kib
            payload["peak_cpu_pid_count"] = cpu_pid_count
        if current_vram_mib >= float(payload.get("peak_vram_mib", 0.0)):
            payload["peak_vram_mib"] = round(current_vram_mib, 3)
            payload["peak_vram_pid_count"] = current_vram_pid_count

        write_json(args.output_json, payload)

        if STOP_REQUESTED or not process_exists(args.root_pid):
            payload["finished"] = True
            payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            payload["elapsed_seconds"] = round(time.time() - start_time, 3)
            write_json(args.output_json, payload)
            return

        time.sleep(max(args.poll_seconds, 0.1))


if __name__ == "__main__":
    main()
