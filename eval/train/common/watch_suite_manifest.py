from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINAL_METHOD_STATES = {"completed", "failed", "cancelled", "timed_out", "inherited"}
FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "FileNotFoundError:",
    "RuntimeError:",
    "ModuleNotFoundError:",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Killed",
)
QUEUE_WAIT_MARKER = "waiting for pid="
QUEUE_LAUNCH_MARKER = "wait finished; launching"
TRAINING_ARTIFACT_DIRS = ("checkpoints", "tb", "analysis")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
        except OSError:
            fields = []
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail_text(path: Path, max_bytes: int = 16384) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def infer_failed(method: dict[str, Any]) -> bool:
    run_dir = Path(method.get("run_dir") or "")
    log_file = Path(method.get("log_file") or "")
    log_tail = tail_text(log_file)
    if any(marker in log_tail for marker in FAILURE_MARKERS):
        return True
    if run_dir.exists():
        if any((run_dir / name).exists() for name in TRAINING_ARTIFACT_DIRS):
            return False
    return not log_file.exists() and not run_dir.exists()


def has_training_artifacts(run_dir: Path) -> bool:
    return run_dir.exists() and any((run_dir / name).exists() for name in TRAINING_ARTIFACT_DIRS)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def find_run_artifact(run_dir: Path, relative_path: str) -> Path | None:
    direct = run_dir / relative_path
    if direct.exists():
        return direct
    search_roots = [run_dir, run_dir.parent]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches = sorted(path for path in search_root.glob(f"**/{relative_path}") if path.is_file())
        if matches:
            return matches[0]
    return None


def load_actual_runtime_hours(run_dir: Path) -> tuple[float | None, str | None]:
    history_path = find_run_artifact(run_dir, 'metrics_history.json')
    if history_path is None:
        return None, None
    try:
        payload = json.loads(history_path.read_text(encoding='utf-8'))
    except Exception:
        return None, str(history_path)
    history = payload.get('history', []) if isinstance(payload, dict) else payload
    if not isinstance(history, list) or not history:
        return None, str(history_path)
    for metric in reversed(history):
        if not isinstance(metric, dict):
            continue
        elapsed_hours = finite_float(metric.get('elapsed_hours'))
        if elapsed_hours is not None and elapsed_hours >= 0.0:
            return elapsed_hours, str(history_path)
    return None, str(history_path)


def load_gpu_monitor_elapsed_hours(run_dir: Path) -> tuple[float | None, str | None]:
    csv_path = find_run_artifact(run_dir, 'analysis/gpu_metrics.csv')
    if csv_path is None:
        return None, None
    last_elapsed_seconds = None
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                value = finite_float(row.get('elapsed_seconds'))
                if value is not None:
                    last_elapsed_seconds = value
    except Exception:
        return None, str(csv_path)
    if last_elapsed_seconds is None:
        return None, str(csv_path)
    return last_elapsed_seconds / 3600.0, str(csv_path)


def refresh_runtime_metadata(method: dict[str, Any]) -> None:
    run_dir = Path(method.get('run_dir') or '')
    if not run_dir.exists():
        return
    actual_runtime_hours, history_path = load_actual_runtime_hours(run_dir)
    if history_path is not None:
        method['metrics_history_path'] = history_path
    if actual_runtime_hours is not None:
        method['actual_runtime_hours'] = actual_runtime_hours
        method['actual_runtime_seconds'] = actual_runtime_hours * 3600.0
    gpu_monitor_hours, gpu_metrics_path = load_gpu_monitor_elapsed_hours(run_dir)
    if gpu_metrics_path is not None:
        method['gpu_metrics_path'] = gpu_metrics_path
    if gpu_monitor_hours is not None:
        method['gpu_monitor_elapsed_hours'] = gpu_monitor_hours
        method['gpu_monitor_elapsed_seconds'] = gpu_monitor_hours * 3600.0


def infer_alive_status(method: dict[str, Any], previous_status: str) -> str:
    run_dir = Path(method.get("run_dir") or "")
    log_file = Path(method.get("log_file") or "")
    log_tail = tail_text(log_file)

    if has_training_artifacts(run_dir):
        return "running"

    if log_tail:
        lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
        if any("[queue]" not in line for line in lines):
            return "running"

        last_wait_index = log_tail.rfind(QUEUE_WAIT_MARKER)
        last_launch_index = log_tail.rfind(QUEUE_LAUNCH_MARKER)
        if last_wait_index != -1 and last_launch_index < last_wait_index:
            return "queued"
        if last_launch_index != -1:
            return "launching"

    if previous_status == "queued":
        return "queued"
    return "launching"


def update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    methods = manifest.get("methods", [])
    active_count = 0
    failed_count = 0

    for method in methods:
        refresh_runtime_metadata(method)
        status = method.get("status") or "launching"
        if status == "inherited":
            continue

        pid = method.get("pid")
        alive = process_is_alive(pid)
        method["last_checked_at_utc"] = utc_now_iso()

        if alive:
            active_count += 1
            if not method.get("started_at_utc"):
                method["started_at_utc"] = utc_now_iso()
            method["status"] = infer_alive_status(method, status)
            method["exit_code"] = None
            continue

        if status not in FINAL_METHOD_STATES:
            if not method.get("started_at_utc") and pid is not None:
                method["started_at_utc"] = utc_now_iso()
            method["finished_at_utc"] = utc_now_iso()
            if infer_failed(method):
                method["status"] = "failed"
                method["exit_code"] = 1
            else:
                method["status"] = "completed"
                method["exit_code"] = 0

        if method.get("status") == "failed":
            failed_count += 1

    plotter_alive = process_is_alive(manifest.get("plotter_pid"))
    manifest["plotter_status"] = "running" if plotter_alive else "finished"

    if active_count == 0 and not plotter_alive:
        manifest["suite_state"] = "finished"
        manifest["suite_finished_at_utc"] = utc_now_iso()
        manifest["suite_status"] = "failed" if failed_count > 0 else "completed"
    else:
        manifest["suite_state"] = "running"
        manifest["suite_status"] = "running"
        manifest.setdefault("suite_started_at_utc", utc_now_iso())

    manifest["watcher_status"] = "running"
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()

    while True:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = update_manifest(manifest)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if manifest.get("suite_state") == "finished":
            break
        time.sleep(args.interval_seconds)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["watcher_status"] = "finished"
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
