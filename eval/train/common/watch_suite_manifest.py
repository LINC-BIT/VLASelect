from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINAL_METHOD_STATES = {"completed", "failed", "inherited"}
FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "FileNotFoundError:",
    "RuntimeError:",
    "ModuleNotFoundError:",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Killed",
)


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
        if (run_dir / "checkpoints").exists() or (run_dir / "tb").exists() or (run_dir / "analysis").exists():
            return False
    return not log_file.exists() and not run_dir.exists()


def update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    methods = manifest.get("methods", [])
    active_count = 0
    failed_count = 0

    for method in methods:
        status = method.get("status") or "launched"
        if status == "inherited":
            continue

        pid = method.get("pid")
        alive = process_is_alive(pid)
        method["last_checked_at_utc"] = utc_now_iso()

        if alive:
            active_count += 1
            if not method.get("started_at_utc"):
                method["started_at_utc"] = utc_now_iso()
            method["status"] = "running"
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
