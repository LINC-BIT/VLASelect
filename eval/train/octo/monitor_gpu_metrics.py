from __future__ import annotations

import argparse
import csv
import json
import os
import signal
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

    start_time = time.time()
    metadata = {
        "label": args.label,
        "pid": args.pid,
        "gpu_index": args.gpu_index,
        "interval_seconds": args.interval_seconds,
        "run_dir": str(args.run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
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
            "gpu_memory_total_mb",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "process_memory_used_mb",
            "process_found_on_gpu",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                alive = process_is_alive(args.pid)
                now = time.time()
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": round(now - start_time, 3),
                    "pid": args.pid,
                    "label": args.label or "",
                    "gpu_index": args.gpu_index,
                    "gpu_name": gpu_name,
                    "gpu_power_w": None,
                    "gpu_memory_used_mb": None,
                    "gpu_memory_total_mb": None,
                    "gpu_utilization_pct": None,
                    "memory_utilization_pct": None,
                    "process_memory_used_mb": None,
                    "process_found_on_gpu": False,
                }

                try:
                    memory = nvmlDeviceGetMemoryInfo(handle)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    row["gpu_power_w"] = round(nvmlDeviceGetPowerUsage(handle) / 1000.0, 3)
                    row["gpu_memory_used_mb"] = round(to_mb(memory.used) or 0.0, 3)
                    row["gpu_memory_total_mb"] = round(to_mb(memory.total) or 0.0, 3)
                    row["gpu_utilization_pct"] = float(util.gpu)
                    row["memory_utilization_pct"] = float(util.memory)

                    process_memory_used = None
                    for proc in nvmlDeviceGetComputeRunningProcesses_v3(handle):
                        if getattr(proc, "pid", None) != args.pid:
                            continue
                        used_gpu_memory = getattr(proc, "usedGpuMemory", None)
                        if used_gpu_memory is not None and used_gpu_memory != 2**64 - 1:
                            process_memory_used = round(to_mb(used_gpu_memory) or 0.0, 3)
                        row["process_found_on_gpu"] = True
                        row["process_memory_used_mb"] = process_memory_used
                        break
                except NVMLError:
                    pass

                writer.writerow(row)
                f.flush()

                if not alive:
                    break
                time.sleep(args.interval_seconds)
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(0)))
    main()
