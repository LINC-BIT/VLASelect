#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.hold_in_hand import HOLD_IN_HAND_VARIANT_ENV_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runtime-hours", type=float, default=1.5)
    parser.add_argument("--early-stop-zero-success-minutes", type=float, default=45.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--env-ids", nargs="*", default=None)
    parser.add_argument("--batch-name", type=str, default=None)
    parser.add_argument("--save-video", type=str, default="false")
    return parser.parse_args()


def safe_name(env_id: str) -> str:
    return env_id.replace("/", "_")


def metric_series(history: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(metric[key]) for metric in history if metric.get(key) is not None]


def first_or_none(values: List[float]) -> float | None:
    return values[0] if values else None


def last_or_none(values: List[float]) -> float | None:
    return values[-1] if values else None


def mean_or_none(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_single_env(
    env_id: str,
    gpu_id: str,
    batch_name: str,
    max_runtime_hours: float,
    early_stop_zero_success_minutes: float,
    save_video: str,
) -> Dict[str, Any]:
    run_name = f"{batch_name}__{safe_name(env_id)}"
    output_dir = THIS_DIR / "outputs" / run_name
    log_dir = THIS_DIR / "batch_logs" / batch_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{safe_name(env_id)}.log"
    script_path = THIS_DIR / "run_online_rl_hold_cube_in_hand.sh"

    env = os.environ.copy()
    env.update(
        {
            "LAUNCH_DIRECT": "1",
            "TAIL_LOG": "0",
            "CUDA_DEVICES": gpu_id,
            "NPROC_PER_NODE": "1",
            "ENV_ID_OVERRIDE": env_id,
            "MAX_RUNTIME_HOURS_OVERRIDE": str(max_runtime_hours),
            "EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE": str(early_stop_zero_success_minutes),
            "RUN_NAME_OVERRIDE": run_name,
            "SAVE_VIDEO_OVERRIDE": save_video,
        }
    )

    started_at = time.time()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ended_at = time.time()
    return {
        "env_id": env_id,
        "gpu_id": gpu_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "returncode": process.returncode,
        "wall_clock_minutes": (ended_at - started_at) / 60.0,
    }


def summarize_run(run_result: Dict[str, Any], curves_dir: Path) -> Dict[str, Any]:
    output_dir = Path(run_result["output_dir"])
    history_path = output_dir / "metrics_history.json"
    summary_path = output_dir / "workload_verify_summary.json"
    payload: Dict[str, Any] = dict(run_result)

    if not history_path.exists():
        payload["status"] = "missing_metrics_history"
        return payload

    history = json.loads(history_path.read_text()).get("history", [])
    payload["num_metric_points"] = len(history)
    payload["status"] = "ok" if run_result["returncode"] == 0 else "failed"

    for metric_name in ("train_success_once", "train_success_at_end"):
        values = metric_series(history, metric_name)
        payload[f"{metric_name}_initial"] = first_or_none(values)
        payload[f"{metric_name}_final"] = last_or_none(values)
        payload[f"{metric_name}_avg"] = mean_or_none(values)
        payload[f"{metric_name}_max"] = max(values) if values else None
        if values:
            payload[f"{metric_name}_improvement"] = values[-1] - values[0]
        else:
            payload[f"{metric_name}_improvement"] = None

    elapsed_hours = [float(metric.get("elapsed_hours", 0.0)) for metric in history]
    payload["elapsed_minutes"] = elapsed_hours[-1] * 60.0 if elapsed_hours else 0.0

    if summary_path.exists():
        verify_summary = json.loads(summary_path.read_text())
        payload["stop_reason"] = verify_summary.get("stop_reason")
        payload["stopped_early_zero_success"] = verify_summary.get("stopped_early_zero_success")
        final_eval = verify_summary.get("final_eval_metrics", {})
        for summary_name in (
            "train_success_once_summary",
            "train_success_at_end_summary",
            "eval_success_once_summary",
            "eval_success_at_end_summary",
        ):
            summary = verify_summary.get(summary_name, {})
            if not isinstance(summary, dict):
                continue
            for key, value in summary.items():
                payload[f"{summary_name}_{key}"] = value
        for metric_name in ("success_once", "success_at_end"):
            final_value = final_eval.get(metric_name)
            payload[f"eval_{metric_name}_final"] = final_value

    curve_path = output_dir / "plots" / "success_time_curve.png"
    if curve_path.exists():
        curves_dir.mkdir(parents=True, exist_ok=True)
        copied_curve = curves_dir / f"{safe_name(run_result['env_id'])}.png"
        shutil.copy2(curve_path, copied_curve)
        payload["curve_path"] = str(copied_curve)

    return payload


def write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_gpu_queue(
    gpu_id: str,
    env_ids: Iterable[str],
    batch_name: str,
    max_runtime_hours: float,
    early_stop_zero_success_minutes: float,
    save_video: str,
) -> List[Dict[str, Any]]:
    results = []
    for env_id in env_ids:
        result = run_single_env(
            env_id=env_id,
            gpu_id=gpu_id,
            batch_name=batch_name,
            max_runtime_hours=max_runtime_hours,
            early_stop_zero_success_minutes=early_stop_zero_success_minutes,
            save_video=save_video,
        )
        results.append(result)
        print(
            f"[verify] env={result['env_id']} gpu={result['gpu_id']} rc={result['returncode']} "
            f"wall_clock_min={result['wall_clock_minutes']:.2f} output={result['output_dir']}"
        )
    return results


def main() -> None:
    args = parse_args()
    batch_name = args.batch_name or time.strftime("%Y%m%d-%H%M%S")
    batch_dir = THIS_DIR / "reports" / batch_name
    curves_dir = batch_dir / "curves"
    batch_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    env_ids = args.env_ids or list(HOLD_IN_HAND_VARIANT_ENV_IDS)
    env_ids = list(env_ids)
    active_gpu_ids = gpu_ids[: min(args.max_workers, len(gpu_ids), len(env_ids))]
    gpu_queues = {gpu_id: [] for gpu_id in active_gpu_ids}
    for index, env_id in enumerate(env_ids):
        gpu_id = active_gpu_ids[index % len(active_gpu_ids)]
        gpu_queues[gpu_id].append(env_id)

    run_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(active_gpu_ids)) as executor:
        futures = [
            executor.submit(
                run_gpu_queue,
                gpu_id=gpu_id,
                env_ids=queued_env_ids,
                batch_name=batch_name,
                max_runtime_hours=args.max_runtime_hours,
                early_stop_zero_success_minutes=args.early_stop_zero_success_minutes,
                save_video=args.save_video,
            )
            for gpu_id, queued_env_ids in gpu_queues.items()
            if queued_env_ids
        ]
        for future in as_completed(futures):
            run_results.extend(future.result())

    run_results.sort(key=lambda item: item["env_id"])
    summaries = [summarize_run(result, curves_dir) for result in run_results]
    summaries.sort(key=lambda item: item["env_id"])

    write_summary_csv(summaries, batch_dir / "summary.csv")
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "batch_name": batch_name,
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "env_count": len(env_ids),
                "gpu_ids": gpu_ids,
                "runs": run_results,
            },
            indent=2,
        )
    )
    print(f"[verify] report_dir={batch_dir}")


if __name__ == "__main__":
    main()
