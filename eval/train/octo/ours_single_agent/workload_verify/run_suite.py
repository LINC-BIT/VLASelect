from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

from train.octo.ours_single_agent.workload_verify.common import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_ENV_CHANGE_MINUTE,
    DEFAULT_MAX_TIME_MINUTES,
    WorkloadRunSpec,
    build_exp_name,
    build_run_dir,
    build_training_command,
    default_suite_name,
    get_workload_env_ids,
    parse_gpu_ids,
    utc_now_iso,
    write_command_script,
    write_json,
)
from train.octo.ours_single_agent.workload_verify.summarize_suite import render_suite_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6")
    parser.add_argument("--env-regex", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--env-change-minute", type=float, default=DEFAULT_ENV_CHANGE_MINUTE)
    parser.add_argument("--max-time-minutes", type=float, default=DEFAULT_MAX_TIME_MINUTES)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def make_suite_manifest_payload(
    *,
    suite_name: str,
    suite_dir: Path,
    gpu_ids: list[int],
    args: argparse.Namespace,
    runs: list[WorkloadRunSpec],
) -> dict:
    return {
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "created_at": utc_now_iso(),
        "gpu_ids": gpu_ids,
        "track": bool(args.track),
        "env_change_minute": float(args.env_change_minute),
        "max_time_minutes": float(args.max_time_minutes),
        "runs": [run.to_dict() for run in runs],
    }


def write_manifest(
    *,
    manifest_path: Path,
    suite_name: str,
    suite_dir: Path,
    gpu_ids: list[int],
    args: argparse.Namespace,
    runs: list[WorkloadRunSpec],
) -> None:
    payload = make_suite_manifest_payload(
        suite_name=suite_name,
        suite_dir=suite_dir,
        gpu_ids=gpu_ids,
        args=args,
        runs=runs,
    )
    write_json(manifest_path, payload)


def terminate_running_processes(active_processes: dict[int, tuple[subprocess.Popen, WorkloadRunSpec, object]]) -> None:
    for _, (process, _, log_handle) in list(active_processes.items()):
        try:
            process.terminate()
        finally:
            log_handle.close()
    for _, (process, _, _) in list(active_processes.items()):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    args = parse_args()
    suite_name = args.suite_name or default_suite_name()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    suite_dir = args.results_root.resolve() / suite_name
    commands_dir = suite_dir / "commands"
    logs_dir = suite_dir / "logs"
    manifest_path = suite_dir / "manifest.json"

    env_ids = get_workload_env_ids()
    if args.env_regex:
        env_pattern = re.compile(args.env_regex)
        env_ids = [env_id for env_id in env_ids if env_pattern.search(env_id)]
    if args.limit is not None:
        env_ids = env_ids[: args.limit]
    if not env_ids:
        raise ValueError("No workload envs matched the provided filters")

    runs: list[WorkloadRunSpec] = []
    for env_id in env_ids:
        exp_name = build_exp_name(suite_name, env_id)
        run_dir = build_run_dir(exp_name)
        log_file = logs_dir / f"{env_id}.log"
        command = build_training_command(
            env_id=env_id,
            exp_name=exp_name,
            env_change_minute=args.env_change_minute,
            max_time_minutes=args.max_time_minutes,
            track=args.track,
        )
        runs.append(
            WorkloadRunSpec(
                env_id=env_id,
                exp_name=exp_name,
                run_dir=str(run_dir),
                log_file=str(log_file),
                command=command,
            )
        )

    commands_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        write_command_script(commands_dir / f"{run.env_id}.sh", run.command, gpu_id=0)
    write_manifest(
        manifest_path=manifest_path,
        suite_name=suite_name,
        suite_dir=suite_dir,
        gpu_ids=gpu_ids,
        args=args,
        runs=runs,
    )
    render_suite_outputs(suite_dir)

    pending_runs = list(runs)
    active_processes: dict[int, tuple[subprocess.Popen, WorkloadRunSpec, object]] = {}
    had_error = False

    try:
        while pending_runs or active_processes:
            for gpu_id in gpu_ids:
                if gpu_id in active_processes or not pending_runs:
                    continue
                run = pending_runs.pop(0)
                command_path = commands_dir / f"{run.env_id}.sh"
                write_command_script(command_path, run.command, gpu_id=gpu_id)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                log_handle = open(run.log_file, "a", encoding="utf-8")
                process = subprocess.Popen(
                    run.command,
                    cwd=str(Path(__file__).resolve().parents[4]),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                run.gpu_id = gpu_id
                run.pid = process.pid
                run.status = "running"
                run.started_at = utc_now_iso()
                active_processes[gpu_id] = (process, run, log_handle)
                print(f"[launch] env={run.env_id} gpu={gpu_id} pid={process.pid}")
                write_manifest(
                    manifest_path=manifest_path,
                    suite_name=suite_name,
                    suite_dir=suite_dir,
                    gpu_ids=gpu_ids,
                    args=args,
                    runs=runs,
                )
                render_suite_outputs(suite_dir)

            if not active_processes:
                continue

            time.sleep(args.poll_seconds)
            finished_gpu_ids: list[int] = []
            for gpu_id, (process, run, log_handle) in active_processes.items():
                returncode = process.poll()
                if returncode is None:
                    continue
                finished_gpu_ids.append(gpu_id)
                log_handle.close()
                run.returncode = int(returncode)
                run.finished_at = utc_now_iso()
                run.status = "completed" if returncode == 0 else "failed"
                if returncode != 0:
                    had_error = True
                print(f"[finish] env={run.env_id} gpu={gpu_id} returncode={returncode}")

            for gpu_id in finished_gpu_ids:
                active_processes.pop(gpu_id, None)

            if finished_gpu_ids:
                write_manifest(
                    manifest_path=manifest_path,
                    suite_name=suite_name,
                    suite_dir=suite_dir,
                    gpu_ids=gpu_ids,
                    args=args,
                    runs=runs,
                )
            render_suite_outputs(suite_dir)

            if had_error and args.stop_on_error:
                raise RuntimeError("At least one workload run failed")

    except KeyboardInterrupt:
        print("Interrupted, terminating active processes...")
        terminate_running_processes(active_processes)
        raise
    except Exception:
        terminate_running_processes(active_processes)
        raise

    write_manifest(
        manifest_path=manifest_path,
        suite_name=suite_name,
        suite_dir=suite_dir,
        gpu_ids=gpu_ids,
        args=args,
        runs=runs,
    )
    summary_path = render_suite_outputs(suite_dir)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
