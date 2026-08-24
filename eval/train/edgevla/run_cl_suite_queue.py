from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train.common.gpu_auto_select import get_method_gpu_override_map, resolve_method_gpu_map


METHOD_ORDER = ["ppo_gen", "ours", "conrft", "flare", "edgeta", "convertnet", "improv_vla", "self_improv", "vla_rft", "world_env"]
DISPLAY_NAMES = {
    "conrft": "ConRFT",
    "convertnet": "ConvertNet",
    "edgeta": "EdgeTA",
    "flare": "FLaRe",
    "improv_vla": "Improv-VLA",
    "ours": "Ours",
    "ppo_gen": "PPO-Gen",
    "self_improv": "Self-Improv",
    "vla_rft": "VLA-RFT",
    "world_env": "WorldEnv",
}
GPU_BY_METHOD = {
    "ppo_gen": 6,
    "conrft": 1,
    "convertnet": 1,
    "edgeta": 6,
    "flare": 6,
    "improv_vla": 7,
    "ours": 1,
    "self_improv": 6,
    "vla_rft": 7,
    "world_env": 6,
}
SCRIPT_BY_METHOD = {
    "conrft": "train/edgevla/conrft/run_online_rl.sh",
    "convertnet": "train/edgevla/convertnet/run_online_rl.sh",
    "edgeta": "train/edgevla/edgeta/run_online_rl.sh",
    "flare": "train/edgevla/flare/run_online_rl.sh",
    "improv_vla": "train/edgevla/improv_vla/run_online_rl.sh",
    "ours": "train/edgevla/ours/run_online_rl_cl.sh",
    "ppo_gen": "train/edgevla/ppo_gen/run_online_rl.sh",
    "self_improv": "train/edgevla/self_improv/run_online_rl.sh",
    "vla_rft": "train/edgevla/vla_rft/run_online_rl.sh",
    "world_env": "train/edgevla/world_env/run_online_rl.sh",
}
def build_gpu_queues(gpu_by_method: dict[str, int]) -> dict[int, list[str]]:
    queues: dict[int, list[str]] = {}
    for method in METHOD_ORDER:
        gpu = gpu_by_method[method]
        queues.setdefault(gpu, []).append(method)
    return queues


class SuiteScheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root_dir = args.root_dir.resolve()
        self.suite_root = args.suite_root
        self.launch_log_dir = self.suite_root / "launch_logs"
        self.pid_dir = self.suite_root / "pids"
        self.manifest_path = args.manifest
        override_map = get_method_gpu_override_map(args.gpu_by_method_override)
        self.gpu_by_method = resolve_method_gpu_map(METHOD_ORDER, GPU_BY_METHOD, override_map)
        self.gpu_queues = build_gpu_queues(self.gpu_by_method)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.current_processes: dict[str, subprocess.Popen[Any]] = {}

    def write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self.initial_manifest()
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def initial_manifest(self) -> dict[str, Any]:
        return {
            "suite_stamp": self.args.suite_stamp,
            "smoke": self.args.smoke,
            "queued_per_gpu": True,
            "scheduler_pid": os.getpid(),
            "scheduler_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "monitor_interval_seconds": self.args.monitor_interval_seconds,
            "plot_interval_seconds": self.args.plot_interval_seconds,
            "envs_id": self.args.envs_id,
            "env_change_time_points": self.args.env_change_time_points,
            "resource_change_time_points": self.args.resource_change_time_points or None,
            "resource_change_directions": self.args.resource_change_directions or None,
            "resource_change_factors": self.args.resource_change_factors or None,
            "inherited_suite": None,
            "gpu_queues": {str(gpu): methods for gpu, methods in self.gpu_queues.items()},
            "methods": [
                {
                    "name": method,
                    "display_name": DISPLAY_NAMES[method],
                    "gpu": self.gpu_by_method[method],
                    "status": "queued",
                    "inherited_from": None,
                    "pid": None,
                    "monitor_pid": None,
                    "exit_code": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                    "run_dir": str(self.suite_root / method / "run"),
                    "log_file": str(self.launch_log_dir / f"{method}.train.log"),
                    "script_path": SCRIPT_BY_METHOD[method],
                    "output_dir_base": str(self.suite_root / method),
                    "run_name": "run",
                }
                for method in METHOD_ORDER
            ],
        }

    def update_manifest(self, method_name: str | None = None, **updates: Any) -> None:
        with self.lock:
            manifest = self.load_manifest()
            manifest["scheduler_pid"] = os.getpid()
            manifest["queued_per_gpu"] = True
            if method_name is None:
                manifest.update(updates)
            else:
                for method in manifest["methods"]:
                    if method["name"] == method_name:
                        method.update(updates)
                        break
            self.write_json_atomic(self.manifest_path, manifest)

    def append_train_log_line(self, method: str, message: str) -> None:
        log_file = self.launch_log_dir / f"{method}.train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_file.open("ab") as handle:
            handle.write(f"[{timestamp}] {message}\n".encode("utf-8", errors="replace"))

    def build_command(self, method: str, gpu: int) -> list[str]:
        env_items = [
            f"CUDA_DEVICES={gpu}",
            f"OUTPUT_DIR_BASE_OVERRIDE={self.suite_root / method}",
            "RUN_NAME_OVERRIDE=run",
            f"LOG_FILE_OVERRIDE={self.launch_log_dir / f'{method}.train.log'}",
            f"TAIL_LOG={self.args.tail_log}",
            "LAUNCH_DIRECT=1",
            "SAVE_VIDEO_OVERRIDE=false",
            f"ENVS_ID_OVERRIDE={self.args.envs_id}",
            f"ENV_CHANGE_TIME_POINTS_OVERRIDE={self.args.env_change_time_points}",
            f"RESOURCE_CHANGE_TIME_POINTS_OVERRIDE={self.args.resource_change_time_points}",
            f"RESOURCE_CHANGE_DIRECTIONS_OVERRIDE={self.args.resource_change_directions}",
            f"RESOURCE_CHANGE_FACTORS_OVERRIDE={self.args.resource_change_factors}",
        ]
        if self.args.smoke:
            env_items.extend(
                [
                    "TOTAL_TIMESTEPS_OVERRIDE=1024",
                    "NUM_ENVS_OVERRIDE=2",
                    "NUM_EVAL_ENVS_OVERRIDE=3",
                    "NUM_STEPS_OVERRIDE=16",
                    "NUM_MINIBATCHES_OVERRIDE=2",
                    "UPDATE_EPOCHS_OVERRIDE=1",
                    "EVAL_EPISODES_OVERRIDE=4",
                    f"MAX_RUNTIME_HOURS_OVERRIDE={self.args.smoke_max_runtime_hours}",
                    "EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE=45",
                    "ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE=2",
                    "EVAL_MICRO_BATCH_SIZE_OVERRIDE=3",
                    "UPDATE_MICRO_BATCH_SIZE_OVERRIDE=1",
                    "ROLLOUT_PROGRESS_LOG_INTERVAL_OVERRIDE=1",
                    "SUPERVISED_UPDATES_PER_ITER_OVERRIDE=1",
                    "SUPERVISED_BATCH_SIZE_OVERRIDE=2",
                    "ONLINE_BUFFER_CAPACITY_OVERRIDE=256",
                    "EXPERT_BUFFER_CAPACITY_OVERRIDE=256",
                    "EXPERT_TARGET_SUCCESS_TRAJECTORIES_OVERRIDE=0",
                    "EXPERT_COLLECT_NUM_ENVS_OVERRIDE=1",
                    "EXPERT_COLLECT_MAX_STEPS_OVERRIDE=128",
                    "MWE_ACTIVE_RUNTIME_ONLY=1",
                ]
            )
        return ["env", *env_items, "bash", SCRIPT_BY_METHOD[method]]

    def start_monitor(self, method: str, gpu: int, train_pid: int) -> subprocess.Popen[Any]:
        monitor_log = self.launch_log_dir / f"{method}.gpu_monitor.log"
        monitor_pid_file = self.pid_dir / f"{method}.gpu_monitor.pid"
        command = [
            sys.executable,
            "-u",
            "-m",
            "train.common.monitor_gpu_metrics",
            "--pid",
            str(train_pid),
            "--gpu-index",
            str(gpu),
            "--run-dir",
            str(self.suite_root / method / "run"),
            "--interval-seconds",
            str(self.args.monitor_interval_seconds),
            "--label",
            method,
        ]
        monitor_log.parent.mkdir(parents=True, exist_ok=True)
        with monitor_log.open("ab") as log_f:
            proc = subprocess.Popen(
                command,
                cwd=str(self.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        monitor_pid_file.write_text(str(proc.pid), encoding="utf-8")
        return proc

    def run_method(self, method: str) -> None:
        if self.stop_event.is_set():
            self.update_manifest(method, status="cancelled", finished_at_utc=datetime.now(timezone.utc).isoformat())
            return

        gpu = self.gpu_by_method[method]
        self.append_train_log_line(method, f"[queue] method={method} gpu={gpu} wait finished; launching")
        log_file = self.launch_log_dir / f"{method}.train.log"
        pid_file = self.pid_dir / f"{method}.pid"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("ab") as log_f:
            train_proc = subprocess.Popen(
                self.build_command(method, gpu),
                cwd=str(self.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        self.current_processes[f"{method}:train"] = train_proc
        pid_file.write_text(str(train_proc.pid), encoding="utf-8")

        monitor_proc = self.start_monitor(method, gpu, train_proc.pid)
        self.current_processes[f"{method}:monitor"] = monitor_proc
        self.update_manifest(
            method,
            status="running",
            pid=train_proc.pid,
            monitor_pid=monitor_proc.pid,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        exit_code = train_proc.wait()

        try:
            monitor_proc.wait(timeout=max(5.0, self.args.monitor_interval_seconds + 5.0))
        except subprocess.TimeoutExpired:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()

        self.current_processes.pop(f"{method}:train", None)
        self.current_processes.pop(f"{method}:monitor", None)
        self.update_manifest(
            method,
            status="completed" if exit_code == 0 else ("timed_out" if exit_code == 124 else "failed"),
            exit_code=exit_code,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def run_gpu_queue(self, gpu: int, methods: list[str]) -> None:
        total = len(methods)
        for index, method in enumerate(methods, start=1):
            if index == 1:
                self.append_train_log_line(method, f"[queue] method={method} gpu={gpu} first in queue; launching when scheduler starts it")
            else:
                waiting_for = ",".join(methods[: index - 1])
                self.append_train_log_line(method, f"[queue] method={method} gpu={gpu} queued at position={index}/{total}; waiting for previous methods on this GPU: {waiting_for}")
            if self.stop_event.is_set():
                self.update_manifest(method, status="cancelled", finished_at_utc=datetime.now(timezone.utc).isoformat())
                continue
            self.run_method(method)

    def terminate_current_processes(self, *_args: Any) -> None:
        self.stop_event.set()
        for proc in list(self.current_processes.values()):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(5)
        for proc in list(self.current_processes.values()):
            if proc.poll() is None:
                proc.kill()

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.terminate_current_processes)
        signal.signal(signal.SIGINT, self.terminate_current_processes)
        self.suite_root.mkdir(parents=True, exist_ok=True)
        self.launch_log_dir.mkdir(parents=True, exist_ok=True)
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.write_json_atomic(self.manifest_path, self.initial_manifest())

        threads = [
            threading.Thread(target=self.run_gpu_queue, args=(gpu, methods), daemon=False)
            for gpu, methods in self.gpu_queues.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_status = "cancelled" if self.stop_event.is_set() else "finished"
        self.update_manifest(scheduler_status=final_status, scheduler_finished_at_utc=datetime.now(timezone.utc).isoformat())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-stamp", required=True)
    parser.add_argument("--envs-id", required=True)
    parser.add_argument("--env-change-time-points", required=True)
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--plot-interval-seconds", type=float, default=60.0)
    parser.add_argument("--tail-log", default="0")
    parser.add_argument("--resource-change-time-points", default="")
    parser.add_argument("--resource-change-directions", default="")
    parser.add_argument("--resource-change-factors", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-by-method-override", default="")
    parser.add_argument("--smoke-max-runtime-hours", type=float, default=0.0084)
    args = parser.parse_args()
    SuiteScheduler(args).run()


if __name__ == "__main__":
    main()
