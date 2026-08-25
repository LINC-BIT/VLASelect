#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


SCRIPT_PATH = Path(__file__).resolve()
TASK_ENV_SCAN_DIR = SCRIPT_PATH.parent
EVAL_ROOT = SCRIPT_PATH.parents[2]

STAMP = os.environ.get("SCAN_STAMP", time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
SCAN_ROOT = f"stage1_first_env_scan/{STAMP}"
SUMMARY_PATH = TASK_ENV_SCAN_DIR / "stage1_vlaselect_only_summary.json"
SUMMARY_STAMP_PATH = TASK_ENV_SCAN_DIR / f"stage1_vlaselect_only_summary_{STAMP}.json"
LOG_DIR = TASK_ENV_SCAN_DIR / "stage1_vlaselect_only_logs" / STAMP
SCAN_CKPT_ROOT = EVAL_ROOT / "ckpt" / SCAN_ROOT

TIME_POINTS = os.environ.get(
    "ENV_CHANGE_TIME_POINTS_OVERRIDE",
    "[31,62,96,131,151,163,207,247,271,300]",
)
RUNTIME_LIMIT_SECONDS = max(
    1,
    int(float(os.environ.get("MWE_WORKLOAD_RUNTIME_LIMIT_SECONDS", "30"))),
)
SCAN_EVAL_EVERY_UPDATES = max(
    1,
    int(float(os.environ.get("SCAN_EVAL_EVERY_UPDATES", "2"))),
)
GPU_LIST = [
    gpu.strip()
    for gpu in os.environ.get("SCAN_GPUS", "1,2,3,4,5,6").split(",")
    if gpu.strip()
]
if not GPU_LIST:
    raise SystemExit("SCAN_GPUS must contain at least one GPU id")

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".safetensors"}
CLEANUP_INTERVAL_SECONDS = max(5, int(float(os.environ.get("SCAN_CLEANUP_INTERVAL_SECONDS", "30"))))


ORIGINAL_ENVS = {
    "octo": [
        "PickCubeObjectScaleUp1p2-v1",
        "PickCubeLightStronger50-v1",
        "PickCubeObjectScaleUp1p4-v1",
        "PickCubeLightWeaker50-v1",
        "PushCubeLightWeaker50-v1",
        "PushCubeLightStronger50-v1",
        "PushCubeColorTempHigher50-v1",
        "PushCubeColorTempLower50-v1",
        "PickCubeColorTempHigher50-v1",
        "PickCubeObjectScaleDown1p2-v1",
    ],
    "vla_adapter_new": [
        "HoldHammerInHandObjectScaleDown1p6-v1",
        "HoldWrenchInHandObjectScaleUp1p2-v1",
        "HoldWoodBlockInHandObjectScaleDown1p6-v1",
        "HoldHammerInHandObjectScaleUp1p6-v1",
        "HoldHammerInHandObjectScaleDown1p4-v1",
        "HoldWrenchInHandObjectScaleUp1p6-v1",
        "HoldWrenchInHandObjectScaleUp1p4-v1",
        "HoldHammerInHandObjectScaleDown1p2-v1",
        "HoldHammerInHandObjectScaleUp1p4-v1",
        "HoldWrenchInHandObjectScaleDown1p6-v1",
    ],
    "tinyvla": [
        "OpenCabinetDrawerCabinet1021Default-v1",
        "OpenCabinetDrawerCabinet1016ScaleUp1p3-v1",
        "OpenCabinetDrawerCabinet1027Default-v1",
        "OpenCabinetDrawerCabinet1016ScaleUp1p3-v1",
        "OpenCabinetDrawerCabinet1032Default-v1",
        "OpenCabinetDrawerCabinet1033ScaleUp1p3-v1",
        "OpenCabinetDrawerCabinet1027Default-v1",
        "OpenCabinetDrawerCabinet1021Default-v1",
        "OpenCabinetDrawerCabinet1032Default-v1",
        "OpenCabinetDrawerCabinet1033ScaleUp1p3-v1",
    ],
    "edgevla": [
        "UnitreeG1LiftCubeObjectScaleDown1p3-v1",
        "UnitreeG1LiftCubeLightWeaker50-v1",
        "UnitreeG1LiftCubeLightWeaker50-v1",
        "UnitreeG1LiftCubeObjectPurple-v1",
        "UnitreeG1LiftSphereLightStronger50-v1",
        "UnitreeG1LiftCubeColorTempLower50-v1",
        "UnitreeG1LiftCubeObjectScaleDown1p1-v1",
        "UnitreeG1LiftSphereObjectScaleDown1p3-v1",
        "UnitreeG1LiftCubeColorTempLower50-v1",
        "UnitreeG1LiftCubeObjectPurple-v1",
    ],
}

FAMILY_CONFIGS = {
    "octo": {
        "workload": "single_arm_robot",
        "script": ["bash", "train/octo/ours_single_agent/online_rl_ours_single_agent_cl.sh"],
        "run_dir_suffix": "[agent]",
        "base_env": {
            "TOTAL_TIMESTEPS_OVERRIDE": "1024",
            "NUM_ENVS_OVERRIDE": "2",
            "NUM_EVAL_ENVS_OVERRIDE": "8",
            "NUM_STEPS_OVERRIDE": "16",
            "NUM_EVAL_STEPS_OVERRIDE": "50",
            "EVAL_FREQ_OVERRIDE": str(SCAN_EVAL_EVERY_UPDATES),
            "NUM_MINIBATCHES_OVERRIDE": "2",
            "UPDATE_EPOCHS_OVERRIDE": "1",
            "SUPERVISED_UPDATES_PER_ITER_OVERRIDE": "1",
            "SUPERVISED_BATCH_SIZE_OVERRIDE": "2",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "ENV_CONFIG_PATH_OVERRIDE": "datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json",
        },
    },
    "vla_adapter_new": {
        "workload": "dexterous_hand",
        "script": ["bash", "train/vla_adapter_new/ours/run_online_rl_cl.sh"],
        "run_dir_suffix": "",
        "base_env": {
            "TOTAL_TIMESTEPS_OVERRIDE": "1024",
            "NUM_ENVS_OVERRIDE": "2",
            "NUM_EVAL_ENVS_OVERRIDE": "8",
            "NUM_STEPS_OVERRIDE": "16",
            "NUM_MINIBATCHES_OVERRIDE": "2",
            "UPDATE_EPOCHS_OVERRIDE": "1",
            "EVAL_EVERY_UPDATES_OVERRIDE": str(SCAN_EVAL_EVERY_UPDATES),
            "EVAL_EPISODES_OVERRIDE": "8",
            "ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE": "2",
            "EVAL_MICRO_BATCH_SIZE_OVERRIDE": "3",
            "UPDATE_MICRO_BATCH_SIZE_OVERRIDE": "1",
            "EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE": "45",
        },
    },
    "tinyvla": {
        "workload": "mobile_manipulator",
        "script": ["bash", "train/tinyvla/ours/run_online_rl_cl.sh"],
        "run_dir_suffix": "",
        "base_env": {
            "TOTAL_TIMESTEPS_OVERRIDE": "1024",
            "NUM_ENVS_OVERRIDE": "2",
            "NUM_EVAL_ENVS_OVERRIDE": "8",
            "NUM_STEPS_OVERRIDE": "16",
            "NUM_MINIBATCHES_OVERRIDE": "2",
            "UPDATE_EPOCHS_OVERRIDE": "1",
            "EVAL_EVERY_UPDATES_OVERRIDE": str(SCAN_EVAL_EVERY_UPDATES),
            "EVAL_EPISODES_OVERRIDE": "8",
            "ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE": "2",
            "EVAL_MICRO_BATCH_SIZE_OVERRIDE": "3",
            "UPDATE_MICRO_BATCH_SIZE_OVERRIDE": "1",
            "EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE": "45",
        },
    },
    "edgevla": {
        "workload": "humanoid_robot",
        "script": ["bash", "train/edgevla/ours/run_online_rl_cl.sh"],
        "run_dir_suffix": "",
        "base_env": {
            "TOTAL_TIMESTEPS_OVERRIDE": "1024",
            "NUM_ENVS_OVERRIDE": "2",
            "NUM_EVAL_ENVS_OVERRIDE": "8",
            "NUM_STEPS_OVERRIDE": "16",
            "NUM_MINIBATCHES_OVERRIDE": "2",
            "UPDATE_EPOCHS_OVERRIDE": "1",
            "EVAL_EVERY_UPDATES_OVERRIDE": str(SCAN_EVAL_EVERY_UPDATES),
            "EVAL_EPISODES_OVERRIDE": "8",
            "ROLLOUT_MICRO_BATCH_SIZE_OVERRIDE": "2",
            "EVAL_MICRO_BATCH_SIZE_OVERRIDE": "3",
            "UPDATE_MICRO_BATCH_SIZE_OVERRIDE": "1",
            "EARLY_STOP_ZERO_SUCCESS_MINUTES_OVERRIDE": "45",
            "ROLLOUT_PROGRESS_LOG_INTERVAL_OVERRIDE": "1",
            "ONLINE_BUFFER_CAPACITY_OVERRIDE": "256",
            "EXPERT_BUFFER_CAPACITY_OVERRIDE": "256",
            "EXPERT_TARGET_SUCCESS_TRAJECTORIES_OVERRIDE": "0",
            "EXPERT_COLLECT_NUM_ENVS_OVERRIDE": "1",
            "EXPERT_COLLECT_MAX_STEPS_OVERRIDE": "128",
        },
    },
}

COMMON_ENV = {
    "PYTHONPATH": str(EVAL_ROOT),
    "LAUNCH_DIRECT": "1",
    "TAIL_LOG": "0",
    "SAVE_VIDEO_OVERRIDE": "false",
    "TRAIN_VIDEO_NUM_ENVS_OVERRIDE": "1",
    "TEST_VIDEO_NUM_ENVS_OVERRIDE": "1",
    "TEST_VIDEO_EPISODES_OVERRIDE": "1",
    "MWE_ACTIVE_RUNTIME_ONLY": "1",
    "VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE": "0.0",
    "VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED": "0",
}


def delete_checkpoint_artifacts(root: Path) -> int:
    if not root.exists():
        return 0
    deleted = 0
    for artifact in root.rglob("*"):
        if not artifact.is_file() or artifact.suffix not in CHECKPOINT_SUFFIXES:
            continue
        try:
            artifact.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return deleted


def cleanup_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(CLEANUP_INTERVAL_SECONDS):
        deleted = delete_checkpoint_artifacts(SCAN_CKPT_ROOT)
        if deleted:
            print(
                f"[scan] cleanup deleted_files={deleted} root={SCAN_CKPT_ROOT}",
                flush=True,
            )


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def move_first(original: list[str], candidate: str) -> list[str]:
    remainder: list[str] = []
    removed = False
    for item in original:
        if not removed and item == candidate:
            removed = True
            continue
        remainder.append(item)
    return [candidate] + remainder


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_accuracy_series(run_dir: Path) -> list[tuple[float, float]]:
    metrics_history_path = run_dir / "metrics_history.json"
    payload = load_json(metrics_history_path)
    series: list[tuple[float, float]] = []
    if isinstance(payload, dict):
        history = payload.get("history", [])
        if isinstance(history, list):
            for idx, metric in enumerate(history):
                if not isinstance(metric, dict):
                    continue
                raw_value = metric.get("eval_success_once")
                if raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                if not math.isfinite(value):
                    continue
                elapsed_hours = metric.get("elapsed_hours")
                try:
                    elapsed_minutes = (
                        float(elapsed_hours) * 60.0
                        if elapsed_hours is not None
                        else float(idx)
                    )
                except Exception:
                    elapsed_minutes = float(idx)
                series.append((elapsed_minutes, value))
    if series:
        return series

    for fallback_path in (run_dir / "final_eval_metrics.json", run_dir / "latest_metrics.json"):
        payload = load_json(fallback_path)
        if not isinstance(payload, dict):
            continue
        raw_value = payload.get("success_once", payload.get("eval_success_once"))
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        return [(0.0, value)]
    return []


def recompute_best(family_payload: dict) -> None:
    candidates = [
        candidate
        for candidate in family_payload["candidates"]
        if candidate.get("avg_accuracy") is not None
    ]
    if not candidates:
        family_payload["best"] = None
        return
    best = max(
        candidates,
        key=lambda item: (
            float(item["avg_accuracy"]),
            float(item.get("final_accuracy") or float("-inf")),
        ),
    )
    family_payload["best"] = {
        "candidate_index": best["candidate_index"],
        "env_id": best["env_id"],
        "avg_accuracy": best["avg_accuracy"],
        "final_accuracy": best.get("final_accuracy"),
        "run_dir": best.get("run_dir"),
        "log_path": best.get("log_path"),
    }


def build_results_payload() -> tuple[dict, list[dict]]:
    results = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "vlaselect_only_first_env_scan",
        "scan_stamp": STAMP,
        "scan_root": SCAN_ROOT,
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "results": {},
    }
    tasks: list[dict] = []
    for family, original_envs in ORIGINAL_ENVS.items():
        payload = {
            "workload": FAMILY_CONFIGS[family]["workload"],
            "candidates": [],
            "best": None,
        }
        for idx, env_id in enumerate(unique_in_order(original_envs)):
            env_sequence = move_first(original_envs, env_id)
            candidate_tag = f"{idx:02d}_{env_id}"
            exp_name = f"{SCAN_ROOT}/{family}/{candidate_tag}"
            candidate_root = EVAL_ROOT / "ckpt" / SCAN_ROOT / family / candidate_tag
            run_dir = candidate_root
            suffix = FAMILY_CONFIGS[family]["run_dir_suffix"]
            if suffix:
                run_dir = run_dir / suffix
            log_path = LOG_DIR / family / f"{candidate_tag}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            candidate = {
                "candidate_index": idx,
                "env_id": env_id,
                "env_sequence": env_sequence,
                "exp_name": exp_name,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "status": "pending",
            }
            payload["candidates"].append(candidate)
            tasks.append(
                {
                    "family": family,
                    "candidate_index": idx,
                    "candidate_tag": candidate_tag,
                    "env_id": env_id,
                    "env_sequence": env_sequence,
                    "exp_name": exp_name,
                    "run_dir": run_dir,
                    "cleanup_root": candidate_root,
                    "log_path": log_path,
                }
            )
        results["results"][family] = payload

    interleaved: list[dict] = []
    pools = {
        family: [task for task in tasks if task["family"] == family]
        for family in ("octo", "vla_adapter_new", "tinyvla", "edgevla")
    }
    while True:
        progressed = False
        for family in ("octo", "vla_adapter_new", "tinyvla", "edgevla"):
            if not pools[family]:
                continue
            interleaved.append(pools[family].pop(0))
            progressed = True
        if not progressed:
            break
    return results, interleaved


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results, tasks = build_results_payload()
    summary_lock = threading.Lock()

    def write_summary() -> None:
        results["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = json.dumps(results, indent=2)
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        SUMMARY_STAMP_PATH.write_text(text, encoding="utf-8")

    write_summary()

    delete_checkpoint_artifacts(SCAN_CKPT_ROOT)
    cleanup_stop_event = threading.Event()
    cleanup_thread = threading.Thread(
        target=cleanup_loop,
        args=(cleanup_stop_event,),
        daemon=True,
    )
    cleanup_thread.start()

    gpu_queue: Queue[str] = Queue()
    for gpu in GPU_LIST:
        gpu_queue.put(gpu)

    def run_task(task: dict) -> None:
        gpu = gpu_queue.get()
        family = task["family"]
        family_cfg = FAMILY_CONFIGS[family]
        start_time = time.time()
        env = os.environ.copy()
        env.update(COMMON_ENV)
        env.update(family_cfg["base_env"])
        env.update(
            {
                "CUDA_DEVICES": gpu,
                "EXP_NAME": task["exp_name"],
                "RUN_NAME_OVERRIDE": task["candidate_tag"],
                "ENV_ID_OVERRIDE": task["env_id"],
                "ENVS_ID_OVERRIDE": repr(task["env_sequence"]),
                "ENV_CHANGE_TIME_POINTS_OVERRIDE": TIME_POINTS,
            }
        )
        if family == "octo":
            env["MAX_TIME_OVERRIDE"] = f"{RUNTIME_LIMIT_SECONDS / 60.0:.6f}"
        else:
            env["MAX_RUNTIME_HOURS_OVERRIDE"] = f"{RUNTIME_LIMIT_SECONDS / 3600.0:.12f}"

        print(
            f"[scan] start family={family} candidate={task['candidate_index']} "
            f"env={task['env_id']} gpu={gpu} runtime_limit_seconds={RUNTIME_LIMIT_SECONDS}",
            flush=True,
        )
        with task["log_path"].open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                family_cfg["script"],
                cwd=str(EVAL_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wallclock_seconds = round(time.time() - start_time, 2)
        series = collect_accuracy_series(task["run_dir"])
        avg_accuracy = (
            sum(value for _, value in series) / len(series) if series else None
        )
        final_accuracy = series[-1][1] if series else None
        deleted_after_task = delete_checkpoint_artifacts(task["cleanup_root"])
        with summary_lock:
            candidate = results["results"][family]["candidates"][task["candidate_index"]]
            candidate.update(
                {
                    "status": "done",
                    "return_code": proc.returncode,
                    "wallclock_seconds": wallclock_seconds,
                    "num_points": len(series),
                    "avg_accuracy": avg_accuracy,
                    "final_accuracy": final_accuracy,
                    "gpu": gpu,
                }
            )
            if proc.returncode != 0 and not series:
                candidate["error"] = f"return_code:{proc.returncode}"
            elif proc.returncode != 0:
                candidate["warning"] = f"return_code:{proc.returncode}"
            recompute_best(results["results"][family])
            write_summary()
        print(
            f"[scan] done family={family} candidate={task['candidate_index']} "
            f"env={task['env_id']} gpu={gpu} rc={proc.returncode} "
            f"points={len(series)} avg={avg_accuracy} deleted_files={deleted_after_task}",
            flush=True,
        )
        gpu_queue.put(gpu)

    try:
        with ThreadPoolExecutor(max_workers=len(GPU_LIST)) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
    finally:
        cleanup_stop_event.set()
        cleanup_thread.join(timeout=1.0)
        delete_checkpoint_artifacts(SCAN_CKPT_ROOT)

    results["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_summary()
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_PATH),
                "stamp_summary": str(SUMMARY_STAMP_PATH),
                "scan_root": SCAN_ROOT,
                "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
