from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from workloads.table_top import TABLE_TOP_VARIANT_ENV_IDS


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKLOAD_VERIFY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = WORKLOAD_VERIFY_DIR / "results"

DEFAULT_GPU_IDS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ENV_CHANGE_MINUTE = 30.0
DEFAULT_MAX_TIME_MINUTES = 35.0

DEFAULT_ENV_CONFIG_PATH = (
    "datasets/PickCube-v1/motionplanning/"
    "trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json"
)
DEFAULT_STATE_NORM_STATS_PATH = "ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth"
DEFAULT_CHECKPOINT_PATH = (
    "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/"
    "20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
)
DEFAULT_TAG = (
    "ours-single-agent-cl-targetsingletraj-feedif0.2-"
    "regen_per_rollout_0.1_4-small_policy-feedback0.1-"
    "arch_update0.05-reset_optimizer"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_suite_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_gpu_ids(raw_value: str | None) -> list[int]:
    if raw_value is None:
        return list(DEFAULT_GPU_IDS)
    gpu_ids = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gpu_ids.append(int(chunk))
    if not gpu_ids:
        raise ValueError("No GPU ids were provided")
    return gpu_ids


def get_workload_env_ids() -> list[str]:
    return list(TABLE_TOP_VARIANT_ENV_IDS)


@dataclass
class WorkloadRunSpec:
    env_id: str
    exp_name: str
    run_dir: str
    log_file: str
    command: list[str]
    gpu_id: int | None = None
    status: str = "queued"
    pid: int | None = None
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_exp_name(suite_name: str, env_id: str) -> str:
    return f"workload_verify/{suite_name}/{env_id}"


def build_run_dir(exp_name: str) -> Path:
    return REPO_ROOT / "ckpt" / exp_name


def build_training_command(
    *,
    env_id: str,
    exp_name: str,
    env_change_minute: float = DEFAULT_ENV_CHANGE_MINUTE,
    max_time_minutes: float = DEFAULT_MAX_TIME_MINUTES,
    track: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "train.octo.ours_single_agent.online_rl_cl",
        "--exp-name",
        exp_name,
        "--env-id",
        env_id,
        "--envs-id",
        json.dumps([env_id]),
        "--env-change-time-points",
        json.dumps([env_change_minute]),
        "--env_config_path",
        DEFAULT_ENV_CONFIG_PATH,
        "--state-norm-stats-path",
        DEFAULT_STATE_NORM_STATS_PATH,
        "--checkpoint",
        DEFAULT_CHECKPOINT_PATH,
        "--total_timesteps",
        "100000000",
        "--learning_rate",
        "3e-5",
        "--eval_freq",
        "1",
        "--max-sparsity",
        "0.8",
        "--num_envs",
        "256",
        "--num_eval_envs",
        "32",
        "--num_minibatches",
        "16",
        "--small_model_generation_strategy",
        "target-single-traj",
        "--small_model_feedback_schedule",
        "before_per_rollout_if_success_improv_is_larger_than_0.2",
        "--small_model_regeneration_schedule",
        "before_per_rollout_if_success_improv_less_than_0.1_for_4_iters",
        "--small_model_feedback_alpha",
        "0.1",
        "--small_model_regeneration_increment_ratio",
        "0.05",
        "--reset_optimizer_after_regeneration",
        "--small_model_generation_policy",
        "small",
        "--tag",
        DEFAULT_TAG,
        "--max_time",
        str(max_time_minutes),
    ]
    if track:
        command.append("--track")
    return command


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_command_script(path: Path, command: Iterable[str], gpu_id: int) -> None:
    rendered = " ".join(json.dumps(part) for part in command)
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"cd {json.dumps(str(REPO_ROOT))}\n"
        f"export CUDA_VISIBLE_DEVICES={gpu_id}\n"
        f"{rendered}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
