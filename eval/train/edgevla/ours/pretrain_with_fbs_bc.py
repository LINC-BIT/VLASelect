from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append(".")

import train.tinyvla.ours.pretrain_with_fbs_bc as bc
from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task
from train.edgevla.ours.model_with_fbs import convert_to_fbs_model


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = "train/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs"
TEACHER_SEARCH_DIR = Path("train/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple")


def backup_run_sources(output_dir: Path) -> None:
    code_dir = bc.mkdir(output_dir / "code")
    sources = {
        "pretrain_with_fbs_bc.py": Path(__file__).resolve(),
        "model_with_fbs.py": THIS_DIR / "model_with_fbs.py",
        "online_rl_unitree_g1_lift_apple.py": Path(human_task.__file__).resolve(),
        "base_pretrain_with_fbs_bc.py": Path(bc.__file__).resolve(),
        "base_model_with_fbs.py": Path("train/tinyvla/ours/model_with_fbs.py"),
        "online_rl_open_cabinet_drawer.py": Path(bc.reference.__file__).resolve(),
        "workloads_human__init__.py": Path(human_task.human_workload.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {"source": str(source_path), "backup": str(destination)}
    bc.save_json(code_dir / "source_manifest.json", manifest)


def configure_human_bc_defaults() -> None:
    human_task.patch_reference_for_humanoid_env()
    bc.convert_to_fbs_model = convert_to_fbs_model
    bc.backup_run_sources = backup_run_sources
    bc.DEFAULT_WORKDIR = DEFAULT_WORKDIR
    bc.TEACHER_SEARCH_DIR = TEACHER_SEARCH_DIR

    defaults = {
        "env_id": human_task.ENV_ID,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "obs_mode": "rgb+state_dict",
        "output_dir": DEFAULT_WORKDIR,
        "teacher_checkpoint": "auto",
        "student_init_policy": "teacher",
        "num_envs": 128,
        "num_eval_envs": 8,
        "num_steps": 64,
        "num_minibatches": 16,
        "update_epochs": 2,
        "backbone_learning_rate": 6e-5,
        "head_learning_rate": 6e-5,
        "state_learning_rate": 6e-5,
        "value_head_learning_rate": 6e-5,
        "eval_episodes": 16,
        "eval_every_updates": 1,
        "max_runtime_hours": 400.0,
        "cuda_device": "1,2,3,4",
        "save_video": False,
        "action_dim": len(human_task.RIGHT_ARM_AND_HAND_ACTION_INDICES),
        "env_action_dim": 25,
        "state_dim": 73,
        "run_setup_smoke": False,
        "early_stop_zero_success_minutes": 45000.0,
    }
    for field_name, value in defaults.items():
        if field_name in bc.Args.__dataclass_fields__:
            bc.Args.__dataclass_fields__[field_name].default = value


def main() -> None:
    configure_human_bc_defaults()
    args = bc.parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")
    try:
        bc.train(args)
    finally:
        bc.cleanup_runtime()


if __name__ == "__main__":
    main()
