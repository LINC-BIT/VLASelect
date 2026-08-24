from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(".")

import train.tinyvla.ppo_gen.online_rl as algo
from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task
from train.edgevla.ours.model_with_fbs import convert_to_fbs_model


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_OUTPUT_DIR = "train/edgevla/ppo_gen/outputs"
DEFAULT_ENVS_ID = "['UnitreeG1LiftApple-v1','UnitreeG1LiftApple-v1']"
DEFAULT_ENV_CHANGE_TIME_POINTS = "[10,20]"
DEFAULT_STATIC_MODEL_CHECKPOINT = (
    "ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt"
)


def copy_run_metadata(output_dir: Path, args: algo.Args) -> None:
    algo.save_json(output_dir / "args.json", asdict(args))
    code_dir = algo.mkdir(output_dir / "code")
    sources = {
        "online_rl.py": Path(__file__).resolve(),
        "base_online_rl.py": Path(algo.__file__).resolve(),
        "online_rl_unitree_g1_lift_apple.py": Path(human_task.__file__).resolve(),
        "model_with_fbs.py": THIS_DIR.parent / "ours" / "model_with_fbs.py",
        "base_model_with_fbs.py": Path("train/tinyvla/ours/model_with_fbs.py"),
        "online_rl_open_cabinet_drawer.py": Path(algo.reference.__file__).resolve(),
        "workloads_human__init__.py": Path(human_task.human_workload.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {"source": str(source_path), "backup": str(destination)}
    algo.save_json(code_dir / "source_manifest.json", manifest)


def configure_human_defaults() -> None:
    human_task.patch_reference_for_humanoid_env()
    algo.convert_to_fbs_model = convert_to_fbs_model
    algo.copy_run_metadata = copy_run_metadata
    algo.DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR
    algo.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    algo.DEFAULT_ENVS_ID = DEFAULT_ENVS_ID
    algo.DEFAULT_ENV_CHANGE_TIME_POINTS = DEFAULT_ENV_CHANGE_TIME_POINTS
    algo.DEFAULT_STATIC_MODEL_CHECKPOINT = DEFAULT_STATIC_MODEL_CHECKPOINT

    defaults = {
        "env_id": human_task.ENV_ID,
        "envs_id": DEFAULT_ENVS_ID,
        "env_change_time_points": DEFAULT_ENV_CHANGE_TIME_POINTS,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "obs_mode": "rgb+state_dict",
        "model_dir": DEFAULT_MODEL_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "static_model_checkpoint": DEFAULT_STATIC_MODEL_CHECKPOINT,
        "num_envs": 128,
        "num_eval_envs": 8,
        "num_steps": 64,
        "num_minibatches": 16,
        "update_epochs": 2,
        "learning_rate": 6e-5,
        "backbone_learning_rate": 6e-5,
        "head_learning_rate": 6e-5,
        "state_learning_rate": 6e-5,
        "value_head_learning_rate": 6e-5,
        "eval_episodes": 16,
        "eval_every_updates": 1,
        "max_runtime_hours": 400.0,
        "cuda_device": "1,2,3,4",
        "run_setup_smoke": False,
        "save_video": False,
        "action_dim": len(human_task.RIGHT_ARM_AND_HAND_ACTION_INDICES),
        "env_action_dim": 25,
        "state_dim": 73,
        "early_stop_zero_success_minutes": 45000.0,
    }
    for field_name, value in defaults.items():
        if field_name in algo.Args.__dataclass_fields__:
            algo.Args.__dataclass_fields__[field_name].default = value


def main() -> None:
    configure_human_defaults()
    args = algo.parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")
    try:
        algo.train(args)
    finally:
        algo.cleanup_runtime()


if __name__ == "__main__":
    main()
