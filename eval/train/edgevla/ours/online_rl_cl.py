from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parents[2]
for candidate in (THIS_DIR, EVAL_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import train.tinyvla.ours.online_rl_cl as algo
from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task
from train.edgevla.ours.model_with_fbs import convert_to_fbs_model


DEFAULT_MODEL_DIR = "ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_OUTPUT_DIR = "train/edgevla/ours/outputs/online_rl_cl"
DEFAULT_ENVS_ID = (
    "['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1',"
    "'UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1',"
    "'UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1',"
    "'UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1',"
    "'UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"
)
DEFAULT_ENV_CHANGE_TIME_POINTS = "[31,62,96,131,151,163,207,247,271,300]"
DEFAULT_FBS_CHECKPOINT = "ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt"
DEFAULT_VERIFY_SUMMARY_NAME = "edgevla_ours_cl_summary.json"


_ORIGINAL_LOAD_POLICY_STATE = algo.load_policy_state_from_checkpoint


def copy_run_metadata(output_dir: Path, args: algo.Args) -> None:
    algo.save_json(output_dir / "args.json", asdict(args))
    code_dir = algo.mkdir(output_dir / "code")
    sources = {
        "online_rl_cl.py": Path(__file__).resolve(),
        "base_online_rl_cl.py": Path(algo.__file__).resolve(),
        "online_rl_unitree_g1_lift_apple.py": Path(human_task.__file__).resolve(),
        "model_with_fbs.py": THIS_DIR / "model_with_fbs.py",
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


def load_policy_state_from_checkpoint_or_default(checkpoint_path: str, policy):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print("[setup] large-agent checkpoint not found; using default initialized model")
        return {}
    return _ORIGINAL_LOAD_POLICY_STATE(checkpoint_path, policy)


def configure_human_defaults() -> None:
    human_task.patch_reference_for_humanoid_env()
    algo.reference = human_task.reference
    algo.convert_to_fbs_model = convert_to_fbs_model
    algo.copy_run_metadata = copy_run_metadata
    algo.load_policy_state_from_checkpoint = load_policy_state_from_checkpoint_or_default
    algo.DEFAULT_WORKDIR = DEFAULT_OUTPUT_DIR
    algo.DEFAULT_FBS_CHECKPOINT = DEFAULT_FBS_CHECKPOINT
    algo.DEFAULT_VERIFY_SUMMARY_NAME = DEFAULT_VERIFY_SUMMARY_NAME

    defaults = {
        "env_id": human_task.ENV_ID,
        "envs_id": DEFAULT_ENVS_ID,
        "env_change_time_points": DEFAULT_ENV_CHANGE_TIME_POINTS,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "obs_mode": "rgb+state_dict",
        "model_dir": DEFAULT_MODEL_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "num_envs": 128,
        "num_eval_envs": 8,
        "num_steps": 64,
        "num_minibatches": 16,
        "update_epochs": 2,
        "eval_episodes": 16,
        "eval_every_updates": 1,
        "max_runtime_hours": 400.0,
        "cuda_device": "1,2,3,4",
        "save_video": False,
        "action_dim": len(human_task.RIGHT_ARM_AND_HAND_ACTION_INDICES),
        "env_action_dim": 25,
        "state_dim": 73,
        "run_setup_smoke": False,
        "large_agent_checkpoint": DEFAULT_FBS_CHECKPOINT,
        "max_sparsity": 0.9,
        "small_model_sparsity": 0.9,
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
        algo.reference.cleanup_runtime()


if __name__ == "__main__":
    main()
