from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append('.')

import train.tinyvla.vla_rft.online_rl as algo
from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task
from train.edgevla.ours.model_with_fbs import convert_to_fbs_model


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = 'ckpt/vla_adapter_new/LIBERO-Object'
DEFAULT_OUTPUT_DIR = 'train/edgevla/vla_rft/outputs/online_rl'
DEFAULT_ENVS_ID = "['UnitreeG1LiftCubeObjectScaleDown1p3-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeLightWeaker50-v1','UnitreeG1LiftCubeObjectPurple-v1','UnitreeG1LiftSphereLightStronger50-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectScaleDown1p1-v1','UnitreeG1LiftSphereObjectScaleDown1p3-v1','UnitreeG1LiftCubeColorTempLower50-v1','UnitreeG1LiftCubeObjectPurple-v1']"
DEFAULT_ENV_CHANGE_TIME_POINTS = '[31,62,96,131,151,163,207,247,271,300]'
DEFAULT_WORLD_MODEL_CHECKPOINT = ''
DEFAULT_FBS_POLICY_CHECKPOINT = (
    'ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt'
)


_ORIGINAL_SAVE_JSON = algo.save_json
_ORIGINAL_LOAD_POLICY_STATE = algo.load_policy_state_from_checkpoint


def save_json_with_sources(path: Path, payload) -> None:
    _ORIGINAL_SAVE_JSON(path, payload)
    if path.name != 'args.json':
        return
    code_dir = path.parent / 'code'
    manifest_path = code_dir / 'source_manifest.json'
    if manifest_path.exists():
        return
    algo.mkdir(code_dir)
    sources = {
        'online_rl.py': Path(__file__).resolve(),
        'base_online_rl.py': Path(algo.__file__).resolve(),
        'online_rl_unitree_g1_lift_apple.py': Path(human_task.__file__).resolve(),
        'model_with_fbs.py': THIS_DIR.parent / 'ours' / 'model_with_fbs.py',
        'base_model_with_fbs.py': Path('train/tinyvla/ours/model_with_fbs.py'),
        'online_rl_open_cabinet_drawer.py': Path(algo.reference.__file__).resolve(),
        'workloads_human__init__.py': Path(human_task.human_workload.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {'source': str(source_path), 'backup': str(destination)}
    _ORIGINAL_SAVE_JSON(manifest_path, manifest)


def load_policy_state_from_checkpoint_or_default(checkpoint_path: str, policy):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print('[setup] static policy checkpoint not provided or not found; using default initialized model')
        return {}
    return _ORIGINAL_LOAD_POLICY_STATE(checkpoint_path, policy)


def configure_human_defaults() -> None:
    human_task.patch_reference_for_humanoid_env()
    algo.convert_to_fbs_model = convert_to_fbs_model
    algo.save_json = save_json_with_sources
    algo.load_policy_state_from_checkpoint = load_policy_state_from_checkpoint_or_default
    algo.DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR
    algo.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    algo.DEFAULT_WORLD_MODEL_CHECKPOINT = DEFAULT_WORLD_MODEL_CHECKPOINT
    algo.DEFAULT_FBS_POLICY_CHECKPOINT = DEFAULT_FBS_POLICY_CHECKPOINT

    defaults = {
        'env_id': human_task.ENV_ID,
        'envs_id': DEFAULT_ENVS_ID,
        'env_change_time_points': DEFAULT_ENV_CHANGE_TIME_POINTS,
        'control_mode': 'pd_joint_delta_pos',
        'reward_mode': 'normalized_dense',
        'obs_mode': 'rgb+state_dict',
        'model_dir': DEFAULT_MODEL_DIR,
        'output_dir': DEFAULT_OUTPUT_DIR,
        'world_model_checkpoint': DEFAULT_WORLD_MODEL_CHECKPOINT,
        'fbs_policy_checkpoint': DEFAULT_FBS_POLICY_CHECKPOINT,
        'num_envs': 64,
        'num_eval_envs': 8,
        'num_steps': 64,
        'num_minibatches': 16,
        'update_epochs': 2,
        'eval_episodes': 16,
        'eval_every_updates': 1,
        'max_runtime_hours': 400.0,
        'cuda_device': '1,2,3,4',
        'save_video': False,
        'action_dim': len(human_task.RIGHT_ARM_AND_HAND_ACTION_INDICES),
        'env_action_dim': 25,
        'state_dim': 73,
        'run_setup_smoke': False,
        'early_stop_zero_success_minutes': 45000.0,
        'static_sparsity': 0.8,
    }
    for field_name, value in defaults.items():
        if field_name in algo.Args.__dataclass_fields__:
            algo.Args.__dataclass_fields__[field_name].default = value


def main() -> None:
    configure_human_defaults()
    args = algo.parse_args()
    if not hasattr(args, 'smoke_steps'):
        args.smoke_steps = 32
    if args.mode != 'train':
        raise ValueError(f'Unsupported mode: {args.mode}')
    algo.train(args)


if __name__ == '__main__':
    main()
