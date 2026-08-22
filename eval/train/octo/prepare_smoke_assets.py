from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

import h5py
import numpy as np
import torch

from train.octo.vla_rft.pretrain_world_model import DynamicsWorldModel


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_env_config(path: Path) -> None:
    ensure_parent(path)
    payload = {
        "env_info": {
            "env_kwargs": {
                "obs_mode": "rgb+depth+state_dict",
                "control_mode": "pd_ee_delta_pos",
                "reward_mode": "normalized_dense",
                "render_mode": "all",
                "num_envs": 1,
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_state_norm_stats(path: Path, state_dim: int) -> None:
    ensure_parent(path)
    state_max = torch.ones(state_dim, dtype=torch.float32)
    state_min = torch.zeros(state_dim, dtype=torch.float32)
    torch.save((state_max, state_min), path)


def create_expert_demo(path: Path, state_dim: int, action_dim: int, steps: int) -> None:
    ensure_parent(path)
    json_path = path.with_suffix('.json')
    rng = np.random.default_rng(0)
    height = 128
    width = 128
    qpos_dim = 9
    qvel_dim = 9
    if state_dim != 42:
        raise ValueError(f'Octo smoke demo expects state_dim=42, got {state_dim}')

    with h5py.File(path, 'w') as h5_file:
        traj = h5_file.create_group('traj_0')
        obs = traj.create_group('obs')
        sensor_data = obs.create_group('sensor_data')
        base_camera = sensor_data.create_group('base_camera')
        base_camera.create_dataset('rgb', data=rng.integers(0, 256, size=(steps, height, width, 3), dtype=np.uint8))
        base_camera.create_dataset('depth', data=rng.integers(0, 1024, size=(steps, height, width, 1), dtype=np.uint16))

        agent = obs.create_group('agent')
        agent.create_dataset('qpos', data=rng.standard_normal((steps, qpos_dim), dtype=np.float32))
        agent.create_dataset('qvel', data=rng.standard_normal((steps, qvel_dim), dtype=np.float32))

        extra = obs.create_group('extra')
        extra.create_dataset('is_grasped', data=rng.integers(0, 2, size=(steps,), dtype=np.int8))
        extra.create_dataset('tcp_pose', data=rng.standard_normal((steps, 7), dtype=np.float32))
        extra.create_dataset('goal_pos', data=rng.standard_normal((steps, 3), dtype=np.float32))
        extra.create_dataset('obj_pose', data=rng.standard_normal((steps, 7), dtype=np.float32))
        extra.create_dataset('tcp_to_obj_pos', data=rng.standard_normal((steps, 3), dtype=np.float32))
        extra.create_dataset('obj_to_goal_pos', data=rng.standard_normal((steps, 3), dtype=np.float32))
        traj.create_dataset('actions', data=rng.standard_normal((steps, action_dim), dtype=np.float32))

    payload = {
        'episodes': [{'episode_id': 0}],
        'env_info': {
            'env_kwargs': {
                'obs_mode': 'rgb+depth+state_dict',
                'control_mode': 'pd_ee_delta_pos',
                'reward_mode': 'normalized_dense',
                'render_mode': 'all',
                'num_envs': 1,
            }
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_world_model(path: Path, state_dim: int, action_dim: int, latent_dim: int) -> None:
    ensure_parent(path)
    model = DynamicsWorldModel(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    payload = {
        'model_config': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'latent_dim': latent_dim,
        },
        'model': model.state_dict(),
        'state_max': torch.ones(state_dim, dtype=torch.float32),
        'state_min': torch.zeros(state_dim, dtype=torch.float32),
        'reference_bank': {
            'latent': torch.randn(16, latent_dim, dtype=torch.float32),
            'state': torch.randn(16, state_dim, dtype=torch.float32),
        },
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-config-out', type=Path)
    parser.add_argument('--state-norm-out', type=Path)
    parser.add_argument('--expert-demo-out', type=Path)
    parser.add_argument('--world-model-out', type=Path)
    parser.add_argument('--state-dim', type=int, default=42)
    parser.add_argument('--action-dim', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--demo-steps', type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_config_out is not None:
        create_env_config(args.env_config_out)
        print(f'[smoke-assets] env_config={args.env_config_out}')
    if args.state_norm_out is not None:
        create_state_norm_stats(args.state_norm_out, args.state_dim)
        print(f'[smoke-assets] state_norm={args.state_norm_out}')
    if args.expert_demo_out is not None:
        create_expert_demo(args.expert_demo_out, args.state_dim, args.action_dim, args.demo_steps)
        print(f'[smoke-assets] expert_demo={args.expert_demo_out}')
    if args.world_model_out is not None:
        create_world_model(args.world_model_out, args.state_dim, args.action_dim, args.latent_dim)
        print(f'[smoke-assets] world_model={args.world_model_out}')


if __name__ == '__main__':
    main()
