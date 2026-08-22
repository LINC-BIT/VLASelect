from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mani_skill.envs  # noqa: F401
import workloads.table_top  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ours.libs.train_with_fbs.lib import set_sparsity
from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn
from ours.utils.dl.common.model import get_module, set_module
from train.octo.conrft import online_rl as reference
from train.octo.model import Actor
from train.octo.ours.evolving_envs import PickCubeEnvMutable  # noqa: F401


STATE_LAYOUT = (
    ("qpos", 9),
    ("qvel", 9),
    ("is_grasped", 1),
    ("tcp_pose", 7),
    ("goal_pos", 3),
    ("obj_pose", 7),
    ("tcp_to_obj_pos", 3),
    ("obj_to_goal_pos", 3),
)
STATE_DIM = sum(size for _, size in STATE_LAYOUT)
ACTION_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--env-config-path', type=Path, required=True)
    parser.add_argument('--state-norm-stats-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--env-id', default='PickCube-v1')
    parser.add_argument('--target-success-trajectories', type=int, default=24)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-sparsity', type=float, default=0.8)
    parser.add_argument('--reuse-if-exists', action='store_true')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--log-prefix', default='teacher-demo')
    return parser.parse_args()


def extract_obs_from_batch(obs: dict[str, Any], env_idx: int) -> dict[str, torch.Tensor]:
    single_obs = {}
    for key in ('rgb', 'depth', 'state'):
        value = obs[key]
        if isinstance(value, torch.Tensor):
            sample = value[env_idx].detach().cpu().clone()
        else:
            sample = torch.as_tensor(value[env_idx]).cpu().clone()
        if key == 'rgb' and sample.ndim == 3 and sample.shape[-1] > 3:
            sample = sample[..., 0:3]
        if key == 'depth' and sample.ndim == 3 and sample.shape[-1] > 1:
            sample = sample[..., 0:1]
        single_obs[key] = sample
    return single_obs


def build_teacher_agent(args: argparse.Namespace, device: torch.device) -> reference.Agent:
    actor = Actor(STATE_DIM, ACTION_DIM, 1, False).to(device=device)
    set_module(actor, 'rgb_encoder.fc.0', svd_decompose_linear(get_module(actor, 'rgb_encoder.fc.0')))
    set_module(actor, 'depth_encoder.fc.0', svd_decompose_linear(get_module(actor, 'depth_encoder.fc.0')))

    actor_example = {
        'rgb': torch.rand((1, 3, 128, 128), device=device),
        'depth': torch.rand((1, 1, 128, 128), device=device),
        'state': torch.rand((1, STATE_DIM), device=device),
    }
    add_FBS_into_cnn(
        actor,
        [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
        ['decoder.0', 'rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
        actor_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample['rgb'], sample['depth'], sample['state']),
    )

    state_max, state_min = torch.load(args.state_norm_stats_path, map_location='cpu')
    state_max = state_max.to(device)
    state_min = state_min.to(device)
    agent = reference.Agent(actor, 256 * 3, state_max, state_min, True, -0.5).to(device)
    actor.decoder = nn.Identity()

    agent_example = {
        'rgb': torch.rand((1, 128, 128, 3)),
        'depth': torch.rand((1, 128, 128, 1)),
        'state': torch.rand((1, STATE_DIM)),
    }
    add_FBS_into_cnn(
        agent,
        [],
        ['actor_mean.0', 'critic.0'],
        agent_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample),
    )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'teacher checkpoint not found: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(agent.load_state_dict(checkpoint['agent'], strict=True))
    set_sparsity(agent, args.max_sparsity)
    for module in agent.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    agent.eval()
    return agent


def load_env_kwargs(env_config_path: Path) -> dict[str, Any]:
    payload = json.loads(env_config_path.read_text(encoding='utf-8'))
    env_kwargs = dict(payload['env_info']['env_kwargs'])
    env_kwargs['sim_backend'] = 'physx_cuda'
    env_kwargs.pop('num_envs', None)
    env_kwargs.pop('reward_mode', None)
    return env_kwargs


def make_collection_env(args: argparse.Namespace, env_kwargs: dict[str, Any]):
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        reconfiguration_freq=None,
        **env_kwargs,
    )
    env = reference.FlattenRGBDObservationWrapper2(env, rgb=True, depth=True, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env,
        args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )
    return env


def new_episode_buffer() -> dict[str, Any]:
    return {
        'obs': [],
        'actions': [],
        'rewards': [],
        'success': [],
        'final_obs': None,
    }


def collect_successful_episodes(args: argparse.Namespace, agent: reference.Agent, envs) -> list[dict[str, Any]]:
    obs, _ = envs.reset(seed=args.seed)
    active = [new_episode_buffer() for _ in range(args.num_envs)]
    completed: list[dict[str, Any]] = []
    total_steps = 0
    start_time = time.time()

    while len(completed) < args.target_success_trajectories and total_steps < args.max_steps:
        step_obs = [extract_obs_from_batch(obs, env_idx) for env_idx in range(args.num_envs)]
        with torch.no_grad():
            actions = agent.get_action(obs, deterministic=args.deterministic)
        action_tensor = actions.detach().cpu()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        total_steps += 1

        for env_idx in range(args.num_envs):
            active[env_idx]['obs'].append(step_obs[env_idx])
            active[env_idx]['actions'].append(action_tensor[env_idx].numpy().astype(np.float32))
            active[env_idx]['rewards'].append(float(rewards[env_idx].item()))
            active[env_idx]['success'].append(0.0)

        if 'final_info' in infos:
            done_mask = infos['_final_info']
            if isinstance(done_mask, torch.Tensor):
                done_indices = torch.arange(args.num_envs, device=done_mask.device)[done_mask].detach().cpu().tolist()
            else:
                done_indices = np.flatnonzero(np.asarray(done_mask)).tolist()
            success_once = infos['final_info']['episode']['success_once']
            final_obs_batch = infos['final_observation']
            for local_idx, env_idx in enumerate(done_indices):
                episode = active[env_idx]
                episode['final_obs'] = extract_obs_from_batch(final_obs_batch, local_idx)
                if float(success_once[env_idx].item()) > 0.0 and episode['actions']:
                    episode['success'][-1] = 1.0
                    completed.append(episode)
                    print(
                        f"[{args.log_prefix}] collected success trajectory "
                        f"{len(completed)}/{args.target_success_trajectories} "
                        f"len={len(episode['actions'])} total_steps={total_steps}"
                    )
                    if len(completed) >= args.target_success_trajectories:
                        break
                active[env_idx] = new_episode_buffer()

        obs = next_obs

    elapsed = time.time() - start_time
    print(
        f"[{args.log_prefix}] finished collection with {len(completed)} successful trajectories "
        f"after {total_steps} environment steps in {elapsed:.1f}s"
    )
    if not completed:
        raise RuntimeError('failed to collect any successful teacher trajectory')
    return completed[: args.target_success_trajectories]


def split_state_vector(state: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f'expected state dim {STATE_DIM}, got {state.shape[0]}')
    offset = 0
    parts: dict[str, np.ndarray] = {}
    for key, size in STATE_LAYOUT:
        parts[key] = state[offset: offset + size]
        offset += size
    return parts


def obs_to_numpy(obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    rgb = obs['rgb'].cpu().numpy()
    depth = obs['depth'].cpu().numpy()
    state = obs['state'].cpu().numpy().astype(np.float32)
    return {
        'rgb': rgb.astype(np.uint8),
        'depth': depth.astype(np.uint16) if np.issubdtype(depth.dtype, np.integer) else depth.astype(np.float32),
        'state': state,
    }


def write_dataset(output_path: Path, env_config_path: Path, checkpoint_path: Path, episodes: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix('.json')
    tmp_h5 = output_path.with_suffix(output_path.suffix + '.tmp')
    tmp_json = json_path.with_suffix(json_path.suffix + '.tmp')

    with h5py.File(tmp_h5, 'w') as h5_file:
        for episode_id, episode in enumerate(episodes):
            if episode['final_obs'] is None:
                raise RuntimeError(f'trajectory {episode_id} is missing final observation')
            obs_sequence = episode['obs'] + [episode['final_obs']]
            obs_np = [obs_to_numpy(sample) for sample in obs_sequence]
            states = [split_state_vector(sample['state']) for sample in obs_np]
            traj = h5_file.create_group(f'traj_{episode_id}')
            obs_group = traj.create_group('obs')
            sensor_data = obs_group.create_group('sensor_data')
            base_camera = sensor_data.create_group('base_camera')
            base_camera.create_dataset('rgb', data=np.stack([sample['rgb'] for sample in obs_np], axis=0))
            base_camera.create_dataset('depth', data=np.stack([sample['depth'] for sample in obs_np], axis=0))

            agent_group = obs_group.create_group('agent')
            agent_group.create_dataset('qpos', data=np.stack([state['qpos'] for state in states], axis=0))
            agent_group.create_dataset('qvel', data=np.stack([state['qvel'] for state in states], axis=0))

            extra_group = obs_group.create_group('extra')
            extra_group.create_dataset('is_grasped', data=np.stack([state['is_grasped'] for state in states], axis=0).reshape(-1))
            extra_group.create_dataset('tcp_pose', data=np.stack([state['tcp_pose'] for state in states], axis=0))
            extra_group.create_dataset('goal_pos', data=np.stack([state['goal_pos'] for state in states], axis=0))
            extra_group.create_dataset('obj_pose', data=np.stack([state['obj_pose'] for state in states], axis=0))
            extra_group.create_dataset('tcp_to_obj_pos', data=np.stack([state['tcp_to_obj_pos'] for state in states], axis=0))
            extra_group.create_dataset('obj_to_goal_pos', data=np.stack([state['obj_to_goal_pos'] for state in states], axis=0))

            traj.create_dataset('actions', data=np.asarray(episode['actions'], dtype=np.float32))
            traj.create_dataset('rewards', data=np.asarray(episode['rewards'], dtype=np.float32))
            traj.create_dataset('success', data=np.asarray(episode['success'], dtype=np.float32))

    config_payload = json.loads(env_config_path.read_text(encoding='utf-8'))
    json_payload = {
        'episodes': [{'episode_id': episode_id} for episode_id in range(len(episodes))],
        'env_info': config_payload['env_info'],
        'metadata': {
            'generator': 'train.octo.generate_teacher_demo',
            'checkpoint': str(checkpoint_path),
        },
    }
    tmp_json.write_text(json.dumps(json_payload, indent=2) + '\n', encoding='utf-8')
    tmp_h5.replace(output_path)
    tmp_json.replace(json_path)


def main() -> None:
    args = parse_args()
    json_path = args.output_path.with_suffix('.json')
    if args.reuse_if_exists and args.output_path.is_file() and json_path.is_file():
        print(f'[{args.log_prefix}] reuse existing dataset: {args.output_path}')
        return

    env_kwargs = load_env_kwargs(args.env_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    print(f'[{args.log_prefix}] device={device}')
    print(f'[{args.log_prefix}] checkpoint={args.checkpoint}')
    print(f'[{args.log_prefix}] output={args.output_path}')
    agent = build_teacher_agent(args, device)
    envs = make_collection_env(args, env_kwargs)
    try:
        episodes = collect_successful_episodes(args, agent, envs)
    finally:
        envs.close()

    write_dataset(args.output_path, args.env_config_path, args.checkpoint, episodes)
    print(f'[{args.log_prefix}] wrote dataset: {args.output_path}')
    print(f'[{args.log_prefix}] wrote metadata: {json_path}')


if __name__ == '__main__':
    main()
