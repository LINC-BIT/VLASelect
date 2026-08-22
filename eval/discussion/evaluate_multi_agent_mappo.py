from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

MANISKILL_ROOT = os.environ.get('MANISKILL_ROOT', '/home/Maniskill')
sys.path.append(os.getcwd())
sys.path.append(MANISKILL_ROOT)
sys.path.append(os.path.join(MANISKILL_ROOT, 'train', 'toy_cnn', 'multi_agents', 'two_robot_pick'))

import torch

from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn
from ours.utils.dl.common.model import get_module, set_module
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.toy_cnn.model import MAPPOAgent
from train.toy_cnn.multi_agents.two_robot_pick.mappo_online_rl import (
    FlattenRGBObservationWrapperForMARL,
    get_agent_info,
    make_collate_fn,
)

CAMERAS = ('hand_camera',)
AGENT_OBS_RULES = {
    'panda_wristcam-0': ['cube_pose', 'cube_to_goal_pos', 'left_arm_tcp_to_cube_pos', 'left_arm_tcp', 'qpos', 'qvel', 'stage'],
    'panda_wristcam-1': ['cube_pose', 'cube_to_goal_pos', 'right_arm_tcp_to_cube_pos', 'right_arm_tcp', 'qpos', 'qvel', 'stage'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='TwoRobotPickCube-v2')
    parser.add_argument('--seed', type=int, default=1788)
    parser.add_argument('--expert-agent-dir', type=str, required=True)
    parser.add_argument('--obs-mode', type=str, default='rgb+state_dict')
    parser.add_argument('--control-mode', type=str, default='pd_joint_delta_pos')
    parser.add_argument('--reward-mode', type=str, default='normalized_dense')
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--num-eval-envs', type=int, default=2)
    parser.add_argument('--eval-episodes', type=int, default=4)
    parser.add_argument('--normalize-state', action='store_true', default=True)
    parser.add_argument('--max-sparsity', type=float, default=0.0)
    parser.add_argument('--output-json', type=str, required=True)
    return parser.parse_args()


def infer_dims_from_checkpoint(checkpoint: dict, agent_names: list[str]) -> tuple[int, int]:
    state_key = f'actor_state_rms.{agent_names[0]}.mean'
    state_dim = int(checkpoint[state_key].numel())
    global_state_dim = int(checkpoint['critic_state_rms.mean'].numel())
    return state_dim, global_state_dim


def build_agent(args, env_kwargs: dict, device: torch.device) -> tuple[MAPPOAgent, callable]:
    ckpt_path = os.path.join(args.expert_agent_dir, 'best_agent.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    collate_fn = make_collate_fn(args, CAMERAS, device)
    infos = get_agent_info(args, env_kwargs, AGENT_OBS_RULES, collate_fn)
    state_dim, global_state_dim = infer_dims_from_checkpoint(checkpoint, infos['agent_names'])
    infos['state_dim'] = state_dim
    infos['global_state_dim'] = global_state_dim

    agent = MAPPOAgent(**infos, normalize_state=args.normalize_state, use_depth=False).to(device)
    set_module(agent, 'rgb_encoder.fc.0', svd_decompose_linear(get_module(agent, 'rgb_encoder.fc.0')))

    dummy_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu' if device.type == 'cuda' else 'cpu',
        env_kwargs=env_kwargs,
        wrappers=[lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=AGENT_OBS_RULES)],
    )
    example_sample, _ = dummy_env.reset(seed=args.seed)
    example_sample = collate_fn(example_sample)
    if example_sample['global_state'].shape[-1] != global_state_dim:
        example_sample['global_state'] = example_sample['global_state'][..., :global_state_dim]
    action_heads_names = [f'actor_heads.{name}.0' for name in agent.actor_heads.keys()]
    add_FBS_into_cnn(
        agent,
        [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]],
        action_heads_names + ['rgb_encoder.fc.0.0', 'critic.0'],
        example_sample,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample),
    )
    dummy_env.close()
    agent.load_state_dict(checkpoint)
    agent.eval()
    return agent, collate_fn


def make_sample_fn(agent, collate_fn):
    def sample_fn(obs):
        batch = collate_fn(obs)
        return agent.get_action(batch, deterministic=True)

    return sample_fn


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_kwargs = {
        'obs_mode': args.obs_mode,
        'control_mode': args.control_mode,
        'reward_mode': args.reward_mode,
        'render_mode': 'rgb_array',
        'max_episode_steps': args.max_episode_steps,
    }
    args_ns = SimpleNamespace(**vars(args))
    agent, collate_fn = build_agent(args_ns, env_kwargs, device)
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend='gpu',
        env_kwargs=env_kwargs,
        wrappers=[lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=AGENT_OBS_RULES)],
    )
    metrics = evaluate(
        n=args.eval_episodes,
        sample_fn=make_sample_fn(agent, collate_fn),
        eval_envs=eval_envs,
    )
    eval_envs.close()
    payload = {k: float(v.mean()) for k, v in metrics.items()}
    print('[Eval] ' + ' '.join(f'{k}={v:.4f}' for k, v in sorted(payload.items())))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
