import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime
import argparse
import time
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mani_skill.utils.io_utils import load_json, dump_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym
import random

from train.common.mwe_runtime import ActiveRuntimeTracker
from train.octo.model import Agent as BaseAgent
from train.octo.model import make_mlp, make_mlp_with_orth_init
from train.reinforcement_learning.utils import collect_rollout, compute_gae, ppo_update_on_policy, get_step_infos
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.evaluate import evaluate
import workloads.table_top  # noqa: F401
import envs.pick_obj_random


model_name = 'ricl'
robot_name = 'panda_wristcam'
sensor_configs = {
    'shader_pack': 'default',
    'width': 128,
    'height': 128,
}
cameras = ('base_camera',)


@dataclass
class RiclBankStats:
    size: int = 0
    last_added: int = 0
    last_mean_distance: float = 0.0


class RiclDemoBank:
    def __init__(self, capacity: int, embedding_dim: int, action_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.embedding_dim = int(embedding_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.embeddings = torch.zeros((capacity, embedding_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.size = 0
        self.cursor = 0
        self.last_mean_distance = 0.0

    def add(self, embeddings: torch.Tensor, actions: torch.Tensor) -> int:
        if embeddings.numel() == 0 or actions.numel() == 0:
            return 0
        embeddings = embeddings.to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        count = min(embeddings.shape[0], actions.shape[0])
        for idx in range(count):
            self.embeddings[self.cursor] = embeddings[idx]
            self.actions[self.cursor] = actions[idx]
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        return count

    def lookup(self, query_embeddings: torch.Tensor, num_neighbors: int, temperature: float):
        batch_size = query_embeddings.shape[0]
        if self.size == 0:
            zero_actions = torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=query_embeddings.device)
            zero_embeddings = torch.zeros((batch_size, self.embedding_dim), dtype=torch.float32, device=query_embeddings.device)
            zero_distances = torch.zeros((batch_size,), dtype=torch.float32, device=query_embeddings.device)
            self.last_mean_distance = 0.0
            return zero_actions, zero_embeddings, zero_distances

        bank_embeddings = self.embeddings[:self.size]
        bank_actions = self.actions[:self.size]
        dists = torch.cdist(query_embeddings.to(self.device), bank_embeddings)
        k = min(num_neighbors, self.size)
        values, indices = torch.topk(dists, k=k, dim=1, largest=False)
        weights = torch.softmax(-temperature * values, dim=1)
        gathered_actions = bank_actions[indices]
        gathered_embeddings = bank_embeddings[indices]
        action_context = (weights.unsqueeze(-1) * gathered_actions).sum(dim=1)
        embedding_context = (weights.unsqueeze(-1) * gathered_embeddings).sum(dim=1)
        mean_distance = values.mean().item() if values.numel() > 0 else 0.0
        self.last_mean_distance = mean_distance
        return action_context.to(query_embeddings.device), embedding_context.to(query_embeddings.device), values.mean(dim=1).to(query_embeddings.device)


def resolve_ckpt_dir(args, model_name_value: str):
    task_dir = os.path.join(args.save_dir, f'{args.task_name}/ppo/{args.robot_name}/{model_name_value}')
    root_dir = os.path.join(task_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(task_dir, exist_ok=True)
    resume_dir = args.resume_dir if args.resume_dir is not None else root_dir
    log_dir = os.path.join(resume_dir, 'tb')
    video_dir = os.path.join(resume_dir, 'videos')
    latest_agent = os.path.join(resume_dir, 'latest_agent.pt')
    latest_opt = os.path.join(resume_dir, 'latest_opt.pt')
    actor_path = os.path.join(args.actor_ckpt_path if args.actor_ckpt_path is not None else '', 'last.pt')
    bank_path = os.path.join(resume_dir, 'ricl_demo_bank.pt')
    return {
        'task_dir': task_dir,
        'log_dir': log_dir,
        'root_dir': root_dir,
        'video_dir': video_dir,
        'latest_agent': latest_agent,
        'latest_opt': latest_opt,
        'best_agent': os.path.join(resume_dir, 'best_agent.pt'),
        'metrics': os.path.join(resume_dir, 'metrics.json'),
        'actor_path': actor_path,
        'bank_path': bank_path,
    }


def _slice_camera_tensor(rgb: torch.Tensor, depth: torch.Tensor):
    idx = (0, 1)
    return rgb[:, idx[0] * 3: idx[1] * 3], depth[:, idx[0]: idx[1]]


def _to_tensor(value, scale: Optional[float] = None):
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = value
    tensor = tensor.float()
    if scale is not None:
        tensor = tensor / scale
    return tensor


def extract_modalities(obs, device: torch.device):
    rgb = _to_tensor(obs['rgb'], scale=255.0).permute(0, 3, 1, 2)
    depth = _to_tensor(obs['depth'], scale=1024.0).permute(0, 3, 1, 2)
    state = _to_tensor(obs['state'])
    rgb, depth = _slice_camera_tensor(rgb, depth)
    rgb = F.interpolate(rgb, size=128, mode='bilinear')
    depth = F.interpolate(depth, size=128, mode='bilinear')
    return rgb.to(device), depth.to(device), state.to(device)


def build_retrieval_query_embeddings(rgb: torch.Tensor, depth: torch.Tensor, state: torch.Tensor, state_dim_cap: int):
    rgb_stats = rgb.mean(dim=(2, 3))
    depth_stats = depth.mean(dim=(2, 3))
    state_slice = state[:, :state_dim_cap]
    if state_slice.shape[1] < state_dim_cap:
        pad = torch.zeros((state_slice.shape[0], state_dim_cap - state_slice.shape[1]), dtype=state_slice.dtype, device=state_slice.device)
        state_slice = torch.cat([state_slice, pad], dim=1)
    return torch.cat([state_slice, rgb_stats, depth_stats], dim=1)


def make_collate_fn(args, device: torch.device, demo_bank: RiclDemoBank):
    def collate_fn(obs):
        rgb, depth, state = extract_modalities(obs, device)
        query_embeddings = build_retrieval_query_embeddings(rgb, depth, state, args.ricl_state_dim_cap)
        retrieval_action, retrieval_embedding, retrieval_distance = demo_bank.lookup(
            query_embeddings,
            num_neighbors=args.ricl_num_neighbors,
            temperature=args.ricl_retrieval_temperature,
        )
        return {
            'rgb': rgb,
            'depth': depth,
            'state': state,
            'ricl_retrieval_action': retrieval_action,
            'ricl_retrieval_embedding': retrieval_embedding,
            'ricl_retrieval_distance': retrieval_distance.unsqueeze(1),
        }
    return collate_fn


def make_sample_fn(agent_model, collate_fn):
    def sample_fn(obs):
        batch = collate_fn(obs)
        action = agent_model.get_action(batch)
        return action
    return sample_fn


def get_agent_info(args, env_kwargs, collate_fn):
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu',
        env_kwargs=env_kwargs,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    obs, _ = test_env.reset()
    batch = collate_fn(obs)
    infos = dict(state_dim=batch['state'].shape[-1], action_dim=test_env.single_action_space.shape[0])
    try:
        test_env.close()
    except Exception:
        pass
    return infos


class RiclAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        retrieval_embedding_dim,
        retrieval_hidden_dim=128,
        camera_count=1,
        normalize_state=True,
        use_depth=True,
        simulate_prompt_feature=False,
        prompt_feature_scale=1.0,
    ):
        self.retrieval_hidden_dim = retrieval_hidden_dim
        self.retrieval_embedding_dim = retrieval_embedding_dim
        self.simulate_prompt_feature = simulate_prompt_feature
        self.prompt_feature_scale = prompt_feature_scale
        super().__init__(state_dim=state_dim, action_dim=action_dim, camera_count=camera_count, normalize_state=normalize_state, use_depth=use_depth)
        self.retrieval_action_proj = make_mlp(action_dim, [retrieval_hidden_dim], last_act=False)
        self.retrieval_embedding_proj = make_mlp(retrieval_embedding_dim, [retrieval_hidden_dim], last_act=False)
        base_feature_dim = 256 * 3 if use_depth else 256 * 2
        self.prompt_feature_proj = make_mlp(action_dim + retrieval_embedding_dim, [256], last_act=False)
        self.prompt_injector = make_mlp(256, [base_feature_dim], last_act=False)
        self.context_fuser = make_mlp(retrieval_hidden_dim * 2, [256], last_act=False)
        total_feature_dim = base_feature_dim + 256
        self.actor_mean = make_mlp_with_orth_init(total_feature_dim, [512, action_dim], last_act=False, is_actor=True)
        self.critic = make_mlp_with_orth_init(total_feature_dim, [512, 1], last_act=False)

    def get_feature(self, batch):
        base = super().get_feature({
            'rgb': batch['rgb'],
            'depth': batch['depth'],
            'state': batch['state'],
        })
        retrieval_action = batch.get('ricl_retrieval_action')
        retrieval_embedding = batch.get('ricl_retrieval_embedding')
        if retrieval_action is None:
            retrieval_action = torch.zeros((base.shape[0], self.action_dim), dtype=base.dtype, device=base.device)
        if retrieval_embedding is None:
            retrieval_embedding = torch.zeros((base.shape[0], self.retrieval_embedding_dim), dtype=base.dtype, device=base.device)
        action_context = self.retrieval_action_proj(retrieval_action)
        embedding_context = self.retrieval_embedding_proj(retrieval_embedding)
        prompt_feature = self.prompt_feature_proj(torch.cat([retrieval_action, retrieval_embedding], dim=1))
        if self.simulate_prompt_feature:
            base = base + self.prompt_feature_scale * self.prompt_injector(prompt_feature)
        context_feature = self.context_fuser(torch.cat([action_context, embedding_context], dim=1))
        return torch.cat([base, context_feature], dim=1)

    def load_actor(self, path, strict=True):
        actor_state = torch.load(path, map_location='cpu')['actor']
        self.rgb_encoder.load_state_dict(
            {k.replace('rgb_encoder.', ''): v for k, v in actor_state.items() if k.startswith('rgb_encoder.')},
            strict=strict,
        )
        if self.depth_encoder is not None:
            self.depth_encoder.load_state_dict(
                {k.replace('depth_encoder.', ''): v for k, v in actor_state.items() if k.startswith('depth_encoder.')},
                strict=strict,
            )
        self.state_encoder.load_state_dict(
            {k.replace('state_encoder.', ''): v for k, v in actor_state.items() if k.startswith('state_encoder.')},
            strict=strict,
        )
        decoder_state = {k.replace('decoder.', ''): v for k, v in actor_state.items() if k.startswith('decoder.')}
        first_linear = self.actor_mean[0]
        second_linear = self.actor_mean[2]
        pretrained_first = decoder_state['0.weight']
        pretrained_first_bias = decoder_state['0.bias']
        with torch.no_grad():
            first_linear.weight.zero_()
            first_linear.weight[:, :pretrained_first.shape[1]].copy_(pretrained_first)
            first_linear.bias.copy_(pretrained_first_bias)
            second_linear.weight.copy_(decoder_state['2.weight'])
            second_linear.bias.copy_(decoder_state['2.bias'])

    def load_pretrained_agent_checkpoint(self, path):
        agent_state = torch.load(path, map_location='cpu')['agent']

        def copy_overlap(dst, src, zero_dst=False):
            with torch.no_grad():
                if zero_dst:
                    dst.zero_()
                slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
                dst[slices].copy_(src[slices])

        with torch.no_grad():
            self.state_encoder[0].weight.zero_()
            self.state_encoder[0].bias.zero_()
            eye = torch.eye(self.state_encoder[0].weight.shape[0], self.state_encoder[0].weight.shape[1])
            self.state_encoder[0].weight.copy_(eye)

        encoder_pairs = [
            ('rgb_encoder.cnn.0.weight', 'feature_net.rgb_encoder.cnn.0.raw_conv2d.weight'),
            ('rgb_encoder.cnn.0.bias', 'feature_net.rgb_encoder.cnn.0.raw_conv2d.bias'),
            ('rgb_encoder.cnn.3.weight', 'feature_net.rgb_encoder.cnn.3.weight'),
            ('rgb_encoder.cnn.3.bias', 'feature_net.rgb_encoder.cnn.3.bias'),
            ('rgb_encoder.cnn.6.weight', 'feature_net.rgb_encoder.cnn.6.raw_conv2d.weight'),
            ('rgb_encoder.cnn.6.bias', 'feature_net.rgb_encoder.cnn.6.raw_conv2d.bias'),
            ('rgb_encoder.cnn.9.weight', 'feature_net.rgb_encoder.cnn.9.weight'),
            ('rgb_encoder.cnn.9.bias', 'feature_net.rgb_encoder.cnn.9.bias'),
            ('rgb_encoder.cnn.12.weight', 'feature_net.rgb_encoder.cnn.12.raw_conv2d.weight'),
            ('rgb_encoder.cnn.12.bias', 'feature_net.rgb_encoder.cnn.12.raw_conv2d.bias'),
            ('rgb_encoder.cnn.15.weight', 'feature_net.rgb_encoder.cnn.15.weight'),
            ('rgb_encoder.cnn.15.bias', 'feature_net.rgb_encoder.cnn.15.bias'),
            ('rgb_encoder.fc.0.weight', 'feature_net.rgb_encoder.fc.0.0.raw_linear.weight'),
            ('rgb_encoder.fc.0.bias', 'feature_net.rgb_encoder.fc.0.0.raw_linear.bias'),
            ('depth_encoder.cnn.0.weight', 'feature_net.depth_encoder.cnn.0.raw_conv2d.weight'),
            ('depth_encoder.cnn.0.bias', 'feature_net.depth_encoder.cnn.0.raw_conv2d.bias'),
            ('depth_encoder.cnn.3.weight', 'feature_net.depth_encoder.cnn.3.weight'),
            ('depth_encoder.cnn.3.bias', 'feature_net.depth_encoder.cnn.3.bias'),
            ('depth_encoder.cnn.6.weight', 'feature_net.depth_encoder.cnn.6.raw_conv2d.weight'),
            ('depth_encoder.cnn.6.bias', 'feature_net.depth_encoder.cnn.6.raw_conv2d.bias'),
            ('depth_encoder.cnn.9.weight', 'feature_net.depth_encoder.cnn.9.weight'),
            ('depth_encoder.cnn.9.bias', 'feature_net.depth_encoder.cnn.9.bias'),
            ('depth_encoder.cnn.12.weight', 'feature_net.depth_encoder.cnn.12.raw_conv2d.weight'),
            ('depth_encoder.cnn.12.bias', 'feature_net.depth_encoder.cnn.12.raw_conv2d.bias'),
            ('depth_encoder.cnn.15.weight', 'feature_net.depth_encoder.cnn.15.weight'),
            ('depth_encoder.cnn.15.bias', 'feature_net.depth_encoder.cnn.15.bias'),
            ('depth_encoder.fc.0.weight', 'feature_net.depth_encoder.fc.0.0.raw_linear.weight'),
            ('depth_encoder.fc.0.bias', 'feature_net.depth_encoder.fc.0.0.raw_linear.bias'),
            ('state_encoder.2.weight', 'feature_net.state_encoder.weight'),
            ('state_encoder.2.bias', 'feature_net.state_encoder.bias'),
        ]
        current_state = self.state_dict()
        for dst_key, src_key in encoder_pairs:
            if src_key in agent_state:
                copy_overlap(current_state[dst_key], agent_state[src_key])

        if 'actor_mean.0.raw_linear.weight' in agent_state:
            copy_overlap(current_state['actor_mean.0.weight'], agent_state['actor_mean.0.raw_linear.weight'], zero_dst=True)
        if 'actor_mean.0.raw_linear.bias' in agent_state:
            copy_overlap(current_state['actor_mean.0.bias'], agent_state['actor_mean.0.raw_linear.bias'])
        if 'critic.0.raw_linear.weight' in agent_state:
            copy_overlap(current_state['critic.0.weight'], agent_state['critic.0.raw_linear.weight'], zero_dst=True)
        if 'critic.0.raw_linear.bias' in agent_state:
            copy_overlap(current_state['critic.0.bias'], agent_state['critic.0.raw_linear.bias'])
        if 'critic.2.weight' in agent_state:
            copy_overlap(current_state['critic.2.weight'], agent_state['critic.2.weight'])
        if 'critic.2.bias' in agent_state:
            copy_overlap(current_state['critic.2.bias'], agent_state['critic.2.bias'])
        print(f'[Train] Loaded shared pretrained weights from {path}')


def update_demo_bank_from_rollout(args, demo_bank: RiclDemoBank, obs_buf, act_buf, rew_buf, device: torch.device):
    flat_obs = obs_buf.reshape((-1,))
    rgb, depth, state = extract_modalities(flat_obs, device)
    embeddings = build_retrieval_query_embeddings(rgb, depth, state, args.ricl_state_dim_cap)
    rewards = rew_buf.reshape(-1)
    actions = torch.as_tensor(act_buf.reshape((-1, act_buf.shape[-1])), dtype=torch.float32, device=device)

    candidate_count = embeddings.shape[0]
    if candidate_count == 0:
        return 0

    k = min(args.ricl_bank_add_per_iter, candidate_count)
    if k <= 0:
        return 0

    top_values, top_indices = torch.topk(rewards, k=k, largest=True)
    selected = top_indices
    if torch.allclose(top_values.abs().sum(), torch.tensor(0.0, device=top_values.device)):
        selected = torch.randperm(candidate_count, device=device)[:k]

    return demo_bank.add(embeddings[selected], actions[selected])


def main(args):
    ckpt = resolve_ckpt_dir(args, model_name)
    step_infos = get_step_infos(args)
    env_kwargs = {
        'obs_mode': 'rgb+depth+state_dict',
        'control_mode': 'pd_joint_delta_pos',
        'render_mode': 'rgb_array',
        'reward_mode': 'normalized_dense',
        'shader_dir': 'default',
        'sim_backend': 'physx_cuda',
        'robot_uids': robot_name,
        'sensor_configs': sensor_configs,
        'max_episode_steps': args.max_episode_steps,
    }
    env_kwargs_for_eval = dict(env_kwargs)
    env_kwargs_for_eval.pop('sim_backend')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    accelerator = Accelerator(mixed_precision='bf16' if args.use_amp else 'no')
    device = accelerator.device

    demo_bank = RiclDemoBank(
        capacity=args.ricl_bank_capacity,
        embedding_dim=args.ricl_state_dim_cap + 4,
        action_dim=1,
        device=device,
    )

    temp_collate = make_collate_fn(args, device, demo_bank)
    infos = get_agent_info(args, env_kwargs_for_eval, temp_collate)
    demo_bank = RiclDemoBank(
        capacity=args.ricl_bank_capacity,
        embedding_dim=args.ricl_state_dim_cap + 4,
        action_dim=infos['action_dim'],
        device=device,
    )
    collate_fn = make_collate_fn(args, device, demo_bank)

    agent = RiclAgent(
        state_dim=infos['state_dim'],
        action_dim=infos['action_dim'],
        retrieval_embedding_dim=demo_bank.embedding_dim,
        retrieval_hidden_dim=args.ricl_context_hidden_dim,
        normalize_state=args.normalize_state,
        simulate_prompt_feature=args.simulate_prompt_feature,
        prompt_feature_scale=args.prompt_feature_scale,
    )

    if os.path.exists(ckpt['latest_agent']):
        print(f"[Train] Resume agent from {ckpt['latest_agent']}")
        agent.load_state_dict(torch.load(ckpt['latest_agent'], map_location='cpu'))
    elif args.pretrained_checkpoint and os.path.isfile(args.pretrained_checkpoint):
        print(f"[Train] Load pretrained agent from {args.pretrained_checkpoint}")
        agent.load_pretrained_agent_checkpoint(args.pretrained_checkpoint)
    elif os.path.exists(ckpt['actor_path']):
        print(f"[Train] Load actor from {ckpt['actor_path']}")
        agent.load_actor(ckpt['actor_path'])

    sta_steps = global_steps = 0
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_time = time.time()
    training_start_time = time.monotonic()
    runtime_tracker = ActiveRuntimeTracker.from_env(wall_clock_start_time=training_start_time)
    if os.path.exists(ckpt['latest_opt']):
        print(f"[Train] Resume optimizer from {ckpt['latest_opt']}")
        resume_opt = torch.load(ckpt['latest_opt'], map_location='cpu')
        optimizer.load_state_dict(resume_opt['opt'])
        sta_steps = global_steps = resume_opt['step']

    agent, optimizer = accelerator.prepare(agent, optimizer)

    if args.reset_logstd:
        print('[Train] Reset actor log std to -0.5')
        agent.reset_logstd(-0.5)

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt['log_dir'])
    else:
        writer = None

    best_score = -1.0
    metrics_log = []
    pbar = tqdm(total=step_infos['total_steps'], initial=global_steps, ascii=True)
    writer = SummaryWriter(ckpt['log_dir'], purge_step=global_steps)
    print(f"[TensorBoard] Logging to {ckpt['log_dir']}")

    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend='gpu',
        env_kwargs=env_kwargs_for_eval,
        video_dir=f"{ckpt['video_dir']}_train",
        wrappers=[FlattenRGBDObservationWrapper],
    )
    _, _ = eval_envs.reset(seed=args.seed)
    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=args.ignore_partial_reset, record_metrics=True)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    if os.path.exists(ckpt['metrics']):
        metrics_log = load_json(ckpt['metrics'])
    resume_skip = True if args.resume_dir is not None else False

    if args.minibatch_size == 0:
        args.minibatch_size = max(1, step_infos['rollot_steps'] // args.num_minibatch // args.grad_accum_steps)

    while global_steps < step_infos['total_steps']:
        agent.eval()
        if not resume_skip and global_steps % step_infos['save_interval_steps'] == 0:
            torch.save(agent.state_dict(), ckpt['latest_agent'])
            torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt['latest_opt'])
            eval_metrics = evaluate(
                n=1,
                sample_fn=make_sample_fn(agent, collate_fn),
                eval_envs=eval_envs,
            )
            for key, value in eval_metrics.items():
                mean = value.mean()
                writer.add_scalar(f'eval/{key}', mean, global_steps)
                print(f'eval_{key}_mean={mean}')
            score = eval_metrics.get('success_rate', eval_metrics[list(eval_metrics.keys())[0]]).mean()
            pbar.set_postfix(eval_score=score)
            metrics_log.append(dict(step=global_steps, score=float(score), demo_bank_size=demo_bank.size))
            dump_json(ckpt['metrics'], metrics_log)
            if score > best_score:
                best_score = score
                torch.save(agent.state_dict(), ckpt['best_agent'])
                print(f'[Eval] New best model saved (score={score:.3f})')

        resume_skip = False
        if args.dynamic_clip:
            frac = 1.0 - (global_steps / step_infos['total_steps'])
            args.clip_eps = 0.1 + (args.clip_eps - 0.1) * frac
        if args.dynamic_ent_coef:
            frac = 1.0 - (global_steps / step_infos['total_steps'])
            args.ent_coef = args.ent_coef / 10 + (args.ent_coef * 0.9) * frac

        if args.max_time is not None:
            elapsed_minutes = runtime_tracker.current_minutes()
            if elapsed_minutes >= args.max_time:
                print(
                    f'[RICL] reached max_time={args.max_time} minutes '
                    f'(elapsed={elapsed_minutes:.2f} minutes) before rollout, stopping training.'
                )
                break

        rollout_time = time.perf_counter()
        rollout = collect_rollout(
            args=args,
            agent=agent,
            collate_fn=collate_fn,
            accelerator=accelerator,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            writer=writer,
            global_step=global_steps,
        )
        rollout_time = time.perf_counter() - rollout_time
        runtime_tracker.add_active_seconds(rollout_time)
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done = rollout

        update_time = time.perf_counter()
        adv_buf, ret_buf = compute_gae(rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done, agent, collate_fn, args, accelerator)
        data = (
            obs_buf.reshape((-1,)),
            act_buf.reshape((-1,) + envs.single_action_space.shape),
            logp_buf.reshape(-1),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )
        agent.train()
        stats = ppo_update_on_policy(args, agent, optimizer, data, collate_fn, accelerator, 'RICL PPO', writer)
        added = update_demo_bank_from_rollout(args, demo_bank, obs_buf, act_buf, rew_buf, device)
        update_time = time.perf_counter() - update_time
        runtime_tracker.add_active_seconds(update_time)

        global_steps += step_infos['rollot_steps']
        pbar.update(step_infos['rollot_steps'])

        y_pred, y_true = val_buf.flatten(0, 1).cpu().numpy(), ret_buf.flatten(0, 1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = (global_steps - sta_steps) / max(time.time() - start_time, 1e-6)
        writer.add_scalar('charts/SPS', sps, global_steps)
        writer.add_scalar('loss/policy', stats['policy_loss'], global_steps)
        writer.add_scalar('loss/value', stats['value_loss'], global_steps)
        writer.add_scalar('loss/entropy', stats['entropy'], global_steps)
        writer.add_scalar('loss/approx_kl', stats['approx_kl'], global_steps)
        writer.add_scalar('loss/clip_frac', stats['clip_frac'], global_steps)
        writer.add_scalar('loss/explained_var', explained_var, global_steps)
        writer.add_scalar('ricl/demo_bank_size', demo_bank.size, global_steps)
        writer.add_scalar('ricl/added_per_iteration', added, global_steps)
        writer.add_scalar('ricl/mean_retrieval_distance', demo_bank.last_mean_distance, global_steps)
        writer.add_scalar('time/elapsed_minutes', runtime_tracker.current_minutes(), global_steps)
        writer.add_scalar('time/rollout_time', rollout_time, global_steps)
        writer.add_scalar('time/update_time', update_time, global_steps)
        writer.add_scalar('time/total_rollout+update_time', runtime_tracker.active_seconds, global_steps)
        print(f'[RICL] demo_bank_size={demo_bank.size} added={added} mean_distance={demo_bank.last_mean_distance:.6f}')

    writer.close()
    torch.save(agent.state_dict(), ckpt['latest_agent'])
    torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt['latest_opt'])
    torch.save(
        {
            'embeddings': demo_bank.embeddings[:demo_bank.size].detach().cpu(),
            'actions': demo_bank.actions[:demo_bank.size].detach().cpu(),
            'size': demo_bank.size,
            'cursor': demo_bank.cursor,
        },
        ckpt['bank_path'],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--actor-ckpt-path', type=str, default=None)
    parser.add_argument('--pretrained-checkpoint', type=str, default=None)
    parser.add_argument('--task-name', type=str, default='PickCube-v1')
    parser.add_argument('--seed', type=int, default=1788)
    parser.add_argument('--total-steps', type=int, default=5_000_000)
    parser.add_argument('--critic-warmup-rollouts', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--num-eval-envs', type=int, default=8)
    parser.add_argument('--ignore-partial-reset', action='store_true')
    parser.add_argument('--ignore-torch-deterministic', action='store_true')
    parser.add_argument('--rollout-steps', type=int, default=16)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--num_minibatch', type=int, default=16)
    parser.add_argument('--minibatch-size', type=int, default=0)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--rollout-minibatch-size', type=int, default=0)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--dynamic-lr', action='store_true')
    parser.add_argument('--dynamic-clip', action='store_true')
    parser.add_argument('--dynamic-ent-coef', action='store_true')
    parser.add_argument('--normalize-state', action='store_true')
    parser.add_argument('--not-normalize-adv', action='store_true')
    parser.add_argument('--reset-logstd', action='store_true')
    parser.add_argument('--finite-horizon-gae', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--clip-vloss', action='store_true')
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.2)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='ckpt')
    parser.add_argument('--resume-dir', type=str, default=None)
    parser.add_argument('--save-interval-per-rollout', type=int, default=25)
    parser.add_argument('--max-episode-steps', type=int, default=50)
    parser.add_argument('--max-time', type=float, default=None)
    parser.add_argument('--robot-name', type=str, default='panda')
    parser.add_argument('--ricl-bank-capacity', type=int, default=4096)
    parser.add_argument('--ricl-bank-add-per-iter', type=int, default=256)
    parser.add_argument('--ricl-num-neighbors', type=int, default=4)
    parser.add_argument('--ricl-retrieval-temperature', type=float, default=10.0)
    parser.add_argument('--ricl-state-dim-cap', type=int, default=32)
    parser.add_argument('--ricl-context-hidden-dim', type=int, default=128)
    parser.add_argument('--simulate-prompt-feature', action='store_true')
    parser.add_argument('--prompt-feature-scale', type=float, default=0.001)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
