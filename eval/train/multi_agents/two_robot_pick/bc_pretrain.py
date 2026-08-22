import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.append(os.getcwd())
sys.path.append("/home/Maniskill/train/toy_cnn/multi_agents/two_robot_pick")

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gymnasium.spaces.utils import flatten
from mani_skill.utils import common
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.replay_trajectory import Args as ReplayArgs, main as replay_main
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.two_robot_pick_cube_v2  # noqa: F401
import gymnasium as gym
from ours.libs.train_with_fbs.lib import set_sparsity
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.toy_cnn.multi_agents.two_robot_pick.mappo_online_rl import process_agent as build_fbs_expert_agent
from train.toy_cnn.model import MAPPOAgent
from train.multi_agents.two_robot_pick.model import (
    MultiAgentVLAMAPPOAgent,
    build_batch_from_obs,
)
from train.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAMultiAgentsSFTAgent,
)

mp.set_start_method("spawn", force=True)


MODEL_NAME = "multi_agents_sft"
DEFAULT_EXPERT_AGENT_DIR = "ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942"


def get_model_name(args) -> str:
    if args.model_backbone == "multi_agents":
        return "multi_agents_sft"
    return MODEL_NAME


def resolve_ckpt_dir(args):
    task_dir = os.path.join(args.save_dir, f"{args.task_name}/sft/{args.robot_name}/{get_model_name(args)}")
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(task_dir, exist_ok=True)
    dataset_dir = os.path.join("datasets", args.task_name, "rl")
    os.makedirs(dataset_dir, exist_ok=True)

    resume_dir = args.resume_dir if args.resume_dir is not None else root_dir
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
        "trajectory_h5": os.path.join(
            dataset_dir,
            f"trajectory.{args.obs_mode}.{args.expert_control_mode}.physx_cuda.h5",
        ),
        "trajectory_json": os.path.join(
            dataset_dir,
            f"trajectory.{args.obs_mode}.{args.expert_control_mode}.physx_cuda.json",
        ),
        "dataset_cache": os.path.join(resume_dir, "expert_sft_dataset.pt"),
    }


def save_checkpoint(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_env_kwargs(args, render_mode: str, sim_backend: str | None) -> Dict[str, Any]:
    kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": render_mode,
        "max_episode_steps": args.max_episode_steps,
    }
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    return kwargs


def create_expert_env_kwargs(args, render_mode: str, sim_backend: str | None) -> Dict[str, Any]:
    kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.expert_control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": render_mode,
        "max_episode_steps": args.max_episode_steps,
    }
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    return kwargs


def make_sample_fn(agent_names: Sequence[str], agent: MultiAgentVLAMAPPOAgent, deterministic=True):
    def sample_fn(obs):
        batch = build_batch_from_obs(obs, list(agent_names))
        return agent.get_action(batch, deterministic=deterministic)

    return sample_fn


def build_expert_batch_from_raw_obs(obs: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    rgb = obs["sensor_data"]["base_camera"]["rgb"][..., :3]
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
    rgb = F.interpolate(rgb, size=128, mode="bilinear")

    agent_states = {}
    for agent_name in obs["agent"].keys():
        is_left = agent_name.endswith("-0")
        tcp_key = "left_arm_tcp" if is_left else "right_arm_tcp"
        tcp_to_cube_key = "left_arm_tcp_to_cube_pos" if is_left else "right_arm_tcp_to_cube_pos"
        flat_state = common.flatten_state_dict(
            {
                "cube_pose": obs["extra"]["cube_pose"],
                "cube_to_goal_pos": obs["extra"]["cube_to_goal_pos"],
                tcp_to_cube_key: obs["extra"][tcp_to_cube_key],
                tcp_key: obs["extra"][tcp_key],
                "qpos": obs["agent"][agent_name]["qpos"],
                "qvel": obs["agent"][agent_name]["qvel"],
                "stage": obs["extra"]["stage"],
            },
            use_torch=True,
            device=rgb.device,
        )
        agent_states[f"agent_states_{agent_name}"] = flat_state

    global_state = common.flatten_state_dict(
        {"agent": obs["agent"], "extra": obs["extra"]},
        use_torch=True,
        device=rgb.device,
    )
    batch = {
        "rgb": rgb.to(device=device, dtype=torch.float32),
        "global_state": global_state.to(device=device, dtype=torch.float32),
    }
    for key, value in agent_states.items():
        batch[key] = value.to(device=device, dtype=torch.float32)
    return batch


def get_vla_agent_info(args) -> Dict[str, Any]:
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": env.single_action_space[agent_names[0]].shape[0],
    }
    env.close()
    return info


def get_expert_agent_info(args, device: torch.device) -> Dict[str, Any]:
    env_kwargs = create_expert_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = env.reset()
    batch = build_expert_batch_from_raw_obs(obs, device)
    agent_names = list(obs["agent"].keys())
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": env.single_action_space[agent_names[0]].shape[0],
    }
    env.close()
    return info


def load_expert_agent(args, device: torch.device) -> MAPPOAgent:
    infos = get_expert_agent_info(args, device)
    loader_args = argparse.Namespace(
        task_name=args.task_name,
        seed=args.seed,
        normalize_state=args.expert_normalize_state,
        max_sparsity=args.expert_max_sparsity,
        pretrained_agent_path=args.expert_agent_dir,
    )
    env_kwargs = create_expert_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    agent_obs_rules = {
        "panda_wristcam-0": [
            "cube_pose",
            "cube_to_goal_pos",
            "left_arm_tcp_to_cube_pos",
            "left_arm_tcp",
            "qpos",
            "qvel",
            "stage",
        ],
        "panda_wristcam-1": [
            "cube_pose",
            "cube_to_goal_pos",
            "right_arm_tcp_to_cube_pos",
            "right_arm_tcp",
            "qpos",
            "qvel",
            "stage",
        ],
    }
    agent = build_fbs_expert_agent(loader_args, infos, env_kwargs, agent_obs_rules, device)
    set_sparsity(agent, 0.0)
    for parameter in agent.parameters():
        parameter.requires_grad = False
    agent.eval()
    return agent


def flatten_expert_actions(action_dict: Dict[str, np.ndarray], agent_names: Sequence[str]) -> np.ndarray:
    flat_parts = []
    for name in agent_names:
        value = action_dict[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        flat_parts.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(flat_parts, axis=-1)


def verify_replayable_trajectory(traj_path: str) -> None:
    print(f"[Replay] verifying {traj_path}")
    replay_main(
        ReplayArgs(
            traj_path=traj_path,
            sim_backend="physx_cpu",
            save_traj=False,
            save_video=False,
            count=1,
            num_envs=1,
            allow_failure=True,
            use_env_states=False,
        )
    )
    print("[Replay] verification passed")


def filter_successful_trajectories(traj_path: str, max_episodes: int | None = None) -> int:
    json_path = traj_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    successful_episodes = [episode for episode in meta["episodes"] if bool(episode.get("success", False))]
    if max_episodes is not None:
        successful_episodes = successful_episodes[:max_episodes]

    tmp_h5_path = f"{traj_path}.tmp"
    tmp_json_path = f"{json_path}.tmp"
    kept_episodes = []

    with h5py.File(traj_path, "r") as src_h5, h5py.File(tmp_h5_path, "w") as dst_h5:
        for new_episode_id, episode in enumerate(successful_episodes):
            src_group_name = f"traj_{episode['episode_id']}"
            dst_group_name = f"traj_{new_episode_id}"
            src_h5.copy(src_group_name, dst_h5, name=dst_group_name)

            new_episode = dict(episode)
            new_episode["episode_id"] = new_episode_id
            kept_episodes.append(new_episode)

    new_meta = dict(meta)
    new_meta["episodes"] = kept_episodes
    with open(tmp_json_path, "w") as f:
        json.dump(new_meta, f, indent=2)

    shutil.move(tmp_h5_path, traj_path)
    shutil.move(tmp_json_path, json_path)
    return len(kept_episodes)


def collect_expert_trajectories(args, ckpt: Dict[str, str], expert_agent: MAPPOAgent | None) -> str:
    traj_path = args.trajectory_h5_path or ckpt["trajectory_h5"]
    json_path = traj_path.replace(".h5", ".json")
    if os.path.exists(traj_path) and os.path.exists(json_path) and not args.force_recollect_trajectories:
        print(f"[Collect] reuse existing trajectories: {traj_path}")
        return traj_path

    if expert_agent is None:
        raise RuntimeError("Expert agent is required when recollecting trajectories.")

    output_dir = os.path.dirname(traj_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    device = next(expert_agent.parameters()).device
    env_kwargs = create_expert_env_kwargs(args, render_mode="rgb_array", sim_backend="physx_cuda")
    base_env = gym.make(args.task_name, num_envs=args.num_collect_envs, **env_kwargs)
    flat_env = FlattenActionSpaceWrapper(base_env)
    env = RecordEpisode(
        flat_env,
        output_dir=output_dir,
        save_trajectory=True,
        trajectory_name=Path(traj_path).stem,
        save_video=False,
        record_reward=False,
        record_env_state=False,
        source_type="rl",
        source_desc="Demonstrations generated by rolling out a pretrained CNN expert policy",
    )

    obs, _ = env.reset(seed=args.seed)
    successful_episodes = 0
    total_finished_episodes = 0
    progress = tqdm(total=args.num_successful_trajectories, ascii=True, desc="Collect Expert")

    while successful_episodes < args.num_successful_trajectories:
        with torch.no_grad():
            expert_batch = build_expert_batch_from_raw_obs(obs, device)
            actions = expert_agent.get_action(expert_batch, deterministic=not args.expert_sample_actions)
            flat_actions = flatten_expert_actions(actions, list(actions.keys()))

        obs, _, terminations, truncations, infos = env.step(flat_actions)
        done = (terminations | truncations).detach().cpu().numpy().astype(bool)

        if done.any():
            done_env_idx = np.flatnonzero(done)
            step_success = infos["success"][done_env_idx].detach().cpu().numpy().astype(bool)
            successful_episodes += int(step_success.sum())
            total_finished_episodes += int(done_env_idx.size)
            progress.n = min(successful_episodes, args.num_successful_trajectories)
            progress.refresh()
            obs, _ = env.reset(options={"env_idx": done_env_idx})

    progress.close()
    env.save_on_reset = False
    env.close()
    kept_successes = filter_successful_trajectories(
        traj_path,
        max_episodes=args.num_successful_trajectories,
    )
    print(
        f"[Collect] successful_episodes={successful_episodes}, "
        f"finished_episodes={total_finished_episodes}, "
        f"kept_success_episodes={kept_successes}, saved={traj_path}"
    )
    if args.verify_replay:
        verify_replayable_trajectory(traj_path)
    return traj_path


def load_h5_item(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        return node[()]
    return {key: load_h5_item(node[key]) for key in node.keys()}


def load_sft_dataset_from_trajectories(
    args,
    ckpt: Dict[str, str],
    trajectory_h5_path: str,
    agent_names: Sequence[str],
) -> Dict[str, Any]:
    cache_path = args.dataset_cache_path or ckpt["dataset_cache"]
    if os.path.exists(cache_path) and not args.force_rebuild_dataset_cache:
        print(f"[Dataset] loading cache {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    json_path = trajectory_h5_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        meta = json.load(f)
    success_episodes = [episode for episode in meta["episodes"] if bool(episode.get("success", False))]
    success_episodes = success_episodes[: args.num_successful_trajectories]
    if not success_episodes:
        raise RuntimeError(f"No successful episodes found in {json_path}")

    rgbs = []
    global_states = []
    trajectory_lengths = []
    per_agent_states = {name: [] for name in agent_names}
    per_agent_actions = {name: [] for name in agent_names}

    with h5py.File(trajectory_h5_path, "r") as h5_file:
        for episode in tqdm(success_episodes, ascii=True, desc="Build SFT Dataset"):
            traj = h5_file[f"traj_{episode['episode_id']}"]
            horizon = int(episode["elapsed_steps"])
            action_node = traj["actions"]
            if isinstance(action_node, h5py.Dataset):
                flat_actions = action_node[:horizon]
                split_dim = flat_actions.shape[-1] // len(agent_names)
                actions = {
                    name: flat_actions[:, idx * split_dim : (idx + 1) * split_dim]
                    for idx, name in enumerate(agent_names)
                }
            else:
                actions = load_h5_item(action_node)
            raw_obs = {
                "sensor_data": {
                    "base_camera": {
                        "rgb": traj["obs"]["sensor_data"]["base_camera"]["rgb"][:horizon],
                    }
                },
                "agent": {},
                "extra": {},
            }
            for name in agent_names:
                raw_obs["agent"][name] = {
                    "qpos": traj["obs"]["agent"][name]["qpos"][:horizon],
                    "qvel": traj["obs"]["agent"][name]["qvel"][:horizon],
                }
            for key in [
                "cube_pose",
                "cube_to_goal_pos",
                "left_arm_tcp",
                "right_arm_tcp",
                "left_arm_tcp_to_cube_pos",
                "right_arm_tcp_to_cube_pos",
                "stage",
            ]:
                raw_obs["extra"][key] = traj["obs"]["extra"][key][:horizon]

            parsed = build_batch_from_obs(raw_obs, list(agent_names))
            rgbs.append(torch.as_tensor(parsed["rgb"]).to(dtype=torch.uint8, device="cpu"))
            global_states.append(torch.as_tensor(parsed["global_state"]).to(dtype=torch.float32, device="cpu"))
            trajectory_lengths.append(horizon)
            for name in agent_names:
                per_agent_states[name].append(
                    torch.as_tensor(parsed[f"agent_states_{name}"]).to(dtype=torch.float32, device="cpu")
                )
                per_agent_actions[name].append(
                    torch.as_tensor(actions[name]).to(dtype=torch.float32, device="cpu")
                )

    dataset = {
        "rgb": torch.cat(rgbs, dim=0),
        "global_state": torch.cat(global_states, dim=0),
        "trajectory_lengths": trajectory_lengths,
        "num_samples": int(sum(trajectory_lengths)),
        "num_trajectories": len(trajectory_lengths),
        "agent_names": list(agent_names),
        "trajectory_h5_path": trajectory_h5_path,
    }
    for name in agent_names:
        dataset[f"agent_states_{name}"] = torch.cat(per_agent_states[name], dim=0)
        dataset[f"actions_{name}"] = torch.cat(per_agent_actions[name], dim=0)

    save_checkpoint(cache_path, dataset)
    dump_json(
        f"{cache_path}.json",
        {
            "num_samples": dataset["num_samples"],
            "num_trajectories": dataset["num_trajectories"],
            "trajectory_lengths": [int(x) for x in dataset["trajectory_lengths"]],
            "trajectory_h5_path": trajectory_h5_path,
        },
    )
    print(f"[Dataset] cached to {cache_path}")
    return dataset


def build_sft_optimizer(args, agent: MultiAgentVLAMAPPOAgent) -> torch.optim.Optimizer:
    if isinstance(agent, MixedTinyVLAMultiAgentsSFTAgent):
        param_groups = [
            {
                "params": [p for p in agent.vla_actor.vla.parameters() if p.requires_grad],
                "lr": args.backbone_learning_rate,
            },
            {
                "params": [p for p in agent.vla_actor.state_projector.parameters() if p.requires_grad],
                "lr": args.state_learning_rate,
            },
            {
                "params": [p for p in agent.vla_actor.context_projector.parameters() if p.requires_grad]
                + [p for p in agent.vla_actor.actor_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
            },
            {
                "params": [p for p in agent.smolvla_actor.vla.parameters() if p.requires_grad],
                "lr": args.backbone_learning_rate,
            },
            {
                "params": [p for p in agent.smolvla_actor.state_projector.parameters() if p.requires_grad],
                "lr": args.state_learning_rate,
            },
            {
                "params": [p for p in agent.smolvla_actor.context_projector.parameters() if p.requires_grad]
                + [p for p in agent.smolvla_actor.action_mean_head.parameters() if p.requires_grad]
                + [p for p in agent.smolvla_actor.log_std_head.parameters() if p.requires_grad],
                "lr": args.head_learning_rate,
            },
        ]
        param_groups = [group for group in param_groups if group["params"]]
        return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)

    param_groups = [
        {
            "params": [p for p in agent.actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
        },
        {
            "params": [p for p in agent.actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
        },
    ]
    if agent.actor.policy_mode == "residual":
        param_groups.extend(
            [
                {
                    "params": [p for p in agent.actor.context_projector.parameters() if p.requires_grad],
                    "lr": args.head_learning_rate,
                },
                {
                    "params": [p for p in agent.actor.actor_head.parameters() if p.requires_grad],
                    "lr": args.head_learning_rate,
                },
            ]
        )
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def add_action_bins_to_dataset(dataset: Dict[str, Any], agent: MultiAgentVLAMAPPOAgent, chunk_size: int) -> None:
    action_bin_requirements = (
        agent.requires_action_bins() if hasattr(agent, "requires_action_bins") else {name: True for name in agent.agent_names}
    )
    target_names = [name for name, required in action_bin_requirements.items() if required]
    if not target_names:
        return
    keys = [f"action_bins_{name}" for name in target_names]
    if all(key in dataset for key in keys):
        return
    for name in target_names:
        chunks = []
        actions = dataset[f"actions_{name}"]
        for start in range(0, actions.shape[0], chunk_size):
            end = min(actions.shape[0], start + chunk_size)
            if isinstance(agent, MixedTinyVLAMultiAgentsSFTAgent):
                chunks.append(agent.vla_actor.env_actions_to_bin_indices(actions[start:end]))
            else:
                chunks.append(agent.actor.env_actions_to_bin_indices(actions[start:end]))
        dataset[f"action_bins_{name}"] = torch.cat(chunks, dim=0).to(dtype=torch.long, device="cpu")


def fit_state_stats_from_dataset(
    agent: MultiAgentVLAMAPPOAgent,
    dataset: Dict[str, Any],
    chunk_size: int,
) -> None:
    if agent.actor_state_rms is None and agent.critic_state_rms is None:
        return

    agent.unfreeze_state_stats()
    target_device = agent.device if hasattr(agent, "device") else agent.actor.device
    with torch.no_grad():
        total = int(dataset["num_samples"])
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            if agent.actor_state_rms is not None:
                for name in agent.agent_names:
                    agent.actor_state_rms[name].update(
                        dataset[f"agent_states_{name}"][start:end].to(
                            device=target_device,
                            dtype=torch.float32,
                        )
                    )
            if agent.critic_state_rms is not None:
                agent.critic_state_rms.update(
                    dataset["global_state"][start:end].to(device=target_device, dtype=torch.float32)
                )
    agent.freeze_state_stats()


def sample_batch(dataset: Dict[str, Any], indices: np.ndarray, agent_names: Sequence[str], device: torch.device):
    torch_indices = torch.as_tensor(indices, dtype=torch.long)
    batch = {
        "rgb": dataset["rgb"][torch_indices].to(device=device),
        "global_state": dataset["global_state"][torch_indices].to(device=device, dtype=torch.float32),
    }
    action_bins = {}
    actions = {}
    for name in agent_names:
        batch[f"agent_states_{name}"] = dataset[f"agent_states_{name}"][torch_indices].to(
            device=device,
            dtype=torch.float32,
        )
        actions[name] = dataset[f"actions_{name}"][torch_indices].to(device=device, dtype=torch.float32)
        action_bin_key = f"action_bins_{name}"
        if action_bin_key in dataset:
            action_bins[name] = dataset[action_bin_key][torch_indices].to(device=device, dtype=torch.long)
    return batch, action_bins, actions


def maybe_run_eval(
    args,
    ckpt: Dict[str, str],
    agent: MultiAgentVLAMAPPOAgent,
    agent_names: Sequence[str],
    train_iter: int,
    writer: SummaryWriter | None,
) -> Dict[str, float]:
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
        video_dir=f"{ckpt['video_dir']}_train",
    )
    agent.eval()
    metrics = evaluate(
        n=args.eval_episodes,
        sample_fn=make_sample_fn(agent_names, agent, deterministic=True),
        eval_envs=eval_envs,
    )
    eval_envs.close()
    payload = {k: float(v.mean()) for k, v in metrics.items()}
    print(
        f"[Eval] iter={train_iter} "
        + " ".join(f"{k}={v:.4f}" for k, v in sorted(payload.items()))
    )
    if writer is not None:
        for key, value in payload.items():
            writer.add_scalar(f"eval/{key}", value, train_iter)
    return payload


def train_sft(args):
    device = get_device()
    ckpt = resolve_ckpt_dir(args)

    if args.control_mode != args.expert_control_mode and not args.collect_only:
        raise ValueError(
            f"SFT currently requires matching control modes, got expert_control_mode={args.expert_control_mode} "
            f"and control_mode={args.control_mode}. Collect-only is supported."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic

    trajectory_h5_path = args.trajectory_h5_path or ckpt["trajectory_h5"]
    trajectory_json_path = trajectory_h5_path.replace(".h5", ".json")
    expert_agent = None
    if args.force_recollect_trajectories or not (
        os.path.exists(trajectory_h5_path) and os.path.exists(trajectory_json_path)
    ):
        expert_agent = load_expert_agent(args, device)
    trajectory_h5_path = collect_expert_trajectories(args, ckpt, expert_agent)
    if args.collect_only:
        print("[Train] collect-only enabled; stop after trajectory generation and replay verification.")
        return
    infos = get_vla_agent_info(args)
    agent_names = infos["agent_names"]
    dataset = load_sft_dataset_from_trajectories(args, ckpt, trajectory_h5_path, agent_names)
    if expert_agent is not None:
        del expert_agent

    if args.model_backbone == "multi_agents":
        agent = MixedTinyVLAMultiAgentsSFTAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        ).to(device)
    else:
        agent = MultiAgentVLAMAPPOAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            model_backbone=args.model_backbone,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        ).to(device)
    total_params = sum(parameter.numel() for parameter in agent.parameters())
    trainable_params = sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
    print(
        f"[Model] backbone={args.model_backbone} total_params={total_params / 1e6:.2f}M "
        f"trainable_params={trainable_params / 1e6:.2f}M"
    )
    optimizer = build_sft_optimizer(args, agent)

    add_action_bins_to_dataset(dataset, agent, args.state_stats_batch_size)
    if args.refit_state_stats or not os.path.exists(ckpt["latest_agent"]):
        fit_state_stats_from_dataset(agent, dataset, args.state_stats_batch_size)

    best_score = -1.0
    global_updates = 0
    processed_samples = 0
    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []

    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] resume agent from {ckpt['latest_agent']}")
        agent.load_checkpoint_state_dict(torch.load(ckpt["latest_agent"], map_location="cpu"))
        agent.to(device)
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        try:
            optimizer.load_state_dict(latest_opt["opt"])
        except ValueError as exc:
            print(
                "[Train] skip optimizer state restore because parameter groups changed: "
                f"{exc}"
            )
        best_score = float(latest_opt.get("best_score", best_score))
        global_updates = int(latest_opt.get("iter", latest_opt.get("updates", 0)))
        processed_samples = int(latest_opt.get("processed_samples", global_updates * args.batch_size))

    writer = SummaryWriter(ckpt["log_dir"])
    total_samples = int(dataset["num_samples"])
    batches_per_pass = max(1, math.ceil(total_samples / args.batch_size))
    total_iters = args.sft_total_iters
    if total_iters is None:
        total_iters = args.sft_epochs * batches_per_pass
        print(
            f"[Train] --sft-total-iters not set; fallback to "
            f"sft_epochs({args.sft_epochs}) * batches_per_pass({batches_per_pass}) = {total_iters}"
        )
    total_iters = int(total_iters)
    if global_updates >= total_iters:
        print(f"[Train] already reached target iterations: {global_updates}/{total_iters}")
        writer.close()
        return

    pbar = tqdm(total=total_iters, initial=global_updates, ascii=True, desc="SFT Iters")
    train_start = time.time()
    shuffled = np.arange(total_samples)
    np.random.shuffle(shuffled)
    cursor = 0
    dataset_pass = 0
    window_loss = 0.0
    window_entropy = 0.0
    window_steps = 0

    while global_updates < total_iters:
        agent.train()
        if cursor >= total_samples:
            np.random.shuffle(shuffled)
            cursor = 0
            dataset_pass += 1

        end = min(cursor + args.batch_size, total_samples)
        mb_inds = shuffled[cursor:end]
        cursor = end
        batch, action_bins_input, actions_input = sample_batch(dataset, mb_inds, agent_names, device)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = args.use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            _, log_probs, entropies, _ = agent.get_action_and_value(
                batch,
                actions_input=actions_input,
                action_bins_input=action_bins_input,
            )
            mean_log_prob = torch.stack([log_probs[name] for name in agent_names], dim=0).mean()
            mean_entropy = torch.stack([entropies[name] for name in agent_names], dim=0).mean()
            bc_loss = (-mean_log_prob) / agent.action_dim

        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        global_updates += 1
        processed_samples += len(mb_inds)
        window_steps += 1
        window_loss += float(bc_loss.detach().item())
        window_entropy += float(mean_entropy.detach().item())
        writer.add_scalar("train/bc_loss_step", float(bc_loss.detach().item()), global_updates)
        writer.add_scalar("train/entropy_step", float(mean_entropy.detach().item()), global_updates)
        pbar.update(1)

        should_log = global_updates % args.log_interval_iters == 0 or global_updates == total_iters
        should_eval = global_updates % args.eval_interval_iters == 0 or global_updates == total_iters

        eval_payload = {}
        score = None
        if should_eval:
            eval_payload = maybe_run_eval(args, ckpt, agent, agent_names, global_updates, writer)
            score = eval_payload.get("success_rate")
            if score is None:
                score = eval_payload.get("success_once")
            if score is None:
                score = next(iter(eval_payload.values()))
            if score >= best_score:
                best_score = score
                save_checkpoint(ckpt["best_agent"], agent.checkpoint_state_dict())
                print(f"[Eval] new best model saved, score={score:.4f}")

        if should_log or should_eval:
            avg_loss = window_loss / max(window_steps, 1)
            avg_entropy = window_entropy / max(window_steps, 1)
            samples_per_sec = processed_samples / max(time.time() - train_start, 1e-6)
            writer.add_scalar("train/bc_loss_iter_window", avg_loss, global_updates)
            writer.add_scalar("train/entropy_iter_window", avg_entropy, global_updates)
            writer.add_scalar("train/samples_per_sec", samples_per_sec, global_updates)

            save_checkpoint(ckpt["latest_agent"], agent.checkpoint_state_dict())
            save_checkpoint(
                ckpt["latest_opt"],
                {
                    "opt": optimizer.state_dict(),
                    "iter": global_updates,
                    "updates": global_updates,
                    "processed_samples": processed_samples,
                    "best_score": best_score,
                },
            )

            row = {
                "iter": global_updates,
                "dataset_pass": dataset_pass,
                "bc_loss": float(avg_loss),
                "entropy": float(avg_entropy),
                "samples_per_sec": float(samples_per_sec),
            }
            if score is not None:
                row["score"] = float(score)
            row.update({f"eval_{k}": float(v) for k, v in eval_payload.items()})
            metrics_log.append(row)
            dump_json(ckpt["metrics"], metrics_log)
            print(
                f"[Train] iter={global_updates}/{total_iters} "
                f"pass={dataset_pass} bc_loss={avg_loss:.6f} "
                f"entropy={avg_entropy:.6f} samples_per_sec={samples_per_sec:.2f}"
            )

        if should_log or global_updates == total_iters:
            window_loss = 0.0
            window_entropy = 0.0
            window_steps = 0

    pbar.close()
    writer.close()


def evaluate_only(args):
    device = get_device()
    ckpt = resolve_ckpt_dir(args)
    infos = get_vla_agent_info(args)
    agent_names = infos["agent_names"]
    if args.model_backbone == "multi_agents":
        agent = MixedTinyVLAMultiAgentsSFTAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        ).to(device)
    else:
        agent = MultiAgentVLAMAPPOAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=args.freeze_vla_backbone,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            model_backbone=args.model_backbone,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        ).to(device)
    if args.eval_agent_dir is None:
        raise ValueError("Please provide --eval-agent-dir for evaluation mode")
    agent.load_checkpoint_state_dict(
        torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu")
    )
    agent.to(device)
    payload = maybe_run_eval(args, ckpt, agent, agent_names, train_iter=0, writer=None)
    dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="TwoRobotPickCube-v2")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="pandas_pandas")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="openvla",
        choices=["openvla", "tiny", "multi_agents"],
        help="Choose the original OpenVLA adapter backbone, the local tiny autoregressive VLA backbone, or the mixed tiny VLA-adapter + tiny SmolVLA SFT backbone. model_dir is required only for openvla.",
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--obs-mode", type=str, default="rgb+state_dict")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--expert-control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--ignore-torch-deterministic", action="store_true")

    parser.add_argument("--expert-agent-dir", type=str, default=DEFAULT_EXPERT_AGENT_DIR)
    parser.add_argument("--expert-max-sparsity", type=float, default=0.0)
    parser.add_argument("--trajectory-h5-path", type=str, default=None)
    parser.add_argument("--dataset-cache-path", type=str, default=None)
    parser.add_argument("--force-recollect-trajectories", action="store_true")
    parser.add_argument("--force-rebuild-dataset-cache", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-replay-verify", action="store_false", dest="verify_replay")
    parser.add_argument("--num-successful-trajectories", type=int, default=10240)
    parser.add_argument("--num-collect-envs", type=int, default=1024)
    parser.add_argument("--expert-sample-actions", action="store_true")
    parser.add_argument("--no-expert-normalize-state", action="store_false", dest="expert_normalize_state")
    parser.set_defaults(expert_normalize_state=True, verify_replay=True)

    parser.add_argument("--sft-total-iters", type=int, default=None)
    parser.add_argument("--sft-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-interval-iters", type=int, default=2000)
    parser.add_argument("--state-stats-batch-size", type=int, default=1024)
    parser.add_argument("--refit-state-stats", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--vision-token-pool-size",
        type=int,
        default=None,
        help="Optionally downsample visual patch tokens before the language model, e.g. 16 to match InternVL-scale visual token counts.",
    )
    parser.add_argument("--tiny-hidden-dim", type=int, default=640)
    parser.add_argument("--tiny-vision-layers", type=int, default=7)
    parser.add_argument("--tiny-decoder-layers", type=int, default=8)
    parser.add_argument("--tiny-attention-heads", type=int, default=10)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--state-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--eval-interval-iters", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-eval-envs", type=int, default=16)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="native",
        choices=["residual", "native"],
        help="BC default uses the native VLA action-token policy so its checkpoints are compatible with native PPO.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.evaluate_mode:
        evaluate_only(args)
        return
    train_sft(args)


if __name__ == "__main__":
    main()
