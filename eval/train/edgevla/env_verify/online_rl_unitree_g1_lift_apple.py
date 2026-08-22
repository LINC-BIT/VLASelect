from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, get_args, get_origin

import gymnasium as gym
import numpy as np
import torch

sys.path.append(".")

import workloads.human as human_workload
import train.tinyvla.model_impl.online_rl_open_cabinet_drawer as reference


TASK_PROMPT = "touch the cube with two right-hand fingers and hold contact for half a second."
DEFAULT_WORKDIR = "train/edgevla/env_verify/outputs/ppo_unitree_g1_lift_apple"
DEFAULT_SMOKE_WORKDIR = "train/edgevla/env_verify/outputs_smoke"
ENV_ID = human_workload.HUMAN_ENV_NAME
RIGHT_ARM_AND_HAND_ACTION_INDICES: Tuple[int, ...] = (
    2,
    4,
    6,
    8,
    10,
    14,
    15,
    16,
    20,
    21,
    22,
    24,
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> torch.Tensor:
    sensor_data = obs["sensor_data"]
    preferred_cameras = ("head_camera", "base_camera")
    rgb_tensors: List[torch.Tensor] = []
    for camera_name in preferred_cameras:
        if camera_name not in sensor_data:
            continue
        rgb = sensor_data[camera_name]["rgb"]
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(np.asarray(rgb)[..., :3].astype(np.uint8, copy=False))
        else:
            rgb = rgb[..., :3].detach().to(device="cpu", dtype=torch.uint8)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        rgb_tensors.append(rgb.contiguous())

    if not rgb_tensors:
        raise KeyError(f"No RGB cameras found in observation sensor_data keys={list(sensor_data.keys())}")
    return torch.cat(rgb_tensors, dim=2).contiguous()


def extract_humanoid_lift_state_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    agent = obs["agent"]
    extra = obs["extra"]

    qpos = _to_numpy(agent["qpos"]).astype(np.float32)
    qvel = _to_numpy(agent["qvel"]).astype(np.float32)
    tcp_pose = _to_numpy(extra["tcp_pose"]).astype(np.float32)
    obj_pose = _to_numpy(extra["obj_pose"]).astype(np.float32)
    tcp_to_obj_pos = _to_numpy(extra["tcp_to_obj_pos"]).astype(np.float32)
    obj_to_lift_goal_pos = _to_numpy(extra["obj_to_lift_goal_pos"]).astype(np.float32)
    apple_lift = _to_numpy(extra["apple_lift"]).astype(np.float32)
    is_grasped = _to_numpy(extra["is_grasped"]).astype(np.float32)
    is_lifted = _to_numpy(extra["is_lifted"]).astype(np.float32)

    if qpos.ndim == 1:
        qpos = qpos[None, :]
        qvel = qvel[None, :]
        tcp_pose = tcp_pose[None, :]
        obj_pose = obj_pose[None, :]
        tcp_to_obj_pos = tcp_to_obj_pos[None, :]
        obj_to_lift_goal_pos = obj_to_lift_goal_pos[None, :]

    batch_size = qpos.shape[0]
    apple_lift = apple_lift.reshape(batch_size, 1)
    is_grasped = is_grasped.reshape(batch_size, 1)
    is_lifted = is_lifted.reshape(batch_size, 1)

    qvel = np.clip(qvel, -10.0, 10.0) / 10.0
    tcp_pose = np.clip(tcp_pose, -2.0, 2.0) / 2.0
    obj_pose = np.clip(obj_pose, -2.0, 2.0) / 2.0
    tcp_to_obj_pos = np.clip(tcp_to_obj_pos, -1.0, 1.0)
    obj_to_lift_goal_pos = np.clip(obj_to_lift_goal_pos, -1.0, 1.0)
    apple_lift = np.clip(apple_lift, -0.2, 0.3) / 0.3

    return np.concatenate(
        [
            qpos,
            qvel,
            tcp_pose,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_lift_goal_pos,
            apple_lift,
            is_grasped,
            is_lifted,
        ],
        axis=-1,
    ).astype(np.float32)


def get_controlled_action_indices(action_mapping: Dict[str, Tuple[int, int]]) -> Tuple[int, ...]:
    if "body" in action_mapping:
        start, end = action_mapping["body"]
        body_indices = tuple(range(int(start), int(end)))
        if len(body_indices) < max(RIGHT_ARM_AND_HAND_ACTION_INDICES) + 1:
            raise ValueError(
                f"Expected at least {max(RIGHT_ARM_AND_HAND_ACTION_INDICES) + 1} body action dims, "
                f"got {len(body_indices)} from action_mapping={action_mapping}"
            )
        return tuple(body_indices[index] for index in RIGHT_ARM_AND_HAND_ACTION_INDICES)
    return reference.get_controlled_action_indices(action_mapping)


def backup_run_sources(output_dir: Path) -> None:
    code_dir = reference.mkdir(output_dir / "code")
    sources = {
        "online_rl_unitree_g1_lift_apple.py": Path(__file__).resolve(),
        "online_rl_open_cabinet_drawer.py": Path(reference.__file__).resolve(),
        "workloads_human__init__.py": Path(human_workload.__file__).resolve(),
    }
    manifest = {}
    for backup_name, source_path in sources.items():
        if not source_path.is_file():
            continue
        destination = code_dir / backup_name
        shutil.copy2(source_path, destination)
        manifest[backup_name] = {"source": str(source_path), "backup": str(destination)}
    reference.save_json(code_dir / "source_manifest.json", manifest)


def patch_reference_for_humanoid_env() -> None:
    reference.TASK_PROMPT = TASK_PROMPT
    reference.backup_run_sources = backup_run_sources
    reference.extract_rgb_batch_from_obs = extract_rgb_batch_from_obs
    reference.extract_cabinet_state_batch_from_obs = extract_humanoid_lift_state_batch_from_obs
    reference.get_controlled_action_indices = get_controlled_action_indices


def set_reference_arg_defaults() -> None:
    defaults = {
        "env_id": ENV_ID,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "obs_mode": "rgb+state_dict",
        "output_dir": DEFAULT_WORKDIR,
        "num_envs": 64,
        "num_eval_envs": 8,
        "num_steps": 100,
        "eval_episodes": 16,
        "eval_every_updates": 1,
        "save_video": False,
        "action_dim": len(RIGHT_ARM_AND_HAND_ACTION_INDICES),
        "env_action_dim": 25,
        "state_dim": 73,
        "cuda_device": "5",
        "run_setup_smoke": True,
    }
    for field_name, value in defaults.items():
        if field_name in reference.Args.__dataclass_fields__:
            reference.Args.__dataclass_fields__[field_name].default = value


def parse_args() -> reference.Args:
    set_reference_arg_defaults()
    parser = argparse.ArgumentParser()
    for field_name, field_def in reference.Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        field_type = field_def.type
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=reference.parse_bool, default=default)
        elif default is None:
            arg_type = str
            origin = get_origin(field_type)
            if origin is not None:
                candidate_types = [candidate for candidate in get_args(field_type) if candidate is not type(None)]
                if len(candidate_types) == 1 and isinstance(candidate_types[0], type):
                    arg_type = candidate_types[0]
            parser.add_argument(arg_name, type=arg_type, default=None)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    namespace = parser.parse_args()
    return reference.Args(**vars(namespace))


def run_env_contract_smoke(args: reference.Args, device: torch.device, output_dir: Path) -> None:
    env_action_dim, state_dim, controlled_action_indices = reference.inspect_env_contract(args, device)
    backend_kwargs = reference.get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        **backend_kwargs,
    )
    try:
        obs, info = env.reset(seed=args.seed)
        rgbs = extract_rgb_batch_from_obs(obs)
        states = extract_humanoid_lift_state_batch_from_obs(obs)
        random_action = torch.as_tensor(
            env.action_space.sample(),
            device=device,
            dtype=torch.float32,
        ).view(1, -1)
        _, reward, terminated, truncated, step_info = env.step(random_action)
        payload = {
            "env_id": args.env_id,
            "rgb_shape": list(rgbs.shape),
            "state_shape": list(states.shape),
            "env_action_dim": env_action_dim,
            "state_dim": state_dim,
            "controlled_action_indices": list(controlled_action_indices),
            "controlled_action_dim": len(controlled_action_indices),
            "fixed_action_dim": env_action_dim - len(controlled_action_indices),
            "initial_success": bool(info["success"].item()),
            "step_reward": float(reward.item()),
            "step_success": bool(step_info["success"].item()),
            "terminated": bool(terminated.item()),
            "truncated": bool(truncated.item()),
        }
    finally:
        env.close()

    print("[human-env-smoke]", payload)
    reference.save_json(output_dir / "env_smoke.json", payload)


def main() -> None:
    patch_reference_for_humanoid_env()
    args = parse_args()

    if args.mode == "train":
        try:
            reference.train(args)
        finally:
            reference.cleanup_runtime()
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_device)
    reference.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    output_root = DEFAULT_SMOKE_WORKDIR if args.output_dir == DEFAULT_WORKDIR else args.output_dir
    output_dir = reference.mkdir(Path(output_root) / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    backup_run_sources(output_dir)
    reference.save_json(output_dir / "args.json", asdict(args))

    if args.mode == "env_smoke":
        run_env_contract_smoke(args, device, output_dir)
        return
    if args.mode == "vla_smoke":
        reference.run_vla_inference_smoke(args, device, output_dir)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
