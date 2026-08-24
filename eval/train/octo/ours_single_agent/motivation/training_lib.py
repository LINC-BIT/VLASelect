from __future__ import annotations

import ast
import json
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from train.octo.ours_single_agent import online_rl_cl as base
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs


# WORKLOAD_ENVS = ['PokeCubeLightWeaker50-v1','PushCubeColorTempLower50-v1','StackCubeObjectBlack-v1','StackCubeLightWeaker50-v1','StackCubeColorTempLower50-v1','RollBallLightWeaker50-v1','RollBallObjectBlack-v1','PickCubeLightWeaker50-v1','RollBallColorTempHigher50-v1','PokeCubeColorTempLower50-v1']
WORKLOAD_ENVS = ['PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PushCubeObjectPurple-v1','PushCubeObjectBlack-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PickCubeColorTempHigher50-v1','PickCubeColorTempLower50-v1','PickCubeObjectPurple-v1','PickCubeObjectBlack-v1']
WORKLOAD_CHANGE_TIME_POINTS = list(range(30, 330, 30))

MOTIVATION_DIR = Path("train/octo/ours_single_agent/motivation")
LATEST_RUN_FILES = {
    "original": MOTIVATION_DIR / "original_model_latest_run.json",
    "small": MOTIVATION_DIR / "small_model_latest_run.json",
}


def _ensure_2d_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    return value


def _slice_xyz(value: torch.Tensor) -> torch.Tensor:
    value = _ensure_2d_tensor(value)
    return value[:, :3]


def _resolve_extra_tensor(
    extra: dict,
    batch_size: int,
    key: str,
    *,
    aliases: Sequence[str] = (),
    default_dim: int,
    device: Optional[torch.device] = None,
    derive=None,
) -> torch.Tensor:
    candidate_keys = (key, *aliases)
    for candidate_key in candidate_keys:
        if candidate_key in extra:
            value = _ensure_2d_tensor(extra[candidate_key])
            if candidate_key == "is_grasped":
                value = value.to(torch.float32)
            if value.shape[1] > default_dim:
                value = value[:, :default_dim]
            elif value.shape[1] < default_dim:
                pad = torch.zeros(
                    (value.shape[0], default_dim - value.shape[1]),
                    dtype=value.dtype,
                    device=value.device,
                )
                value = torch.cat([value, pad], dim=1)
            return value
    if derive is not None:
        value = _ensure_2d_tensor(derive(extra))
        if value.shape[1] > default_dim:
            value = value[:, :default_dim]
        elif value.shape[1] < default_dim:
            pad = torch.zeros(
                (value.shape[0], default_dim - value.shape[1]),
                dtype=value.dtype,
                device=value.device,
            )
            value = torch.cat([value, pad], dim=1)
        return value
    return torch.zeros((batch_size, default_dim), dtype=torch.float32, device=device)


def patch_observation_wrapper_for_multi_task_state() -> None:
    if getattr(base.FlattenRGBDObservationWrapper2, "_motivation_patched", False):
        return

    def patched_observation(self, observation: dict):
        observation = dict(observation)
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = []
        depth_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)

        agent = observation["agent"]
        extra = observation["extra"]
        state_parts = []
        for key in ["qpos", "qvel"]:
            state_parts.append(_ensure_2d_tensor(agent[key]).to(torch.float32))

        batch_size = state_parts[0].shape[0]
        device = state_parts[0].device
        tcp_pose = _resolve_extra_tensor(
            extra,
            batch_size,
            "tcp_pose",
            default_dim=7,
            device=device,
        )
        goal_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "goal_pos",
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(obs_extra["cubeB_pose"])
            if "cubeB_pose" in obs_extra
            else None,
        )
        obj_pose = _resolve_extra_tensor(
            extra,
            batch_size,
            "obj_pose",
            aliases=("cube_pose", "ball_pose", "cubeA_pose"),
            default_dim=7,
            device=device,
        )
        tcp_to_obj_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "tcp_to_obj_pos",
            aliases=("tcp_to_ball_pos", "tcp_to_cubeA_pos", "peghead_to_cube_pos", "tcp_to_peg_pos"),
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(obj_pose) - _slice_xyz(tcp_pose),
        )
        obj_to_goal_pos = _resolve_extra_tensor(
            extra,
            batch_size,
            "obj_to_goal_pos",
            aliases=("ball_to_goal_pos", "cubeA_to_cubeB_pos", "cube_to_goal_pos"),
            default_dim=3,
            device=device,
            derive=lambda obs_extra: _slice_xyz(goal_pos) - _slice_xyz(obj_pose),
        )
        is_grasped = _resolve_extra_tensor(
            extra,
            batch_size,
            "is_grasped",
            default_dim=1,
            device=device,
        )

        state_parts.extend(
            [
                is_grasped.to(torch.float32),
                tcp_pose.to(torch.float32),
                goal_pos.to(torch.float32),
                obj_pose.to(torch.float32),
                tcp_to_obj_pos.to(torch.float32),
                obj_to_goal_pos.to(torch.float32),
            ]
        )
        state = torch.cat(state_parts, dim=1)

        ret = {}
        if self.include_state:
            ret["state"] = state
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb_images
        elif self.include_rgb and self.include_depth:
            if self.sep_depth:
                ret["rgb"] = rgb_images
                ret["depth"] = depth_images
            else:
                ret["rgbd"] = torch.concat([rgb_images, depth_images], axis=-1)
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth_images
        return ret

    base.FlattenRGBDObservationWrapper2.observation = patched_observation
    base.FlattenRGBDObservationWrapper2._motivation_patched = True


@dataclass
class MotivationArgs(base.Args):
    metrics_filename: str = "motivation_eval_metrics.jsonl"
    importance_dirname: str = "motivation_neuron_importance"


def apply_fixed_workload(args: MotivationArgs) -> None:
    if args.envs_id is not None:
        env_sequence = list(ast.literal_eval(args.envs_id))
    else:
        env_sequence = WORKLOAD_ENVS
        override_envs = os.environ.get("MOTIVATION_ENV_SEQUENCE")
        if override_envs:
            env_sequence = list(ast.literal_eval(override_envs))
        args.envs_id = repr(env_sequence)

    if args.env_change_time_points is not None:
        change_points = list(ast.literal_eval(args.env_change_time_points))
    else:
        change_points = WORKLOAD_CHANGE_TIME_POINTS
        override_change_points = os.environ.get("MOTIVATION_CHANGE_POINTS")
        if override_change_points:
            change_points = list(ast.literal_eval(override_change_points))
        args.env_change_time_points = repr(change_points)

    if len(env_sequence) == 0:
        raise ValueError("`envs_id` must contain at least one environment")

    args.env_id = env_sequence[0]


def apply_mode_overrides(args: MotivationArgs, mode: Literal["original", "small"]) -> None:
    if mode == "original":
        args.max_sparsity = 0.
        args.small_model_generation_strategy = "source"
        args.small_model_generation_policy = "large"
    elif mode == "small":
        args.max_sparsity = 0.8
        args.small_model_generation_strategy = "source"
        args.small_model_generation_policy = "large"
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def make_run_name(args: MotivationArgs, script_path: str) -> str:
    if args.exp_name is None:
        args.exp_name = Path(script_path).stem
        run_name = (
            f"{args.env_id}/ours/octo/{args.exp_name}/"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        if args.tag is not None:
            run_name += f"-{args.tag}"
        return run_name
    return args.exp_name


def backup_code_snapshot(run_name: str, script_path: str, args: MotivationArgs) -> None:
    code_dir = Path(f"ckpt/{run_name}/code")
    code_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(script_path, code_dir / "script.py")
    shutil.copyfile(
        "train/octo/ours_single_agent/online_rl_cl.py",
        code_dir / "online_rl_cl.py",
    )
    shutil.copyfile(
        "train/octo/ours_single_agent/motivation/training_lib.py",
        code_dir / "training_lib.py",
    )
    with open(code_dir / "args.txt", "w", encoding="utf-8") as handle:
        for arg in vars(args):
            handle.write(f"{arg}: {getattr(args, arg)}\n")

    motivation_snapshot_dir = code_dir / "motivation"
    if motivation_snapshot_dir.exists():
        shutil.rmtree(motivation_snapshot_dir)
    shutil.copytree(
        MOTIVATION_DIR,
        motivation_snapshot_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "res.png",
            "res_images",
            "*_latest_run.json",
            "*.log",
            "*.pt",
            "*.jsonl",
        ),
    )


def write_latest_run_manifest(mode: Literal["original", "small"], run_name: str) -> None:
    manifest_path = LATEST_RUN_FILES[mode]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "run_name": run_name,
        "run_dir": f"ckpt/{run_name}",
        "updated_at": time.time(),
    }
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def list_importance_layers(agent: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in agent.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


def build_zero_importance_snapshot(agent: nn.Module) -> Dict[str, object]:
    layer_names: List[str] = []
    layer_types: List[str] = []
    values: List[List[float]] = []
    for name, module in list_importance_layers(agent):
        layer_names.append(name)
        layer_types.append(type(module).__name__)
        values.append([0.0] * int(module.weight.shape[0]))
    return {
        "layer_names": layer_names,
        "layer_types": layer_types,
        "values": values,
    }


def init_importance_accumulator(agent: nn.Module) -> Dict[str, object]:
    layers = list_importance_layers(agent)
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    layer_types: Dict[str, str] = {}
    for name, module in layers:
        sums[name] = torch.zeros(int(module.weight.shape[0]), dtype=torch.float64)
        counts[name] = 0
        layer_types[name] = type(module).__name__
    return {
        "layer_order": [name for name, _ in layers],
        "layer_types": layer_types,
        "sums": sums,
        "counts": counts,
    }


def update_importance_accumulator(accumulator: Dict[str, object], agent: nn.Module) -> None:
    sums = accumulator["sums"]
    counts = accumulator["counts"]
    for name, module in list_importance_layers(agent):
        if module.weight.grad is None:
            continue
        importance = (module.weight.detach() * module.weight.grad.detach()).abs()
        importance = importance.reshape(importance.shape[0], -1).mean(dim=1)
        sums[name] = sums[name] + importance.cpu().double()
        counts[name] += 1


def finalize_importance_accumulator(accumulator: Dict[str, object]) -> Dict[str, object]:
    layer_names: List[str] = list(accumulator["layer_order"])
    layer_types: List[str] = []
    values: List[List[float]] = []
    for name in layer_names:
        count = accumulator["counts"][name]
        total = accumulator["sums"][name]
        if count > 0:
            averaged = total / count
        else:
            averaged = total
        layer_types.append(accumulator["layer_types"][name])
        values.append(averaged.tolist())
    return {
        "layer_names": layer_names,
        "layer_types": layer_types,
        "values": values,
    }


def save_importance_snapshot(
    run_name: str,
    args: MotivationArgs,
    iteration: int,
    global_step: int,
    elapsed_minutes: float,
    snapshot: Dict[str, object],
) -> None:
    importance_dir = Path(f"ckpt/{run_name}/{args.importance_dirname}")
    importance_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "global_step": global_step,
        "elapsed_minutes": float(elapsed_minutes),
        "saved_at": time.time(),
        **snapshot,
    }
    output_path = importance_dir / f"iter_{iteration:06d}.pt"
    tmp_path = output_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def append_eval_metric(
    run_name: str,
    args: MotivationArgs,
    iteration: int,
    global_step: int,
    elapsed_minutes: float,
    success_once: float,
    success_at_end: float,
) -> None:
    metrics_path = Path(f"ckpt/{run_name}/{args.metrics_filename}")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "global_step": int(global_step),
        "elapsed_minutes": float(elapsed_minutes),
        "success_once": float(success_once),
        "success_at_end": float(success_at_end),
        "saved_at": time.time(),
    }
    with open(metrics_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_logger(
    args: MotivationArgs,
    current_env_id: str,
    env_kwargs: dict,
    max_episode_steps: int,
    run_name: str,
) -> base.Logger:
    if args.track:
        config = vars(args).copy()
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_envs,
            env_id=current_env_id,
            reward_mode="normalized_dense",
            env_horizon=max_episode_steps,
            partial_reset=args.partial_reset,
        )
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=current_env_id,
            reward_mode="normalized_dense",
            env_horizon=max_episode_steps,
            partial_reset=args.partial_reset,
        )
        continual_env_schedule = base.build_continual_env_schedule(args)
        if continual_env_schedule is not None:
            config["continual_env_ids"] = list(continual_env_schedule.env_ids)
            config["continual_env_change_time_points"] = list(
                continual_env_schedule.change_time_points
            )

        wandb_api_key = os.environ.get("WANDB_API_KEY", None)
        if wandb_api_key and len(wandb_api_key) == 40:
            base.wandb.login(key=wandb_api_key)
        base.wandb.init(
            project="EuroSys2026",
            config=config,
            name=run_name,
            save_code=True,
            group=f"Maniskill/{current_env_id}/ours/octo/{args.exp_name}",
        )

    writer = SummaryWriter(f"ckpt/{run_name}/tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    return base.Logger(log_wandb=args.track, tensorboard=writer)


def prepare_args_and_context(
    mode: Literal["original", "small"],
    script_path: str,
) -> Tuple[MotivationArgs, torch.device, str]:
    args = tyro.cli(MotivationArgs)
    apply_fixed_workload(args)
    apply_mode_overrides(args, mode)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print(
        f"Use continual env schedule with first env `{args.env_id}`, "
        f"envs={ast.literal_eval(args.envs_id)}, "
        f"change_time_points={ast.literal_eval(args.env_change_time_points)}"
    )
    print(
        f"batch size: {args.batch_size}, minibatch_size: {args.minibatch_size}, "
        f"num_iterations: {args.num_iterations}"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    run_name = make_run_name(args, script_path)
    backup_code_snapshot(run_name, script_path, args)
    write_latest_run_manifest(mode, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    base.args = args
    base.device = device
    return args, device, run_name


def initialize_agent(
    mode: Literal["original", "small"],
    args: MotivationArgs,
    eval_envs: gym.Env,
    env_kwargs: dict,
    device: torch.device,
) -> nn.Module:
    print("Loading large/original agent")
    large_agent = base.load_agent()
    static_model_name = "original" if mode == "original" else "small"
    print(
        f"Generating static {static_model_name} model once before training "
        f"(sparsity={args.max_sparsity})"
    )
    from ours.pretrain_fbs_model.main import generate_small_cnn_with_verify

    from ours.libs.train_with_fbs.lib import set_sparsity
    set_sparsity(large_agent, args.max_sparsity)

    sample_for_gen_small_model = base.collect_sample_for_small_model_generation(
        args=args,
        large_agent=large_agent,
        small_agent=None,
        eval_envs=eval_envs,
        env_kwargs=env_kwargs,
        device=device,
    )
    agent, _ = generate_small_cnn_with_verify(
        large_agent,
        args.max_sparsity,
        sample_for_gen_small_model,
        lambda model, sample: model(sample),
        return_pruning_info=True,
    )

    for module in agent.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    for parameter in agent.parameters():
        parameter.requires_grad = True

    from ours.utils.dl.common.model import get_model_size
    print(f"Initialized agent with size {get_model_size(agent, True):.2f}MB")

    return agent


def run_training(mode: Literal["original", "small"], script_path: str) -> None:
    args, device, run_name = prepare_args_and_context(mode, script_path)
    patch_observation_wrapper_for_multi_task_state()

    import json as json_lib

    with open(args.env_config_path, "r", encoding="utf-8") as handle:
        demo_info = json_lib.load(handle)
    env_kwargs = demo_info["env_info"]["env_kwargs"]
    env_kwargs["sim_backend"] = "physx_cuda"
    del env_kwargs["num_envs"]
    del env_kwargs["reward_mode"]

    continual_env_schedule = base.build_continual_env_schedule(args)
    current_env_index = 0
    current_env_id = args.env_id if continual_env_schedule is None else continual_env_schedule.env_ids[0]
    envs, eval_envs = base.make_envs_for_env_id(
        args,
        current_env_id,
        env_kwargs,
        run_name,
        current_env_index,
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_episode_steps = base.gym_utils.find_max_episode_steps_value(envs._env)
    logger = build_logger(args, current_env_id, env_kwargs, max_episode_steps, run_name)

    obs = base.DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    training_start_time = time.monotonic()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    print("####")
    print(
        f"args.num_iterations={args.num_iterations} "
        f"args.num_envs={args.num_envs} "
        f"args.num_eval_envs={args.num_eval_envs}"
    )
    print(
        f"args.minibatch_size={args.minibatch_size} "
        f"args.batch_size={args.batch_size} "
        f"args.update_epochs={args.update_epochs}"
    )
    print("####")
    if continual_env_schedule is not None:
        print(
            f"Continual env schedule enabled: envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )

    def current_elapsed_minutes() -> float:
        return (time.monotonic() - training_start_time) / 60.0

    def maybe_switch_envs() -> Tuple[bool, bool, Optional[float]]:
        nonlocal envs, eval_envs, next_obs, eval_obs, next_done, current_env_id, current_env_index
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = current_elapsed_minutes()
        if current_env_index >= len(continual_env_schedule.env_ids) - 1:
            should_stop = (
                elapsed_minutes > continual_env_schedule.change_time_points[-1]
            )
            return False, should_stop, elapsed_minutes

        next_switch_time = continual_env_schedule.change_time_points[current_env_index]
        if elapsed_minutes <= next_switch_time:
            return False, False, elapsed_minutes

        previous_env_id = current_env_id
        current_env_index += 1
        current_env_id = continual_env_schedule.env_ids[current_env_index]
        print(
            f"Switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        envs, eval_envs = base.make_envs_for_env_id(
            args,
            current_env_id,
            env_kwargs,
            run_name,
            current_env_index,
        )
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        eval_obs, _ = eval_envs.reset(seed=args.seed + current_env_index)
        next_done = torch.zeros(args.num_envs, device=device)
        return True, False, elapsed_minutes

    agent = initialize_agent(mode, args, eval_envs, env_kwargs, device)
    initial_importance = build_zero_importance_snapshot(agent)
    save_importance_snapshot(
        run_name=run_name,
        args=args,
        iteration=0,
        global_step=global_step,
        elapsed_minutes=current_elapsed_minutes(),
        snapshot=initial_importance,
    )

    trainable_parameters = []
    for _, parameter in agent.named_parameters():
        parameter.requires_grad = True
        trainable_parameters.append(parameter)
    optimizer = optim.Adam([{"params": trainable_parameters, "lr": args.learning_rate}], eps=1e-5)

    start_iter_idx = 1
    cumulative_times = defaultdict(float)
    best_success_once = 0.0
    best_success_end = 0.0
    iteration = 0

    from ours.libs.train_with_fbs.lib import set_sparsity

    try:
        for iteration in range(start_iter_idx, args.num_iterations + 1):
            switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
            if elapsed_minutes is not None:
                logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
                logger.add_scalar("continual/current_env_index", current_env_index, global_step)
            if should_stop_for_schedule:
                print(
                    f"Reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes, "
                    "stopping training."
                )
                break

            print(f"Iteration: {iteration}, global_step={global_step}")
            final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

            agent.eval()
            if iteration % args.eval_freq == 1 or iteration == start_iter_idx or args.eval_freq == 1:
                print("Evaluating")
                avg_success_once = 0.0
                avg_success_end = 0.0

                sparsity_list = [args.max_sparsity]
                for test_sparsity in sparsity_list:
                    test_sparsity_str = f"{test_sparsity:.4f}"
                    set_sparsity(agent, test_sparsity)

                    stime = time.perf_counter()
                    eval_obs, _ = eval_envs.reset()
                    eval_metrics = defaultdict(list)
                    for _ in range(args.num_eval_steps):
                        with torch.no_grad():
                            eval_obs, _, _, _, eval_infos = eval_envs.step(
                                agent.get_action(eval_obs, deterministic=True)
                            )
                            if "final_info" in eval_infos:
                                for key, value in eval_infos["final_info"]["episode"].items():
                                    eval_metrics[key].append(value)

                    for key, value in eval_metrics.items():
                        mean_value = torch.stack(value).float().mean()
                        logger.add_scalar(f"eval/{key}_{test_sparsity_str}", mean_value, global_step)
                        if key == "success_once":
                            avg_success_once += mean_value
                        if key == "success_at_end":
                            avg_success_end += mean_value

                    if test_sparsity < 0.001:
                        eval_time = time.perf_counter() - stime
                        cumulative_times["eval_time"] += eval_time
                        logger.add_scalar("time/eval_time", eval_time, global_step)

                avg_success_once = float(avg_success_once / len(sparsity_list))
                avg_success_end = float(avg_success_end / len(sparsity_list))
                logger.add_scalar("eval/success_once", avg_success_once, global_step)
                logger.add_scalar("eval/success_end", avg_success_end, global_step)
                elapsed_for_metric = current_elapsed_minutes()
                append_eval_metric(
                    run_name=run_name,
                    args=args,
                    iteration=iteration,
                    global_step=global_step,
                    elapsed_minutes=elapsed_for_metric,
                    success_once=avg_success_once,
                    success_at_end=avg_success_end,
                )
                print(
                    f"eval success_once={avg_success_once:.4f}, "
                    f"success_at_end={avg_success_end:.4f}"
                )

                if avg_success_once >= best_success_once:
                    best_success_once = avg_success_once
                    os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "success_once": best_success_once,
                        },
                        f"ckpt/{run_name}/checkpoints/best_success_once.pt",
                    )
                if avg_success_end >= best_success_end:
                    best_success_end = avg_success_end
                    os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iteration": iteration,
                            "success_at_end": best_success_end,
                        },
                        f"ckpt/{run_name}/checkpoints/best_success_end.pt",
                    )

                if args.evaluate:
                    break

            if args.save_model and (iteration % args.eval_freq == 1 or args.eval_freq == 1):
                os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    f"ckpt/{run_name}/checkpoints/last.pt",
                )

            switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
            if elapsed_minutes is not None:
                logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
                logger.add_scalar("continual/current_env_index", current_env_index, global_step)
            if should_stop_for_schedule:
                print(
                    f"Reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes "
                    "after evaluation, stopping training."
                )
                break

            agent.train()
            if args.max_time is not None:
                elapsed_minutes = current_elapsed_minutes()
                if elapsed_minutes >= args.max_time:
                    print(
                        f"Reached max_time={args.max_time} minutes "
                        f"(elapsed={elapsed_minutes:.2f} minutes) before rollout, stopping training."
                    )
                    break

            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            rollout_time = time.perf_counter()
            print("rollout...")
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value(
                        next_obs,
                        return_action_mean=True,
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
                rewards[step] = reward.view(-1) * args.reward_scale

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for key, value in final_info["episode"].items():
                        logger.add_scalar(f"train/{key}", value[done_mask].float().mean(), global_step)

                    for key in infos["final_observation"]:
                        infos["final_observation"][key] = infos["final_observation"][key][done_mask]
                    with torch.no_grad():
                        final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = (
                            agent.get_value(infos["final_observation"]).view(-1)
                        )
            rollout_time = time.perf_counter() - rollout_time
            cumulative_times["rollout_time"] += rollout_time

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t]
                    if args.finite_horizon_gae:
                        if t == args.num_steps - 1:
                            lam_coef_sum = 0.0
                            reward_term_sum = 0.0
                            value_term_sum = 0.0
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done
                        lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                        reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                        value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                    else:
                        delta = rewards[t] + args.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = (
                            delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                        )
                returns = advantages + values

            b_obs = obs.reshape((-1,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            agent.train()
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            update_time = time.perf_counter()
            update_times = 0
            importance_accumulator = init_importance_accumulator(agent)

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    update_times += 1
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        print(
                            f"early stop (kl={approx_kl:.6f}) after {update_times} updates"
                        )
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()
                        ) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    update_importance_accumulator(importance_accumulator, agent)
                    nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            print(
                f"updated: {update_times} steps, v_loss={v_loss.item():.4f}, "
                f"kl={approx_kl.item():.4f}"
            )

            snapshot = finalize_importance_accumulator(importance_accumulator)
            save_importance_snapshot(
                run_name=run_name,
                args=args,
                iteration=iteration,
                global_step=global_step,
                elapsed_minutes=current_elapsed_minutes(),
                snapshot=snapshot,
            )

            update_time = time.perf_counter() - update_time
            cumulative_times["update_time"] += update_time
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
            for key, value in cumulative_times.items():
                logger.add_scalar(f"time/total_{key}", value, global_step)
            logger.add_scalar(
                "time/total_rollout+update_time",
                cumulative_times["rollout_time"] + cumulative_times["update_time"],
                global_step,
            )

    finally:
        if args.save_model and not args.evaluate:
            os.makedirs(f"ckpt/{run_name}/checkpoints", exist_ok=True)
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                },
                f"ckpt/{run_name}/checkpoints/last.pt",
            )

        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        logger.close()
