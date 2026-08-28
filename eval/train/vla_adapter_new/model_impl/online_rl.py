import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import gymnasium as gym
import h5py
import mani_skill.envs
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from PIL import Image
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from train.common.random_init_vla import maybe_build_random_init_vla_bundle
from train.common.time_breakdown import write_time_breakdown_from_metrics_history



matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_PROMPT = "Grasp a red cube and move it to a target goal position."
DEFAULT_MODEL_DIR = "ckpt/vla_adapter_new/LIBERO-Object"
DEFAULT_DEMO_H5 = "datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5"
DEFAULT_WORKDIR = "train/vla_adapter_new/model_impl/outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).copy()
    quat_xyzw[3] = np.clip(quat_xyzw[3], -1.0, 1.0)
    den = np.sqrt(max(1e-12, 1.0 - quat_xyzw[3] * quat_xyzw[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat_xyzw[:3] * 2.0 * math.acos(quat_xyzw[3])) / den


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            serializable[key] = np.asarray(value).tolist()
        else:
            serializable[key] = value
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))


def iter_slices(total: int, chunk_size: int):
    chunk_size = max(1, min(chunk_size, total))
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def distributed_barrier() -> None:
    if is_distributed():
        dist.barrier()


def distributed_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return float(tensor.item())


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    pred = predictions.detach().reshape(-1).to(device=device, dtype=torch.float64)
    target = targets.detach().reshape(-1).to(device=device, dtype=torch.float64)
    error = target - pred

    target_stats = torch.stack(
        (
            torch.tensor(float(target.numel()), device=device, dtype=torch.float64),
            target.sum(),
            (target * target).sum(),
        )
    )
    error_stats = torch.stack(
        (
            torch.tensor(float(error.numel()), device=device, dtype=torch.float64),
            error.sum(),
            (error * error).sum(),
        )
    )
    if is_distributed():
        dist.all_reduce(target_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(error_stats, op=dist.ReduceOp.SUM)

    total_count = float(target_stats[0].item())
    if total_count <= 0:
        return float("nan")

    target_mean = target_stats[1] / total_count
    target_var = target_stats[2] / total_count - target_mean * target_mean
    if float(target_var.item()) <= 1e-12:
        return float("nan")

    error_mean = error_stats[1] / total_count
    error_var = error_stats[2] / total_count - error_mean * error_mean
    return float((1.0 - error_var / target_var).item())


def gather_metric_summary(local_summary: Dict[str, Tuple[float, int] | float]) -> Dict[str, float]:
    if not local_summary:
        return {}

    first_value = next(iter(local_summary.values()))
    values_are_pairs = isinstance(first_value, (tuple, list))

    if not is_distributed():
        if values_are_pairs:
            return {
                key: (value_sum / count)
                for key, (value_sum, count) in local_summary.items()
                if count > 0
            }
        return {key: float(value) for key, value in local_summary.items()}

    gathered: List[Optional[Dict[str, Tuple[float, int] | float]]] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_summary)
    merged: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for summary in gathered:
        if summary is None:
            continue
        for key, value in summary.items():
            if isinstance(value, (tuple, list)):
                value_sum, count = value
                merged[key][0] += float(value_sum)
                merged[key][1] += float(count)
            else:
                merged[key][0] += float(value)
                merged[key][1] += 1.0
    return {
        key: merged_value[0] / merged_value[1]
        for key, merged_value in merged.items()
        if merged_value[1] > 0
    }


def broadcast_object(value: Any) -> Any:
    if not is_distributed():
        return value
    payload = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def init_runtime(args: "Args") -> Tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        visible_devices = [device_id.strip() for device_id in args.cuda_device.split(",") if device_id.strip()]
        if len(visible_devices) < world_size:
            raise ValueError(
                f"cuda_device={args.cuda_device!r} provides {len(visible_devices)} devices, "
                f"but torchrun world_size={world_size}"
            )
        if not dist.is_initialized():
            timeout_hours = float(os.environ.get("TORCH_DIST_TIMEOUT_HOURS", "6"))
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=timeout_hours))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return device, get_rank(), get_world_size()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return device, 0, 1


def cleanup_runtime() -> None:
    if is_distributed():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def save_metrics_history(output_dir: Path, metrics_history: List[Dict[str, Any]]) -> None:
    save_json(output_dir / "metrics_history.json", {"history": metrics_history})
    write_time_breakdown_from_metrics_history(output_dir, metrics_history)


def _metric_series(history: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for metric in history:
        value = metric.get(key)
        if value is None:
            continue
        xs.append(int(metric["update"]))
        ys.append(float(value))
    return xs, ys


def _plot_single_metric(history: List[Dict[str, Any]], plot_path: Path, title: str, ylabel: str, keys: List[str]) -> bool:
    series = []
    for key in keys:
        xs, ys = _metric_series(history, key)
        if xs:
            series.append((key, xs, ys))
    if not series:
        return False

    plt.figure(figsize=(9, 6))
    for key, xs, ys in series:
        plt.plot(xs, ys, marker="o", linewidth=2, label=key)
    plt.xlabel("PPO Update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return True


def plot_metrics_history(output_dir: Path, metrics_history: List[Dict[str, Any]]) -> None:
    plots_dir = mkdir(output_dir / "plots")
    _plot_single_metric(
        metrics_history,
        plots_dir / "reward_curve.png",
        title="Reward Curve",
        ylabel="Reward",
        keys=["reward_mean"],
    )
    _plot_single_metric(
        metrics_history,
        plots_dir / "return_curve.png",
        title="Return Curve",
        ylabel="Return",
        keys=["train_return", "eval_return"],
    )
    _plot_single_metric(
        metrics_history,
        plots_dir / "success_curve.png",
        title="Success Curve",
        ylabel="Success Rate",
        keys=["train_success_once", "train_success", "eval_success_once", "eval_success", "eval_success_at_end"],
    )
    _plot_single_metric(
        metrics_history,
        plots_dir / "optimization_curve.png",
        title="Optimization Diagnostics",
        ylabel="Value",
        keys=["approx_kl", "clipfrac", "explained_variance"],
    )

    _plot_single_metric(
        metrics_history,
        plots_dir / "entropy.png",
        title="Entropy",
        ylabel="Entropy",
        keys=["entropy"],
    )

    plt.figure(figsize=(12, 10))
    subplot_specs = [
        ("Reward", "Reward", ["reward_mean"]),
        ("Return", "Return", ["train_return", "eval_return"]),
        ("Success", "Success Rate", ["train_success_once", "eval_success_once", "eval_success"]),
        ("Optimization", "Value", ["approx_kl", "clipfrac", "explained_variance"]),
    ]
    plotted_any = False
    for subplot_idx, (title, ylabel, keys) in enumerate(subplot_specs, start=1):
        ax = plt.subplot(2, 2, subplot_idx)
        subplot_has_series = False
        for key in keys:
            xs, ys = _metric_series(metrics_history, key)
            if not xs:
                continue
            plotted_any = True
            subplot_has_series = True
            ax.plot(xs, ys, marker="o", linewidth=2, label=key)
        ax.set_title(title)
        ax.set_xlabel("PPO Update")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if subplot_has_series:
            ax.legend()
    plt.tight_layout()
    if plotted_any:
        plt.savefig(plots_dir / "overview.png", dpi=200)
    plt.close()


def save_rollout_progress(
    output_dir: Path,
    update: int,
    num_updates: int,
    rollout_step: int,
    num_steps: int,
    elapsed_hours: float,
    partial_reward_means: List[float],
) -> None:
    plots_dir = mkdir(output_dir / "plots")
    save_json(
        output_dir / "rollout_progress.json",
        {
            "update": update,
            "num_updates": num_updates,
            "rollout_step": rollout_step,
            "num_steps": num_steps,
            "elapsed_hours": elapsed_hours,
            "partial_reward_means": partial_reward_means,
        },
    )

    if partial_reward_means:
        xs = list(range(1, len(partial_reward_means) + 1))
        plt.figure(figsize=(9, 6))
        plt.plot(xs, partial_reward_means, marker="o", linewidth=2)
        plt.xlabel("Rollout Step In Current Update")
        plt.ylabel("Mean Reward")
        plt.title(f"Partial Reward Curve (update {update}/{num_updates}, step {rollout_step}/{num_steps})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "partial_rollout_reward_curve.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.text(
        0.02,
        0.90,
        (
            f"Update: {update}/{num_updates}\n"
            f"Rollout step: {rollout_step}/{num_steps}\n"
            f"Elapsed hours: {elapsed_hours:.2f}\n"
            f"Partial reward mean: {partial_reward_means[-1]:.6f}" if partial_reward_means else
            f"Update: {update}/{num_updates}\nRollout step: {rollout_step}/{num_steps}\nElapsed hours: {elapsed_hours:.2f}"
        ),
        va="top",
        ha="left",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_status.png", dpi=200)
    plt.close()


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def ensure_package(package_name: str, package_dir: Path) -> None:
    if package_name in sys.modules:
        return
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def build_state_feature_from_parts(
    qpos: np.ndarray,
    qvel: np.ndarray,
    tcp_pose: np.ndarray,
    goal_pos: np.ndarray,
    obj_pose: np.ndarray,
    tcp_to_obj_pos: np.ndarray,
    obj_to_goal_pos: np.ndarray,
    is_grasped: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            qpos.astype(np.float32),
            qvel.astype(np.float32),
            tcp_pose.astype(np.float32),
            goal_pos.astype(np.float32),
            obj_pose.astype(np.float32),
            tcp_to_obj_pos.astype(np.float32),
            obj_to_goal_pos.astype(np.float32),
            is_grasped.astype(np.float32).reshape(1),
        ],
        axis=0,
    )


def extract_state_feature_from_obs(obs: Dict[str, Any]) -> torch.Tensor:
    qpos = obs["agent"]["qpos"]
    qvel = obs["agent"]["qvel"]
    tcp_pose = obs["extra"]["tcp_pose"]
    goal_pos = obs["extra"]["goal_pos"]
    obj_pose = obs["extra"]["obj_pose"]
    tcp_to_obj_pos = obs["extra"]["tcp_to_obj_pos"]
    obj_to_goal_pos = obs["extra"]["obj_to_goal_pos"]
    is_grasped = obs["extra"]["is_grasped"].to(torch.float32).unsqueeze(-1)
    return torch.cat(
        [qpos, qvel, tcp_pose, goal_pos, obj_pose, tcp_to_obj_pos, obj_to_goal_pos, is_grasped],
        dim=-1,
    )


def extract_vla_proprio_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float32)
    tcp_pose = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32)
    if qpos.ndim == 2:
        qpos = qpos[0]
    if tcp_pose.ndim == 2:
        tcp_pose = tcp_pose[0]
    return np.concatenate(
        [tcp_pose[:3], quat2axisangle(tcp_pose[3:]), qpos[7:9]],
        axis=0,
    ).astype(np.float32)


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb[..., :3].detach().cpu().numpy()
    else:
        rgb = np.asarray(rgb)[..., :3]
    return rgb.astype(np.uint8)


def extract_vla_proprio_batch_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    qpos = obs["agent"]["qpos"]
    tcp_pose = obs["extra"]["tcp_pose"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    else:
        qpos = np.asarray(qpos)
    if isinstance(tcp_pose, torch.Tensor):
        tcp_pose = tcp_pose.detach().cpu().numpy()
    else:
        tcp_pose = np.asarray(tcp_pose)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if tcp_pose.ndim == 1:
        tcp_pose = tcp_pose[None, :]
    proprios = [
        np.concatenate([tcp_pose[i, :3], quat2axisangle(tcp_pose[i, 3:]), qpos[i, 7:9]], axis=0).astype(np.float32)
        for i in range(qpos.shape[0])
    ]
    return np.stack(proprios, axis=0)


def project_vla_pose_action_to_delta_pos(action_chunk_7d: np.ndarray, xyz_scale: float) -> np.ndarray:
    action_chunk_7d = np.asarray(action_chunk_7d, dtype=np.float32)
    projected = np.zeros((action_chunk_7d.shape[0], 4), dtype=np.float32)
    projected[:, :3] = np.clip(action_chunk_7d[:, :3] * xyz_scale, -1.0, 1.0)
    projected[:, 3] = np.clip(1.0 - 2.0 * action_chunk_7d[:, 6], -1.0, 1.0)
    return projected


def project_vla_pose_action_to_delta_pos_torch(action_chunk_7d: torch.Tensor, xyz_scale: float) -> torch.Tensor:
    projected = torch.zeros((*action_chunk_7d.shape[:-1], 4), device=action_chunk_7d.device, dtype=action_chunk_7d.dtype)
    projected[..., :3] = torch.clamp(action_chunk_7d[..., :3] * xyz_scale, -1.0, 1.0)
    projected[..., 3] = torch.clamp(1.0 - 2.0 * action_chunk_7d[..., 6], -1.0, 1.0)
    return projected


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(proprio)
        projected = self.act_fn1(projected)
        return self.fc2(projected)


def learnable_random_perturbations(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, h_t: torch.Tensor, h_a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)
        h = torch.cat([h_a, p], dim=1)
        batch, seq_len, dim = x.shape
        cond_len = h.shape[1]
        task_len = h_t.shape[1]
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = self.k_proj(h).view(batch, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = self.v_proj(h).view(batch, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_adapter = self.k_proj(h_t).view(batch, task_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = self.v_proj(h_t).view(batch, task_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.cat(
            [
                torch.matmul(q, k_tokens.transpose(-2, -1)),
                torch.matmul(q, k_task.transpose(-2, -1)),
                torch.matmul(q, k_adapter.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        ) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        values = torch.cat([v_tokens, v_task, v_adapter], dim=2)
        attn_out = torch.matmul(attn_weights, values).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        attn_out = self.o_proj(attn_out)
        return x + attn_out + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h_a: torch.Tensor, h_t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim: int = 896, hidden_dim: int = 896, action_dim: int = 7, num_task_tokens: int = 512) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_task_tokens = num_task_tokens
        self.num_actions_chunk = 8
        self.model = MLPResNet(
            num_blocks=24,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor, proprio: torch.Tensor, proprio_projector: nn.Module, phase: str = "Inference") -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
        proprio_features = proprio_projector(proprio).unsqueeze(1)
        task_hidden_states = actions_hidden_states[:, :, : self.num_task_tokens, :]
        action_hidden_states = actions_hidden_states[:, :, self.num_task_tokens :, :]
        cond_action_states = torch.zeros(
            (batch_size, self.action_dim * self.num_actions_chunk, self.hidden_dim),
            device=device,
            dtype=action_hidden_states.dtype,
        ).detach()
        rearranged = cond_action_states.reshape(batch_size, self.num_actions_chunk, -1)
        if phase == "Training":
            _, seq_len, dim = rearranged.shape
            rearranged = rearranged + learnable_random_perturbations(seq_len, dim, rearranged.device, rearranged.dtype)
        return self.model(rearranged, h_a=action_hidden_states, p=proprio_features, h_t=task_hidden_states)


class VLAAdapterActorCritic(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device, xyz_scale: float = 0.25):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        self.xyz_scale = xyz_scale
        self.prompt = f"In: What action should the robot take to {TASK_PROMPT.lower()}?\nOut:"
        fallback_bundle = maybe_build_random_init_vla_bundle(
            model_dir=model_dir,
            prompt=self.prompt,
            device=device,
            num_action_tokens=8,
            action_stats_dim=7,
        )
        if fallback_bundle is None:
            self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            self.tokenizer_vocab_size = int(self.processor.tokenizer.vocab_size)
            self.action_vocab_size = int(self.action_tokenizer.vocab_size)
            ensure_package("local_vla_pkg", model_dir)
            config_mod = load_module_from_path("local_vla_pkg.configuration_prismatic", model_dir / "configuration_prismatic.py")
            model_mod = load_module_from_path("local_vla_pkg.modeling_prismatic", model_dir / "modeling_prismatic.py")
            self.ignore_index = int(getattr(model_mod, "IGNORE_INDEX", -100))
            self.num_tokens = int(getattr(model_mod, "NUM_TOKENS", 64))
            self.num_actions_chunk = int(getattr(model_mod, "NUM_ACTIONS_CHUNK", 8))
            self.action_dim = int(getattr(model_mod, "ACTION_DIM", 7))
            self.action_norm_type = str(getattr(model_mod, "ACTION_PROPRIO_NORMALIZATION_TYPE", "bounds_q99")).lower()

            self.vla = model_mod.OpenVLAForActionPrediction.from_pretrained(
                str(model_dir),
                config=config_mod.OpenVLAConfig.from_pretrained(str(model_dir)),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).to(device)
            self.vla.set_version("v1")
            self.full_vocab_size = int(self.vla.vocab_size)

            with open(model_dir / "dataset_statistics.json", "r") as f:
                dataset_statistics = json.load(f)
            self.vla.norm_stats = dataset_statistics
            self.unnorm_key = "libero_object_no_noops"
        else:
            self.processor = fallback_bundle["processor"]
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            self.tokenizer_vocab_size = int(self.processor.tokenizer.vocab_size)
            self.action_vocab_size = int(self.action_tokenizer.vocab_size)
            self.ignore_index = int(fallback_bundle["ignore_index"])
            self.num_tokens = int(fallback_bundle["num_tokens"])
            self.num_actions_chunk = int(fallback_bundle["num_actions_chunk"])
            self.action_dim = int(fallback_bundle["action_dim"])
            self.action_norm_type = str(fallback_bundle["action_norm_type"]).lower()
            self.vla = fallback_bundle["vla"]
            self.full_vocab_size = int(self.vla.vocab_size)
            self.unnorm_key = "fallback_random_init"

        stats = self.vla.get_action_stats(self.unnorm_key)
        if "q99" in self.action_norm_type:
            action_high = np.asarray(stats["q99"], dtype=np.float32)
            action_low = np.asarray(stats["q01"], dtype=np.float32)
        else:
            action_high = np.asarray(stats["max"], dtype=np.float32)
            action_low = np.asarray(stats["min"], dtype=np.float32)
        action_mask = np.asarray(stats.get("mask", np.ones_like(action_high, dtype=bool)), dtype=bool)
        self.register_buffer("action_high", torch.from_numpy(action_high), persistent=False)
        self.register_buffer("action_low", torch.from_numpy(action_low), persistent=False)
        self.register_buffer("action_mask", torch.from_numpy(action_mask), persistent=False)
        selected_action_dims = torch.tensor([0, 1, 2, 6], dtype=torch.long)
        self.register_buffer("selected_action_dims", selected_action_dims, persistent=False)
        self.register_buffer("selected_action_high", self.action_high.index_select(0, selected_action_dims), persistent=False)
        self.register_buffer("selected_action_low", self.action_low.index_select(0, selected_action_dims), persistent=False)
        self.register_buffer("selected_action_mask", self.action_mask.index_select(0, selected_action_dims), persistent=False)
        self.register_buffer(
            "action_bin_centers",
            torch.from_numpy(self.action_tokenizer.bin_centers.astype(np.float32)),
            persistent=False,
        )

        critic_hidden_dim = max(256, self.vla.llm_dim // 4)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        ).to(device=device, dtype=torch.float32)
        self.eval_micro_batch_size = 64

    def configure_trainable_modules(
        self,
        freeze_vla_backbone: bool,
        freeze_proprio_projector: bool,
        train_action_head: bool = True,
    ) -> None:
        for parameter in self.vla.parameters():
            parameter.requires_grad = not freeze_vla_backbone
        for parameter in self.value_head.parameters():
            parameter.requires_grad = True

    def trainable_parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        modules = {
            "vla": self.vla,
            "value_head": self.value_head,
        }
        summary: Dict[str, Tuple[int, int]] = {}
        for name, module in modules.items():
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            summary[name] = (total, trainable)
        return summary

    @staticmethod
    def _prepare_image(rgb: np.ndarray) -> Image.Image:
        image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        original_w, original_h = image.size
        crop_scale = math.sqrt(0.9)
        crop_w = max(1, int(original_w * crop_scale))
        crop_h = max(1, int(original_h * crop_scale))
        left = max(0, (original_w - crop_w) // 2)
        top = max(0, (original_h - crop_h) // 2)
        image = image.crop((left, top, left + crop_w, top + crop_h))
        return image.resize((original_w, original_h), Image.Resampling.BILINEAR)

    def _prepare_policy_inputs(self, rgbs: np.ndarray) -> Dict[str, torch.Tensor]:
        images = [self._prepare_image(rgb[..., :3]) for rgb in np.asarray(rgbs)]
        prompts = [self.prompt] * len(images)
        processor_outputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        return {
            "input_ids": processor_outputs["input_ids"].to(self.device),
            "attention_mask": processor_outputs["attention_mask"].to(self.device),
            "pixel_values": processor_outputs["pixel_values"].to(self.device, dtype=torch.bfloat16),
        }

    def _unnormalize_selected_actions_torch(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        action_low = self.selected_action_low.to(device=normalized_actions.device, dtype=normalized_actions.dtype)
        action_high = self.selected_action_high.to(device=normalized_actions.device, dtype=normalized_actions.dtype)
        action_mask = self.selected_action_mask.to(device=normalized_actions.device)
        actions = torch.where(
            action_mask.view(1, -1),
            0.5 * (normalized_actions + 1.0) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions

    def _forward_token_policy(self, rgbs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        labels = input_ids.clone()
        labels[:] = self.ignore_index
        num_prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = self.vla._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self.vla._prepare_labels_for_action_prediction(labels, input_ids)

        input_embeddings = self.vla.get_input_embeddings()(input_ids)
        all_actions_mask = self.vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = self.vla._process_vision_features(pixel_values, language_embeddings, use_film=False)
        action_queries = self.vla.action_queries.weight.view(1, self.num_tokens, self.vla.llm_dim).expand(
            input_embeddings.shape[0], -1, -1
        )
        input_embeddings = self.vla._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)
        multimodal_embeddings, multimodal_attention_mask = self.vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = self.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        num_patches = self.vla.vision_backbone.get_num_patches() * self.vla.vision_backbone.get_num_images_in_input()
        token_start = num_patches + num_prompt_tokens
        action_logits = language_model_output.logits[
            :,
            token_start : token_start + self.action_dim,
            : self.full_vocab_size,
        ].to(torch.float32)
        action_logits = action_logits.index_select(1, self.selected_action_dims)
        critic_features = language_model_output.hidden_states[-1][:, token_start, :].to(torch.float32)
        value = self.value_head(critic_features).squeeze(-1)
        return action_logits, value

    def tokenize_env_actions(self, env_actions: np.ndarray) -> torch.Tensor:
        env_actions = np.asarray(env_actions, dtype=np.float32)
        if env_actions.ndim == 1:
            env_actions = env_actions[None, :]
        selected_raw_actions = np.zeros((env_actions.shape[0], 4), dtype=np.float32)
        selected_raw_actions[:, :3] = env_actions[:, :3] / self.xyz_scale
        selected_raw_actions[:, 3] = 0.5 * (1.0 - env_actions[:, 3])
        selected_raw_actions = np.where(
            self.selected_action_mask.detach().cpu().numpy()[None, :],
            np.clip(
                selected_raw_actions,
                self.selected_action_low.detach().cpu().numpy()[None, :],
                self.selected_action_high.detach().cpu().numpy()[None, :],
            ),
            selected_raw_actions,
        )
        normalized_actions = np.where(
            self.selected_action_mask.detach().cpu().numpy()[None, :],
            2.0
            * (selected_raw_actions - self.selected_action_low.detach().cpu().numpy()[None, :])
            / (
                self.selected_action_high.detach().cpu().numpy()[None, :]
                - self.selected_action_low.detach().cpu().numpy()[None, :]
                + 1e-8
            )
            - 1.0,
            selected_raw_actions,
        )
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        token_ids = np.asarray(self.action_tokenizer(normalized_actions, use_minivlm=True), dtype=np.int64)
        return torch.from_numpy(token_ids.astype(np.int64))

    def token_ids_to_env_actions(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        if action_token_ids.ndim == 1:
            action_token_ids = action_token_ids.unsqueeze(0)
        action_token_ids = action_token_ids.to(self.device, dtype=torch.long)
        discretized_actions = self.full_vocab_size - action_token_ids - 1
        discretized_actions = torch.clamp(discretized_actions, min=0, max=self.action_bin_centers.numel() - 1)
        normalized_actions = self.action_bin_centers.to(action_token_ids.device)[discretized_actions]
        raw_actions = self._unnormalize_selected_actions_torch(normalized_actions)
        env_actions = torch.zeros((raw_actions.shape[0], 4), device=self.device, dtype=torch.float32)
        env_actions[:, :3] = torch.clamp(raw_actions[:, :3] * self.xyz_scale, -1.0, 1.0)
        env_actions[:, 3] = torch.clamp(1.0 - 2.0 * raw_actions[:, 3], -1.0, 1.0)
        return env_actions

    def forward_policy(self, rgbs: np.ndarray, proprio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        del proprio
        return self._forward_token_policy(rgbs)

    def get_value(self, rgbs: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
        _, value = self.forward_policy(rgbs, proprio)
        return value

    def get_action_and_value(
        self,
        rgbs: np.ndarray,
        proprio: np.ndarray,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, value = self.forward_policy(rgbs, proprio)
        categorical = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action_tokens = action_logits.argmax(dim=-1) if deterministic else categorical.sample()
        else:
            action_tokens = action.to(self.device, dtype=torch.long)
        log_prob = categorical.log_prob(action_tokens).sum(dim=-1)
        entropy = categorical.entropy().mean(dim=-1)
        env_actions = self.token_ids_to_env_actions(action_tokens)
        return env_actions, log_prob, entropy, value, action_tokens

    def forward(
        self,
        rgbs: np.ndarray,
        proprio: np.ndarray,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        mode: str = "action_and_value",
    ):
        if mode == "action_and_value":
            return self.get_action_and_value(rgbs=rgbs, proprio=proprio, action=action, deterministic=deterministic)
        if mode == "value":
            return self.get_value(rgbs=rgbs, proprio=proprio)
        if mode == "policy":
            return self.forward_policy(rgbs=rgbs, proprio=proprio)
        raise ValueError(f"Unsupported forward mode: {mode}")

    def predict_delta_pos_action(self, rgb: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            action, _, _, _, _ = self.get_action_and_value(
                rgbs=np.expand_dims(rgb, axis=0),
                proprio=np.expand_dims(proprio, axis=0),
                deterministic=True,
            )
        return action[0].detach().cpu().numpy().astype(np.float32)


class DemoDataset(Dataset):
    def __init__(self, h5_path: Path, max_frames: int, seed: int):
        self.samples: List[Dict[str, Any]] = []
        rng = np.random.default_rng(seed)
        with h5py.File(h5_path, "r") as f:
            traj_keys = list(f.keys())
            rng.shuffle(traj_keys)
            for traj_key in traj_keys:
                traj = f[traj_key]
                rgb = traj["obs"]["sensor_data"]["base_camera"]["rgb"][:]
                qpos = traj["obs"]["agent"]["qpos"][:]
                qvel = traj["obs"]["agent"]["qvel"][:]
                tcp_pose = traj["obs"]["extra"]["tcp_pose"][:]
                goal_pos = traj["obs"]["extra"]["goal_pos"][:]
                obj_pose = traj["obs"]["extra"]["obj_pose"][:]
                tcp_to_obj_pos = traj["obs"]["extra"]["tcp_to_obj_pos"][:]
                obj_to_goal_pos = traj["obs"]["extra"]["obj_to_goal_pos"][:]
                is_grasped = traj["obs"]["extra"]["is_grasped"][:]
                actions = traj["actions"][:]
                length = actions.shape[0]
                frame_indices = np.arange(length)
                rng.shuffle(frame_indices)
                for i in frame_indices:
                    state_feature = build_state_feature_from_parts(
                        qpos=qpos[i],
                        qvel=qvel[i],
                        tcp_pose=tcp_pose[i],
                        goal_pos=goal_pos[i],
                        obj_pose=obj_pose[i],
                        tcp_to_obj_pos=tcp_to_obj_pos[i],
                        obj_to_goal_pos=obj_to_goal_pos[i],
                        is_grasped=np.asarray(is_grasped[i], dtype=np.float32),
                    )
                    proprio = np.concatenate([tcp_pose[i][:3], quat2axisangle(tcp_pose[i][3:]), qpos[i][7:9]], axis=0).astype(np.float32)
                    self.samples.append(
                        {
                            "rgb": rgb[i],
                            "state": state_feature,
                            "proprio": proprio,
                            "action": actions[i].astype(np.float32),
                        }
                    )
                    if max_frames > 0 and len(self.samples) >= max_frames:
                        return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "rgb": sample["rgb"],
            "state": torch.from_numpy(sample["state"]),
            "proprio": torch.from_numpy(sample["proprio"]),
            "action": torch.from_numpy(sample["action"]),
        }


def collate_demo_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rgb": [item["rgb"] for item in batch],
        "state": torch.stack([item["state"] for item in batch], dim=0),
        "proprio": torch.stack([item["proprio"] for item in batch], dim=0),
        "action": torch.stack([item["action"] for item in batch], dim=0),
    }


def pretrain_policy_with_demos(
    args: "Args",
    policy: VLAAdapterActorCritic,
    dataset: DemoDataset,
    optimizer: optim.Optimizer,
    output_dir: Path,
) -> Optional[Path]:
    if args.demo_pretrain_epochs <= 0 or len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=args.demo_pretrain_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_demo_batch,
    )
    history = []
    policy.train()
    for epoch in range(args.demo_pretrain_epochs):
        losses = []
        for batch in loader:
            action_logits, _ = policy.forward_policy(batch["rgb"], batch["proprio"])
            expert_action_tokens = policy.tokenize_env_actions(batch["action"].cpu().numpy()).to(
                policy.device, dtype=torch.long
            )
            loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                expert_action_tokens.reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append(epoch_loss)
        print(f"[demo-pretrain] epoch={epoch + 1}/{args.demo_pretrain_epochs} loss={epoch_loss:.6f}")

    ckpt_path = output_dir / "demo_pretrain_policy.pt"
    torch.save({"policy": policy.state_dict(), "history": history}, ckpt_path)
    return ckpt_path


def broadcast_model_state(module: nn.Module) -> None:
    if not is_distributed():
        return
    with torch.no_grad():
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


def build_optimizer(args: "Args", policy: VLAAdapterActorCritic) -> optim.Optimizer:
    value_head_lr = args.value_head_learning_rate if args.value_head_learning_rate is not None else args.learning_rate
    param_groups = []

    vla_lr = args.backbone_learning_rate if not args.freeze_vla_backbone else args.head_learning_rate
    vla_params = [parameter for parameter in policy.vla.parameters() if parameter.requires_grad]
    if vla_params:
        param_groups.append({"params": vla_params, "lr": vla_lr, "group_name": "vla"})

    value_head_params = [parameter for parameter in policy.value_head.parameters() if parameter.requires_grad]
    if value_head_params:
        param_groups.append({"params": value_head_params, "lr": value_head_lr, "group_name": "value_head"})

    if not param_groups:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    return optim.Adam(param_groups, eps=1e-5)


def set_optimizer_group_lr(optimizer: optim.Optimizer, group_name: str, lr: float) -> None:
    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == group_name:
            param_group["lr"] = lr
            return
    raise KeyError(f"Optimizer param group {group_name!r} not found")


@dataclass
class Args:
    mode: str = "train"
    seed: int = 1
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+depth+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    demo_h5: str = DEFAULT_DEMO_H5
    output_dir: str = DEFAULT_WORKDIR
    num_envs: int = 256
    num_eval_envs: int = 64
    num_steps: int = 50
    total_timesteps: int = 100_000_000
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-6
    head_learning_rate: float = 1e-5
    action_head_learning_rate: Optional[float] = None
    proprio_learning_rate: Optional[float] = None
    value_head_learning_rate: Optional[float] = None
    log_std_learning_rate: Optional[float] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 1
    num_minibatches: int = 16
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    minibatch_target_kl_factor: float = 1.5
    freeze_action_head_updates: int = 10
    demo_pretrain_frames: int = 4096
    demo_pretrain_epochs: int = 20
    demo_pretrain_batch_size: int = 8
    eval_episodes: int = 100
    eval_every_updates: int = 5
    max_episode_steps: Optional[int] = None
    vla_xyz_scale: float = 0.25
    cuda_device: str = "4"
    smoke_steps: int = 32
    save_video: bool = False
    save_train_video_freq: int = 1
    train_video_num_envs: int = 4
    test_video_num_envs: int = 4
    test_video_episodes: int = 4
    max_runtime_hours: float = 5.0
    rollout_micro_batch_size: int = 64
    eval_micro_batch_size: int = 64
    update_micro_batch_size: int = 32
    rollout_progress_log_interval: int = 5
    freeze_vla_backbone: bool = True
    freeze_proprio_projector: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        field_type = field_def.type
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=parse_bool, default=default)
        elif default is None:
            arg_type = int
            origin = get_origin(field_type)
            if origin is not None:
                candidate_types = [candidate for candidate in get_args(field_type) if candidate is not type(None)]
                if len(candidate_types) == 1 and isinstance(candidate_types[0], type):
                    arg_type = candidate_types[0]
            parser.add_argument(arg_name, type=arg_type, default=None)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    namespace = parser.parse_args()
    if namespace.action_head_learning_rate is None:
        namespace.action_head_learning_rate = namespace.head_learning_rate
    if namespace.proprio_learning_rate is None:
        namespace.proprio_learning_rate = namespace.head_learning_rate
    if namespace.value_head_learning_rate is None:
        namespace.value_head_learning_rate = namespace.learning_rate
    if namespace.log_std_learning_rate is None:
        namespace.log_std_learning_rate = namespace.value_head_learning_rate
    return Args(**vars(namespace))


def get_maniskill_backend_kwargs(device: torch.device) -> Dict[str, str]:
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        return {
            "sim_backend": f"physx_cuda:{device_index}",
            "render_backend": f"cuda:{device_index}",
        }
    return {
        "sim_backend": "physx_cpu",
        "render_backend": "sapien_cpu",
    }


def make_vector_env(
    args: Args,
    device: torch.device,
    num_envs: int,
    record_metrics: bool = True,
    video_output_dir: Optional[Path] = None,
    video_max_steps: Optional[int] = None,
) -> ManiSkillVectorEnv:
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        **backend_kwargs,
    )
    if video_output_dir is not None:
        resolved_video_max_steps = video_max_steps
        if resolved_video_max_steps is None:
            resolved_video_max_steps = args.max_episode_steps or gym_utils.find_max_episode_steps_value(env)
        env = RecordEpisode(
            env,
            output_dir=str(video_output_dir),
            save_trajectory=False,
            max_steps_per_video=resolved_video_max_steps,
            video_fps=30,
        )
    return ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False, record_metrics=record_metrics)


def unwrap_policy(policy: nn.Module) -> VLAAdapterActorCritic:
    return policy.module if isinstance(policy, DDP) else policy


def policy_get_action_and_value(
    policy: nn.Module,
    rgbs: np.ndarray,
    proprio: np.ndarray,
    action: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(policy, DDP):
        return policy(rgbs, proprio, action=action, deterministic=deterministic, mode="action_and_value")
    return policy.get_action_and_value(rgbs=rgbs, proprio=proprio, action=action, deterministic=deterministic)


def policy_get_value(policy: nn.Module, rgbs: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
    if isinstance(policy, DDP):
        return policy(rgbs, proprio, mode="value")
    return policy.get_value(rgbs=rgbs, proprio=proprio)


def get_completed_episode_metrics(infos: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    final_info = infos.get("final_info")
    done_mask = infos.get("_final_info")
    if final_info is None or done_mask is None:
        return None, {}
    episode_metrics = final_info.get("episode")
    if not isinstance(episode_metrics, dict):
        return done_mask, {}
    return done_mask, episode_metrics


def evaluate_policy(policy: nn.Module, envs: ManiSkillVectorEnv, target_episodes: int) -> Dict[str, float]:
    metrics = defaultdict(list)
    obs, _ = envs.reset(seed=0)
    episodes = 0
    raw_policy = unwrap_policy(policy)
    raw_policy.eval()
    with torch.no_grad():
        while episodes < target_episodes:
            rgbs = extract_rgb_batch_from_obs(obs)
            proprio = extract_vla_proprio_batch_from_obs(obs)
            action_chunks = []
            for start, end in iter_slices(len(rgbs), raw_policy.eval_micro_batch_size):
                action_chunk, _, _, _, _ = policy_get_action_and_value(
                    raw_policy,
                    rgbs=rgbs[start:end],
                    proprio=proprio[start:end],
                    deterministic=True,
                )
                action_chunks.append(action_chunk)
            action = torch.cat(action_chunks, dim=0)
            obs, _, _, _, infos = envs.step(action)
            done_mask, episode_metrics = get_completed_episode_metrics(infos)
            if done_mask is None:
                continue
            episodes += int(done_mask.sum().item())
            for key, value in episode_metrics.items():
                metrics[key].append(value[done_mask].float().detach().cpu())
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = torch.cat(values).mean().item()
    return result


def record_policy_rollout_video(
    policy: nn.Module,
    envs: ManiSkillVectorEnv,
    num_steps: int,
    seed: int,
) -> None:
    obs, _ = envs.reset(seed=seed)
    raw_policy = unwrap_policy(policy)
    raw_policy.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            rgbs = extract_rgb_batch_from_obs(obs)
            proprio = extract_vla_proprio_batch_from_obs(obs)
            action_chunks = []
            for start, end in iter_slices(len(rgbs), raw_policy.eval_micro_batch_size):
                action_chunk, _, _, _, _ = policy_get_action_and_value(
                    raw_policy,
                    rgbs=rgbs[start:end],
                    proprio=proprio[start:end],
                    deterministic=False,
                )
                action_chunks.append(action_chunk)
            action = torch.cat(action_chunks, dim=0)
            obs, _, _, _, _ = envs.step(action)


def batched_get_action_and_value_no_grad(
    policy: nn.Module,
    rgbs: np.ndarray,
    proprio: np.ndarray,
    micro_batch_size: int,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = []
    log_probs = []
    entropies = []
    values = []
    action_tokens = []
    with torch.no_grad():
        for start, end in iter_slices(len(rgbs), micro_batch_size):
            action, log_prob, entropy, value, tokens = policy_get_action_and_value(
                policy,
                rgbs=rgbs[start:end],
                proprio=proprio[start:end],
                deterministic=deterministic,
            )
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            action_tokens.append(tokens)
    return (
        torch.cat(actions, dim=0),
        torch.cat(log_probs, dim=0),
        torch.cat(entropies, dim=0),
        torch.cat(values, dim=0),
        torch.cat(action_tokens, dim=0),
    )


def batched_get_value_no_grad(
    policy: nn.Module,
    rgbs: np.ndarray,
    proprio: np.ndarray,
    micro_batch_size: int,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start, end in iter_slices(len(rgbs), micro_batch_size):
            values.append(policy_get_value(policy, rgbs[start:end], proprio[start:end]))
    return torch.cat(values, dim=0)


def ppo_update_with_micro_batches(
    args: Args,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    b_rgbs: np.ndarray,
    b_proprio: np.ndarray,
    b_action_tokens: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    minibatch_inds: np.ndarray,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    local_advantages = b_advantages[minibatch_inds]
    local_advantages = (local_advantages - local_advantages.mean()) / (local_advantages.std(unbiased=False) + 1e-8)
    total = len(minibatch_inds)
    stats = defaultdict(float)
    skipped_on_kl = False
    micro_slices = list(iter_slices(total, args.update_micro_batch_size))

    for slice_idx, (local_start, local_end) in enumerate(micro_slices):
        micro_inds = minibatch_inds[local_start:local_end]
        micro_weight = (local_end - local_start) / total
        is_last_micro = slice_idx == len(micro_slices) - 1
        sync_context = nullcontext() if not isinstance(policy, DDP) or is_last_micro else policy.no_sync()
        with sync_context:
            _, newlogprob, entropy, newvalue, _ = policy_get_action_and_value(
                policy,
                b_rgbs[micro_inds],
                b_proprio[micro_inds],
                b_action_tokens[micro_inds],
            )
            logratio = newlogprob - b_logprobs[micro_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                micro_approx_kl = ((ratio - 1) - logratio).mean().item()
                stats["approx_kl"] += micro_approx_kl * micro_weight
                stats["clipfrac"] += (((ratio - 1.0).abs() > args.clip_coef).float().mean().item()) * micro_weight

            micro_kl_limit = None
            if args.target_kl is not None:
                micro_kl_limit = args.target_kl * args.minibatch_target_kl_factor
            global_micro_approx_kl = distributed_max(micro_approx_kl, unwrap_policy(policy).device)
            if micro_kl_limit is not None and global_micro_approx_kl > micro_kl_limit:
                skipped_on_kl = True
                break

            micro_adv = local_advantages[local_start:local_end]
            pg_loss1 = -micro_adv * ratio
            pg_loss2 = -micro_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[micro_inds]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            (loss * micro_weight).backward()

        stats["pg_loss"] += float(pg_loss.detach().item()) * micro_weight
        stats["v_loss"] += float(v_loss.detach().item()) * micro_weight
        stats["entropy"] += float(entropy_loss.detach().item()) * micro_weight

    if skipped_on_kl:
        optimizer.zero_grad(set_to_none=True)
        stats["skipped_on_kl"] = 1.0
        return dict(stats)

    nn.utils.clip_grad_norm_(unwrap_policy(policy).parameters(), args.max_grad_norm)
    optimizer.step()
    stats["skipped_on_kl"] = 0.0
    return dict(stats)


def run_vla_inference_smoke(
    args: Args,
    device: torch.device,
    output_dir: Path,
    policy: Optional[VLAAdapterActorCritic] = None,
) -> None:
    if policy is None:
        policy = VLAAdapterActorCritic(Path(args.model_dir), device=device, xyz_scale=args.vla_xyz_scale)
    backend_kwargs = get_maniskill_backend_kwargs(device)
    env = gym.make(
        args.env_id,
        obs_mode="rgb+depth+state_dict",
        control_mode=args.control_mode,
        reward_mode="dense",
        render_mode="rgb_array",
        **backend_kwargs,
    )
    obs, _ = env.reset(seed=args.seed)
    returns = 0.0
    success = False
    for step in range(args.smoke_steps):
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()
        proprio = extract_vla_proprio_from_obs(
            {
                "agent": {"qpos": obs["agent"]["qpos"][0].detach().cpu().numpy()},
                "extra": {"tcp_pose": obs["extra"]["tcp_pose"][0].detach().cpu().numpy()},
            }
        )
        action = policy.predict_delta_pos_action(rgb=rgb, proprio=proprio)
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action).view(1, -1).to(obs["agent"]["qpos"].device))
        returns += float(reward.item())
        success = bool(info["success"].item())
        if terminated.item() or truncated.item():
            break
    env.close()
    payload = {"smoke_return": returns, "smoke_success": success, "steps": step + 1}
    print("[vla-smoke]", payload)
    save_json(output_dir / "vla_smoke.json", payload)


def train(args: Args) -> None:
    device, rank, world_size = init_runtime(args)
    set_seed(args.seed + rank)

    if args.num_envs % world_size != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by world_size={world_size}")
    local_num_envs = args.num_envs // world_size

    timestamp = broadcast_object(time.strftime("%Y%m%d-%H%M%S") if is_main_process() else None)
    output_dir = mkdir(Path(args.output_dir) / timestamp)
    if is_main_process():
        args_payload = asdict(args)
        args_payload.update({"world_size": world_size, "local_num_envs": local_num_envs})
        save_json(output_dir / "args.json", args_payload)
        print("[setup] loading VLA-Adapter policy")

    raw_policy = VLAAdapterActorCritic(Path(args.model_dir), device=device, xyz_scale=args.vla_xyz_scale).to(device)
    action_head_trainable = args.freeze_action_head_updates <= 0
    raw_policy.configure_trainable_modules(
        freeze_vla_backbone=args.freeze_vla_backbone,
        freeze_proprio_projector=args.freeze_proprio_projector,
        train_action_head=action_head_trainable,
    )
    raw_policy.eval_micro_batch_size = args.eval_micro_batch_size
    if is_main_process():
        summary = raw_policy.trainable_parameter_summary()
        summary_text = " ".join(
            f"{name}={trainable}/{total}"
            for name, (total, trainable) in summary.items()
        )
        print(
            f"[setup] trainable_params {summary_text} "
            f"freeze_vla_updates={args.freeze_action_head_updates}"
        )

    if is_main_process() and world_size == 1:
        run_vla_inference_smoke(args, device, output_dir, policy=raw_policy)
    elif is_main_process():
        print("[setup] distributed mode: skip VLA smoke")
    distributed_barrier()

    if args.demo_pretrain_epochs > 0 and args.demo_pretrain_frames > 0:
        if is_main_process():
            dataset = DemoDataset(Path(args.demo_h5), max_frames=args.demo_pretrain_frames, seed=args.seed)
            print(f"[setup] demo pretrain frames={len(dataset)}")
            demo_optimizer = build_optimizer(args, raw_policy)
            pretrain_policy_with_demos(args, raw_policy, dataset, demo_optimizer, output_dir)
        broadcast_model_state(raw_policy)
    elif is_main_process():
        print("[setup] demo pretrain skipped")
    distributed_barrier()

    policy: nn.Module = raw_policy
    if world_size > 1:
        policy = DDP(
            raw_policy,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            # The wrapped OpenVLA module contains parameters that are not traversed by
            # this PPO objective on every iteration; DDP must track unused params.
            find_unused_parameters=True,
        )

    envs = make_vector_env(args, device, local_num_envs, record_metrics=True)
    eval_envs = make_vector_env(args, device, args.num_eval_envs, record_metrics=True) if is_main_process() else None
    train_video_envs = None
    test_video_envs = None
    if args.save_video and is_main_process():
        train_video_envs = make_vector_env(
            args,
            device,
            min(args.train_video_num_envs, args.num_envs),
            record_metrics=False,
            video_output_dir=output_dir / "train_videos",
            video_max_steps=args.num_steps,
        )
        test_video_envs = make_vector_env(
            args,
            device,
            min(args.test_video_num_envs, args.num_eval_envs),
            record_metrics=False,
            video_output_dir=output_dir / "test_videos",
        )
        print(
            f"[setup] video previews train_videos={output_dir / 'train_videos'} "
            f"test_videos={output_dir / 'test_videos'} "
            f"train_video_num_envs={min(args.train_video_num_envs, args.num_envs)} "
            f"test_video_num_envs={min(args.test_video_num_envs, args.num_eval_envs)} "
            f"device={device}"
        )

    global_batch_size = args.num_envs * args.num_steps
    local_batch_size = local_num_envs * args.num_steps
    global_minibatch_size = max(1, global_batch_size // args.num_minibatches)
    if global_minibatch_size % world_size != 0:
        raise ValueError(
            f"global minibatch size {global_minibatch_size} must be divisible by world_size={world_size}"
        )
    local_minibatch_size = max(1, global_minibatch_size // world_size)
    num_updates = max(1, args.total_timesteps // global_batch_size)
    optimizer = build_optimizer(args, raw_policy)
    if not action_head_trainable and not args.freeze_vla_backbone:
        set_optimizer_group_lr(optimizer, "vla", 0.0)
    if is_main_process():
        optimizer_lrs = " ".join(
            f"{param_group.get('group_name', 'unnamed')}={param_group['lr']}"
            for param_group in optimizer.param_groups
        )
        print(f"[setup] optimizer_lrs {optimizer_lrs}")

    env_actions_buf = torch.zeros((args.num_steps, local_num_envs, 4), device=device)
    action_tokens_buf = torch.zeros((args.num_steps, local_num_envs, 4), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, local_num_envs), device=device)
    final_values = torch.zeros((args.num_steps, local_num_envs), device=device)

    next_obs, _ = envs.reset(seed=args.seed + rank)
    next_done = torch.zeros(local_num_envs, device=device)
    global_step = 0
    best_success_once = -1.0
    metrics_history: List[Dict[str, Any]] = []
    train_start_time = time.time()
    if is_main_process():
        print(
            f"[setup] world_size={world_size} local_num_envs={local_num_envs} num_updates={num_updates} "
            f"global_batch_size={global_batch_size} local_batch_size={local_batch_size} "
            f"global_minibatch_size={global_minibatch_size} local_minibatch_size={local_minibatch_size} "
            f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs} max_runtime_hours={args.max_runtime_hours}"
        )

    for update in range(1, num_updates + 1):
        if not action_head_trainable and update > args.freeze_action_head_updates:
            action_head_trainable = True
            raw_policy.configure_trainable_modules(
                freeze_vla_backbone=args.freeze_vla_backbone,
                freeze_proprio_projector=args.freeze_proprio_projector,
                train_action_head=True,
            )
            if not args.freeze_vla_backbone:
                set_optimizer_group_lr(
                    optimizer,
                    "vla",
                    args.backbone_learning_rate,
                )
            if is_main_process():
                print(
                    f"[setup] unfreezing token policy at update={update} "
                    f"lr={args.backbone_learning_rate if not args.freeze_vla_backbone else 0.0}"
                )
        raw_policy.train()
        final_values.zero_()
        rollout_rgbs: List[np.ndarray] = []
        rollout_proprio: List[np.ndarray] = []
        train_episode_metrics = defaultdict(list)
        partial_reward_means: List[float] = []
        logged_partial_reward_means: List[float] = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            step_rgbs = extract_rgb_batch_from_obs(next_obs)
            step_proprio = extract_vla_proprio_batch_from_obs(next_obs)
            rollout_rgbs.append(step_rgbs.copy())
            rollout_proprio.append(step_proprio.copy())
            dones_buf[step] = next_done
            action, logprob, _, value, action_tokens = batched_get_action_and_value_no_grad(
                policy,
                step_rgbs,
                step_proprio,
                micro_batch_size=args.rollout_micro_batch_size,
            )
            env_actions_buf[step] = action
            action_tokens_buf[step] = action_tokens
            logprobs_buf[step] = logprob
            values_buf[step] = value
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = (terminations | truncations).to(torch.float32)
            rewards_buf[step] = reward.view(-1)
            partial_reward_means.append(float(rewards_buf[: step + 1].mean().item()))

            if (
                (step + 1) % args.rollout_progress_log_interval == 0
                or step == 0
                or step + 1 == args.num_steps
            ):
                elapsed_hours = (time.time() - train_start_time) / 3600.0
                reward_mean_so_far = distributed_mean(partial_reward_means[-1], device)
                if is_main_process():
                    logged_partial_reward_means.append(reward_mean_so_far)
                    print(
                        f"[rollout] update={update}/{num_updates} step={step + 1}/{args.num_steps} "
                        f"reward_mean_so_far={reward_mean_so_far:.4f} elapsed_h={elapsed_hours:.2f}"
                    )
                    save_rollout_progress(
                        output_dir=output_dir,
                        update=update,
                        num_updates=num_updates,
                        rollout_step=step + 1,
                        num_steps=args.num_steps,
                        elapsed_hours=elapsed_hours,
                        partial_reward_means=logged_partial_reward_means,
                    )

            done_mask, episode_metrics = get_completed_episode_metrics(infos)
            if done_mask is not None:
                if done_mask.any():
                    for key, value_tensor in episode_metrics.items():
                        train_episode_metrics[key].append(value_tensor[done_mask].float().detach().cpu())
                if "final_observation" in infos and done_mask.any():
                    final_obs = infos["final_observation"]
                    done_idx = done_mask.detach().cpu().numpy().astype(bool)
                    final_rgbs = extract_rgb_batch_from_obs(final_obs)[done_idx]
                    final_proprio = extract_vla_proprio_batch_from_obs(final_obs)[done_idx]
                    final_values[step, done_mask] = batched_get_value_no_grad(
                        policy,
                        final_rgbs,
                        final_proprio,
                        micro_batch_size=args.eval_micro_batch_size,
                    ).view(-1)

        with torch.no_grad():
            next_value = batched_get_value_no_grad(
                policy,
                extract_rgb_batch_from_obs(next_obs),
                extract_vla_proprio_batch_from_obs(next_obs),
                micro_batch_size=args.eval_micro_batch_size,
            ).view(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                real_next_values = next_nonterminal * next_values + final_values[t]
                delta = rewards_buf[t] + args.gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            returns = advantages + values_buf

        b_rgbs = np.concatenate(rollout_rgbs, axis=0)
        b_proprio = np.concatenate(rollout_proprio, axis=0)
        b_action_tokens = action_tokens_buf.reshape(-1, 4)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        ev = explained_variance(values_buf, returns, device)

        inds = np.arange(local_batch_size)
        approx_kl = 0.0
        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_value = 0.0
        clipfrac_value = 0.0
        stopped_on_minibatch_kl = False
        skipped_updates_on_kl = 0

        update_i = 0

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            epoch_stats = defaultdict(list)
            for start in range(0, local_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = inds[start:end]

                if is_main_process():
                    print(f'[update] {update_i} / {args.update_epochs * (local_batch_size // local_minibatch_size)}')
                    update_i += 1

                mb_stats = ppo_update_with_micro_batches(
                    args=args,
                    policy=policy,
                    optimizer=optimizer,
                    b_rgbs=b_rgbs,
                    b_proprio=b_proprio,
                    b_action_tokens=b_action_tokens,
                    b_logprobs=b_logprobs,
                    b_advantages=b_advantages,
                    b_returns=b_returns,
                    minibatch_inds=mb_inds,
                )
                for key, value in mb_stats.items():
                    epoch_stats[key].append(value)
                skipped_updates_on_kl += int(mb_stats.get("skipped_on_kl", 0.0) > 0.0)
                minibatch_kl = distributed_max(float(mb_stats.get("approx_kl", 0.0)), device)
                if mb_stats.get("skipped_on_kl", 0.0) > 0.0 or minibatch_kl > args.target_kl * args.minibatch_target_kl_factor:
                    stopped_on_minibatch_kl = True
                    break

            approx_kl = distributed_mean(float(np.mean(epoch_stats["approx_kl"])) if epoch_stats["approx_kl"] else 0.0, device)
            clipfrac_value = distributed_mean(float(np.mean(epoch_stats["clipfrac"])) if epoch_stats["clipfrac"] else 0.0, device)
            pg_loss_value = distributed_mean(float(np.mean(epoch_stats["pg_loss"])) if epoch_stats["pg_loss"] else 0.0, device)
            v_loss_value = distributed_mean(float(np.mean(epoch_stats["v_loss"])) if epoch_stats["v_loss"] else 0.0, device)
            entropy_value = distributed_mean(float(np.mean(epoch_stats["entropy"])) if epoch_stats["entropy"] else 0.0, device)
            if stopped_on_minibatch_kl or approx_kl > args.target_kl:
                break

        metric = {
            "update": update,
            "global_step": global_step,
            "reward_mean": distributed_mean(rewards_buf.mean().item(), device),
            "return_mean": distributed_mean(returns.mean().item(), device),
            "value_mean": distributed_mean(values_buf.mean().item(), device),
            "explained_variance": ev,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac_value,
            "pg_loss": pg_loss_value,
            "v_loss": v_loss_value,
            "entropy": entropy_value,
            "stopped_on_minibatch_kl": stopped_on_minibatch_kl,
            "skipped_updates_on_kl": skipped_updates_on_kl,
            "elapsed_hours": (time.time() - train_start_time) / 3600.0,
        }

        local_train_summary = {}
        for key, values in train_episode_metrics.items():
            if values:
                cat = torch.cat(values)
                local_train_summary[f"train_{key}"] = (float(cat.sum().item()), int(cat.numel()))
        metric.update(gather_metric_summary(local_train_summary))

        if (
            is_main_process()
            and train_video_envs is not None
            and args.save_train_video_freq > 0
            and update % args.save_train_video_freq == 0
        ):
            record_policy_rollout_video(
                raw_policy,
                train_video_envs,
                num_steps=args.num_steps,
                seed=args.seed + update,
            )

        if update % args.eval_every_updates == 0 or update == 1 or update == num_updates:
            if is_main_process() and eval_envs is not None:
                eval_metrics = evaluate_policy(raw_policy, eval_envs, args.eval_episodes)
                metric.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                if test_video_envs is not None:
                    evaluate_policy(
                        raw_policy,
                        test_video_envs,
                        min(args.test_video_episodes, max(1, args.test_video_num_envs)),
                    )
                success_once = eval_metrics.get("success_once", eval_metrics.get("success", 0.0))
                if success_once >= best_success_once:
                    best_success_once = success_once
                    torch.save(
                        {
                            "policy": raw_policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_success_once": best_success_once,
                        },
                        output_dir / "best_policy.pt",
                    )

        if is_main_process():
            metrics_history.append(metric)
            print(
                f"[train] update={update}/{num_updates} step={global_step} "
                f"reward={metric['reward_mean']:.4f} return={metric['return_mean']:.4f} "
                f"value_mean={metric['value_mean']:.4f} explained_variance={metric['explained_variance']:.4f} "
                f"approx_kl={metric['approx_kl']:.5f} eval_success_once={metric.get('eval_success_once', float('nan')):.4f} "
                f"elapsed_h={metric['elapsed_hours']:.2f}"
            )
            save_json(output_dir / "latest_metrics.json", metric)
            save_metrics_history(output_dir, metrics_history)
            plot_metrics_history(output_dir, metrics_history)
            if update % 10 == 0 or update == num_updates:
                torch.save(
                    {"policy": raw_policy.state_dict(), "optimizer": optimizer.state_dict(), "update": update},
                    output_dir / "latest_policy.pt",
                )

        reached_time_limit = metric["elapsed_hours"] >= args.max_runtime_hours
        reached_time_limit = broadcast_object(reached_time_limit)
        if reached_time_limit:
            if is_main_process():
                print(f"[train] reached time limit: {metric['elapsed_hours']:.2f}h >= {args.max_runtime_hours:.2f}h")
            break

    if is_main_process():
        save_metrics_history(output_dir, metrics_history)
        plot_metrics_history(output_dir, metrics_history)
    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    if train_video_envs is not None:
        train_video_envs.close()
    if test_video_envs is not None:
        test_video_envs.close()
    distributed_barrier()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        try:
            train(args)
        finally:
            cleanup_runtime()
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = mkdir(Path(args.output_dir) / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    save_json(output_dir / "args.json", asdict(args))
    if args.mode == "vla_smoke":
        run_vla_inference_smoke(args, device, output_dir)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
