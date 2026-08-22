from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_state_dict_with_space(state_dict: dict) -> np.ndarray:
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(f"The dimension of {key} should not be more than 2.")
            state = value
        else:
            raise TypeError(f"Unsupported type: {type(value)}")
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    try:
        return np.hstack(states)
    except Exception:
        return np.column_stack(states)


def extract_state_at_step(traj, obs_idx: int) -> np.ndarray:
    agent = traj["obs"]["agent"]
    extra = traj["obs"]["extra"]
    parts = [
        agent["qpos"][obs_idx],
        agent["qvel"][obs_idx],
        np.asarray([extra["is_grasped"][obs_idx]], dtype=np.float32),
        extra["tcp_pose"][obs_idx],
        extra["goal_pos"][obs_idx],
        extra["obj_pose"][obs_idx],
        extra["tcp_to_obj_pos"][obs_idx],
        extra["obj_to_goal_pos"][obs_idx],
    ]
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts], axis=0)


class StateNormalizer:
    def __init__(self, state_max: torch.Tensor, state_min: torch.Tensor):
        self.state_max = state_max.float()
        self.state_min = state_min.float()

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_min.to(state.device)) / (
            self.state_max.to(state.device) - self.state_min.to(state.device) + 1e-8
        )

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        return state * (self.state_max.to(state.device) - self.state_min.to(state.device) + 1e-8) + self.state_min.to(state.device)


class TransitionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        max_trajectories: int = -1,
        reference_stride: int = 1,
    ):
        self.dataset_path = dataset_path
        self.reference_stride = max(1, reference_stride)
        json_path = dataset_path.replace(".h5", ".json")
        with open(json_path, "r") as f:
            self.json_data = json.load(f)
        self.episodes = self.json_data["episodes"]
        if max_trajectories is None or max_trajectories < 0:
            max_trajectories = len(self.episodes)
        self.max_trajectories = min(max_trajectories, len(self.episodes))
        self.index: List[Tuple[str, int]] = []
        self.success_reference_index: List[Tuple[str, int]] = []
        self._h5: Optional[h5py.File] = None

        with h5py.File(self.dataset_path, "r") as f:
            for episode in self.episodes[: self.max_trajectories]:
                traj_key = f"traj_{episode['episode_id']}"
                traj = f[traj_key]
                traj_len = traj["actions"].shape[0]
                for t in range(traj_len):
                    self.index.append((traj_key, t))

                success = traj["success"][:]
                success_steps = np.flatnonzero(success)
                if len(success_steps) > 0:
                    first_success_obs_idx = min(int(success_steps[0] + 1), traj_len)
                    self.success_reference_index.append((traj_key, first_success_obs_idx))
                else:
                    self.success_reference_index.append((traj_key, traj_len))

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __len__(self):
        return len(self.index)

    def _load_obs(self, traj, obs_idx: int) -> Dict[str, np.ndarray]:
        rgb = traj["obs"]["sensor_data"]["base_camera"]["rgb"][obs_idx]
        depth = traj["obs"]["sensor_data"]["base_camera"]["depth"][obs_idx]
        state = extract_state_at_step(traj, obs_idx)
        return {
            "rgb": rgb.astype(np.uint8),
            "depth": depth.astype(np.int16),
            "state": state.astype(np.float32),
        }

    def __getitem__(self, idx):
        traj_key, step_idx = self.index[idx]
        traj = self._file()[traj_key]
        obs = self._load_obs(traj, step_idx)
        next_obs = self._load_obs(traj, step_idx + 1)
        action = traj["actions"][step_idx].astype(np.float32)
        reward = np.float32(traj["rewards"][step_idx])
        success = np.float32(traj["success"][step_idx])
        return {
            "obs_rgb": torch.from_numpy(obs["rgb"]),
            "obs_depth": torch.from_numpy(obs["depth"]),
            "obs_state": torch.from_numpy(obs["state"]),
            "next_rgb": torch.from_numpy(next_obs["rgb"]),
            "next_depth": torch.from_numpy(next_obs["depth"]),
            "next_state": torch.from_numpy(next_obs["state"]),
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "success": torch.tensor(success, dtype=torch.float32),
        }

    def build_success_reference_bank(self, limit: int = 256):
        selected = self.success_reference_index[:: self.reference_stride][:limit]
        refs = []
        with h5py.File(self.dataset_path, "r") as f:
            for traj_key, obs_idx in selected:
                traj = f[traj_key]
                obs = self._load_obs(traj, obs_idx)
                refs.append(obs)
        return refs


class ObservationEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        state_feat = self.state_encoder(state)
        return self.fusion(torch.cat([rgb_feat, depth_feat, state_feat], dim=1))


class DynamicsWorldModel(nn.Module):
    def __init__(self, state_dim: int = 42, action_dim: int = 4, latent_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = ObservationEncoder(state_dim=state_dim, latent_dim=latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.transition = nn.GRUCell(latent_dim + 128, latent_dim)
        self.next_state_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, state_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    @staticmethod
    def preprocess_rgb(rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float().permute(0, 3, 1, 2)[:, 0:3] / 255.0
        return F.interpolate(rgb, size=128, mode="bilinear", align_corners=False)

    @staticmethod
    def preprocess_depth(depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float().permute(0, 3, 1, 2)[:, 0:1] / 1024.0
        return F.interpolate(depth, size=128, mode="bilinear", align_corners=False)

    def encode_observation(self, obs: Dict[str, torch.Tensor], normalizer: StateNormalizer) -> torch.Tensor:
        rgb = self.preprocess_rgb(obs["rgb"])
        depth = self.preprocess_depth(obs["depth"])
        state = normalizer.normalize(obs["state"].float())
        return self.encoder(rgb, depth, state)

    def imagine_next(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        normalizer: StateNormalizer,
    ) -> Dict[str, torch.Tensor]:
        latent = self.encode_observation(obs, normalizer)
        action_feat = self.action_encoder(action.float())
        next_latent = self.transition(torch.cat([latent, action_feat], dim=1), latent)
        pred_next_state = self.next_state_head(next_latent)
        pred_reward = self.reward_head(next_latent).squeeze(-1)
        pred_success_logit = self.success_head(next_latent).squeeze(-1)
        return {
            "latent": latent,
            "pred_next_latent": next_latent,
            "pred_next_state": pred_next_state,
            "pred_reward": pred_reward,
            "pred_success_logit": pred_success_logit,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        normalizer: StateNormalizer,
    ) -> Dict[str, torch.Tensor]:
        obs = {
            "rgb": batch["obs_rgb"],
            "depth": batch["obs_depth"],
            "state": batch["obs_state"],
        }
        next_obs = {
            "rgb": batch["next_rgb"],
            "depth": batch["next_depth"],
            "state": batch["next_state"],
        }
        outputs = self.imagine_next(obs, batch["action"], normalizer)
        with torch.no_grad():
            target_next_latent = self.encode_observation(next_obs, normalizer)
        target_next_state = normalizer.normalize(batch["next_state"].float())
        latent_loss = F.smooth_l1_loss(outputs["pred_next_latent"], target_next_latent)
        state_loss = F.smooth_l1_loss(outputs["pred_next_state"], target_next_state)
        reward_loss = F.smooth_l1_loss(outputs["pred_reward"], batch["reward"].float())
        success_loss = F.binary_cross_entropy_with_logits(
            outputs["pred_success_logit"],
            batch["success"].float(),
        )
        total_loss = latent_loss + 2.0 * state_loss + 0.5 * reward_loss + 0.5 * success_loss
        return {
            "loss": total_loss,
            "latent_loss": latent_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "success_loss": success_loss,
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = dicts[0].keys()
    return {key: float(np.mean([item[key] for item in dicts])) for key in keys}


def build_reference_bank(
    model: DynamicsWorldModel,
    dataset: TransitionDataset,
    normalizer: StateNormalizer,
    device: torch.device,
    max_refs: int,
):
    references = dataset.build_success_reference_bank(limit=max_refs)
    if not references:
        raise RuntimeError("No successful reference observations found in offline dataset.")
    latent_bank = []
    state_bank = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(references), 32):
            chunk = references[start : start + 32]
            rgb = torch.stack([torch.from_numpy(item["rgb"]) for item in chunk]).to(device)
            depth = torch.stack([torch.from_numpy(item["depth"]) for item in chunk]).to(device)
            state = torch.stack([torch.from_numpy(item["state"]) for item in chunk]).to(device)
            latent = model.encode_observation({"rgb": rgb, "depth": depth, "state": state}, normalizer)
            latent_bank.append(latent.cpu())
            state_bank.append(normalizer.normalize(state).cpu())
    return {
        "latent": torch.cat(latent_bank, dim=0),
        "state": torch.cat(state_bank, dim=0),
    }


def train_epoch(
    model: DynamicsWorldModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    normalizer: StateNormalizer,
    device: torch.device,
):
    model.train()
    metrics = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_loss(batch, normalizer)
        optimizer.zero_grad()
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.append({key: float(value.detach().item()) for key, value in losses.items()})
    return mean_dict(metrics)


@torch.no_grad()
def eval_epoch(
    model: DynamicsWorldModel,
    loader: DataLoader,
    normalizer: StateNormalizer,
    device: torch.device,
):
    model.eval()
    metrics = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_loss(batch, normalizer)
        metrics.append({key: float(value.detach().item()) for key, value in losses.items()})
    return mean_dict(metrics)


def save_checkpoint(
    ckpt_path: Path,
    model: DynamicsWorldModel,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: dict,
    state_max: torch.Tensor,
    state_min: torch.Tensor,
    reference_bank: Optional[Dict[str, torch.Tensor]] = None,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": args,
        "state_max": state_max.cpu(),
        "state_min": state_min.cpu(),
        "model_config": {
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "latent_dim": model.latent_dim,
        },
    }
    if reference_bank is not None:
        payload["reference_bank"] = {
            "latent": reference_bank["latent"].cpu(),
            "state": reference_bank["state"].cpu(),
        }
    torch.save(payload, ckpt_path)


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    dataset_path: str = "datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.h5"
    state_norm_stats_path: str = "ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth"
    max_trajectories: int = -1
    val_ratio: float = 0.1

    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    latent_dim: int = 256
    max_reference_bank_size: int = 256
    reference_stride: int = 1
    tag: Optional[str] = None


def main():
    args = tyro.cli(Args)
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.exp_name is None:
        run_name = f"PickCube-v1/baselines/vla_rft/world_model/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.tag is not None:
            run_name += f"-{args.tag}"
    else:
        run_name = args.exp_name

    run_dir = Path("ckpt") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(args).items()])),
    )

    dataset = TransitionDataset(
        dataset_path=args.dataset_path,
        max_trajectories=args.max_trajectories,
        reference_stride=args.reference_stride,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    state_max, state_min = torch.load(args.state_norm_stats_path, map_location="cpu")
    normalizer = StateNormalizer(state_max, state_min)
    model = DynamicsWorldModel(latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_metrics = None
    start_time = time.time()

    with open(run_dir / "args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, normalizer, device)
        val_metrics = eval_epoch(model, val_loader, normalizer, device)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_state={val_metrics['state_loss']:.4f}"
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            asdict(args),
            state_max,
            state_min,
            reference_bank=None,
        )
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            reference_bank = build_reference_bank(
                model=model,
                dataset=dataset,
                normalizer=normalizer,
                device=device,
                max_refs=args.max_reference_bank_size,
            )
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                asdict(args),
                state_max,
                state_min,
                reference_bank=reference_bank,
            )

    elapsed = time.time() - start_time
    summary = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
