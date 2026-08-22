import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import train.vla_adapter_new.model_impl.env as hold_cube_env  # noqa: F401
import train.vla_adapter_new.model_impl.workload_verify.online_rl_hold_cube_in_hand as reference
import workloads.hold_in_hand  # noqa: F401
from train.vla_adapter_new.model_impl.online_rl import parse_bool, save_json, strip_module_prefix


DEFAULT_MODEL_DIR = reference.DEFAULT_MODEL_DIR
DEFAULT_OUTPUT_DIR = "train/vla_adapter_new/world_env/outputs/world_model"
DEFAULT_TEACHER_CHECKPOINT = (
    "ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt"
)
DEFAULT_SUMMARY_NAME = "world_model_training_summary.json"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StateNormalizer:
    def __init__(self, state_max: torch.Tensor, state_min: torch.Tensor):
        self.state_max = state_max.float()
        self.state_min = state_min.float()

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        scale = self.state_max.to(state.device) - self.state_min.to(state.device)
        return (state - self.state_min.to(state.device)) / (scale + 1e-8)

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        scale = self.state_max.to(state.device) - self.state_min.to(state.device)
        return state * (scale + 1e-8) + self.state_min.to(state.device)


class HandObservationEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.rgb_encoder(rgb), self.state_encoder(state)], dim=1))


class HandDynamicsWorldModel(nn.Module):
    def __init__(self, state_dim: int = 105, action_dim: int = 16, latent_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = HandObservationEncoder(state_dim=state_dim, latent_dim=latent_dim)
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
        rgb = rgb.float().permute(0, 3, 1, 2) / 255.0
        return torch.clamp(rgb, 0.0, 1.0)

    def encode_observation(self, obs: Dict[str, torch.Tensor], normalizer: StateNormalizer) -> torch.Tensor:
        rgb = self.preprocess_rgb(obs["rgb"])
        state = normalizer.normalize(obs["state"].float())
        return self.encoder(rgb, state)

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
        obs = {"rgb": batch["obs_rgb"], "state": batch["obs_state"]}
        next_obs = {"rgb": batch["next_rgb"], "state": batch["next_state"]}
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


class TransitionDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._h5: Optional[h5py.File] = None
        with h5py.File(dataset_path, "r") as f:
            self.length = int(f["action"].shape[0])

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._file()
        return {
            "obs_rgb": torch.from_numpy(f["obs_rgb"][idx]),
            "obs_state": torch.from_numpy(f["obs_state"][idx]),
            "next_rgb": torch.from_numpy(f["next_rgb"][idx]),
            "next_state": torch.from_numpy(f["next_state"][idx]),
            "action": torch.from_numpy(f["action"][idx]),
            "reward": torch.tensor(f["reward"][idx], dtype=torch.float32),
            "success": torch.tensor(f["success"][idx], dtype=torch.float32),
        }

    def load_reference_samples(self, limit: int) -> List[Dict[str, np.ndarray]]:
        with h5py.File(self.dataset_path, "r") as f:
            total = int(f["reference_rgb"].shape[0])
            take = min(limit, total)
            return [
                {
                    "rgb": f["reference_rgb"][idx],
                    "state": f["reference_state"][idx],
                }
                for idx in range(take)
            ]


class H5TransitionWriter:
    def __init__(self, path: Path, image_size: int, state_dim: int, action_dim: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._datasets = {
            "obs_rgb": self.file.create_dataset(
                "obs_rgb",
                shape=(0, image_size, image_size, 3),
                maxshape=(None, image_size, image_size, 3),
                dtype=np.uint8,
                compression="gzip",
                chunks=(1, image_size, image_size, 3),
            ),
            "obs_state": self.file.create_dataset(
                "obs_state",
                shape=(0, state_dim),
                maxshape=(None, state_dim),
                dtype=np.float32,
                compression="gzip",
                chunks=(128, state_dim),
            ),
            "next_rgb": self.file.create_dataset(
                "next_rgb",
                shape=(0, image_size, image_size, 3),
                maxshape=(None, image_size, image_size, 3),
                dtype=np.uint8,
                compression="gzip",
                chunks=(1, image_size, image_size, 3),
            ),
            "next_state": self.file.create_dataset(
                "next_state",
                shape=(0, state_dim),
                maxshape=(None, state_dim),
                dtype=np.float32,
                compression="gzip",
                chunks=(128, state_dim),
            ),
            "action": self.file.create_dataset(
                "action",
                shape=(0, action_dim),
                maxshape=(None, action_dim),
                dtype=np.float32,
                compression="gzip",
                chunks=(128, action_dim),
            ),
            "reward": self.file.create_dataset(
                "reward",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                compression="gzip",
                chunks=(256,),
            ),
            "success": self.file.create_dataset(
                "success",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                compression="gzip",
                chunks=(256,),
            ),
            "episode_id": self.file.create_dataset(
                "episode_id",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                compression="gzip",
                chunks=(256,),
            ),
            "reference_rgb": self.file.create_dataset(
                "reference_rgb",
                shape=(0, image_size, image_size, 3),
                maxshape=(None, image_size, image_size, 3),
                dtype=np.uint8,
                compression="gzip",
                chunks=(1, image_size, image_size, 3),
            ),
            "reference_state": self.file.create_dataset(
                "reference_state",
                shape=(0, state_dim),
                maxshape=(None, state_dim),
                dtype=np.float32,
                compression="gzip",
                chunks=(64, state_dim),
            ),
        }
        self.transition_count = 0
        self.reference_count = 0

    def append_batch(self, batch: Dict[str, np.ndarray]) -> None:
        count = int(batch["action"].shape[0])
        start = self.transition_count
        end = start + count
        for key in ("obs_rgb", "obs_state", "next_rgb", "next_state", "action", "reward", "success", "episode_id"):
            dataset = self._datasets[key]
            dataset.resize((end,) + dataset.shape[1:])
            dataset[start:end] = batch[key]
        self.transition_count = end

    def append_reference_batch(self, rgb: np.ndarray, state: np.ndarray) -> None:
        count = int(rgb.shape[0])
        start = self.reference_count
        end = start + count
        self._datasets["reference_rgb"].resize((end,) + self._datasets["reference_rgb"].shape[1:])
        self._datasets["reference_state"].resize((end,) + self._datasets["reference_state"].shape[1:])
        self._datasets["reference_rgb"][start:end] = rgb
        self._datasets["reference_state"][start:end] = state
        self.reference_count = end

    def set_attrs(self, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            self.file.attrs[key] = value

    def close(self) -> None:
        self.file.close()


@dataclass
class Args:
    mode: str = "all"
    seed: int = 1
    cuda_device: str = "0"
    torch_deterministic: bool = True
    env_id: str = "HoldCubeInHand-v1"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT
    output_dir: str = DEFAULT_OUTPUT_DIR
    run_name: Optional[str] = None
    dataset_path: Optional[str] = None
    num_collect_envs: int = 8
    num_eval_envs: int = 1
    target_transitions: int = 4000
    target_episodes: int = 80
    max_episode_steps: Optional[int] = 100
    image_size: int = 64
    latent_dim: int = 256
    val_ratio: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    collect_deterministic: bool = True
    eval_micro_batch_size: int = 32
    max_reference_bank_size: int = 256
    save_dataset_copy: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name.replace('_', '-')}"
        field_type = field_def.type
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=parse_bool, default=default)
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
    return Args(**vars(parser.parse_args()))


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = dicts[0].keys()
    return {key: float(np.mean([item[key] for item in dicts])) for key in keys}


def resize_rgb_batch(rgb_batch: torch.Tensor, image_size: int) -> np.ndarray:
    resized = F.interpolate(
        rgb_batch.to(torch.float32).permute(0, 3, 1, 2) / 255.0,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    resized = torch.clamp((resized * 255.0).round(), 0, 255).to(torch.uint8)
    return resized.permute(0, 2, 3, 1).cpu().numpy()


def load_policy_state_from_checkpoint(checkpoint_path: str, policy: nn.Module) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
    else:
        policy_state = strip_module_prefix(checkpoint)
    policy.load_state_dict(policy_state, strict=True)


def build_teacher_policy(args: Args, device: torch.device) -> reference.HandVLAAdapterActorCritic:
    policy = reference.HandVLAAdapterActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=105,
        action_dim=16,
    ).to(device)
    load_policy_state_from_checkpoint(args.teacher_checkpoint, policy)
    policy.eval_micro_batch_size = args.eval_micro_batch_size
    policy.eval()
    return policy


def extract_transition_next_obs(
    next_obs: Dict[str, Any],
    infos: Dict[str, Any],
) -> Tuple[torch.Tensor, np.ndarray]:
    next_rgb = reference.extract_rgb_batch_from_obs(next_obs).clone()
    next_state = reference.extract_hand_state_batch_from_obs(next_obs).copy()
    done_mask, _ = reference.get_completed_episode_metrics(infos)
    if done_mask is not None and bool(done_mask.any()):
        done_np = done_mask.detach().cpu().numpy().astype(bool)
        done_cpu = done_mask.detach().cpu()
        final_obs = infos["final_observation"]
        final_rgb = reference.extract_rgb_batch_from_obs(final_obs)
        final_state = reference.extract_hand_state_batch_from_obs(final_obs)
        next_rgb[done_cpu] = final_rgb[done_cpu]
        next_state[done_np] = final_state[done_np]
    return next_rgb, next_state


def collect_teacher_dataset(args: Args, device: torch.device, dataset_path: Path) -> Dict[str, Any]:
    teacher_policy = build_teacher_policy(args, device)
    envs = reference.make_vector_env(args, device, args.num_collect_envs, record_metrics=True)
    writer = H5TransitionWriter(dataset_path, args.image_size, state_dim=105, action_dim=16)
    writer.set_attrs(
        {
            "env_id": args.env_id,
            "image_size": args.image_size,
            "state_dim": 105,
            "action_dim": 16,
            "teacher_checkpoint": args.teacher_checkpoint,
        }
    )

    state_min = np.full((105,), np.inf, dtype=np.float32)
    state_max = np.full((105,), -np.inf, dtype=np.float32)
    episode_counters = np.zeros(args.num_collect_envs, dtype=np.int32)
    completed_episodes = 0
    total_success_steps = 0.0
    reference_rgbs: List[np.ndarray] = []
    reference_states: List[np.ndarray] = []
    pending_first_success_refs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * args.num_collect_envs

    try:
        obs, _ = envs.reset(seed=args.seed)
        start_time = time.time()
        while writer.transition_count < args.target_transitions or completed_episodes < args.target_episodes:
            rgbs = reference.extract_rgb_batch_from_obs(obs)
            states = reference.extract_hand_state_batch_from_obs(obs)
            actions, _, _, _, _ = reference.batched_get_action_and_value_no_grad(
                teacher_policy,
                rgbs,
                states,
                micro_batch_size=args.eval_micro_batch_size,
                deterministic=args.collect_deterministic,
            )

            next_obs, reward, _, _, infos = envs.step(actions)
            next_rgb, next_state = extract_transition_next_obs(next_obs, infos)
            success = infos.get("success")
            if success is None:
                success = torch.zeros(args.num_collect_envs, device=device, dtype=torch.float32)
            success_np = success.detach().cpu().numpy().astype(np.float32)
            total_success_steps += float(success_np.sum())

            resized_obs_rgb = resize_rgb_batch(rgbs, args.image_size)
            resized_next_rgb = resize_rgb_batch(next_rgb, args.image_size)
            state_min = np.minimum(state_min, np.minimum(states.min(axis=0), next_state.min(axis=0)))
            state_max = np.maximum(state_max, np.maximum(states.max(axis=0), next_state.max(axis=0)))

            writer.append_batch(
                {
                    "obs_rgb": resized_obs_rgb,
                    "obs_state": states.astype(np.float32),
                    "next_rgb": resized_next_rgb,
                    "next_state": next_state.astype(np.float32),
                    "action": actions.detach().cpu().numpy().astype(np.float32),
                    "reward": reward.detach().cpu().numpy().reshape(-1).astype(np.float32),
                    "success": success_np.reshape(-1),
                    "episode_id": episode_counters.copy(),
                }
            )

            done_mask, _ = reference.get_completed_episode_metrics(infos)
            done_np = (
                done_mask.detach().cpu().numpy().astype(bool)
                if done_mask is not None
                else np.zeros(args.num_collect_envs, dtype=bool)
            )
            for env_idx in range(args.num_collect_envs):
                if success_np[env_idx] > 0.0 and pending_first_success_refs[env_idx] is None:
                    pending_first_success_refs[env_idx] = (
                        resized_next_rgb[env_idx].copy(),
                        next_state[env_idx].astype(np.float32).copy(),
                    )
                if done_np[env_idx]:
                    ref_sample = pending_first_success_refs[env_idx]
                    if ref_sample is None:
                        ref_sample = (
                            resized_next_rgb[env_idx].copy(),
                            next_state[env_idx].astype(np.float32).copy(),
                        )
                    reference_rgbs.append(ref_sample[0])
                    reference_states.append(ref_sample[1])
                    pending_first_success_refs[env_idx] = None
                    episode_counters[env_idx] += 1
                    completed_episodes += 1

            obs = next_obs
            if writer.transition_count % max(args.num_collect_envs * 20, 1) == 0:
                elapsed = time.time() - start_time
                print(
                    f"[collect] transitions={writer.transition_count}/{args.target_transitions} "
                    f"episodes={completed_episodes}/{args.target_episodes} "
                    f"success_step_rate={total_success_steps / max(1, writer.transition_count):.4f} "
                    f"elapsed_min={elapsed / 60.0:.2f}"
                )
    finally:
        envs.close()

    if reference_rgbs:
        writer.append_reference_batch(
            np.stack(reference_rgbs, axis=0),
            np.stack(reference_states, axis=0).astype(np.float32),
        )
    writer.set_attrs(
        {
            "total_transitions": writer.transition_count,
            "completed_episodes": completed_episodes,
            "reference_samples": writer.reference_count,
        }
    )
    state_stats = {
        "state_max": torch.from_numpy(state_max),
        "state_min": torch.from_numpy(state_min),
    }
    writer.close()
    return {
        "dataset_path": str(dataset_path),
        "state_max": state_stats["state_max"],
        "state_min": state_stats["state_min"],
        "total_transitions": writer.transition_count,
        "completed_episodes": completed_episodes,
        "reference_samples": len(reference_rgbs),
        "success_step_rate": total_success_steps / max(1, writer.transition_count),
    }


def load_state_stats(dataset_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(dataset_path, "r") as f:
        obs_state = f["obs_state"][:]
        next_state = f["next_state"][:]
    stacked = np.concatenate([obs_state, next_state], axis=0)
    return torch.from_numpy(stacked.max(axis=0)), torch.from_numpy(stacked.min(axis=0))


def build_reference_bank(
    model: HandDynamicsWorldModel,
    dataset: TransitionDataset,
    normalizer: StateNormalizer,
    device: torch.device,
    max_refs: int,
) -> Dict[str, torch.Tensor]:
    references = dataset.load_reference_samples(limit=max_refs)
    if not references:
        raise RuntimeError("No reference samples found in collected dataset.")
    latent_bank = []
    state_bank = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(references), 32):
            chunk = references[start : start + 32]
            rgb = torch.stack([torch.from_numpy(item["rgb"]) for item in chunk]).to(device)
            state = torch.stack([torch.from_numpy(item["state"]) for item in chunk]).to(device)
            latent = model.encode_observation({"rgb": rgb, "state": state}, normalizer)
            latent_bank.append(latent.cpu())
            state_bank.append(normalizer.normalize(state).cpu())
    return {
        "latent": torch.cat(latent_bank, dim=0),
        "state": torch.cat(state_bank, dim=0),
    }


def train_epoch(
    model: HandDynamicsWorldModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    normalizer: StateNormalizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    metrics = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_loss(batch, normalizer)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.append({key: float(value.detach().item()) for key, value in losses.items()})
    return mean_dict(metrics)


@torch.no_grad()
def eval_epoch(
    model: HandDynamicsWorldModel,
    loader: DataLoader,
    normalizer: StateNormalizer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_loss(batch, normalizer)
        metrics.append({key: float(value.detach().item()) for key, value in losses.items()})
    return mean_dict(metrics)


def save_checkpoint(
    ckpt_path: Path,
    model: HandDynamicsWorldModel,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: Args,
    state_max: torch.Tensor,
    state_min: torch.Tensor,
    dataset_path: str,
    reference_bank: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": asdict(args),
        "dataset_path": dataset_path,
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


def copy_run_metadata(output_dir: Path, args: Args) -> None:
    save_json(output_dir / "args.json", asdict(args))
    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, code_dir / "pretrain_world_model.py")


def train_world_model(
    args: Args,
    device: torch.device,
    output_dir: Path,
    dataset_path: Path,
    state_max: torch.Tensor,
    state_min: torch.Tensor,
) -> Dict[str, Any]:
    dataset = TransitionDataset(str(dataset_path))
    if len(dataset) < 2:
        raise RuntimeError(f"dataset too small: {len(dataset)} transitions")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
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

    normalizer = StateNormalizer(state_max, state_min)
    model = HandDynamicsWorldModel(
        state_dim=int(state_max.numel()),
        action_dim=16,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(str(output_dir / "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(args).items()])),
    )

    best_val_loss = float("inf")
    best_metrics: Optional[Dict[str, float]] = None
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, normalizer, device)
        val_metrics = eval_epoch(model, val_loader, normalizer, device)
        print(
            f"[train] epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_state={val_metrics['state_loss']:.4f}"
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            args,
            state_max,
            state_min,
            str(dataset_path),
        )
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                args,
                state_max,
                state_min,
                str(dataset_path),
            )

    reference_bank = build_reference_bank(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        device=device,
        max_refs=args.max_reference_bank_size,
    )
    final_metrics = best_metrics if best_metrics is not None else val_metrics
    save_checkpoint(
        output_dir / "checkpoints" / "best_with_reference.pt",
        model,
        optimizer,
        args.epochs,
        final_metrics,
        args,
        state_max,
        state_min,
        str(dataset_path),
        reference_bank=reference_bank,
    )
    writer.close()
    elapsed = time.time() - start_time
    summary = {
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "elapsed_minutes": elapsed / 60.0,
        "checkpoint": str(output_dir / "checkpoints" / "best_with_reference.pt"),
        "dataset_path": str(dataset_path),
        "reference_bank_size": int(reference_bank["latent"].shape[0]),
    }
    save_json(output_dir / DEFAULT_SUMMARY_NAME, summary)
    return summary


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_run_metadata(output_dir, args)
    dataset_path = Path(args.dataset_path) if args.dataset_path else output_dir / "teacher_rollouts.h5"

    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] dataset_path={dataset_path}")
    print(f"[setup] device={device}")

    collect_summary: Optional[Dict[str, Any]] = None
    if args.mode in ("collect", "all"):
        collect_summary = collect_teacher_dataset(args, device, dataset_path)
        save_json(
            output_dir / "dataset_collection_summary.json",
            {
                key: value
                for key, value in collect_summary.items()
                if key not in {"state_max", "state_min"}
            },
        )
        state_max = collect_summary["state_max"]
        state_min = collect_summary["state_min"]
    else:
        state_max, state_min = load_state_stats(dataset_path)

    if args.mode in ("train", "all"):
        train_summary = train_world_model(args, device, output_dir, dataset_path, state_max, state_min)
        if collect_summary is not None:
            train_summary["collection"] = {
                key: value
                for key, value in collect_summary.items()
                if key not in {"state_max", "state_min"}
            }
            save_json(output_dir / DEFAULT_SUMMARY_NAME, train_summary)

    if args.save_dataset_copy and dataset_path.parent != output_dir:
        shutil.copyfile(dataset_path, output_dir / dataset_path.name)


if __name__ == "__main__":
    main()
