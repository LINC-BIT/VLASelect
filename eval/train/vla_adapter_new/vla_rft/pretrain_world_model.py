from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_args, get_origin

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import train.vla_adapter_new.model_impl.env as hold_cube_env  # noqa: F401
import train.vla_adapter_new.model_impl.workload_verify.online_rl_hold_cube_in_hand as reference
import workloads.hold_in_hand  # noqa: F401
from train.vla_adapter_new.model_impl.online_rl import mkdir, parse_bool, save_json, strip_module_prefix


DEFAULT_MODEL_DIR = reference.DEFAULT_MODEL_DIR
DEFAULT_TEACHER_CHECKPOINT = (
    "ckpt/vla_adapter_new/model_impl/outputs/ppo_hold_cube_in_hand/20260430-103518/best_policy.pt"
)
DEFAULT_OUTPUT_DIR = "train/vla_adapter_new/vla_rft/outputs/world_model"
DEFAULT_SUMMARY_NAME = "world_model_summary.json"


@dataclass
class Args:
    mode: str = "train"
    seed: int = 1
    env_id: str = "HoldCubeInHand-v1"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "rgb+state_dict"
    model_dir: str = DEFAULT_MODEL_DIR
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT
    output_dir: str = DEFAULT_OUTPUT_DIR
    run_name: Optional[str] = None
    dataset_path: Optional[str] = None
    reuse_dataset: bool = False
    num_collect_envs: int = 8
    collect_episodes: int = 128
    collect_micro_batch_size: int = 32
    max_episode_steps: Optional[int] = 100
    collect_deterministic: bool = True
    batch_size: int = 64
    num_workers: int = 0
    val_ratio: float = 0.1
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    latent_dim: int = 256
    max_reference_bank_size: int = 256
    cuda_device: str = "0"
    action_dim: int = 16
    state_dim: int = 105


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cuda_device: str) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return device


def load_policy_state_from_checkpoint(checkpoint_path: str, policy: nn.Module) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = strip_module_prefix(checkpoint["policy"])
    else:
        policy_state = strip_module_prefix(checkpoint)
    policy.load_state_dict(policy_state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def make_vector_env(args: Args, device: torch.device, num_envs: int, record_metrics: bool = True):
    runtime_args = reference.Args(
        env_id=args.env_id,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
    )
    return reference.make_vector_env(runtime_args, device, num_envs, record_metrics=record_metrics)


def clone_step_observation(obs: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray]:
    rgbs = reference.extract_rgb_batch_from_obs(obs).clone()
    states = reference.extract_hand_state_batch_from_obs(obs).copy()
    return rgbs, states


def collect_teacher_dataset(
    args: Args,
    device: torch.device,
    policy: nn.Module,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    envs = make_vector_env(args, device, args.num_collect_envs, record_metrics=True)
    try:
        obs, _ = envs.reset(seed=args.seed)
        active_episodes: List[List[Dict[str, Any]]] = [[] for _ in range(args.num_collect_envs)]
        completed_episodes: List[Dict[str, np.ndarray]] = []
        completed_successes = 0
        total_steps = 0
        start_time = time.time()

        while len(completed_episodes) < args.collect_episodes:
            step_rgbs, step_states = clone_step_observation(obs)
            action, _, _, _, _ = reference.batched_get_action_and_value_no_grad(
                policy,
                step_rgbs,
                step_states,
                micro_batch_size=args.collect_micro_batch_size,
                deterministic=args.collect_deterministic,
            )
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            total_steps += int(args.num_collect_envs)

            next_rgbs = reference.extract_rgb_batch_from_obs(next_obs).clone()
            next_states = reference.extract_hand_state_batch_from_obs(next_obs).copy()
            done_mask = torch.logical_or(terminations, truncations).detach().cpu().bool().numpy()

            if "final_observation" in infos:
                final_rgbs = reference.extract_rgb_batch_from_obs(infos["final_observation"]).clone()
                final_states = reference.extract_hand_state_batch_from_obs(infos["final_observation"]).copy()
                next_rgbs[done_mask] = final_rgbs[done_mask]
                next_states[done_mask] = final_states[done_mask]

            reward_np = torch.as_tensor(reward).detach().cpu().view(-1).numpy().astype(np.float32)
            success_np = torch.as_tensor(infos.get("success", torch.zeros_like(reward))).detach().cpu().view(-1)
            success_np = success_np.numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)

            for env_idx in range(args.num_collect_envs):
                active_episodes[env_idx].append(
                    {
                        "obs_rgb": step_rgbs[env_idx].numpy().astype(np.uint8, copy=False),
                        "obs_state": step_states[env_idx].astype(np.float32, copy=False),
                        "action": action_np[env_idx],
                        "reward": reward_np[env_idx],
                        "success": success_np[env_idx],
                        "next_rgb": next_rgbs[env_idx].numpy().astype(np.uint8, copy=False),
                        "next_state": next_states[env_idx].astype(np.float32, copy=False),
                    }
                )

            if done_mask.any():
                done_indices = np.flatnonzero(done_mask)
                for env_idx in done_indices.tolist():
                    episode_steps = active_episodes[env_idx]
                    if not episode_steps:
                        active_episodes[env_idx] = []
                        continue
                    episode = {
                        key: np.stack([step[key] for step in episode_steps], axis=0)
                        if key not in {"reward", "success"}
                        else np.asarray([step[key] for step in episode_steps], dtype=np.float32)
                        for key in episode_steps[0].keys()
                    }
                    completed_episodes.append(episode)
                    completed_successes += int(float(episode["success"].max()) > 0.0)
                    active_episodes[env_idx] = []
                    if len(completed_episodes) >= args.collect_episodes:
                        break

            obs = next_obs

        stats = {
            "num_episodes": len(completed_episodes),
            "num_success_episodes": completed_successes,
            "num_transitions": int(sum(int(episode["action"].shape[0]) for episode in completed_episodes)),
            "success_rate": float(completed_successes / max(1, len(completed_episodes))),
            "elapsed_seconds": float(time.time() - start_time),
            "total_env_steps": total_steps,
        }
        return completed_episodes, stats
    finally:
        envs.close()


class StateNormalizer:
    def __init__(self, state_max: torch.Tensor, state_min: torch.Tensor):
        self.state_max = state_max.to(dtype=torch.float32)
        self.state_min = state_min.to(dtype=torch.float32)

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_min.to(state.device)) / (
            self.state_max.to(state.device) - self.state_min.to(state.device) + 1e-6
        )


class TransitionDataset(Dataset):
    def __init__(self, episodes: Sequence[Dict[str, np.ndarray]]):
        self.episodes = list(episodes)
        self.index: List[Tuple[int, int]] = []
        self.success_reference_index: List[Tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            traj_len = int(episode["action"].shape[0])
            for step_idx in range(traj_len):
                self.index.append((episode_idx, step_idx))
            success_steps = np.flatnonzero(episode["success"] > 0.0)
            if success_steps.size > 0:
                self.success_reference_index.append((episode_idx, int(success_steps[0])))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, step_idx = self.index[idx]
        episode = self.episodes[episode_idx]
        return {
            "obs_rgb": torch.from_numpy(episode["obs_rgb"][step_idx]),
            "obs_state": torch.from_numpy(episode["obs_state"][step_idx]),
            "next_rgb": torch.from_numpy(episode["next_rgb"][step_idx]),
            "next_state": torch.from_numpy(episode["next_state"][step_idx]),
            "action": torch.from_numpy(episode["action"][step_idx]),
            "reward": torch.tensor(float(episode["reward"][step_idx]), dtype=torch.float32),
            "success": torch.tensor(float(episode["success"][step_idx]), dtype=torch.float32),
        }

    def compute_state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        states = np.concatenate(
            [
                np.concatenate([episode["obs_state"], episode["next_state"]], axis=0)
                for episode in self.episodes
            ],
            axis=0,
        )
        return (
            torch.from_numpy(states.max(axis=0).astype(np.float32)),
            torch.from_numpy(states.min(axis=0).astype(np.float32)),
        )

    def build_success_reference_bank(self, limit: int) -> List[Dict[str, np.ndarray]]:
        references: List[Dict[str, np.ndarray]] = []
        for episode_idx, step_idx in self.success_reference_index[:limit]:
            episode = self.episodes[episode_idx]
            references.append(
                {
                    "rgb": episode["next_rgb"][step_idx],
                    "state": episode["next_state"][step_idx],
                }
            )
        return references


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
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_encoder(rgb)
        state_feat = self.state_encoder(state)
        return self.fusion(torch.cat([rgb_feat, state_feat], dim=1))


class DynamicsWorldModel(nn.Module):
    def __init__(self, state_dim: int = 105, action_dim: int = 16, latent_dim: int = 256):
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
        rgb = rgb.float().permute(0, 3, 1, 2)[:, :3] / 255.0
        return F.interpolate(rgb, size=128, mode="bilinear", align_corners=False)

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
        success_loss = F.binary_cross_entropy_with_logits(outputs["pred_success_logit"], batch["success"].float())
        loss = latent_loss + 2.0 * state_loss + 0.5 * reward_loss + 0.5 * success_loss
        return {
            "loss": loss,
            "latent_loss": latent_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "success_loss": success_loss,
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mean_dict(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(np.mean([metric[key] for metric in metrics])) for key in keys}


def train_epoch(
    model: DynamicsWorldModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    normalizer: StateNormalizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    metrics: List[Dict[str, float]] = []
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
    model: DynamicsWorldModel,
    loader: DataLoader,
    normalizer: StateNormalizer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics: List[Dict[str, float]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_loss(batch, normalizer)
        metrics.append({key: float(value.detach().item()) for key, value in losses.items()})
    return mean_dict(metrics)


@torch.no_grad()
def build_reference_bank(
    model: DynamicsWorldModel,
    dataset: TransitionDataset,
    normalizer: StateNormalizer,
    device: torch.device,
    max_refs: int,
) -> Dict[str, torch.Tensor]:
    refs = dataset.build_success_reference_bank(limit=max_refs)
    if not refs:
        raise RuntimeError("No successful transitions collected for reference_bank.")
    latent_bank = []
    state_bank = []
    model.eval()
    for start in range(0, len(refs), 32):
        chunk = refs[start : start + 32]
        rgb = torch.stack([torch.from_numpy(item["rgb"]) for item in chunk]).to(device)
        state = torch.stack([torch.from_numpy(item["state"]) for item in chunk]).to(device)
        latent = model.encode_observation({"rgb": rgb, "state": state}, normalizer)
        latent_bank.append(latent.cpu())
        state_bank.append(normalizer.normalize(state).cpu())
    return {
        "latent": torch.cat(latent_bank, dim=0),
        "state": torch.cat(state_bank, dim=0),
    }


def save_checkpoint(
    output_path: Path,
    model: DynamicsWorldModel,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: Args,
    state_max: torch.Tensor,
    state_min: torch.Tensor,
    reference_bank: Optional[Dict[str, torch.Tensor]],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": asdict(args),
            "state_max": state_max.cpu(),
            "state_min": state_min.cpu(),
            "model_config": {
                "state_dim": model.state_dim,
                "action_dim": model.action_dim,
                "latent_dim": model.latent_dim,
            },
            "reference_bank": None
            if reference_bank is None
            else {
                "latent": reference_bank["latent"].cpu(),
                "state": reference_bank["state"].cpu(),
            },
        },
        output_path,
    )


def save_dataset(dataset_path: Path, episodes: Sequence[Dict[str, np.ndarray]], stats: Dict[str, Any]) -> None:
    torch.save({"episodes": list(episodes), "stats": dict(stats)}, dataset_path)


def load_dataset(dataset_path: Path) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    payload = torch.load(dataset_path, map_location="cpu")
    episodes = payload["episodes"]
    stats = payload.get("stats", {})
    return episodes, stats


def main() -> None:
    args = parse_args()
    if args.mode != "train":
        raise ValueError(f"Unsupported mode: {args.mode}")

    set_seed(args.seed)
    device = get_device(args.cuda_device)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = mkdir(Path(args.output_dir) / run_name)
    dataset_path = Path(args.dataset_path) if args.dataset_path else output_dir / "teacher_rollouts.pt"
    save_json(output_dir / "args.json", asdict(args))
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] device={device}")
    print(f"[setup] dataset_path={dataset_path}")

    teacher_policy = reference.HandVLAAdapterActorCritic(
        Path(args.model_dir),
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    ).to(device)
    load_policy_state_from_checkpoint(args.teacher_checkpoint, teacher_policy)
    teacher_policy.eval_micro_batch_size = args.collect_micro_batch_size
    teacher_policy.eval()
    for parameter in teacher_policy.parameters():
        parameter.requires_grad = False

    if args.reuse_dataset and dataset_path.exists():
        episodes, collect_stats = load_dataset(dataset_path)
        print(f"[collect] reusing existing dataset with {len(episodes)} episodes")
    else:
        print("[collect] generating teacher rollout dataset")
        episodes, collect_stats = collect_teacher_dataset(args, device, teacher_policy)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset_path, episodes, collect_stats)
    save_json(output_dir / "dataset_stats.json", collect_stats)

    dataset = TransitionDataset(episodes)
    state_max, state_min = dataset.compute_state_bounds()
    normalizer = StateNormalizer(state_max, state_min)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = DynamicsWorldModel(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    metrics_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_reference_bank: Optional[Dict[str, torch.Tensor]] = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, normalizer, device)
        val_metrics = eval_epoch(model, val_loader, normalizer, device)
        reference_bank = build_reference_bank(
            model,
            dataset,
            normalizer,
            device,
            max_refs=args.max_reference_bank_size,
        )
        metric = {
            "epoch": epoch,
            "elapsed_seconds": float(time.time() - start_time),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "reference_bank_size": int(reference_bank["latent"].shape[0]),
        }
        metrics_history.append(metric)
        print(
            f"[train] epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_state={val_metrics['state_loss']:.4f} refs={metric['reference_bank_size']}"
        )
        save_json(output_dir / "latest_metrics.json", metric)
        save_json(output_dir / "metrics_history.json", {"metrics": metrics_history})
        save_checkpoint(
            output_dir / "latest_world_model.pt",
            model,
            optimizer,
            epoch,
            metric,
            args,
            state_max,
            state_min,
            reference_bank,
        )
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_reference_bank = {
                "latent": reference_bank["latent"].clone(),
                "state": reference_bank["state"].clone(),
            }
            save_checkpoint(
                output_dir / "best_world_model.pt",
                model,
                optimizer,
                epoch,
                metric,
                args,
                state_max,
                state_min,
                best_reference_bank,
            )

    save_json(
        output_dir / DEFAULT_SUMMARY_NAME,
        {
            "output_dir": str(output_dir),
            "dataset_path": str(dataset_path),
            "best_val_loss": best_val_loss,
            "num_epochs": args.epochs,
            "num_transitions": len(dataset),
            "num_success_references": 0 if best_reference_bank is None else int(best_reference_bank["latent"].shape[0]),
            "collect_stats": collect_stats,
        },
    )


if __name__ == "__main__":
    main()
