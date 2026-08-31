# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
from pathlib import Path
import torch.multiprocessing as mp
import ast
import bisect
from copy import deepcopy
import os
import random
import re
import time

from train.common.mwe_runtime import ActiveRuntimeTracker
from train.common.mwe_checkpoint import maybe_save_model_checkpoint
from train.common.env_cleanup import clear_torch_cuda_cache, close_envs
from train.common.memory_accounting import (
    DEFAULT_EXCLUDED_RUNTIME_PHASE_NAMES,
    MemoryPhaseTracker,
    write_module_memory_exclusion_metadata,
)
from train.common.time_breakdown import (
    empty_module_breakdown,
    snapshot_time_breakdown_to_metric,
    update_combined_search_enhancement_seconds,
    write_time_breakdown,
)
from dataclasses import dataclass
from typing import List, Optional
import wandb

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
import workloads.table_top
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.utils.dl.common.model import get_module, set_module

from train.octo.metrics_json import JsonMetricsLogger, build_metric_entry
from train.common.mwe_eval import append_episode_metric_batch, summarize_episode_metric_tensors, use_train_success_only
from train.octo.ours.evolving_envs import PickCubeEnvMutable


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `ckpt/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    envs_id: Optional[str] = None
    """optional env sequence for continual online RL, e.g. \"['EnvA', 'EnvB']\""""
    env_change_time_points: Optional[str] = None
    """optional env switch/end minutes, e.g. \"[20, 35]\""""
    include_state: bool = True
    """whether to include state information in observations"""

    normalize_states: bool = True
    """whether to normalize the state in the input data"""

    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    max_time: Optional[float] = None
    """maximum total training time in minutes; None disables the limit"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    bc_pretrained_fbs_model_path: str = ''

    tag: Optional[str] = None

    env_config_path: str = ''

    continue_train_from: Optional[str] = None

    state_norm_stats_path: str = ''

    actor_logstd: float = -0.5
    use_pretrained_decoder_as_actor_mean: bool = False

    max_sparsity: float = 0.9

    ppo_pretrained_model1_path: str = ''
    ppo_pretrained_model2_path: str = ''

    data_manager_url: str = 'http://localhost:8000'

    reinit_head: bool = False

    head_learning_rate: float = 0.0
    """learning rate for head layers (critic/actor). 0 means frozen."""

    gate_reg_coef: float = 0.1
    """coefficient for gate regularization loss. Encourages gate to stay open so aggregator contributes."""

    aggregator_target_kl: float = 2.0
    """separate target KL for aggregator phase. Higher than head target_kl to allow more aggregator updates."""

    feature_selector_topk_trajectories: int = 4
    """number of highest-return trajectories to upload as remote features"""

    feature_selector_temporal_pool_steps: Optional[int] = 8
    """temporal pooling target length for uploaded trajectories; None disables pooling"""

    feature_selector_strategy: str = 'topk_return'
    """trajectory selection strategy: topk_return, random, return_span"""

    feature_aggregator_attention_num_heads: int = 1
    """number of attention heads used in feature aggregator"""

    feature_aggregator_gate_type: str = 'single-layer'
    """gate architecture in feature aggregator: single-layer or two-layers"""

    feature_aggregator_gate_activation: str = 'relu'
    """activation used inside two-layers gate: relu, gelu, silu, tanh"""

    feature_aggregator_norm_type: str = 'none'
    """normalization used inside feature aggregator: none or layernorm"""

    enable_feature_fusion: bool = False
    small_model_generation_strategy: str = 'target-batch'  # 'source', 'target-batch', 'target-single', 'target-single-traj'
    small_model_generation_policy: str = 'small'
    """which policy collects target trajectories for small model generation: small, large, or better"""
    small_model_feedback_schedule: Optional[str] = None
    """when to feedback small model into large model before rollout. None follows small_model_regeneration_schedule for backward compatibility"""
    small_model_regeneration_schedule: str = 'once'
    """when to regenerate small model: once, before_per_rollout, before_per_rollout_if_success_improv_less_than_xx_for_yy_iters, or legacy before_per_rollout_if_success_improv_is_larger_than_xx"""
    small_model_feedback_alpha: float = 1.0
    reset_optimizer_after_regeneration: bool = True
    """feedback strength from small model to large model before regeneration"""
    small_model_regeneration_increment_ratio: float = 1.0
    """fraction of selected channels replaced during regeneration. 1.0 means full reselection, 0.0 means keep previous subnet"""
    small_model_training_variant: str = 'pruned'
    """how to realize the trainable small model: pruned (static compact subnet) or frozen (gate-masked FBS model)"""
    small_model_ab_strategy: Optional[str] = None
    """optional ablation strategy for small-model channel selection: random, inverse, or default"""
    small_model_regeneration_ab_strategy: Optional[str] = None
    """optional ablation strategy used only during regeneration/swapping; None reuses small_model_ab_strategy"""
    update_feature_aggregator_lr: float = 0.
    enable_ricl_injection: bool = False
    """enable RICL-style retrieval injection on top of the VLASelect runtime path"""
    ricl_bank_capacity: int = 4096
    """maximum number of retrieval entries kept in the demo bank"""
    ricl_bank_add_per_iter: int = 128
    """number of rollout samples inserted into the bank after each iteration"""
    ricl_num_neighbors: int = 4
    """number of nearest neighbors used for retrieval"""
    ricl_retrieval_temperature: float = 10.0
    """softmax temperature used to aggregate retrieved neighbors"""
    ricl_state_dim_cap: int = 32
    """maximum number of state dimensions used in the retrieval query"""
    ricl_context_hidden_dim: int = 128
    """hidden dimension of the lightweight retrieval injector"""
    ricl_prompt_feature_scale: float = 0.12
    """strength of the retrieval feature injected into the VLASelect latent"""

    data_manager_port: str = 8000


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _ensure_2d_tensor(value):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(1)
    return value


def _slice_xyz(value):
    value = _ensure_2d_tensor(value)
    return value[:, :3]


def _resolve_extra_tensor(
    extra,
    batch_size,
    key,
    *,
    aliases=(),
    default_dim,
    device=None,
    derive=None,
):
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
        derived_value = derive(extra)
        if derived_value is None:
            return torch.zeros((batch_size, default_dim), dtype=torch.float32, device=device)
        value = _ensure_2d_tensor(derived_value)
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


ABLATION_VARIANTS_KEY = "vlaselect_ablation_checkpoint_variants"
ABLATION_METADATA_KEY = "vlaselect_ablation_checkpoint_metadata"
ABLATION_VARIANTS_SUFFIX = ".ablation_variants.pt"

KNOWN_ABLATION_VARIANT_CURVES = {
    "scaling_law_function:without_scaling_law",
    "neuron_grained_scaling_up:random",
    "neuron_grained_scaling_up:inverse",
    "scaling_down_freezing_vs_pruning:pruning",
    "neuron_swapping:random_swapping",
    "knowledge_accumulation:no_accumulation",
    "knowledge_accumulation:accumulate_every_rollout",
}


def resolve_ablation_curve_key(tag: Optional[str]) -> Optional[str]:
    if not tag or '-' not in tag:
        return None
    panel_id, curve_id = tag.split('-', 1)
    curve_key = f"{panel_id}:{curve_id}"
    known_ours_curves = {
        "scaling_law_function:with_scaling_law",
        "neuron_grained_scaling_up:neuron_grained",
        "scaling_down_freezing_vs_pruning:freezing",
        "neuron_swapping:with_swapping",
        "knowledge_accumulation:selective_accumulation",
    }
    if curve_key not in KNOWN_ABLATION_VARIANT_CURVES and curve_key not in known_ours_curves:
        return None
    return curve_key


def resolve_ablation_variant_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.name.endswith(ABLATION_VARIANTS_SUFFIX):
        return checkpoint_path
    return checkpoint_path.with_suffix(ABLATION_VARIANTS_SUFFIX)


def maybe_load_packaged_ablation_checkpoint_payload(checkpoint_path: Path, curve_key: Optional[str]):
    variant_checkpoint_path = resolve_ablation_variant_checkpoint_path(checkpoint_path.resolve())
    if not variant_checkpoint_path.exists():
        return None
    payload = torch.load(variant_checkpoint_path, map_location='cpu')
    if not isinstance(payload, dict) or ABLATION_VARIANTS_KEY not in payload:
        return None
    if curve_key:
        variants = payload.get(ABLATION_VARIANTS_KEY, {})
        if isinstance(variants, dict) and curve_key in variants:
            print(f"[ablation] loaded packaged checkpoint variant curve={curve_key} src={variant_checkpoint_path}")
            return deepcopy(variants[curve_key])
    base_payload = payload.get('base_payload')
    if base_payload is not None:
        print(f"[ablation] loaded packaged base checkpoint src={variant_checkpoint_path}")
        return deepcopy(base_payload)
    return None


def maybe_load_ablation_checkpoint_payload(checkpoint_path: Path, args: Args):
    resolved_checkpoint_path = checkpoint_path.resolve()
    curve_key = resolve_ablation_curve_key(getattr(args, "tag", None))
    packaged_payload = maybe_load_packaged_ablation_checkpoint_payload(resolved_checkpoint_path, curve_key)
    if packaged_payload is not None:
        return packaged_payload
    return torch.load(resolved_checkpoint_path, map_location='cpu')


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
        distances = torch.cdist(query_embeddings.to(self.device), bank_embeddings)
        k = min(num_neighbors, self.size)
        values, indices = torch.topk(distances, k=k, dim=1, largest=False)
        weights = torch.softmax(-temperature * values, dim=1)
        gathered_actions = bank_actions[indices]
        gathered_embeddings = bank_embeddings[indices]
        action_context = (weights.unsqueeze(-1) * gathered_actions).sum(dim=1)
        embedding_context = (weights.unsqueeze(-1) * gathered_embeddings).sum(dim=1)
        self.last_mean_distance = values.mean().item() if values.numel() > 0 else 0.0
        return (
            action_context.to(query_embeddings.device),
            embedding_context.to(query_embeddings.device),
            values.mean(dim=1).to(query_embeddings.device),
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "cursor": self.cursor,
            "last_mean_distance": self.last_mean_distance,
            "embeddings": self.embeddings.detach().cpu(),
            "actions": self.actions.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.size = int(state.get("size", 0))
        self.cursor = int(state.get("cursor", 0))
        self.last_mean_distance = float(state.get("last_mean_distance", 0.0))
        embeddings = state.get("embeddings")
        actions = state.get("actions")
        if embeddings is not None:
            count = min(self.embeddings.shape[0], embeddings.shape[0])
            self.embeddings[:count].copy_(embeddings[:count].to(self.device))
        if actions is not None:
            count = min(self.actions.shape[0], actions.shape[0])
            self.actions[:count].copy_(actions[:count].to(self.device))


def _build_ricl_query_embeddings(processed_obs, state_dim_cap: int):
    rgb_stats = processed_obs["rgb"].mean(dim=(2, 3))
    depth_stats = processed_obs["depth"].mean(dim=(2, 3))
    state_slice = processed_obs["state"][:, :state_dim_cap]
    if state_slice.shape[1] < state_dim_cap:
        pad = torch.zeros(
            (state_slice.shape[0], state_dim_cap - state_slice.shape[1]),
            dtype=state_slice.dtype,
            device=state_slice.device,
        )
        state_slice = torch.cat([state_slice, pad], dim=1)
    return torch.cat([state_slice, rgb_stats, depth_stats], dim=1)


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

# class NatureCNN(nn.Module):
#     def __init__(self, sample_obs):
#         super().__init__()

#         extractors = {}

#         self.out_features = 0
#         feature_size = 256
#         in_channels=sample_obs["rgb"].shape[-1]
#         image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


#         # here we use a NatureCNN architecture to process images, but any architecture is permissble here
#         cnn = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=32,
#                 kernel_size=8,
#                 stride=4,
#                 padding=0,
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
#             ),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # to easily figure out the dimensions after flattening, we pass a test tensor
#         with torch.no_grad():
#             n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
#             fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         extractors["rgb"] = nn.Sequential(cnn, fc)
#         self.out_features += feature_size

#         if "state" in sample_obs:
#             # for state data we simply pass it through a single linear layer
#             state_size = sample_obs["state"].shape[-1]
#             extractors["state"] = nn.Linear(state_size, 256)
#             self.out_features += 256

#         self.extractors = nn.ModuleDict(extractors)

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             obs = observations[key]
#             if key == "rgb":
#                 obs = obs.float().permute(0,3,1,2)
#                 obs = obs / 255
#             encoded_tensor_list.append(extractor(obs))
#         return torch.cat(encoded_tensor_list, dim=1)


class DepthFeatureFilter(nn.Module):
    def forward(self, x):
        return x[:, torch.cat([torch.arange(0, 256), torch.arange(256 * 2, 256 * 3)])]



class Agent(nn.Module):
    def __init__(self, feature_net, latent_size, state_max=None, state_min=None, normalize_states=True, actor_logstd=-0.5):
        super().__init__()
        # self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        # latent_size = self.feature_net.out_features
        self.feature_net = feature_net
        self.normalize_states = normalize_states
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 4), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 4) * actor_logstd)

        self.state_max = state_max
        self.state_min = state_min

    def preprocess(self, obs):

        device = self.actor_logstd.device

        if isinstance(obs['rgb'], np.ndarray):
            rgb, depth = torch.from_numpy(obs['rgb']), torch.from_numpy(obs['depth'])
            state = torch.from_numpy(obs['state'])
        else:
            rgb, depth, state = obs['rgb'], obs['depth'], obs['state']

        rgb = rgb / 255.
        depth = depth / 1024.

        # print(obs)
        rgb = rgb.permute(0, 3, 1, 2)[:, 0: 3].float()
        depth = depth.permute(0, 3, 1, 2)[:, 0: 1].float()
        import torch.nn.functional as F
        def _resize(img, size=128):
            # img = img.unsqueeze(0)          # (1, C, H, W)
            img = F.interpolate(
                img,
                size=size,
                mode='bilinear',
                # align_corners=align_corners if mode != "nearest" else None,
            )
            return img

        rgb = _resize(rgb).to(device)
        depth = _resize(depth).to(device)

        res = {
            'rgb': rgb,
            'depth': depth
        }
        # obs['rgb'] = rgb
        # obs['depth'] = depth

        if not hasattr(self, 'debuged'):
            self.debuged = 0
        # if self.debuged < 10:
        #     from ours.utils.dl.common.vis import save_tensor_image
        #     save_tensor_image(rgb, f'ckpt/{run_name}/sample-rgb-{self.debuged}.png')
        #     save_tensor_image(depth, f'ckpt/{run_name}/sample-depth-{self.debuged}.png')
        #     self.debuged += 1

        res['state'] = state.to(device)
        def minmax_normalize(x, eps=1e-8):
            return (x - self.state_min) / (self.state_max - self.state_min + eps)
        if self.normalize_states:
            # print(self.state_max, self.state_min)
            res['state'] = minmax_normalize(res['state'])

        return res

    def get_features(self, x):
        x = self.preprocess(x)
        return self.feature_net(x)
    def get_value(self, x):
        x = self.preprocess(x)
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.preprocess(x)
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None, return_action_mean=False):
        # print(11, x['rgb'][0], x['depth'][0], x['state'][0])
        x = self.preprocess(x)
        # print(22, x['rgb'][0], x['depth'][0], x['state'][0])
        # print(111, x['rgb'].mean())
        x = self.feature_net(x)
        # print(222, x.mean())
        action_mean = self.actor_mean(x)
        # print(333, action_mean.mean())
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if return_action_mean:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        value = self.critic(x)
        return torch.cat([action_mean, value], dim=1).sum()


class RiclInjectedAgent(Agent):
    def __init__(
        self,
        feature_net,
        latent_size,
        state_max=None,
        state_min=None,
        normalize_states=True,
        actor_logstd=-0.5,
        ricl_state_dim_cap=32,
        ricl_num_neighbors=4,
        ricl_retrieval_temperature=10.0,
        ricl_context_hidden_dim=128,
        ricl_prompt_feature_scale=0.12,
    ):
        super().__init__(feature_net, latent_size, state_max, state_min, normalize_states, actor_logstd)
        self.ricl_state_dim_cap = ricl_state_dim_cap
        self.ricl_num_neighbors = ricl_num_neighbors
        self.ricl_retrieval_temperature = ricl_retrieval_temperature
        self.ricl_prompt_feature_scale = ricl_prompt_feature_scale
        self.ricl_action_dim = self.actor_mean[-1].out_features
        self.ricl_retrieval_embedding_dim = ricl_state_dim_cap + 4
        injector_in_dim = self.ricl_action_dim + self.ricl_retrieval_embedding_dim + 1
        self.register_buffer(
            'ricl_injector_weight',
            torch.randn(latent_size, injector_in_dim) / np.sqrt(max(injector_in_dim, 1)),
        )
        self.demo_bank = None
        self._pending_ricl_state = None
        self.last_ricl_mean_distance = 0.0

    def set_demo_bank(self, demo_bank):
        self.demo_bank = demo_bank
        if self._pending_ricl_state is not None:
            self.demo_bank.load_state_dict(self._pending_ricl_state)
            self._pending_ricl_state = None

    def export_ricl_state(self):
        if self.demo_bank is None:
            return None
        return {"demo_bank": self.demo_bank.state_dict()}

    def load_ricl_state(self, state):
        if not state:
            return
        bank_state = state.get("demo_bank", state)
        if self.demo_bank is None:
            self._pending_ricl_state = bank_state
        else:
            self.demo_bank.load_state_dict(bank_state)

    def build_query_embeddings(self, obs, already_processed=False):
        processed_obs = obs if already_processed else self.preprocess(obs)
        return _build_ricl_query_embeddings(processed_obs, self.ricl_state_dim_cap)

    def _encode_with_ricl(self, obs):
        processed_obs = self.preprocess(obs)
        latent = self.feature_net(processed_obs)
        if self.training or self.demo_bank is None or self.demo_bank.size == 0 or self.ricl_prompt_feature_scale == 0:
            self.last_ricl_mean_distance = 0.0
            return latent
        query_embeddings = self.build_query_embeddings(processed_obs, already_processed=True)
        retrieval_action, retrieval_embedding, retrieval_distance = self.demo_bank.lookup(
            query_embeddings,
            self.ricl_num_neighbors,
            self.ricl_retrieval_temperature,
        )
        self.last_ricl_mean_distance = float(retrieval_distance.mean().item()) if retrieval_distance.numel() > 0 else 0.0
        retrieval_feature = torch.cat(
            [retrieval_action, retrieval_embedding, retrieval_distance.unsqueeze(1)],
            dim=1,
        )
        return latent + self.ricl_prompt_feature_scale * torch.matmul(retrieval_feature, self.ricl_injector_weight.t())

    def get_features(self, x):
        return self._encode_with_ricl(x)

    def get_value(self, x):
        x = self._encode_with_ricl(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self._encode_with_ricl(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None, return_action_mean=False):
        x = self._encode_with_ricl(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if return_action_mean:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def forward(self, x):
        x = self._encode_with_ricl(x)
        action_mean = self.actor_mean(x)
        value = self.critic(x)
        return torch.cat([action_mean, value], dim=1).sum()


class AgentWithoutDepthInput(nn.Module):
    def __init__(self, feature_net, latent_size, state_max=None, state_min=None, normalize_states=True, actor_logstd=-0.5):
        super().__init__()
        # self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        # latent_size = self.feature_net.out_features
        self.feature_net = feature_net
        self.normalize_states = normalize_states
        self.feature_filter = DepthFeatureFilter()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 4), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 4) * actor_logstd)

        self.state_max = state_max
        self.state_min = state_min

    def preprocess(self, obs):

        device = self.actor_logstd.device

        if isinstance(obs['rgb'], np.ndarray):
            rgb, depth = torch.from_numpy(obs['rgb']), torch.from_numpy(obs['depth'])
            state = torch.from_numpy(obs['state'])
        else:
            rgb, depth, state = obs['rgb'], obs['depth'], obs['state']

        rgb = rgb / 255.
        depth = depth / 1024.

        # print(obs)
        rgb = rgb.permute(0, 3, 1, 2)[:, 0: 3].float()
        depth = depth.permute(0, 3, 1, 2)[:, 0: 1].float()
        import torch.nn.functional as F
        def _resize(img, size=128):
            # img = img.unsqueeze(0)          # (1, C, H, W)
            img = F.interpolate(
                img,
                size=size,
                mode='bilinear',
                # align_corners=align_corners if mode != "nearest" else None,
            )
            return img

        rgb = _resize(rgb).to(device)
        depth = _resize(depth).to(device)

        res = {
            'rgb': rgb,
            'depth': depth
        }
        # obs['rgb'] = rgb
        # obs['depth'] = depth

        if not hasattr(self, 'debuged'):
            self.debuged = 0
        # if self.debuged < 10:
        #     from ours.utils.dl.common.vis import save_tensor_image
        #     save_tensor_image(rgb, f'ckpt/{run_name}/sample-rgb-{self.debuged}.png')
        #     save_tensor_image(depth, f'ckpt/{run_name}/sample-depth-{self.debuged}.png')
        #     self.debuged += 1

        res['state'] = state.to(device)
        def minmax_normalize(x, eps=1e-8):
            return (x - self.state_min) / (self.state_max - self.state_min + eps)
        if self.normalize_states:
            # print(self.state_max, self.state_min)
            res['state'] = minmax_normalize(res['state'])

        return res

    def get_features(self, x):
        x = self.preprocess(x)
        return self.feature_filter(self.feature_net(x))
    def get_value(self, x):
        x = self.preprocess(x)
        x = self.feature_filter(self.feature_net(x))
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.preprocess(x)
        x = self.feature_filter(self.feature_net(x))
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None, return_action_mean=False):
        # print(11, x['rgb'][0], x['depth'][0], x['state'][0])
        x = self.preprocess(x)
        # print(22, x['rgb'][0], x['depth'][0], x['state'][0])
        # print(111, x['rgb'].mean())
        x = self.feature_filter(self.feature_net(x))
        # print(222, x.mean())
        action_mean = self.actor_mean(x)
        # print(333, action_mean.mean())
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if return_action_mean:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.feature_filter(self.feature_net(x))
        action_mean = self.actor_mean(x)
        value = self.critic(x)
        return torch.cat([action_mean, value], dim=1).sum()


def maybe_export_ricl_state(agent):
    export_fn = getattr(agent, 'export_ricl_state', None)
    if export_fn is None:
        return None
    return export_fn()


def build_agent_checkpoint_payload(agent, optimizer, iteration, **extra):
    payload = {
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    ricl_state = maybe_export_ricl_state(agent)
    if ricl_state is not None:
        payload['ricl_state'] = ricl_state
    payload.update(extra)
    return payload


def update_ricl_demo_bank_from_rollout(args, agent, obs_buf, act_buf, rew_buf):
    demo_bank = getattr(agent, 'demo_bank', None)
    build_query_embeddings = getattr(agent, 'build_query_embeddings', None)
    if demo_bank is None or build_query_embeddings is None or args.ricl_bank_add_per_iter <= 0:
        return 0

    flat_obs = obs_buf.reshape((-1,))
    with torch.no_grad():
        embeddings = build_query_embeddings(flat_obs)
    rewards = rew_buf.reshape(-1)
    actions = act_buf.reshape((-1, act_buf.shape[-1])).to(dtype=torch.float32)
    candidate_count = embeddings.shape[0]
    if candidate_count == 0:
        return 0

    k = min(args.ricl_bank_add_per_iter, candidate_count)
    if k <= 0:
        return 0
    top_values, top_indices = torch.topk(rewards, k=k, largest=True)
    selected = top_indices
    if torch.allclose(top_values.abs().sum(), torch.tensor(0.0, device=top_values.device)):
        selected = torch.randperm(candidate_count, device=embeddings.device)[:k]
    return demo_bank.add(embeddings[selected], actions[selected])


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def add_histogram(self, tag, values, step):
        if self.log_wandb:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
        self.writer.add_histogram(tag, values.cpu().numpy(), step)

    def close(self):
        self.writer.close()


@dataclass
class ContinualEnvSchedule:
    env_ids: List[str]
    change_time_points: List[float]


def _parse_cli_sequence(raw_value, arg_name, cast_fn):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raw_text = str(raw_value).strip()
        if raw_text == "":
            return []
        try:
            parsed_value = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed_value = [item.strip() for item in raw_text.split(",") if item.strip()]
        if isinstance(parsed_value, (list, tuple)):
            values = list(parsed_value)
        else:
            values = [parsed_value]
    try:
        return [cast_fn(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse `{arg_name}` from {raw_value!r}") from exc


def build_continual_env_schedule(args: Args) -> Optional[ContinualEnvSchedule]:
    env_ids = _parse_cli_sequence(args.envs_id, "envs_id", str)
    time_points = _parse_cli_sequence(args.env_change_time_points, "env_change_time_points", float)
    if env_ids is None and time_points is None:
        return None
    if env_ids is None or time_points is None:
        raise ValueError("`envs_id` and `env_change_time_points` must be provided together")
    if len(env_ids) == 0:
        raise ValueError("`envs_id` must contain at least one environment")
    if len(env_ids) != len(time_points):
        raise ValueError(
            f"`envs_id` and `env_change_time_points` must have the same length, "
            f"got {len(env_ids)} and {len(time_points)}"
        )
    last_time_point = None
    for time_point in time_points:
        if time_point <= 0:
            raise ValueError("All `env_change_time_points` must be positive")
        if last_time_point is not None and time_point <= last_time_point:
            raise ValueError("`env_change_time_points` must be strictly increasing")
        last_time_point = time_point
    return ContinualEnvSchedule(env_ids=env_ids, change_time_points=time_points)


def _sanitize_env_name_for_path(env_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", env_id)


def make_envs_for_env_id(args: Args, env_id: str, env_kwargs: dict, run_name: str, env_index: int):
    print(f"making gym for env[{env_index}]={env_id}...")
    eval_envs = gym.make(
        env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    envs = gym.make(
        env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    print("gym made!")

    envs = FlattenRGBDObservationWrapper2(envs, rgb=True, depth=True, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper2(eval_envs, rgb=True, depth=True, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        env_dir_name = f"env{env_index:02d}-{_sanitize_env_name_for_path(env_id)}"
        eval_output_dir = f"ckpt/{run_name}/videos/{env_dir_name}"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos/{env_dir_name}"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"ckpt/{run_name}/train_videos/{env_dir_name}",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    return envs, eval_envs


def _extract_single_env_obs(obs, env_idx, keys=("rgb", "depth", "state")):
    single_obs = {}
    for key in keys:
        if key not in obs:
            continue
        value = obs[key]
        if isinstance(value, torch.Tensor):
            single_obs[key] = value[env_idx].detach().clone()
        else:
            single_obs[key] = torch.as_tensor(value[env_idx]).clone()
    return single_obs


def _stack_trajectory_obs(trajectory_obs, device):
    if len(trajectory_obs) == 0:
        raise ValueError("trajectory_obs must contain at least one step")
    return {
        key: torch.stack([step_obs[key] for step_obs in trajectory_obs], dim=0).to(device)
        for key in trajectory_obs[0]
    }


def collect_best_return_trajectory(agent, eval_envs, num_steps, reset_seed=None):
    was_training = agent.training
    agent.eval()

    if reset_seed is None:
        eval_obs, _ = eval_envs.reset()
    else:
        eval_obs, _ = eval_envs.reset(seed=reset_seed)
    num_envs = eval_obs["rgb"].shape[0]
    running_trajectories = [[] for _ in range(num_envs)]
    running_returns = [0.0 for _ in range(num_envs)]
    finished_trajectories = []
    forward_seconds = 0.0

    with torch.no_grad():
        for _ in range(num_steps):
            for env_idx in range(num_envs):
                running_trajectories[env_idx].append(_extract_single_env_obs(eval_obs, env_idx))

            forward_start_time = time.perf_counter()
            actions = agent.get_action(eval_obs, deterministic=True)
            forward_seconds += time.perf_counter() - forward_start_time
            eval_obs, rewards, terminations, truncations, _ = eval_envs.step(actions)

            reward_values = torch.as_tensor(rewards).detach().cpu().view(-1)
            done_mask = torch.logical_or(
                torch.as_tensor(terminations),
                torch.as_tensor(truncations),
            ).detach().cpu().view(-1).bool()

            for env_idx in range(num_envs):
                running_returns[env_idx] += reward_values[env_idx].item()
                if done_mask[env_idx]:
                    finished_trajectories.append(
                        {
                            "return": running_returns[env_idx],
                            "obs": running_trajectories[env_idx],
                        }
                    )
                    running_trajectories[env_idx] = []
                    running_returns[env_idx] = 0.0

    for env_idx in range(num_envs):
        if len(running_trajectories[env_idx]) > 0:
            finished_trajectories.append(
                {
                    "return": running_returns[env_idx],
                    "obs": running_trajectories[env_idx],
                }
            )

    if was_training:
        agent.train()

    if len(finished_trajectories) == 0:
        raise RuntimeError("Failed to collect any trajectory for small model generation")

    return max(finished_trajectories, key=lambda item: item["return"]), forward_seconds


def collect_best_return_trajectory_sample(agent, eval_envs, num_steps, device, reset_seed=None):
    best_trajectory, forward_seconds = collect_best_return_trajectory(
        agent=agent,
        eval_envs=eval_envs,
        num_steps=num_steps,
        reset_seed=reset_seed,
    )
    sample_for_gen_small_model = _stack_trajectory_obs(best_trajectory["obs"], device)
    return sample_for_gen_small_model, best_trajectory["return"], forward_seconds


def resolve_generation_policy_agent(args, large_agent, small_agent=None):
    if args.small_model_generation_policy == 'small':
        if small_agent is not None:
            return small_agent, 'small'
        print('small model generation policy requested small agent, but small agent is unavailable; fallback to large agent')
        return large_agent, 'large'

    if args.small_model_generation_policy == 'large':
        return large_agent, 'large'

    if args.small_model_generation_policy == 'better':
        if small_agent is not None:
            return None, 'better'
        print('small model generation policy requested better policy, but small agent is unavailable; fallback to large agent')
        return large_agent, 'large'

    raise NotImplementedError(
        f"Unknown small_model_generation_policy: {args.small_model_generation_policy}"
    )


def collect_sample_for_small_model_generation(args, large_agent, small_agent, eval_envs, env_kwargs, device):
    if args.small_model_generation_strategy == 'target-batch':
        target_eval_obs, _ = eval_envs.reset()
        return {
            'rgb': target_eval_obs['rgb'].to(device),
            'depth': target_eval_obs['depth'].to(device),
            'state': target_eval_obs['state'].to(device)
        }, 0.0

    if args.small_model_generation_strategy == 'target-single':
        target_eval_obs, _ = eval_envs.reset()
        return {
            'rgb': target_eval_obs['rgb'].to(device)[0: 1],
            'depth': target_eval_obs['depth'].to(device)[0: 1],
            'state': target_eval_obs['state'].to(device)[0: 1]
        }, 0.0

    if args.small_model_generation_strategy == 'target-single-traj':
        generation_agent, generation_policy = resolve_generation_policy_agent(
            args,
            large_agent=large_agent,
            small_agent=small_agent,
        )
        if generation_policy == 'better':
            comparison_seed = random.randint(0, 2**31 - 1)
            print('use better policy to collect best target trajectory for small model generation')
            large_sample, large_return, large_forward_seconds = collect_best_return_trajectory_sample(
                agent=large_agent,
                eval_envs=eval_envs,
                num_steps=args.num_eval_steps,
                device=device,
                reset_seed=comparison_seed,
            )
            small_sample, small_return, small_forward_seconds = collect_best_return_trajectory_sample(
                agent=small_agent,
                eval_envs=eval_envs,
                num_steps=args.num_eval_steps,
                device=device,
                reset_seed=comparison_seed,
            )
            if small_return >= large_return:
                chosen_policy = 'small'
                chosen_sample = small_sample
                chosen_return = small_return
            else:
                chosen_policy = 'large'
                chosen_sample = large_sample
                chosen_return = large_return
            print(
                f"use best target trajectory for small model generation: "
                f"chosen_policy={chosen_policy}, "
                f"large_return={large_return:.4f}, "
                f"small_return={small_return:.4f}, "
                f"chosen_return={chosen_return:.4f}, "
                f"steps={chosen_sample['rgb'].shape[0]}"
            )
            forward_seconds = large_forward_seconds + small_forward_seconds
            return chosen_sample, forward_seconds

        print(f'use {generation_policy} model policy to collect best target trajectory for small model generation')
        sample_for_gen_small_model, best_return, forward_seconds = collect_best_return_trajectory_sample(
            agent=generation_agent,
            eval_envs=eval_envs,
            num_steps=args.num_eval_steps,
            device=device,
        )
        print(
            f"use best target trajectory for small model generation: "
            f"return={best_return:.4f}, steps={sample_for_gen_small_model['rgb'].shape[0]}"
        )
        return sample_for_gen_small_model, forward_seconds

    if args.small_model_generation_strategy == 'source':
        env_kwargs_for_source = dict(env_kwargs)
        source_eval_envs = gym.make(
            'PickCube-v1',
            num_envs=args.num_eval_envs,
            reconfiguration_freq=args.eval_reconfiguration_freq,
            **env_kwargs_for_source,
        )
        try:
            source_eval_envs = FlattenRGBDObservationWrapper2(
                source_eval_envs,
                rgb=True,
                depth=True,
                state=args.include_state,
            )
            if isinstance(source_eval_envs.action_space, gym.spaces.Dict):
                source_eval_envs = FlattenActionSpaceWrapper(source_eval_envs)
            source_eval_envs = ManiSkillVectorEnv(
                source_eval_envs,
                args.num_eval_envs,
                ignore_terminations=not args.eval_partial_reset,
                record_metrics=True,
            )
            source_eval_obs, _ = source_eval_envs.reset()
            print('use source domain data for small model generation')
            return {
                'rgb': source_eval_obs['rgb'].to(device)[0: 1],
                'depth': source_eval_obs['depth'].to(device)[0: 1],
                'state': source_eval_obs['state'].to(device)[0: 1]
            }, 0.0
        finally:
            source_eval_envs.close()

    raise NotImplementedError(f"Unknown small_model_generation_strategy: {args.small_model_generation_strategy}")


def resolve_regeneration_ab_strategy(args) -> Optional[str]:
    if args.small_model_regeneration_ab_strategy is not None:
        return args.small_model_regeneration_ab_strategy
    return args.small_model_ab_strategy


def set_trainable_small_model_sparsity(model: nn.Module, k: float) -> None:
    for module in model.modules():
        k_takes_all = getattr(module, 'k_takes_all', None)
        if k_takes_all is not None and hasattr(k_takes_all, 'k'):
            k_takes_all.k = k


def build_initial_trainable_small_model(args, large_agent, eval_envs, env_kwargs, device):
    search_start_time = time.perf_counter()
    sample_for_gen_small_model, forward_seconds = collect_sample_for_small_model_generation(
        args=args,
        large_agent=large_agent,
        small_agent=None,
        eval_envs=eval_envs,
        env_kwargs=env_kwargs,
        device=device,
    )
    search_seconds = time.perf_counter() - search_start_time

    enhancer_start_time = time.perf_counter()

    if args.small_model_training_variant == 'pruned':
        from ours.pretrain_fbs_model.main import generate_small_cnn_with_verify

        small_agent, pruning_info = generate_small_cnn_with_verify(
            large_agent,
            args.max_sparsity,
            sample_for_gen_small_model,
            lambda model, sample: model(sample),
            return_pruning_info=True,
            ab_strategy=args.small_model_ab_strategy,
        )
        enhancer_seconds = time.perf_counter() - enhancer_start_time
        return small_agent, pruning_info, forward_seconds, enhancer_seconds

    if args.small_model_training_variant == 'frozen':
        small_agent = deepcopy(large_agent)
        set_trainable_small_model_sparsity(small_agent, args.max_sparsity)
        enhancer_seconds = time.perf_counter() - enhancer_start_time
        return small_agent, None, forward_seconds, enhancer_seconds

    raise NotImplementedError(
        f"Unknown small_model_training_variant: {args.small_model_training_variant}"
    )


def should_regenerate_small_model_before_rollout(
    schedule: str,
    iteration: int,
    start_iter_idx: int,
    current_success_end: Optional[float] = None,
    success_end_at_last_regeneration: Optional[float] = None,
    iteration_at_last_regeneration: Optional[int] = None,
) -> bool:
    if schedule == 'once':
        return False
    if schedule == 'before_per_rollout':
        return iteration > start_iter_idx
    threshold_prefix = 'before_per_rollout_if_success_improv_is_larger_than_'
    if schedule.startswith(threshold_prefix):
        try:
            threshold = float(schedule[len(threshold_prefix):])
        except ValueError as exc:
            raise ValueError(
                f"Invalid small_model_regeneration_schedule threshold: {schedule}"
            ) from exc
        return (
            iteration > start_iter_idx
            and current_success_end is not None
            and success_end_at_last_regeneration is not None
            and current_success_end - success_end_at_last_regeneration > threshold
        )
    threshold_match = re.fullmatch(
        r'before_per_rollout_if_success_improv_less_than_'
        r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)_for_(\d+)_iters',
        schedule,
    )
    if threshold_match is not None:
        threshold = float(threshold_match.group(1))
        num_iters = int(threshold_match.group(2))
        return (
            iteration > start_iter_idx
            and current_success_end is not None
            and success_end_at_last_regeneration is not None
            and iteration_at_last_regeneration is not None
            and iteration - iteration_at_last_regeneration >= num_iters
            and current_success_end - success_end_at_last_regeneration < threshold
        )
    raise NotImplementedError(
        f"Unknown small_model_regeneration_schedule: {schedule}"
    )


def should_feedback_small_model_before_rollout(
    schedule: str,
    iteration: int,
    start_iter_idx: int,
    current_success_end: Optional[float] = None,
    success_end_at_last_feedback: Optional[float] = None,
) -> bool:
    if schedule == 'once':
        return False
    if schedule == 'before_per_rollout':
        return iteration > start_iter_idx
    threshold_prefix = 'before_per_rollout_if_success_improv_is_larger_than_'
    if schedule.startswith(threshold_prefix):
        try:
            threshold = float(schedule[len(threshold_prefix):])
        except ValueError as exc:
            raise ValueError(
                f"Invalid small_model_feedback_schedule threshold: {schedule}"
            ) from exc
        return (
            iteration > start_iter_idx
            and current_success_end is not None
            and success_end_at_last_feedback is not None
            and current_success_end - success_end_at_last_feedback > threshold
        )
    raise NotImplementedError(
        f"Unknown small_model_feedback_schedule: {schedule}"
    )


def resolve_small_model_feedback_schedule(args) -> str:
    if args.small_model_feedback_schedule is not None:
        return args.small_model_feedback_schedule
    legacy_feedback_compatible_prefix = 'before_per_rollout_if_success_improv_is_larger_than_'
    if (
        args.small_model_regeneration_schedule in {'once', 'before_per_rollout'}
        or args.small_model_regeneration_schedule.startswith(legacy_feedback_compatible_prefix)
    ):
        return args.small_model_regeneration_schedule
    return 'once'


def reset_optimizer_state_for_model(
    optimizer,
    model: nn.Module,
    new_pruning_info: Optional[dict] = None,
    previous_pruning_info: Optional[dict] = None,
):
    from ours.libs.gen_scaling_law_data_points_cnn import remap_small_cnn_optimizer_state

    optimizer.zero_grad(set_to_none=True)
    if new_pruning_info is not None and previous_pruning_info is not None:
        remap_small_cnn_optimizer_state(
            optimizer,
            model,
            new_pruning_info,
            previous_pruning_info,
        )
        return

    for param in model.parameters():
        optimizer.state.pop(param, None)


@torch.no_grad()
def feedback_small_model_to_large_model(
    large_agent,
    small_agent,
    current_pruning_info,
    args,
):
    from ours.libs.gen_scaling_law_data_points_cnn import small_cnn_feedback

    small_cnn_feedback(
        large_agent,
        small_agent,
        current_pruning_info,
        alpha=args.small_model_feedback_alpha,
    )


@torch.no_grad()
def regenerate_small_model_in_place(
    large_agent,
    small_agent,
    current_pruning_info,
    optimizer,
    args,
    eval_envs,
    env_kwargs,
    device,
):
    from ours.libs.gen_scaling_law_data_points_cnn import (
        inherit_small_cnn_retained_channels,
    )
    from ours.pretrain_fbs_model.main import generate_small_cnn_with_verify

    search_start_time = time.perf_counter()
    sample_for_gen_small_model, forward_seconds = collect_sample_for_small_model_generation(
        args=args,
        large_agent=large_agent,
        small_agent=small_agent,
        eval_envs=eval_envs,
        env_kwargs=env_kwargs,
        device=device,
    )
    search_seconds = time.perf_counter() - search_start_time

    enhancer_start_time = time.perf_counter()
    regenerated_small_agent, new_pruning_info = generate_small_cnn_with_verify(
        large_agent,
        args.max_sparsity,
        sample_for_gen_small_model,
        lambda model, sample: model(sample),
        return_pruning_info=True,
        previous_pruning_info=current_pruning_info,
        regeneration_increment_ratio=args.small_model_regeneration_increment_ratio,
        ab_strategy=resolve_regeneration_ab_strategy(args),
    )
    if args.small_model_regeneration_increment_ratio < 1.0:
        inherit_small_cnn_retained_channels(
            regenerated_small_agent,
            small_agent,
            new_pruning_info,
            current_pruning_info,
        )
    merge_stats = new_pruning_info.get('merge_stats', {})
    if merge_stats:
        replaced_ratios = []
        for layer_name, layer_stats in merge_stats.items():
            merged_count = max(layer_stats.get('merged_count', 0), 1)
            replaced_ratios.append(layer_stats.get('replaced_count', 0) / merged_count)
        if replaced_ratios:
            print(
                f"incremental regeneration replaced ratio: "
                f"avg={sum(replaced_ratios) / len(replaced_ratios):.4f}, "
                f"min={min(replaced_ratios):.4f}, max={max(replaced_ratios):.4f}"
            )
    small_agent.load_state_dict(regenerated_small_agent.state_dict(), strict=True)
    if args.reset_optimizer_after_regeneration:
        reset_optimizer_state_for_model(
            optimizer,
            small_agent,
            new_pruning_info=new_pruning_info,
            previous_pruning_info=current_pruning_info,
        )
    enhancer_seconds = time.perf_counter() - enhancer_start_time
    return new_pruning_info, forward_seconds, enhancer_seconds

import copy

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class FlattenRGBDObservationWrapper2(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
        sep_depth (bool): Whether to separate depth and rgb images in the observation. Default is True.

    Note that the returned observations will have a "rgb" or "depth" key depending on the rgb/depth bool flags, and will
    always have a "state" key. If sep_depth is False, rgb and depth will be merged into a single "rgbd" key.
    """

    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.sep_depth = sep_depth
        self.include_state = state

        # check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
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
        # flatten the rest of the data which should just be state data
        # observation = common.flatten_state_dict(
        #     observation, use_torch=True, device=self.base_env.device
        # )

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
        observation = state

        ret = dict()
        if self.include_state:
            ret["state"] = observation
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


def load_agent():
    from train.octo.model import Actor
    actor = Actor(42, 4, 1, False).to(
        device=device
    )
    # add FBS
    set_module(actor, 'rgb_encoder.fc.0', svd_decompose_linear(
        get_module(actor, 'rgb_encoder.fc.0')
    ))
    set_module(actor, 'depth_encoder.fc.0', svd_decompose_linear(
        get_module(actor, 'depth_encoder.fc.0')
    ))
    rgb, depth, state = torch.rand((1, 3, 128, 128)), torch.rand((1, 1, 128, 128)), torch.rand((1, 42))
    example_sample = {
        'rgb': rgb.to(device),
        'depth': depth.to(device),
        'state': state.to(device)
    }

    from ours.pretrain_fbs_model.main import add_FBS_into_cnn
    add_FBS_into_cnn(
        actor,
        [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
        # [f'state_encoder.{i}' for i in [2]] + ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
        ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
        example_sample,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample['rgb'], sample['depth'], sample['state'])
    )
    # actor.load_state_dict(torch.load(args.bc_pretrained_fbs_model_path)['actor'])
    # # actor.decoder = nn.Identity()
    # print(f'load bc pretrained fbs model from {args.bc_pretrained_fbs_model_path}')

    state_max, state_min = torch.load(args.state_norm_stats_path)
    if args.enable_ricl_injection:
        agent = RiclInjectedAgent(
            actor,
            256 * 3,
            state_max,
            state_min,
            args.normalize_states,
            args.actor_logstd,
            args.ricl_state_dim_cap,
            args.ricl_num_neighbors,
            args.ricl_retrieval_temperature,
            args.ricl_context_hidden_dim,
            args.ricl_prompt_feature_scale,
        ).to(device)
    else:
        agent = Agent(actor, 256 * 3, state_max, state_min, args.normalize_states, args.actor_logstd).to(device)
    if args.use_pretrained_decoder_as_actor_mean:
        print('use_pretrained_decoder_as_actor_mean')
        agent.actor_mean = deepcopy(actor.decoder)
    actor.decoder = nn.Identity()
    rgb, depth, state = torch.rand((1, 128, 128, 3)), torch.rand((1, 128, 128, 1)), torch.rand((1, 42))
    example_sample = {
        'rgb': rgb,
        'depth': depth,
        'state': state
    }
    
    # print(example_sample)
    add_FBS_into_cnn(
        agent,
        # [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
        [],
        # [f'state_encoder.{i}' for i in [2]] + ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
        ['actor_mean.0', 'critic.0'] if not args.use_pretrained_decoder_as_actor_mean else ['critic.0'],
        example_sample,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample)
    )

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint_payload = maybe_load_ablation_checkpoint_payload(checkpoint_path, args)
        if args.enable_ricl_injection:
            incompatible = agent.load_state_dict(checkpoint_payload['agent'], strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    f"RICL-injected agent loaded with missing={incompatible.missing_keys} "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            load_ricl_state = getattr(agent, 'load_ricl_state', None)
            if load_ricl_state is not None:
                load_ricl_state(checkpoint_payload.get('ricl_state'))
        else:
            agent.load_state_dict(checkpoint_payload['agent'])
    else:
        print(f"checkpoint not found at {checkpoint_path}; keep current initialization")
    for m in agent.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    print('agent1: ', agent)

    return agent


def ppo_agent(args: Args, device, base_runname, agent, agent_name, layer_name_of_output_features: str, layers_name_of_head: List[str],
              local_feature_dim: int, feature_selector_alpha: float, pretrained_feature_aggregator_path):
    

    # if args.exp_name is None:
    #     args.exp_name = os.path.basename(__file__)[: -len(".py")]
    #     run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # else:
    #     run_name = args.exp_name

    # if args.exp_name is None:
    #     args.exp_name = os.path.basename(__file__)[: -len(".py")]
        
    #     # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    #     from datetime import datetime
    #     run_name = f'{args.env_id}/ours/octo/{args.exp_name}/{datetime.now().strftime("%Y%m%d-%H%M%S")}/{agent_name}'
    #     # if args.eval_model_only is not None:
    #     #     run_name += '-eval'
    #     if args.tag is not None:
    #         run_name += f'-{args.tag}'
    # else:
    #     run_name = args.exp_name

    run_name = f"{base_runname}/{agent_name}"
    output_dir = Path(f"ckpt/{run_name}")
    memory_phase_tracker = MemoryPhaseTracker(output_dir)
    memory_phase_tracker.mark("setup", force=True)
    json_metrics = JsonMetricsLogger(output_dir)

    

    
    import json
    with open(args.env_config_path, "r") as f:
        demo_info = json.load(f)
        env_kwargs = demo_info['env_info']['env_kwargs']
    env_kwargs['sim_backend'] = 'physx_cuda'
    # env setup
    # env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda")
    # if args.control_mode is not None:
    #     env_kwargs["control_mode"] = args.control_mode
    del env_kwargs['num_envs']
    del env_kwargs['reward_mode']

    continual_env_schedule = build_continual_env_schedule(args)
    current_env_index = 0
    current_env_id = args.env_id if continual_env_schedule is None else continual_env_schedule.env_ids[0]
    module_breakdown = empty_module_breakdown()
    memory_phase_tracker.mark("workload_initialization")
    workload_init_start_time = time.perf_counter()
    envs, eval_envs = make_envs_for_env_id(
        args,
        current_env_id,
        env_kwargs,
        run_name,
        current_env_index,
    )
    module_breakdown["workload_initialization_seconds"] += time.perf_counter() - workload_init_start_time
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        wandb_mode = os.environ.get("WANDB_MODE", "").lower()
        wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        enable_wandb = args.track and wandb_mode not in {"disabled", "offline"} and not wandb_disabled
        if enable_wandb:
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=current_env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=current_env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            if continual_env_schedule is not None:
                config["continual_env_ids"] = list(continual_env_schedule.env_ids)
                config["continual_env_change_time_points"] = list(continual_env_schedule.change_time_points)
            # wandb.init(
            #     project=args.wandb_project_name,
            #     entity=args.wandb_entity,
            #     sync_tensorboard=False,
            #     config=config,
            #     name=run_name,
            #     save_code=True,
            #     group=args.wandb_group,
            #     tags=["ppo", "walltime_efficient"]
            # )
            # wandb.tensorboard.patch(root_logdir=f"ckpt/{run_name}/tb")
            wandb_api_key = os.environ.get('WANDB_API_KEY', None)
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
            wandb.init(
                project='EuroSys2026',
                # sync_tensorboard=True,
                config=config,
                name=run_name,
                save_code=True,
                group=f'Maniskill/{current_env_id}/ours/octo/{args.exp_name}',
                # tags=[f'Maniskill/{args.env_id}/ours/octo/{args.exp_name}'.split('/')],
            )
        writer = SummaryWriter(f"ckpt/{run_name}/tb")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=enable_wandb, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    training_start_time = time.monotonic()
    runtime_tracker = ActiveRuntimeTracker.from_env(wall_clock_start_time=training_start_time)
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    if continual_env_schedule is not None:
        print(
            f"Continual env schedule enabled: envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, eval_obs, next_done, current_env_id, current_env_index
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = runtime_tracker.current_minutes()
        scheduled_env_index = bisect.bisect_right(
            continual_env_schedule.change_time_points,
            elapsed_minutes,
        )
        if scheduled_env_index >= len(continual_env_schedule.env_ids):
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes

        previous_env_id = current_env_id
        current_env_index = scheduled_env_index
        current_env_id = continual_env_schedule.env_ids[current_env_index]
        print(
            f"Client {agent_name} switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        memory_phase_tracker.mark("workload_initialization")
        workload_init_start_time = time.perf_counter()
        close_envs(envs, eval_envs)
        envs = None
        eval_envs = None
        clear_torch_cuda_cache()
        envs, eval_envs = make_envs_for_env_id(
            args,
            current_env_id,
            env_kwargs,
            run_name,
            current_env_index,
        )
        next_obs, _ = envs.reset(seed=args.seed + current_env_index)
        eval_obs, _ = eval_envs.reset(seed=args.seed + current_env_index)
        next_done = torch.zeros(args.num_envs, device=device)
        module_breakdown["workload_initialization_seconds"] += time.perf_counter() - workload_init_start_time
        return True, False, elapsed_minutes

    # 多agent逻辑，暂不需要
    # from ours.de_feature_fusion.client import Client


    # if args.reinit_head:
    #     print("reinitialize head layers")
    #     # print(agent)
    #     for layer_name in layers_name_of_head:
    #         # print(layer_name)
    #         if get_module(agent, layer_name) is None:
    #             continue
    #         for m in get_module(agent, layer_name).modules():
    #             if isinstance(m, nn.Linear):
    #                 m.reset_parameters()

    # feature_selector1 = FeatureSelector(agent1, 'feature_net.decoder', 0.5)
    # feature_selector2 = FeatureSelector(agent2, 'feature_filter', 0.5)
    # feature_aggregator1 = FeatureAggregator(agent1, 'feature_net.decoder', 256 * 3, 256 * 2)
    # feature_aggregator2 = FeatureAggregator(agent2, 'feature_filter', 256 * 2, 256 * 3)

    # generate small model
    print(f'generate small model for online RL')
    large_agent = agent
    memory_exclusion_path = write_module_memory_exclusion_metadata(
        output_dir,
        module=large_agent,
        label="large_agent",
        reason="VLASelect large model can be offloaded during small-model online training; exclude its resident parameter/buffer memory from memory-footprint plots.",
        excluded_runtime_phase_names=DEFAULT_EXCLUDED_RUNTIME_PHASE_NAMES,
    )
    print(f"[setup] memory exclusion metadata saved to {memory_exclusion_path}")
    memory_phase_tracker.mark("large_model_runtime_excluded")
    agent, current_small_model_pruning_info, forward_seconds, enhancer_seconds = build_initial_trainable_small_model(
        args=args,
        large_agent=large_agent,
        eval_envs=eval_envs,
        env_kwargs=env_kwargs,
        device=device,
    )
    module_breakdown["large_model_forward_seconds"] += forward_seconds
    module_breakdown["small_model_generation_seconds"] += enhancer_seconds
    update_combined_search_enhancement_seconds(module_breakdown)
    memory_phase_tracker.mark("evaluation")
    ricl_demo_bank = None
    if args.enable_ricl_injection:
        ricl_demo_bank = RiclDemoBank(
            capacity=args.ricl_bank_capacity,
            embedding_dim=args.ricl_state_dim_cap + 4,
            action_dim=int(np.prod(envs.single_action_space.shape)),
            device=device,
        )
        for ricl_agent in (large_agent, agent):
            set_demo_bank = getattr(ricl_agent, 'set_demo_bank', None)
            if set_demo_bank is not None:
                set_demo_bank(ricl_demo_bank)
    for m in agent.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    for p in agent.parameters():
        p.requires_grad = True


    # 多agent逻辑，暂不需要
    # client = Client(
    #     agent_name,
    #     agent,
    #     layer_name_of_output_features,
    #     local_feature_dim,
    #     feature_selector_alpha,
    #     args.data_manager_url,
    #     device,
    #     local_action_dim=int(np.prod(envs.single_action_space.shape)),
    #     feature_aggregator_attention_num_heads=args.feature_aggregator_attention_num_heads,
    #     feature_aggregator_gate_type=args.feature_aggregator_gate_type,
    #     feature_aggregator_gate_activation=args.feature_aggregator_gate_activation,
    #     feature_aggregator_norm_type=args.feature_aggregator_norm_type,
    #     feature_selector_topk_trajectories=args.feature_selector_topk_trajectories,
    #     feature_selector_temporal_pool_steps=args.feature_selector_temporal_pool_steps,
    #     feature_selector_strategy=args.feature_selector_strategy,
    # )
    # client.load_feature_aggregators(pretrained_feature_aggregator_path)
    # client.before_training_start(agent)

    training_feature_aggregator_modules = []

    trainable_parameters = []
    for n, p in agent.named_parameters():
        p.requires_grad = True
        trainable_parameters += [p]

    #     else:
    #         p.requires_grad = False
    # if args.head_learning_rate > 0:
    #     print(f"Head parameters trainable (lr={args.head_learning_rate}): {[n for n, p in agent.named_parameters() if p.requires_grad]}")
    # else:
    #     print(f"All model parameters frozen. Only feature aggregators will be trained.")

    # Optimizer: head params with small LR, aggregator params added later with full LR
    if len(trainable_parameters) > 0:
        optimizer = optim.Adam(
            [{'params': trainable_parameters, 'lr': args.learning_rate}], eps=1e-5
        )
    else:
        optimizer = optim.Adam(
            [{'params': [torch.zeros(1, requires_grad=True)], 'lr': args.learning_rate}], eps=1e-5  # placeholder
        )
    

    # if args.checkpoint:
    #     agent.load_state_dict(torch.load(args.checkpoint))
    start_iter_idx = 1
    cumulative_times = defaultdict(float)
    best_success_once = 0.
    best_success_end = 0.
    last_eval_metrics = None
    current_success_end = None
    last_train_metrics = {}
    if args.small_model_training_variant == 'frozen':
        if args.small_model_feedback_schedule not in {None, 'once'} or args.small_model_regeneration_schedule != 'once':
            print('frozen small-model ablation disables knowledge accumulation and neuron swapping; forcing feedback/regeneration schedules to once')
        feedback_schedule = 'once'
        regeneration_schedule = 'once'
    else:
        feedback_schedule = resolve_small_model_feedback_schedule(args)
        regeneration_schedule = args.small_model_regeneration_schedule
    success_end_at_last_small_model_feedback = None
    success_end_at_last_small_model_regeneration = None
    iteration_at_last_small_model_regeneration = None

    for iteration in range(start_iter_idx, args.num_iterations + 1):
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/current_env_index", current_env_index, global_step)
        if switched_env:
            current_success_end = None
            success_end_at_last_small_model_feedback = None
            success_end_at_last_small_model_regeneration = None
            iteration_at_last_small_model_regeneration = None
        if should_stop_for_schedule:
            print(
                f"Client {agent_name} reached continual schedule end "
                f"at elapsed={elapsed_minutes:.2f} minutes, stopping training."
            )
            break
        print(f"Client {agent_name}, Iteration: {iteration}, global_step={global_step}")

        sparsity_list = [args.max_sparsity]

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        train_episode_metrics = defaultdict(list)
        agent.eval()
        if iteration % args.eval_freq == 1 or iteration == start_iter_idx or args.eval_freq == 1:
            avg_success_once = None
            avg_success_end = None
            metric_snapshot_source = "eval"
            skip_metric_snapshot = use_train_success_only() and not last_train_metrics

            if skip_metric_snapshot:
                print(f"Client {agent_name} train-success-only snapshot skipped because no completed training episodes were observed yet")
            elif use_train_success_only():
                metric_snapshot_source = "train"
                if "success_once" in last_train_metrics:
                    avg_success_once = float(last_train_metrics["success_once"])
                if "success_at_end" in last_train_metrics:
                    avg_success_end = float(last_train_metrics["success_at_end"])
            else:
                memory_phase_tracker.mark("evaluation")
                print(f"Client {agent_name} Evaluating")

                success_once_values = []
                success_end_values = []
                for test_sparsity in sparsity_list:
                    test_sparsity_str = f'{test_sparsity:.4f}'
                    set_trainable_small_model_sparsity(agent, test_sparsity)

                    stime = time.perf_counter()
                    eval_obs, _ = eval_envs.reset()
                    eval_metrics = defaultdict(list)
                    num_episodes = 0
                    for _ in range(args.num_eval_steps):
                        with torch.no_grad():
                            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                            if "final_info" in eval_infos:
                                mask = eval_infos["_final_info"]
                                num_episodes += mask.sum()
                                for k, v in eval_infos["final_info"]["episode"].items():
                                    eval_metrics[k].append(v)
                    for k, v in eval_metrics.items():
                        mean = torch.stack(v).float().mean()
                        if logger is not None:
                            logger.add_scalar(f"eval/{k}_{test_sparsity_str}", mean, global_step)

                        if k == 'success_once':
                            success_once_values.append(float(mean))
                        if k == 'success_at_end':
                            success_end_values.append(float(mean))

                        if logger is not None and test_sparsity < 0.001:
                            eval_time = time.perf_counter() - stime
                            cumulative_times["eval_time"] += eval_time
                            logger.add_scalar("time/eval_time", eval_time, global_step)

                if success_once_values:
                    avg_success_once = float(sum(success_once_values) / len(success_once_values))
                    logger.add_scalar(f"eval/success_once", avg_success_once, global_step)
                if success_end_values:
                    avg_success_end = float(sum(success_end_values) / len(success_end_values))
                    logger.add_scalar(f"eval/success_end", avg_success_end, global_step)
            if not skip_metric_snapshot:
                if avg_success_once is not None:
                    print(f"Client {agent_name} {metric_snapshot_source} success_once={avg_success_once:.4f}")
                if avg_success_end is not None:
                    current_success_end = float(avg_success_end)
                    if success_end_at_last_small_model_feedback is None:
                        success_end_at_last_small_model_feedback = current_success_end
                    if success_end_at_last_small_model_regeneration is None:
                        success_end_at_last_small_model_regeneration = current_success_end
                        iteration_at_last_small_model_regeneration = iteration
                    print(f"Client {agent_name} eval success_at_end={avg_success_end:.4f}")
                last_eval_metrics = {}
                if avg_success_once is not None:
                    last_eval_metrics["success_once"] = float(avg_success_once)
                if avg_success_end is not None:
                    last_eval_metrics["success_at_end"] = float(avg_success_end)
                if last_eval_metrics:
                    current_elapsed_minutes = elapsed_minutes
                    if current_elapsed_minutes is None:
                        current_elapsed_minutes = runtime_tracker.current_minutes()
                    metric_entry = build_metric_entry(
                        update=iteration,
                        global_step=global_step,
                        current_env_id=current_env_id,
                        current_env_index=current_env_index,
                        elapsed_minutes=current_elapsed_minutes,
                        eval_metrics=last_eval_metrics,
                        extras={
                            "best_success_once": best_success_once,
                            "best_success_at_end": best_success_end,
                            "eval_time": cumulative_times.get("eval_time", 0.0),
                        },
                    )
                    module_breakdown["online_rl_completion_seconds"] = float(
                        cumulative_times.get("rollout_time", 0.0) + cumulative_times.get("update_time", 0.0)
                    )
                    snapshot_time_breakdown_to_metric(
                        metric_entry,
                        rollout_seconds=0.0,
                        training_seconds=0.0,
                        cumulative_rollout_seconds=float(cumulative_times.get("rollout_time", 0.0)),
                        cumulative_training_seconds=float(cumulative_times.get("update_time", 0.0)),
                        module_breakdown=module_breakdown,
                    )
                    json_metrics.append(metric_entry)

                if avg_success_once is not None and avg_success_once >= best_success_once:
                    best_success_once = avg_success_once
                    os.makedirs(f'ckpt/{run_name}/checkpoints', exist_ok=True)
                    maybe_save_model_checkpoint(
                        build_agent_checkpoint_payload(
                            agent,
                            optimizer,
                            iteration,
                            success_once=best_success_once,
                        ),
                        f"ckpt/{run_name}/checkpoints/best_success_once.pt",
                    )
                if avg_success_end is not None and avg_success_end >= best_success_end:
                    best_success_end = avg_success_end
                    os.makedirs(f'ckpt/{run_name}/checkpoints', exist_ok=True)
                    maybe_save_model_checkpoint(
                        build_agent_checkpoint_payload(
                            agent,
                            optimizer,
                            iteration,
                            success_at_end=best_success_end,
                        ),
                        f"ckpt/{run_name}/checkpoints/best_success_end.pt",
                    )

                if args.evaluate:
                    break
        if args.save_model and (iteration % args.eval_freq == 1 or args.eval_freq == 1):
            # model_path = f"ckpt/{run_name}/ckpt_{iteration}.pt"
            # torch.save(agent.state_dict(), model_path)
            os.makedirs(f'ckpt/{run_name}/checkpoints', exist_ok=True)
            maybe_save_model_checkpoint(
                build_agent_checkpoint_payload(
                    agent,
                    optimizer,
                    iteration,
                ),
                f"ckpt/{run_name}/checkpoints/last.pt",
            )
            # 多agent逻辑，暂不需要
            # client.save_feature_aggregators(f"ckpt/{run_name}/checkpoints/last.pt.feature_aggregators")

        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        if logger is not None and elapsed_minutes is not None:
            logger.add_scalar("time/elapsed_minutes", elapsed_minutes, global_step)
            logger.add_scalar("continual/current_env_index", current_env_index, global_step)
        if switched_env:
            current_success_end = None
            success_end_at_last_small_model_feedback = None
            success_end_at_last_small_model_regeneration = None
            iteration_at_last_small_model_regeneration = None
        if should_stop_for_schedule:
            print(
                f"Client {agent_name} reached continual schedule end "
                f"at elapsed={elapsed_minutes:.2f} minutes after evaluation, stopping training."
            )
            break

        if should_feedback_small_model_before_rollout(
            feedback_schedule,
            iteration,
            start_iter_idx,
            current_success_end=current_success_end,
            success_end_at_last_feedback=success_end_at_last_small_model_feedback,
        ):
            memory_phase_tracker.mark("large_model_runtime_excluded")
            print(f'Client {agent_name} feedback small model before rollout')
            feedback_start_time = time.perf_counter()
            feedback_small_model_to_large_model(
                large_agent=large_agent,
                small_agent=agent,
                current_pruning_info=current_small_model_pruning_info,
                args=args,
            )
            module_breakdown["small_model_feedback_seconds"] += time.perf_counter() - feedback_start_time
            success_end_at_last_small_model_feedback = current_success_end

        if should_regenerate_small_model_before_rollout(
            regeneration_schedule,
            iteration,
            start_iter_idx,
            current_success_end=current_success_end,
            success_end_at_last_regeneration=success_end_at_last_small_model_regeneration,
            iteration_at_last_regeneration=iteration_at_last_small_model_regeneration,
        ):
            memory_phase_tracker.mark("large_model_runtime_excluded")
            print(f'Client {agent_name} regenerate small model before rollout')
            current_small_model_pruning_info, forward_seconds, enhancer_seconds = regenerate_small_model_in_place(
                large_agent=large_agent,
                small_agent=agent,
                current_pruning_info=current_small_model_pruning_info,
                optimizer=optimizer,
                args=args,
                eval_envs=eval_envs,
                env_kwargs=env_kwargs,
                device=device,
            )
            module_breakdown["large_model_forward_seconds"] += forward_seconds
            module_breakdown["small_model_generation_seconds"] += enhancer_seconds
            update_combined_search_enhancement_seconds(module_breakdown)
            success_end_at_last_small_model_regeneration = current_success_end
            iteration_at_last_small_model_regeneration = iteration

        # Switch back to train mode for rollout and PPO update
        memory_phase_tracker.mark("online_rl_rollout")
        agent.train()

        if args.max_time is not None:
            elapsed_minutes = runtime_tracker.current_minutes()
            if elapsed_minutes >= args.max_time:
                print(
                    f"Client {agent_name} reached max_time={args.max_time} minutes "
                    f"(elapsed={elapsed_minutes:.2f} minutes) before rollout, stopping training."
                )
                break

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        
        # if iteration % 4 == 0:
        #     cur_sparsity = 0.
        # elif 1 <= iteration % 4 <= 2:
        #     cur_sparsity = random.random() * (0.9 - 0.) + 0.
        # elif iteration % 4 == 3:
        #     cur_sparsity = 0.9
        # cur_sparsity = random.choice(sparsity_list)
        # cur_sparsity = random.uniform(sparsity_list[0], sparsity_list[-1])
        # set_sparsity(large_agent, cur_sparsity)


        rollout_time = time.perf_counter()
        import tqdm
        print(f'Client {agent_name} rollout...')
        for step in tqdm.tqdm(range(0, args.num_steps), desc='rollout...', leave=True, dynamic_ncols=True, disable=True):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, action_mean = agent.get_action_and_value(
                    next_obs,
                    return_action_mean=True,
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            # 多agent逻辑，暂不需要
            # if args.enable_feature_fusion:
            #     client.after_each_forward_during_rollout(
            #         reward.view(-1).detach(),
            #         next_done.detach(),
            #         action_mean=action_mean.detach(),
            #     )
            

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                append_episode_metric_batch(train_episode_metrics, final_info["episode"], done_mask)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        last_train_metrics = summarize_episode_metric_tensors(train_episode_metrics)
        runtime_tracker.add_active_seconds(rollout_time)
        cumulative_times["rollout_time"] += rollout_time

        ricl_added = 0
        if args.enable_ricl_injection:
            ricl_added = update_ricl_demo_bank_from_rollout(args, agent, obs, actions, rewards)
            if logger is not None and ricl_demo_bank is not None:
                logger.add_scalar("ricl/demo_bank_size", ricl_demo_bank.size, global_step)
                logger.add_scalar("ricl/added_per_iteration", ricl_added, global_step)
                logger.add_scalar("ricl/mean_retrieval_distance", ricl_demo_bank.last_mean_distance, global_step)


        # bootstrap value according to termination and truncation
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
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # debug
        # mb_inds = [0, 1]
        # action, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds])
        # _, newlogprob2, entropy2, newvalue2 = agent.get_action_and_value(b_obs[mb_inds], action)
        # logratio = newlogprob - newlogprob2
        # ratio = logratio.exp()
        # with torch.no_grad():
        #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
        #     old_approx_kl = (-logratio).mean()
        #     approx_kl = ((ratio - 1) - logratio).mean()
        #     print(approx_kl)
        #     exit()

        # Optimizing the policy and value network
        agent.train()
        memory_phase_tracker.mark("online_rl_training")
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()

        # all_aggregator_params = []
        # for _, fa in client.feature_aggregators.items():
        #     for p in fa.module.parameters():
        #         all_aggregator_params.append(p)

        # has_active_aggregator = any(
        #     fa.remote_features is not None for fa in client.feature_aggregators.values()
        # )

        # print(f'Client {agent_name} updating (heads_lr={args.head_learning_rate}, agg_active={has_active_aggregator})...')

        # Phase 1: PPO update for head parameters (aggregator frozen)
        if True:
            # Freeze aggregator during PPO head update
            # for p in all_aggregator_params:
            #     p.requires_grad = False

            update_times = 0
            pg_loss = torch.tensor(0.0, device=device)
            v_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
            old_approx_kl = torch.tensor(0.0, device=device)
            approx_kl = torch.tensor(0.0, device=device)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    update_times += 1
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        print(f'Client {agent_name} head early stop (kl={approx_kl:.6f}) after {update_times} updates')
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = mb_advantages - mb_advantages.mean()
                        if mb_advantages.numel() > 1:
                            mb_advantages = mb_advantages / (mb_advantages.std(unbiased=False) + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            print(f'Client {agent_name} head updated: {update_times} steps, v_loss={v_loss.item():.4f}, kl={approx_kl.item():.4f}')

        # # Phase 2: Aggregator update (heads frozen, aggregator unfrozen)
        # if has_active_aggregator and len(all_aggregator_params) > 0:
        #     # Freeze heads, unfreeze aggregator
        #     for p in trainable_parameters:
        #         p.requires_grad = False
        #     for p in all_aggregator_params:
        #         p.requires_grad = True

        #     agg_update_times = 0
        #     for epoch in range(args.update_epochs):
        #         np.random.shuffle(b_inds)
        #         for start in range(0, args.batch_size, args.minibatch_size):
        #             agg_update_times += 1
        #             end = start + args.minibatch_size
        #             mb_inds = b_inds[start:end]

        #             _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
        #             logratio = newlogprob - b_logprobs[mb_inds]
        #             ratio = logratio.exp()

        #             with torch.no_grad():
        #                 approx_kl = ((ratio - 1) - logratio).mean()

        #             if args.aggregator_target_kl is not None and approx_kl > args.aggregator_target_kl:
        #                 print(f'Client {agent_name} aggregator early stop (kl={approx_kl:.6f}) after {agg_update_times} updates')
        #                 break

        #             mb_advantages = b_advantages[mb_inds]
        #             if args.norm_adv:
        #                 mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        #             pg_loss1 = -mb_advantages * ratio
        #             pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        #             pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        #             newvalue = newvalue.view(-1)
        #             v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        #             entropy_loss = entropy.mean()
        #             loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        #             # Gate regularization: encourage gate to stay open so aggregator actually contributes
        #             # Without this, the optimizer learns to close the gate since pretrained agent works without aggregation
        #             if args.gate_reg_coef > 0:
        #                 gate_mean_sum = torch.tensor(0.0, device=device)
        #                 gate_count = 0
        #                 for fa in client.feature_aggregators.values():
        #                     if fa.module.gate_mean is not None:
        #                         gate_mean_sum = gate_mean_sum + fa.module.gate_mean
        #                         gate_count += 1
        #                 if gate_count > 0:
        #                     gate_reg_loss = -args.gate_reg_coef * (gate_mean_sum / gate_count)
        #                     loss = loss + gate_reg_loss

        #             optimizer.zero_grad()
        #             loss.backward()
        #             nn.utils.clip_grad_norm_(all_aggregator_params, args.max_grad_norm)
        #             optimizer.step()

        #         if args.aggregator_target_kl is not None and approx_kl > args.aggregator_target_kl:
        #             break
        #     print(f'Client {agent_name} aggregator updated: {agg_update_times} steps, v_loss={v_loss.item():.4f}, kl={approx_kl.item():.4f}')

        #     # Restore head requires_grad
        #     for p in trainable_parameters:
        #         p.requires_grad = True
        # else:
        #     if not has_active_aggregator:
        #         print(f'Client {agent_name} no active aggregator, skipping aggregator update')
        #     # Set defaults for logging
        #     if args.head_learning_rate <= 0:
        #         v_loss = torch.tensor(0.0)
        #         pg_loss = torch.tensor(0.0)
        #         approx_kl = torch.tensor(0.0)
        #         old_approx_kl = torch.tensor(0.0)
        #         entropy_loss = torch.tensor(0.0)
        #         update_times = 0

        # 多agent逻辑，暂不需要
        # client.refresh_features()
        #
        # if args.update_feature_aggregator_lr > 1e-7:
        #     feature_aggregators_parameters = client.get_feature_aggregators_parameters()
        #     for client_id, fap in feature_aggregators_parameters.items():
        #         # for p in fap:
        #         #     p.requires_grad = True
        #         if client_id not in training_feature_aggregator_modules:
        #             training_feature_aggregator_modules.append(client_id)
        #             fap_list = list(fap)  # must convert to list BEFORE iterating, generator can only be consumed once
        #             for p in fap_list:
        #                 p.requires_grad = True
        #             optimizer.add_param_group({'params': fap_list, 'lr': args.update_feature_aggregator_lr, 'eps': 1e-5})
        #             print(f"Client {agent_name} start training feature aggregator {client_id} with {len(fap_list)} parameters")
        #
        # feature_aggregators_gate_g = client.debug_feature_aggregators()
        # gate_g_strs = []
        # for client_id, gate_info in feature_aggregators_gate_g.items():
        #     for stream_name, gate_g in gate_info.items():
        #         if gate_g is None:
        #             continue
        #         metric_prefix = f"train/feature_aggregator_{client_id}_{stream_name}_gate_g"
        #         logger.add_histogram(metric_prefix, gate_g, global_step)
        #         logger.add_scalar(f"{metric_prefix}_mean", gate_g.mean(), global_step)
        #         logger.add_scalar(f"{metric_prefix}_std", gate_g.std(), global_step)
        #         gate_g_strs.append(f"{client_id}.{stream_name}={gate_g.mean().item():.4f}")
        # if gate_g_strs:
        #     print(f'Client {agent_name} gate_g_mean: {", ".join(gate_g_strs)}')

        update_time = time.perf_counter() - update_time
        runtime_tracker.add_active_seconds(update_time)
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
        # print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"ckpt/{run_name}/final_ckpt.pt"
        # torch.save(agent.state_dict(), model_path)
        maybe_save_model_checkpoint(
            build_agent_checkpoint_payload(
                agent,
                optimizer,
                iteration,
            ),
            f"ckpt/{run_name}/checkpoints/last.pt",
        )
        print(f"model saved to {model_path}")

    # 多agent逻辑，暂不需要
    # client.close()
    if last_eval_metrics is not None:
        json_metrics.save_final_eval(last_eval_metrics)
    module_breakdown["online_rl_completion_seconds"] = float(cumulative_times["rollout_time"] + cumulative_times["update_time"])
    write_time_breakdown(
        output_dir,
        sampling_seconds=float(cumulative_times["rollout_time"]),
        training_seconds=float(cumulative_times["update_time"]),
        module_breakdown=module_breakdown,
    )
    close_envs(envs, eval_envs)
    envs = None
    eval_envs = None
    clear_torch_cuda_cache()
    if logger is not None:
        logger.close()


def apply_mwe_overrides(args: Args) -> Args:
    if os.environ.get("MWE", "0") == "1":
        os.environ.setdefault("VLASELECT_MWE_USE_TRAIN_SUCCESS_ONLY", "1")
        # Keep the same training path while using a deliberately tiny footprint for
        # verification runs. VLA language-model activations dominate memory during
        # PPO, so MWE prioritizes proving the path is runnable over throughput.
        args.num_envs = 4
        args.num_eval_envs = 1
        args.num_steps = 4
        args.num_eval_steps = 4
        args.update_epochs = 1
        args.num_minibatches = 2
        # Initialization still exercises the selected scaling method, while the
        # repeated regeneration path is outside this minimal run and can require
        # architecture-specific checkpoint shapes.
        args.small_model_feedback_schedule = "once"
        args.small_model_regeneration_schedule = "once"
        args.total_timesteps = max(args.total_timesteps, 10**12)
        mwe_runtime_minutes = float(os.environ.get("MWE_MAX_RUNTIME_MINUTES", "5.0"))
        if mwe_runtime_minutes <= 0:
            raise ValueError("MWE_MAX_RUNTIME_MINUTES must be positive")
        args.max_time = mwe_runtime_minutes
    return args

if __name__ == "__main__":
    args = apply_mwe_overrides(tyro.cli(Args))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    continual_env_schedule = build_continual_env_schedule(args)
    if continual_env_schedule is not None:
        args.env_id = continual_env_schedule.env_ids[0]
        print(
            f"Use continual env schedule with first env `{args.env_id}`, "
            f"envs={continual_env_schedule.env_ids}, "
            f"change_time_points={continual_env_schedule.change_time_points}"
        )

    print(f'batch size: {args.batch_size}, minibatch_size: {args.minibatch_size}, num_iterations: {args.num_iterations}')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        
        # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        from datetime import datetime
        run_name = f'{args.env_id}/ours/octo/{args.exp_name}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # if args.eval_model_only is not None:
        #     run_name += '-eval'
        if args.tag is not None:
            run_name += f'-{args.tag}'
    else:
        run_name = args.exp_name

    import shutil
    os.makedirs(f"ckpt/{run_name}/code", exist_ok=True)
    shutil.copyfile(__file__, f"ckpt/{run_name}/code/script.py")
    with open(f"ckpt/{run_name}/code/args.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    shutil.copytree('ours', f"ckpt/{run_name}/code/ours", ignore=shutil.ignore_patterns('__pycache__', 'logs', 'videos', 'utils'))
    current_log_dir = f"ckpt/{run_name}"
    log_root_path = Path(os.path.join(f'ckpt/{run_name}', '..')).resolve()
    all_experiments = sorted(
        [d for d in log_root_path.iterdir() if d.is_dir()],
        key=os.path.getmtime
    )
    last_log_dir = None
    if len(all_experiments) > 1 and os.path.exists(os.path.join(all_experiments[-2], 'code')):
        last_log_dir = all_experiments[-2]
        print(f"Running Diff against last experiment: {last_log_dir.name}")
    else:
        print("No previous experiment found. Skipping diff.")
    if last_log_dir:
        # 上次备份代码的路径 (假设上次也把代码存在了 'src_backup' 文件夹下)
        # 注意：你需要根据你实际的备份习惯修改这里的子目录名，如果直接放在根目录则不需要 / 'src_backup'
        last_code_path = os.path.join(last_log_dir, 'code')
        
        diff_file = os.path.join(current_log_dir, "code_diff.patch")
        
        # 排除列表 (非常重要，防止 diff 包含无关的大文件)
        # 根据你的项目调整
        excludes = [
            ".git", "__pycache__", "logs", "checkpoints", 
            "data", "datasets", "*.pth", "*.pt", "*.jpg", "*.png"
        ]
        exclude_args = " ".join([f"--exclude='{x}'" for x in excludes])
        
        # 只有当上次的代码备份目录存在时才进行 diff
        if os.path.exists(last_code_path):
            # diff -ruN old_dir new_dir
            # -r: 递归, -u: 统一格式(易读), -N: 将缺失文件视为这是新建文件
            cmd = f"diff -ruN {exclude_args} {last_code_path} {os.path.join(current_log_dir, 'code')}"
            
            try:
                import subprocess
                with open(diff_file, "w") as f:
                    # diff 命令返回值：0=无差异, 1=有差异, 2=错误
                    # 我们不检查 returncode，因为有差异是正常的
                    subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.PIPE)
                print(f"Diff saved to: {diff_file}")
            except Exception as e:
                print(f"Error generating diff: {e}")
        else:
            print(f"Warning: Previous code backup not found at {last_code_path}")


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 多agent逻辑，暂不需要
    # Start data manager server in a background process
    # import subprocess
    # data_manager_proc = subprocess.Popen(
    #     ['python', '-m', 'uvicorn', 'ours.de_feature_fusion.data_manager:app', '--host', '0.0.0.0', '--port', args.data_manager_url.split(':')[-1]],
    #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # )
    # import atexit
    # atexit.register(lambda: data_manager_proc.terminate())
    # time.sleep(3)  # Wait for the server to start
    # print(f"Data manager started (pid={data_manager_proc.pid})")
    #
    # mp.set_start_method('spawn', force=True)
    #
    # client1_process = torch.multiprocessing.Process(target=ppo_agent, args=(
    #     args, device, run_name,
    #     load_agent(),
    #     '[agent1]',
    #     'feature_net.decoder',
    #     ['critic', 'actor_mean', 'actor_logstd'],
    #     256 * 3,
    #     0.2,
    #     args.ppo_pretrained_model1_path + '.feature_aggregators'
    # ))
    # client2_process = torch.multiprocessing.Process(target=ppo_agent, args=(
    #     args, device, run_name,
    #     load_agent2(),
    #     'feature_filter',
    #     ['critic', 'actor_mean', 'actor_logstd'],
    #     256 * 2,
    #     0.2,
    #     args.ppo_pretrained_model2_path + '.feature_aggregators'
    # ))
    # client1_process.start()
    # client2_process.start()
    # client1_process.join()
    # client2_process.join()

    ppo_agent(
        args,
        device,
        run_name,
        load_agent(),
        '[agent]',
        'feature_net.decoder',
        ['critic', 'actor_mean', 'actor_logstd'],
        256 * 3,
        0.2,
        None,
    )
