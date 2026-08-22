import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ours.utils.dl.common.model import LayerActivation3, get_module, get_model_device


class FeatureSelector:
    SUPPORTED_SELECTION_STRATEGIES = {
        "topk_return",
        "random",
        "return_span",
    }

    def __init__(self,
                 model: nn.Module,
                 layer_name_of_output_features: str,
                 alpha: float,
                 max_trajectory_count: int = None,
                 temporal_pool_steps: int = None,
                 selection_strategy: str = "topk_return",
                 max_episode_steps: int = None):

        self.model = model
        self.feature_hook = LayerActivation3(
            layer=get_module(model, layer_name_of_output_features)
        )
        self.alpha = alpha
        self.max_trajectory_count = max_trajectory_count
        self.temporal_pool_steps = temporal_pool_steps
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        if selection_strategy not in self.SUPPORTED_SELECTION_STRATEGIES:
            raise ValueError(
                f"Unsupported selection_strategy={selection_strategy}. "
                f"Supported: {sorted(self.SUPPORTED_SELECTION_STRATEGIES)}"
            )
        self.selection_strategy = selection_strategy

        self.completed_trajectories = []
        self.current_trajectory_features = None
        self.current_trajectory_actions = None
        self.current_trajectory_returns = None
        self.current_trajectory_success = None
        self.failed_return_running_max = None
        self.failed_return_running_max_decay = 0.995

    def reset_cache(self):
        self.completed_trajectories = []
        self.current_trajectory_features = None
        self.current_trajectory_actions = None
        self.current_trajectory_returns = None
        self.current_trajectory_success = None

    def get_runtime_state(self):
        return {
            "failed_return_running_max": self.failed_return_running_max,
        }

    def load_runtime_state(self, state):
        if not isinstance(state, dict):
            return
        running_max = state.get("failed_return_running_max")
        self.failed_return_running_max = None if running_max is None else float(running_max)

    def _get_selection_count(self, num_trajectories: int):
        if self.max_trajectory_count is not None:
            return min(self.max_trajectory_count, num_trajectories)
        return max(1, math.ceil(num_trajectories * self.alpha))

    def _compute_task_scores(self, trajectories, update_running_max: bool):
        if len(trajectories) == 0:
            return torch.empty(0, dtype=torch.float32)

        success_once = torch.tensor(
            [bool(item.get("success_once", False)) for item in trajectories],
            dtype=torch.bool,
        )
        episode_lens = torch.tensor(
            [max(int(item.get("episode_len", item["features"].shape[0])), 1) for item in trajectories],
            dtype=torch.float32,
        )
        returns = torch.tensor(
            [float(item["return"]) for item in trajectories],
            dtype=torch.float32,
        )

        max_episode_steps = self.max_episode_steps
        if max_episode_steps is None or max_episode_steps <= 0:
            max_episode_steps = max(int(episode_lens.max().item()), 1)

        success_floor = 0.6
        failure_ceiling = 0.4
        success_speed = 1.0 - (episode_lens - 1.0) / max(max_episode_steps, 1)
        success_speed = success_speed.clamp(0.0, 1.0)

        failed_mask = ~success_once
        normalized_failed_returns = torch.zeros_like(returns)
        if failed_mask.any():
            positive_failed_returns = returns[failed_mask].clamp_min(0.0)
            current_max = positive_failed_returns.max()
            running_max = self.failed_return_running_max
            if running_max is None:
                running_max = float(current_max.item()) if current_max.item() > 0 else 0.0
            elif update_running_max:
                running_max = max(
                    float(current_max.item()),
                    float(running_max) * self.failed_return_running_max_decay,
                )
            if update_running_max:
                self.failed_return_running_max = running_max
            if running_max > 0:
                normalized_failed_returns[failed_mask] = (
                    positive_failed_returns / running_max
                ).clamp(0.0, 1.0)

        return torch.where(
            success_once,
            success_floor + (1.0 - success_floor) * success_speed,
            failure_ceiling * normalized_failed_returns,
        )

    def _select_trajectories(self):
        if len(self.completed_trajectories) == 0:
            return []

        k = self._get_selection_count(len(self.completed_trajectories))
        if self.selection_strategy == "random":
            return random.sample(self.completed_trajectories, k)

        # When task metadata is available, rank trajectories by the same q_task
        # prior that will be sent to the remote aggregator, rather than raw return.
        task_scores = self._compute_task_scores(self.completed_trajectories, update_running_max=True)
        for item, score in zip(self.completed_trajectories, task_scores.tolist()):
            item["q_task"] = float(score)
        sorted_trajectories = sorted(
            self.completed_trajectories,
            key=lambda x: x.get("q_task", x["return"]),
            reverse=True,
        )
        if self.selection_strategy == "topk_return":
            return sorted_trajectories[:k]

        if self.selection_strategy == "return_span":
            if k == 1:
                return sorted_trajectories[:1]
            selected_indices = []
            max_index = len(sorted_trajectories) - 1
            for idx in range(k):
                selected_index = round(idx * max_index / (k - 1))
                if selected_index not in selected_indices:
                    selected_indices.append(selected_index)
            selected_trajectories = [sorted_trajectories[idx] for idx in selected_indices]
            random.shuffle(selected_trajectories)
            return selected_trajectories

        raise RuntimeError(f"Unexpected selection_strategy={self.selection_strategy}")

    def _temporal_pool_trajectory(self, trajectory_tensor: torch.Tensor):
        if self.temporal_pool_steps is None:
            return trajectory_tensor

        if trajectory_tensor.ndim != 2:
            raise ValueError(f"trajectory tensor should be 2D [T, D], got {trajectory_tensor.shape}")

        if self.temporal_pool_steps <= 0:
            raise ValueError(f"temporal_pool_steps should be positive, got {self.temporal_pool_steps}")

        if trajectory_tensor.size(0) <= self.temporal_pool_steps:
            return trajectory_tensor

        pooled = F.adaptive_avg_pool1d(
            trajectory_tensor.transpose(0, 1).unsqueeze(0),
            self.temporal_pool_steps,
        )
        return pooled.squeeze(0).transpose(0, 1).contiguous()

    def _init_trajectory_cache(self, num_envs: int, has_actions: bool):
        self.current_trajectory_features = [[] for _ in range(num_envs)]
        self.current_trajectory_actions = [[] for _ in range(num_envs)] if has_actions else None
        self.current_trajectory_returns = torch.zeros(num_envs, dtype=torch.float32)
        self.current_trajectory_success = torch.zeros(num_envs, dtype=torch.bool)

    def cache_features_of_high_reward_after_each_forward_during_rollout(self, rewards, dones=None, action_mean=None, success=None):
        """
        rollout中每次模型做forward后调用，缓存完整轨迹上的逐步特征。
        """
        features = self.feature_hook.output.detach()
        rewards = rewards.detach().view(-1).cpu()
        actions = None if action_mean is None else action_mean.detach().cpu()

        assert features.ndim == 2, "当前FeatureSelector仅支持二维特征 (B, D)"
        assert len(features) == len(rewards), "features和rewards的长度应该相同"
        if actions is not None:
            assert actions.ndim == 2, "当前FeatureSelector仅支持二维动作表征 (B, D)"
            assert len(actions) == len(rewards), "actions和rewards的长度应该相同"

        features = features.cpu()
        num_envs = len(rewards)

        if dones is None:
            # 兼容旧调用方式：若没有轨迹边界信息，则退化为长度为1的“轨迹”。
            for env_idx in range(num_envs):
                self.completed_trajectories.append({
                    'return': float(rewards[env_idx].item()),
                    'features': features[env_idx].unsqueeze(0),
                    'actions': None if actions is None else actions[env_idx].unsqueeze(0),
                    'success_once': False if success is None else bool(success[env_idx].item()),
                    'episode_len': 1,
                })
            return

        dones = dones.detach().view(-1).to(torch.bool).cpu()
        assert len(dones) == num_envs, "dones和rewards的长度应该相同"
        if success is not None:
            success = success.detach().view(-1).to(torch.bool).cpu()
            assert len(success) == num_envs, "success和rewards的长度应该相同"

        if self.current_trajectory_features is None:
            self._init_trajectory_cache(num_envs, has_actions=actions is not None)

        assert len(self.current_trajectory_features) == num_envs, "num_envs发生变化，当前FeatureSelector不支持动态切换"
        if (self.current_trajectory_actions is None) != (actions is None):
            raise ValueError("action_mean availability changed within one rollout, which is not supported")

        for env_idx in range(num_envs):
            self.current_trajectory_features[env_idx].append(features[env_idx])
            if actions is not None:
                self.current_trajectory_actions[env_idx].append(actions[env_idx])
            self.current_trajectory_returns[env_idx] += rewards[env_idx]
            if success is not None:
                self.current_trajectory_success[env_idx] |= success[env_idx]
                # tmp = self.current_trajectory_success[env_idx]

            if dones[env_idx]:
                self.completed_trajectories.append({
                    'return': float(self.current_trajectory_returns[env_idx].item()),
                    'features': torch.stack(self.current_trajectory_features[env_idx], dim=0),
                    'actions': None if actions is None else torch.stack(self.current_trajectory_actions[env_idx], dim=0),
                    'success_once': bool(self.current_trajectory_success[env_idx].item()),
                    'episode_len': len(self.current_trajectory_features[env_idx]),
                })
                self.current_trajectory_features[env_idx] = []
                if actions is not None:
                    self.current_trajectory_actions[env_idx] = []
                self.current_trajectory_returns[env_idx] = 0
                self.current_trajectory_success[env_idx] = False

    def select_message(self):
        """
        返回其它客户端需要的消息。
        - feature: (num_traj, num_steps, feature_dim)
        - action: (num_traj, num_steps, action_dim) 或 None
        """
        if len(self.completed_trajectories) == 0:
            return None

        selected_trajectories = self._select_trajectories()
        if len(selected_trajectories) == 0:
            return None

        selected_trajectory_features = [
            self._temporal_pool_trajectory(item['features']) for item in selected_trajectories
        ]
        if any("q_task" not in item for item in selected_trajectories):
            task_scores = self._compute_task_scores(selected_trajectories, update_running_max=True)
            for item, score in zip(selected_trajectories, task_scores.tolist()):
                item["q_task"] = float(score)
        selected_returns = torch.tensor(
            [float(item['return']) for item in selected_trajectories],
            dtype=torch.float32,
        )
        selected_success_once = torch.tensor(
            [bool(item.get('success_once', False)) for item in selected_trajectories],
            dtype=torch.bool,
        )
        selected_episode_lens = torch.tensor(
            [int(item.get('episode_len', item['features'].shape[0])) for item in selected_trajectories],
            dtype=torch.long,
        )
        selected_q_task = torch.tensor(
            [float(item.get("q_task", 0.0)) for item in selected_trajectories],
            dtype=torch.float32,
        )
        has_actions = all(item['actions'] is not None for item in selected_trajectories)
        selected_trajectory_actions = None
        if has_actions:
            selected_trajectory_actions = [
                self._temporal_pool_trajectory(item['actions']) for item in selected_trajectories
            ]
        self.completed_trajectories = []

        message = {
            'feature': pad_sequence(
                selected_trajectory_features,
                batch_first=True,
                padding_value=0.0,
            ),
            'action': None,
            'meta': {
                'returns': selected_returns,
                'success_once': selected_success_once,
                'episode_lens': selected_episode_lens,
                'max_episode_steps': self.max_episode_steps,
                'q_task': selected_q_task,
            },
        }
        if selected_trajectory_actions is not None:
            message['action'] = pad_sequence(
                selected_trajectory_actions,
                batch_first=True,
                padding_value=0.0,
            )
        return message

    def select_feature(self):
        message = self.select_message()
        if message is None:
            return None
        return message['feature']
