import os
import itertools
import math
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ours.de_feature_fusion.feature_aggregator import FeatureAggregator
from ours.utils.dl.common.model import LayerActivation3, get_model_device, get_module


def _cast_floating_tensors_for_save(obj, dtype: torch.dtype):
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype) if torch.is_floating_point(obj) else obj
    if isinstance(obj, dict):
        return {key: _cast_floating_tensors_for_save(value, dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_cast_floating_tensors_for_save(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_floating_tensors_for_save(value, dtype) for value in obj)
    return obj


class VLAActionPositionFeatureSelector:
    SUPPORTED_SELECTION_STRATEGIES = {
        "topk_return",
        "random",
        "return_span",
    }

    def __init__(
        self,
        model,
        layer_name_prefix: str,
        num_action_positions: int,
        alpha: float,
        max_trajectory_count: Optional[int] = None,
        temporal_pool_steps: Optional[int] = None,
        selection_strategy: str = "topk_return",
        max_episode_steps: Optional[int] = None,
    ):
        self.model = model
        self.layer_name_prefix = layer_name_prefix
        self.num_action_positions = int(num_action_positions)
        self.feature_hooks = [
            LayerActivation3(get_module(model, f"{layer_name_prefix}.{idx}"))
            for idx in range(self.num_action_positions)
        ]
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
        self.current_trajectory_action_matches = None
        self.current_trajectory_returns = None
        self.current_trajectory_success = None
        self.failed_return_running_max = None
        self.failed_return_running_max_decay = 0.995
        self.last_selected_payload_stats = None

    def reset_cache(self):
        self.completed_trajectories = []
        self.current_trajectory_features = None
        self.current_trajectory_actions = None
        self.current_trajectory_action_matches = None
        self.current_trajectory_returns = None
        self.current_trajectory_success = None
        self.last_selected_payload_stats = None

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

        if trajectory_tensor.ndim == 3:
            time_steps, num_positions, hidden_dim = trajectory_tensor.shape
            if time_steps <= self.temporal_pool_steps:
                return trajectory_tensor
            pooled = F.adaptive_avg_pool1d(
                trajectory_tensor.permute(1, 2, 0).reshape(1, num_positions * hidden_dim, time_steps),
                self.temporal_pool_steps,
            )
            return pooled.reshape(num_positions, hidden_dim, self.temporal_pool_steps).permute(2, 0, 1).contiguous()

        if trajectory_tensor.ndim == 2:
            time_steps, action_dim = trajectory_tensor.shape
            if time_steps <= self.temporal_pool_steps:
                return trajectory_tensor
            pooled = F.adaptive_avg_pool1d(
                trajectory_tensor.transpose(0, 1).unsqueeze(0),
                self.temporal_pool_steps,
            )
            return pooled.squeeze(0).transpose(0, 1).contiguous()

        raise ValueError(
            f"trajectory tensor should be 2D [T, A] or 3D [T, A, H], got {trajectory_tensor.shape}"
        )

    def _init_trajectory_cache(self, num_envs: int, has_actions: bool, has_action_match: bool):
        self.current_trajectory_features = [[] for _ in range(num_envs)]
        self.current_trajectory_actions = [[] for _ in range(num_envs)] if has_actions else None
        self.current_trajectory_action_matches = [[] for _ in range(num_envs)] if has_action_match else None
        self.current_trajectory_returns = torch.zeros(num_envs, dtype=torch.float32)
        self.current_trajectory_success = torch.zeros(num_envs, dtype=torch.bool)

    def _collect_current_features(self) -> torch.Tensor:
        outputs = []
        for hook in self.feature_hooks:
            if hook.output is None:
                raise RuntimeError("Action-position feature hook has no output for the current forward pass.")
            outputs.append(hook.output.detach())
        feature_tensor = torch.stack(outputs, dim=1)
        if feature_tensor.ndim != 3:
            raise ValueError(
                f"Expected stacked action-position features to have shape [B, A, H], got {feature_tensor.shape}"
            )
        return feature_tensor.cpu()

    def cache_features_of_high_reward_after_each_forward_during_rollout(
        self,
        rewards,
        dones=None,
        action_mean=None,
        success=None,
        action_match=None,
    ):
        features = self._collect_current_features()
        rewards = rewards.detach().view(-1).cpu()
        actions = None if action_mean is None else action_mean.detach().cpu()
        match_flags = None if action_match is None else action_match.detach().view(-1).to(torch.bool).cpu()

        assert len(features) == len(rewards), "features和rewards的长度应该相同"
        if actions is not None:
            assert actions.ndim == 2, "当前VLA selector要求动作表征为二维 (B, A)"
            assert len(actions) == len(rewards), "actions和rewards的长度应该相同"
        if match_flags is not None:
            assert len(match_flags) == len(rewards), "action_match和rewards的长度应该相同"

        num_envs = len(rewards)

        if dones is None:
            for env_idx in range(num_envs):
                self.completed_trajectories.append(
                    {
                        "return": float(rewards[env_idx].item()),
                        "features": features[env_idx].unsqueeze(0),
                        "actions": None if actions is None else actions[env_idx].unsqueeze(0),
                        "action_match": None if match_flags is None else match_flags[env_idx].view(1),
                        "success_once": False if success is None else bool(success[env_idx].item()),
                        "episode_len": 1,
                    }
                )
            return

        dones = dones.detach().view(-1).to(torch.bool).cpu()
        assert len(dones) == num_envs, "dones和rewards的长度应该相同"
        if success is not None:
            success = success.detach().view(-1).to(torch.bool).cpu()
            assert len(success) == num_envs, "success和rewards的长度应该相同"

        if self.current_trajectory_features is None:
            self._init_trajectory_cache(
                num_envs,
                has_actions=actions is not None,
                has_action_match=match_flags is not None,
            )

        assert len(self.current_trajectory_features) == num_envs, "num_envs发生变化，当前FeatureSelector不支持动态切换"
        if (self.current_trajectory_actions is None) != (actions is None):
            raise ValueError("action_mean availability changed within one rollout, which is not supported")
        if (self.current_trajectory_action_matches is None) != (match_flags is None):
            raise ValueError("action_match availability changed within one rollout, which is not supported")

        for env_idx in range(num_envs):
            self.current_trajectory_features[env_idx].append(features[env_idx])
            if actions is not None:
                self.current_trajectory_actions[env_idx].append(actions[env_idx])
            if match_flags is not None:
                self.current_trajectory_action_matches[env_idx].append(bool(match_flags[env_idx].item()))
            self.current_trajectory_returns[env_idx] += rewards[env_idx]
            if success is not None:
                self.current_trajectory_success[env_idx] |= success[env_idx]

            if dones[env_idx]:
                self.completed_trajectories.append(
                    {
                        "return": float(self.current_trajectory_returns[env_idx].item()),
                        "features": torch.stack(self.current_trajectory_features[env_idx], dim=0),
                        "actions": None
                        if actions is None
                        else torch.stack(self.current_trajectory_actions[env_idx], dim=0),
                        "action_match": None
                        if match_flags is None
                        else torch.as_tensor(self.current_trajectory_action_matches[env_idx], dtype=torch.bool),
                        "success_once": bool(self.current_trajectory_success[env_idx].item()),
                        "episode_len": len(self.current_trajectory_features[env_idx]),
                    }
                )
                self.current_trajectory_features[env_idx] = []
                if actions is not None:
                    self.current_trajectory_actions[env_idx] = []
                if match_flags is not None:
                    self.current_trajectory_action_matches[env_idx] = []
                self.current_trajectory_returns[env_idx] = 0
                self.current_trajectory_success[env_idx] = False

    @staticmethod
    def _compute_selected_payload_stats(
        selected_trajectories,
        selected_trajectory_features,
        selected_trajectory_actions,
    ):
        total_feature_bytes = sum(
            int(feature.numel() * feature.element_size())
            for feature in selected_trajectory_features
        )
        total_action_bytes = 0
        if selected_trajectory_actions is not None:
            total_action_bytes = sum(
                int(action.numel() * action.element_size())
                for action in selected_trajectory_actions
            )

        total_forwards = 0
        matched_forwards = 0

        for item in selected_trajectories:
            action_match = item.get("action_match")

            if action_match is None:
                continue

            action_match = action_match.to(torch.bool).view(-1)
            if action_match.numel() == 0:
                continue

            num_steps = int(action_match.numel())
            matched_steps = int(action_match.sum().item())
            total_forwards += num_steps
            matched_forwards += matched_steps

        total_bytes = total_feature_bytes + total_action_bytes
        match_ratio = float(matched_forwards) / float(total_forwards) if total_forwards > 0 else 0.0
        matched_bytes = int(round(total_bytes * match_ratio))
        return {
            "total_bytes": int(total_bytes),
            "matched_bytes": int(matched_bytes),
            "effective_ratio": float(match_ratio if total_bytes > 0 else 0.0),
            "total_forwards": int(total_forwards),
            "matched_forwards": int(matched_forwards),
            "feature_bytes": int(total_feature_bytes),
            "action_bytes": int(total_action_bytes),
        }

    def select_message(self):
        if len(self.completed_trajectories) == 0:
            self.last_selected_payload_stats = None
            return None

        selected_trajectories = self._select_trajectories()
        if len(selected_trajectories) == 0:
            self.last_selected_payload_stats = None
            return None

        selected_trajectory_features = [
            self._temporal_pool_trajectory(item["features"]) for item in selected_trajectories
        ]
        if any("q_task" not in item for item in selected_trajectories):
            task_scores = self._compute_task_scores(selected_trajectories, update_running_max=True)
            for item, score in zip(selected_trajectories, task_scores.tolist()):
                item["q_task"] = float(score)
        selected_returns = torch.tensor(
            [float(item["return"]) for item in selected_trajectories],
            dtype=torch.float32,
        )
        selected_success_once = torch.tensor(
            [bool(item.get("success_once", False)) for item in selected_trajectories],
            dtype=torch.bool,
        )
        selected_episode_lens = torch.tensor(
            [int(item.get("episode_len", item["features"].shape[0])) for item in selected_trajectories],
            dtype=torch.long,
        )
        selected_q_task = torch.tensor(
            [float(item.get("q_task", 0.0)) for item in selected_trajectories],
            dtype=torch.float32,
        )
        has_actions = all(item["actions"] is not None for item in selected_trajectories)
        selected_trajectory_actions = None
        if has_actions:
            selected_trajectory_actions = [
                self._temporal_pool_trajectory(item["actions"]) for item in selected_trajectories
            ]
        self.last_selected_payload_stats = self._compute_selected_payload_stats(
            selected_trajectories,
            selected_trajectory_features,
            selected_trajectory_actions,
        )
        self.completed_trajectories = []

        message = {
            "feature": pad_sequence(
                selected_trajectory_features,
                batch_first=True,
                padding_value=0.0,
            ),
            "action": None,
            "meta": {
                "returns": selected_returns,
                "success_once": selected_success_once,
                "episode_lens": selected_episode_lens,
                "max_episode_steps": self.max_episode_steps,
                "q_task": selected_q_task,
            },
        }
        if selected_trajectory_actions is not None:
            message["action"] = pad_sequence(
                selected_trajectory_actions,
                batch_first=True,
                padding_value=0.0,
            )
        return message


class VLAClientForMultiAgent:
    def __init__(
        self,
        name: str,
        large_model,
        action_position_layer_prefix: str,
        action_position_actor_layer_prefix: str,
        local_feature_dim: int,
        num_action_positions: int,
        feature_selector_alpha: float,
        device=None,
        local_action_dim: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        feature_aggregator_attention_num_heads: int = 1,
        feature_aggregator_gate_type: str = "single-layer",
        feature_aggregator_gate_activation: str = "relu",
        feature_aggregator_norm_type: str = "none",
        feature_aggregator_feature_gate_open_max: float = 0.25,
        feature_aggregator_action_gate_open_max: float = 0.10,
        feature_aggregator_q_ret_weight: float = 0.85,
        feature_aggregator_q_attn_weight: float = 0.15,
        feature_aggregator_remote_dropout_prob: float = 0.0,
        feature_aggregator_remote_noise_std: float = 0.0,
        feature_aggregator_remote_stale_shift_max: int = 0,
        feature_selector_topk_trajectories: Optional[int] = None,
        feature_selector_temporal_pool_steps: Optional[int] = None,
        feature_selector_strategy: str = "topk_return",
        eval_feature_selector_strategy: Optional[str] = None,
    ):
        self.name = name
        self.large_model = large_model
        self.action_position_layer_prefix = action_position_layer_prefix
        self.action_position_actor_layer_prefix = action_position_actor_layer_prefix
        self.local_feature_dim = int(local_feature_dim)
        self.num_action_positions = int(num_action_positions)
        self.local_action_dim = self.num_action_positions if local_action_dim is None else int(local_action_dim)
        self.feature_selector_alpha = feature_selector_alpha
        self.feature_selector_topk_trajectories = feature_selector_topk_trajectories
        self.feature_selector_temporal_pool_steps = feature_selector_temporal_pool_steps
        self.train_feature_selector_strategy = feature_selector_strategy
        self.eval_feature_selector_strategy = (
            feature_selector_strategy if eval_feature_selector_strategy is None else eval_feature_selector_strategy
        )
        self.current_feature_selector_strategy = self.train_feature_selector_strategy
        self.max_episode_steps = max_episode_steps
        self.feature_aggregator_attention_num_heads = feature_aggregator_attention_num_heads
        self.feature_aggregator_gate_type = feature_aggregator_gate_type
        self.feature_aggregator_gate_activation = feature_aggregator_gate_activation
        self.feature_aggregator_norm_type = feature_aggregator_norm_type
        self.feature_aggregator_feature_gate_open_max = feature_aggregator_feature_gate_open_max
        self.feature_aggregator_action_gate_open_max = feature_aggregator_action_gate_open_max
        self.feature_aggregator_q_ret_weight = feature_aggregator_q_ret_weight
        self.feature_aggregator_q_attn_weight = feature_aggregator_q_attn_weight
        self.feature_aggregator_remote_dropout_prob = feature_aggregator_remote_dropout_prob
        self.feature_aggregator_remote_noise_std = feature_aggregator_remote_noise_std
        self.feature_aggregator_remote_stale_shift_max = feature_aggregator_remote_stale_shift_max
        self.device = device if device is not None else get_model_device(large_model)

        self.feature_selector = None
        self.feature_aggregators: Dict[str, FeatureAggregator] = {}
        self.pretrained_state_dict = {}
        self.pretrained_feature_selector_state = None

    @staticmethod
    def _merge_legacy_position_state_dicts(saved_states):
        if isinstance(saved_states, dict):
            return saved_states
        if not isinstance(saved_states, (list, tuple)):
            raise TypeError(f"Unsupported saved aggregator state type: {type(saved_states)}")
        if len(saved_states) == 0:
            return {}
        merged = {}
        first_state = saved_states[0]
        for key in first_state.keys():
            values = [state[key] for state in saved_states if key in state]
            if len(values) == 0:
                continue
            if torch.is_tensor(values[0]) and torch.is_floating_point(values[0]):
                merged[key] = torch.stack([value.float() for value in values], dim=0).mean(dim=0).to(values[0].dtype)
            else:
                merged[key] = values[0]
        return merged

    @staticmethod
    def _adapt_state_dict_for_shared_low_rank_module(state_dict, module: FeatureAggregator):
        target_state_dict = module.module.state_dict()
        if (
            "remote_feature_proj.weight" not in state_dict
            or "remote_feature_proj.left.weight" not in target_state_dict
            or "remote_feature_proj.right.weight" not in target_state_dict
        ):
            return state_dict

        adapted = dict(state_dict)
        full_weight = adapted.pop("remote_feature_proj.weight")
        full_bias = adapted.pop("remote_feature_proj.bias", None)

        left_template = target_state_dict["remote_feature_proj.left.weight"]
        right_template = target_state_dict["remote_feature_proj.right.weight"]
        right_bias_template = target_state_dict.get("remote_feature_proj.right.bias")

        weight_fp32 = full_weight.float()
        u, s, vh = torch.linalg.svd(weight_fp32, full_matrices=False)
        rank = min(left_template.shape[0], s.numel())
        sqrt_s = s[:rank].sqrt()

        left_weight = torch.zeros_like(left_template, dtype=torch.float32)
        right_weight = torch.zeros_like(right_template, dtype=torch.float32)
        left_weight[:rank] = sqrt_s.unsqueeze(1) * vh[:rank]
        right_weight[:, :rank] = u[:, :rank] * sqrt_s.unsqueeze(0)

        adapted["remote_feature_proj.left.weight"] = left_weight.to(left_template.dtype)
        adapted["remote_feature_proj.right.weight"] = right_weight.to(right_template.dtype)
        if right_bias_template is not None:
            if full_bias is None:
                adapted["remote_feature_proj.right.bias"] = torch.zeros_like(right_bias_template)
            else:
                adapted["remote_feature_proj.right.bias"] = full_bias.to(right_bias_template.dtype)
        return adapted

    def before_training_start(self, small_model):
        self.feature_selector = VLAActionPositionFeatureSelector(
            small_model,
            self.action_position_layer_prefix,
            num_action_positions=self.num_action_positions,
            alpha=self.feature_selector_alpha,
            max_trajectory_count=self.feature_selector_topk_trajectories,
            temporal_pool_steps=self.feature_selector_temporal_pool_steps,
            selection_strategy=self.current_feature_selector_strategy,
            max_episode_steps=self.max_episode_steps,
        )
        self.feature_selector.load_runtime_state(self.pretrained_feature_selector_state)
        print(f"Client {self.name} initialized (VLA action-position communication mode)")
        return {
            "local_feature_dim": self.local_feature_dim,
            "num_action_positions": self.num_action_positions,
            "local_action_dim": self.local_action_dim,
        }

    @torch.no_grad()
    def after_each_forward_during_rollout(self, rewards, dones=None, action_mean=None, success=None, action_match=None):
        self.feature_selector.cache_features_of_high_reward_after_each_forward_during_rollout(
            rewards,
            dones=dones,
            action_mean=action_mean,
            success=success,
            action_match=action_match,
        )

    def reset_feature_selector_cache(self):
        if self.feature_selector is not None:
            self.feature_selector.reset_cache()

    def set_feature_selector_strategy(self, strategy: str):
        self.current_feature_selector_strategy = strategy
        if self.feature_selector is not None:
            self.feature_selector.selection_strategy = strategy

    def use_train_feature_selector_strategy(self):
        self.set_feature_selector_strategy(self.train_feature_selector_strategy)

    def use_eval_feature_selector_strategy(self):
        self.set_feature_selector_strategy(self.eval_feature_selector_strategy)

    def export_feature_and_action(self):
        if self.feature_selector is None:
            return None
        msg = self.feature_selector.select_message()
        if msg is None or msg.get("feature") is None:
            return None
        return {
            "client_id": self.name,
            "feature": msg["feature"],
            "action": msg.get("action"),
            "meta": msg.get("meta"),
        }

    def debug_last_selected_payload_stats(self):
        if self.feature_selector is None:
            return None
        return self.feature_selector.last_selected_payload_stats

    def receive_feature_and_action(self, sender_name: str, feature_action_msg):
        if feature_action_msg is None:
            return
        if sender_name not in self.feature_aggregators:
            return

        remote_feature = feature_action_msg.get("feature")
        remote_action = feature_action_msg.get("action")
        remote_meta = feature_action_msg.get("meta")
        if remote_feature is None:
            self.feature_aggregators[sender_name].set_remote_message({"feature": None, "action": None, "meta": None})
            return

        if remote_feature.ndim != 4:
            raise ValueError(
                f"Expected remote VLA feature message to have shape [traj, step, action_pos, hidden], got {remote_feature.shape}"
            )

        remote_action_positions = remote_feature.shape[2]
        if remote_action_positions != self.num_action_positions:
            aligned_feature = remote_feature.new_zeros(
                remote_feature.shape[0],
                remote_feature.shape[1],
                self.num_action_positions,
                remote_feature.shape[3],
            )
            shared_positions = min(remote_action_positions, self.num_action_positions)
            aligned_feature[:, :, :shared_positions] = remote_feature[:, :, :shared_positions]
            remote_feature = aligned_feature

        if remote_action is not None and remote_action.ndim != 3:
            raise ValueError(
                f"Expected remote VLA action message to have shape [traj, step, action_dim], got {remote_action.shape}"
            )

        self.feature_aggregators[sender_name].set_remote_message(
            {
                "feature": remote_feature,
                "action": remote_action,
                "meta": remote_meta,
            }
        )

    def clear_messages(self):
        for aggregator in self.feature_aggregators.values():
            aggregator.set_remote_message({"feature": None, "action": None, "meta": None})

    def snapshot_remote_messages(self):
        snapshots = {}
        for remote_name, aggregator in self.feature_aggregators.items():
            snapshots[remote_name] = aggregator.snapshot_remote_message()
        return snapshots

    def restore_remote_messages(self, snapshots):
        if snapshots is None:
            return
        for remote_name, aggregator_snapshot in snapshots.items():
            if remote_name not in self.feature_aggregators:
                continue
            self.feature_aggregators[remote_name].restore_remote_message(aggregator_snapshot)

    def add_feature_aggregator(self, client_name: str, client_info: dict):
        if client_name == self.name:
            return
        if client_name in self.feature_aggregators:
            return

        remote_dim = int(client_info["local_feature_dim"])
        remote_action_dim = int(client_info.get("local_action_dim", self.local_action_dim))
        layer_names = [
            f"{self.action_position_layer_prefix}.{action_idx}"
            for action_idx in range(self.num_action_positions)
        ]
        actor_layer_names = [
            f"{self.action_position_actor_layer_prefix}.{action_idx}"
            for action_idx in range(self.num_action_positions)
        ]
        aggregator = FeatureAggregator(
            self.large_model,
            layer_names[0],
            self.local_feature_dim,
            remote_dim,
            remote_action_dim=remote_action_dim,
            actor_layer_name=actor_layer_names[0],
            action_position_layer_names=layer_names,
            action_position_actor_layer_names=actor_layer_names,
            attention_num_heads=self.feature_aggregator_attention_num_heads,
            gate_type=self.feature_aggregator_gate_type,
            gate_activation=self.feature_aggregator_gate_activation,
            norm_type=self.feature_aggregator_norm_type,
            feature_gate_open_max=self.feature_aggregator_feature_gate_open_max,
            action_gate_open_max=self.feature_aggregator_action_gate_open_max,
            q_ret_weight=self.feature_aggregator_q_ret_weight,
            q_attn_weight=self.feature_aggregator_q_attn_weight,
            remote_dropout_prob=self.feature_aggregator_remote_dropout_prob,
            remote_noise_std=self.feature_aggregator_remote_noise_std,
            remote_stale_shift_max=self.feature_aggregator_remote_stale_shift_max,
        )
        aggregator.module.to(self.device)

        self.feature_aggregators[client_name] = aggregator
        if client_name in getattr(self, "pretrained_state_dict", {}):
            saved_states = self.pretrained_state_dict[client_name]
            merged_state_dict = self._merge_legacy_position_state_dicts(saved_states)
            merged_state_dict = self._adapt_state_dict_for_shared_low_rank_module(merged_state_dict, aggregator)
            aggregator.module.load_state_dict(merged_state_dict)
            print(f"Client {self.name} loaded pretrained VLA feature aggregators for client {client_name}")

    def get_feature_aggregators_parameters(self):
        params = {}
        for remote_name, aggregator in self.feature_aggregators.items():
            params[remote_name] = aggregator.module.parameters()
        return params

    def save_feature_aggregators(self, path):
        state = {
            "feature_selector_runtime_state": None
            if self.feature_selector is None
            else self.feature_selector.get_runtime_state(),
            "aggregators": {
                remote_name: aggregator.module.state_dict()
                for remote_name, aggregator in self.feature_aggregators.items()
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(_cast_floating_tensors_for_save(state, torch.bfloat16), path)

    def load_feature_aggregators(self, path):
        state_dict = torch.load(path)
        if isinstance(state_dict, dict) and "aggregators" in state_dict:
            self.pretrained_state_dict = state_dict["aggregators"]
            self.pretrained_feature_selector_state = state_dict.get("feature_selector_runtime_state")
        else:
            self.pretrained_state_dict = state_dict
            self.pretrained_feature_selector_state = None

    def debug_feature_aggregators(self):
        outputs = {}
        for remote_name, aggregator in self.feature_aggregators.items():
            outputs[remote_name] = {
                "feature": aggregator.module.cached_feature_g,
                "action": aggregator.module.cached_action_g,
            }
        return outputs

    @staticmethod
    def _mean_position_summary(summaries):
        valid = [tensor.float() for tensor in summaries if tensor is not None]
        if not valid:
            return None
        return torch.stack(valid, dim=0).mean(dim=0)

    def debug_feature_aggregator_feature_summaries(self):
        outputs = {}
        for remote_name, aggregator in self.feature_aggregators.items():
            outputs[remote_name] = {
                "feature_local": aggregator.module.cached_feature_local_summary,
                "feature_fused": aggregator.module.cached_feature_fused_summary,
                "action_local": aggregator.module.cached_action_local_summary,
                "action_fused": aggregator.module.cached_action_fused_summary,
            }
        return outputs

    def eval(self):
        for aggregator in self.feature_aggregators.values():
            aggregator.module.eval()

    def train(self):
        for aggregator in self.feature_aggregators.values():
            aggregator.module.train()
