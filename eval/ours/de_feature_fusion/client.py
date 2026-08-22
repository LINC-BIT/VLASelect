import torch
import requests
from torch import nn
from ours.utils.dl.common.model import LayerActivation3, get_module, get_model_device
from typing import Self
import queue
import threading
import time

from .feature_selector import FeatureSelector
from .feature_aggregator import FeatureAggregator
from .comm_utils import serialize_tensor, deserialize_tensor


class Client:
    def __init__(self,
                 name: str,
                 large_model: nn.Module,
                 layer_name_of_output_features: str,
                 local_feature_dim: int,
                 feature_selector_alpha: float,
                 data_manager_url: str,
                 device=None,
                 local_action_dim: int = None,
                 actor_layer_name: str = 'actor_mean',
                 max_episode_steps: int = None,
                 feature_aggregator_attention_num_heads: int = 1,
                 feature_aggregator_gate_type: str = 'single-layer',
                 feature_aggregator_gate_activation: str = 'relu',
                 feature_aggregator_norm_type: str = 'none',
                 feature_aggregator_feature_gate_open_max: float = 0.25,
                 feature_aggregator_action_gate_open_max: float = 0.10,
                 feature_aggregator_q_ret_weight: float = 0.85,
                 feature_aggregator_q_attn_weight: float = 0.15,
                 feature_aggregator_remote_dropout_prob: float = 0.0,
                 feature_aggregator_remote_noise_std: float = 0.0,
                 feature_aggregator_remote_stale_shift_max: int = 0,
                 feature_selector_topk_trajectories: int = None,
                 feature_selector_temporal_pool_steps: int = None,
                 feature_selector_strategy: str = 'topk_return'):

        self.name = name
        self.large_model = large_model
        self.layer_name_of_output_features = layer_name_of_output_features
        self.local_feature_dim = local_feature_dim
        self.data_manager_url = data_manager_url
        self.feature_selector_alpha = feature_selector_alpha
        self.feature_selector_topk_trajectories = feature_selector_topk_trajectories
        self.feature_selector_temporal_pool_steps = feature_selector_temporal_pool_steps
        self.feature_selector_strategy = feature_selector_strategy
        self.local_action_dim = local_action_dim
        self.actor_layer_name = actor_layer_name
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

        self.small_model = None
        self.feature_selector = None
        self.feature_aggregators = {} # 其它客户端的name: 面向其它客户端的feature_aggregator
        self.device = device if device is not None else get_model_device(large_model)

        self.comm_queue = queue.Queue()
        self.comm_thread = None
        self._closed = False
        self.pretrained_state_dict = {}
        self.pretrained_feature_selector_state = None
        
    def before_training_start(self, small_model: nn.Module):
        if self.local_action_dim is None:
            if hasattr(small_model, 'actor_logstd'):
                self.local_action_dim = int(small_model.actor_logstd.shape[-1])
            elif hasattr(small_model, 'actor_mean') and isinstance(getattr(small_model, 'actor_mean'), nn.Sequential):
                self.local_action_dim = int(small_model.actor_mean[-1].out_features)
            else:
                self.local_action_dim = 0
        # 向数据管理器提交自己的信息
        requests.post(f'{self.data_manager_url}/register_client', json={
            'name': self.name,
            'local_feature_dim': self.local_feature_dim,
            'local_action_dim': self.local_action_dim,
        })

        self.small_model = small_model
        self.feature_selector = FeatureSelector(
            self.small_model,
            self.layer_name_of_output_features,
            alpha=self.feature_selector_alpha,
            max_trajectory_count=self.feature_selector_topk_trajectories,
            temporal_pool_steps=self.feature_selector_temporal_pool_steps,
            selection_strategy=self.feature_selector_strategy,
            max_episode_steps=self.max_episode_steps,
        )
        self.feature_selector.load_runtime_state(self.pretrained_feature_selector_state)
        
        # 启动通信进程
        self.comm_thread = threading.Thread(target=self._comm_worker, daemon=True)
        self.comm_thread.start()

        print(f'Client {self.name} is ready for training with small model and feature selector/aggregators')

    @torch.no_grad()
    def after_each_forward_during_rollout(self, rewards, dones=None, action_mean=None, success=None):
        """
        rollout中每次模型做forward后调用，用于筛选和缓存高reward的特征。
        """
        # print(f'Client {self.name} selects and caches features based on rewards')
        self.feature_selector.cache_features_of_high_reward_after_each_forward_during_rollout(
            rewards,
            dones=dones,
            action_mean=action_mean,
            success=success,
        )

    def _comm_worker(self):
        while True:
            # 从数据管理器获取其它客户端的信息，并初始化面向其它客户端的feature_aggregator
            clients_info = requests.get(f'{self.data_manager_url}/clients_info').json()
            # print(clients_info)
            for client_name, client_info in clients_info.items():
                if client_name != self.name and client_name not in self.feature_aggregators:
                    self.feature_aggregators[client_name] = FeatureAggregator(self.small_model, 
                                                                    self.layer_name_of_output_features, 
                                                                    self.local_feature_dim, 
                                                                    client_info['local_feature_dim'],
                                                                    remote_action_dim=client_info.get('local_action_dim', self.local_action_dim or 0),
                                                                    actor_layer_name=self.actor_layer_name,
                                                                    attention_num_heads=self.feature_aggregator_attention_num_heads,
                                                                    gate_type=self.feature_aggregator_gate_type,
                                                                    gate_activation=self.feature_aggregator_gate_activation,
                                                                    norm_type=self.feature_aggregator_norm_type,
                                                                    feature_gate_open_max=self.feature_aggregator_feature_gate_open_max,
                                                                    action_gate_open_max=self.feature_aggregator_action_gate_open_max,
                                                                    q_ret_weight=self.feature_aggregator_q_ret_weight,
                                                                    q_attn_weight=self.feature_aggregator_q_attn_weight)
                    self.feature_aggregators[client_name].module.to(self.device)
                    print(f'Client {self.name} initializes feature aggregator for client {client_name}')
                    print(client_name, getattr(self, 'pretrained_state_dict', {}).keys())
                    if client_name in getattr(self, 'pretrained_state_dict', {}):
                        self.feature_aggregators[client_name].module.load_state_dict(self.pretrained_state_dict[client_name])
                        print(f'Client {self.name} loaded pretrained feature aggregator for client {client_name}')
                    
            comm_task = self.comm_queue.get()

            if comm_task == 'STOP':
                break

            elif comm_task == 'COMM':
                local_message = self.feature_selector.select_message()
                if local_message is None or local_message['feature'] is None:
                    print(f'Client {self.name} has no feature to upload')
                    continue
                local_message = serialize_tensor(local_message)
                # print({
                #     'client_id': self.name,
                #     'feature': local_feature
                # })
                requests.post(f'{self.data_manager_url}/upload_feature', json={
                    'client_id': self.name,
                    'feature': local_message
                })
                print(f'Client {self.name} uploads its selected feature to data manager')

                remote_features = {}
                for client_name in self.feature_aggregators.keys():
                    response = requests.get(f'{self.data_manager_url}/get_feature', params={
                        'client_id': client_name
                    })
                    remote_features[client_name] = response.json()['feature']
                    if remote_features[client_name] is not None:
                        remote_features[client_name] = deserialize_tensor(remote_features[client_name])
                        self.feature_aggregators[client_name].set_remote_message(remote_features[client_name])
                        print(f'Client {self.name} gets feature from client {client_name} and updates its feature aggregator')

            time.sleep(1)

    def refresh_features(self):
        """
        上传自己的最新特征、并从其它客户端获取最新特征
        """
        print(f'Client {self.name} refreshes features from other clients')
        self.comm_queue.put('COMM')

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.comm_thread is not None and self.comm_thread.is_alive():
            self.comm_queue.put('STOP')
            self.comm_thread.join(timeout=5.0)

    def get_feature_aggregators_parameters(self):
        """
        获取面向其它客户端的feature_aggregator的参数，用于更新feature_aggregator。
        由于feature_aggregator会不定时的被新建，
        因此该函数最好在训练中不断的调用，以检查是否存在新增feature_aggregators、并及时将其参数加入优化器
        """
        params = {}
        for key, feature_aggregator in self.feature_aggregators.items():
            params[key] = feature_aggregator.module.parameters()
        return params

    def report_metrics(self, time: int, metrics: dict):
        """
        向数据管理器上报指标，数据管理器会把这些指标转发给其它客户端，供它们做分析和可视化。
        """
        requests.post(f'{self.data_manager_url}/report_metrics', json={
            'client_id': self.name,
            'time': time,
            'metrics': metrics
        })

    def save_feature_aggregators(self, path):
        """
        保存面向其它客户端的feature_aggregator的参数
        """
        state_dict = {
            "feature_selector_runtime_state": None if self.feature_selector is None else self.feature_selector.get_runtime_state(),
            "aggregators": {},
        }
        for key, feature_aggregator in self.feature_aggregators.items():
            state_dict["aggregators"][key] = feature_aggregator.module.state_dict()
        torch.save(state_dict, path)

    def load_feature_aggregators(self, path):
        """
        加载面向其它客户端的feature_aggregator的参数
        """
        state_dict = torch.load(path)
        if isinstance(state_dict, dict) and "aggregators" in state_dict:
            self.pretrained_state_dict = state_dict["aggregators"]
            self.pretrained_feature_selector_state = state_dict.get("feature_selector_runtime_state")
        else:
            self.pretrained_state_dict = state_dict
            self.pretrained_feature_selector_state = None
        # for key, feature_aggregator in self.feature_aggregators.items():
        #     if key in state_dict:
        #         feature_aggregator.module.load_state_dict(state_dict[key])
        #         print(f'Client {self.name} loaded feature aggregator for {key}')

    def debug_feature_aggregators(self):
        """
        打印当前feature_aggregator的门控权重分布，供调试使用
        """
        cached_gs = {}
        for key, feature_aggregator in self.feature_aggregators.items():
            cached_gs[key] = {
                'feature': feature_aggregator.module.cached_feature_g,
                'action': feature_aggregator.module.cached_action_g,
            }
        return cached_gs

class ClientForMultiAgent:
    def __init__(self,
                 name: str,
                 large_model: nn.Module,
                 layer_name_of_output_features: str,
                 local_feature_dim: int,
                 feature_selector_alpha: float,
                 device=None,
                 local_action_dim: int = None,
                 actor_layer_name: str = 'actor_mean',
                 max_episode_steps: int = None,
                 feature_aggregator_attention_num_heads: int = 1,
                 feature_aggregator_gate_type: str = 'single-layer',
                 feature_aggregator_gate_activation: str = 'relu',
                 feature_aggregator_norm_type: str = 'none',
                 feature_aggregator_feature_gate_open_max: float = 0.25,
                 feature_aggregator_action_gate_open_max: float = 0.10,
                 feature_aggregator_q_ret_weight: float = 0.85,
                 feature_aggregator_q_attn_weight: float = 0.15,
                 feature_aggregator_remote_dropout_prob: float = 0.0,
                 feature_aggregator_remote_noise_std: float = 0.0,
                 feature_aggregator_remote_stale_shift_max: int = 0,
                 feature_selector_topk_trajectories: int = None,
                 feature_selector_temporal_pool_steps: int = None,
                 feature_selector_strategy: str = 'topk_return',
                 eval_feature_selector_strategy: str = None):

        self.name = name
        self.large_model = large_model
        self.layer_name_of_output_features = layer_name_of_output_features
        self.local_feature_dim = local_feature_dim

        self.feature_selector_alpha = feature_selector_alpha
        self.feature_selector_topk_trajectories = feature_selector_topk_trajectories
        self.feature_selector_temporal_pool_steps = feature_selector_temporal_pool_steps
        self.train_feature_selector_strategy = feature_selector_strategy
        self.eval_feature_selector_strategy = (
            feature_selector_strategy
            if eval_feature_selector_strategy is None else eval_feature_selector_strategy
        )
        self.current_feature_selector_strategy = self.train_feature_selector_strategy

        self.local_action_dim = local_action_dim
        self.actor_layer_name = actor_layer_name
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

        self.small_model = None
        self.feature_selector = None

        # key: other client name
        self.feature_aggregators = {}

        self.device = device if device is not None else get_model_device(large_model)
        self.pretrained_state_dict = {}
        self.pretrained_feature_selector_state = None

    # --------------------------------------------------
    # init
    # --------------------------------------------------
    def before_training_start(self, small_model: nn.Module):

        self.small_model = small_model

        if self.local_action_dim is None:
            if hasattr(small_model, 'actor_logstd'):
                self.local_action_dim = int(small_model.actor_logstd.shape[-1])
            elif hasattr(small_model, 'actor_mean') and isinstance(getattr(small_model, 'actor_mean'), nn.Sequential):
                self.local_action_dim = int(small_model.actor_mean[-1].out_features)
            else:
                self.local_action_dim = 0

        self.feature_selector = FeatureSelector(
            self.small_model,
            self.layer_name_of_output_features,
            alpha=self.feature_selector_alpha,
            max_trajectory_count=self.feature_selector_topk_trajectories,
            temporal_pool_steps=self.feature_selector_temporal_pool_steps,
            selection_strategy=self.current_feature_selector_strategy,
            max_episode_steps=self.max_episode_steps,
        )
        self.feature_selector.load_runtime_state(self.pretrained_feature_selector_state)

        print(f'Client {self.name} initialized (function-based communication mode)')
        
        client_info = {
            'local_feature_dim': self.local_feature_dim,
            'local_action_dim': self.local_action_dim,
        }
        return client_info

    # --------------------------------------------------
    # rollout hook
    # --------------------------------------------------
    @torch.no_grad()
    def after_each_forward_during_rollout(self, rewards, dones=None, action_mean=None, success=None):
        self.feature_selector.cache_features_of_high_reward_after_each_forward_during_rollout(
            rewards,
            dones=dones,
            action_mean=action_mean,
            success=success,
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

    # --------------------------------------------------
    # 1. EXPORT FEATURE (replacement of upload)
    # --------------------------------------------------
    def export_feature(self):
        """
        输出当前筛选出的 feature（供外部 client 调用）
        """
        msg = self.feature_selector.select_message()

        if msg is None or msg.get('feature', None) is None:
            return None

        return {
            "client_id": self.name,
            "feature": msg["feature"],
            "meta": msg.get("meta", None)
        }
    
    def export_feature_and_action(self):
        """
        输出当前筛选出的 feature（供外部 client 调用）
        """
        msg = self.feature_selector.select_message()

        if msg is None or msg.get('feature', None) is None:
            return None

        return {
            "client_id": self.name,
            "feature": msg["feature"],
            "action": msg.get("action", None),
            "meta": msg.get("meta", None)
        }

    # --------------------------------------------------
    # 2. RECEIVE FEATURE (replacement of download)
    # --------------------------------------------------
    def receive_feature(self, sender_name: str, feature_msg):
        """
        接收其他 client 的 feature
        """
        if feature_msg is None:
            return

        if sender_name not in self.feature_aggregators:
            # 如果没有 aggregator，直接忽略或延迟初始化
            return

        self.feature_aggregators[sender_name].set_remote_message(
            feature_msg["feature"]
        )

    def receive_feature_and_action(self, sender_name: str, feature_action_msg):
        """
        接收其他 client 的 feature 和 action
        """
        if feature_action_msg is None:
            return

        if sender_name not in self.feature_aggregators:
            # 如果没有 aggregator，直接忽略或延迟初始化
            return

        self.feature_aggregators[sender_name].set_remote_message(
            feature_action_msg
        )

    def clear_messages(self):
        for fa in self.feature_aggregators.values():
            fa.set_remote_message({"feature": None, "action": None})

    # --------------------------------------------------
    # aggregator registration
    # --------------------------------------------------
    def add_feature_aggregator(self, client_name: str, client_info: dict):

        if client_name == self.name:
            return

        if client_name in self.feature_aggregators:
            return

        self.feature_aggregators[client_name] = FeatureAggregator(
            self.small_model,
            self.layer_name_of_output_features,
            self.local_feature_dim,
            client_info['local_feature_dim'],
            remote_action_dim=client_info.get('local_action_dim', self.local_action_dim or 0),
            actor_layer_name=self.actor_layer_name,
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

        self.feature_aggregators[client_name].module.to(self.device)
        print(f'Client {self.name} initializes feature aggregator for client {client_name}')
        print(client_name, getattr(self, 'pretrained_state_dict', {}).keys())
        if client_name in getattr(self, 'pretrained_state_dict', {}):
            self.feature_aggregators[client_name].module.load_state_dict(self.pretrained_state_dict[client_name])
            print(f'Client {self.name} loaded pretrained feature aggregator for client {client_name}')

    # --------------------------------------------------
    # parameter access
    # --------------------------------------------------
    def get_feature_aggregators_parameters(self):
        return {
            k: v.module.parameters()
            for k, v in self.feature_aggregators.items()
        }

    # --------------------------------------------------
    # save/load
    # --------------------------------------------------
    def save_feature_aggregators(self, path):
        torch.save(
            {
                "feature_selector_runtime_state": None if self.feature_selector is None else self.feature_selector.get_runtime_state(),
                "aggregators": {k: v.module.state_dict() for k, v in self.feature_aggregators.items()},
            },
            path
        )

    def load_feature_aggregators(self, path):
        """
        加载面向其它客户端的feature_aggregator的参数
        """
        state_dict = torch.load(path)
        if isinstance(state_dict, dict) and "aggregators" in state_dict:
            self.pretrained_state_dict = state_dict["aggregators"]
            self.pretrained_feature_selector_state = state_dict.get("feature_selector_runtime_state")
        else:
            self.pretrained_state_dict = state_dict
            self.pretrained_feature_selector_state = None
        # for key, feature_aggregator in self.feature_aggregators.items():
        #     if key in state_dict:
        #         feature_aggregator.module.load_state_dict(state_dict[key])
        #         print(f'Client {self.name} loaded feature aggregator for {key}')

    # --------------------------------------------------
    # debug
    # --------------------------------------------------
    def debug_feature_aggregators(self):
        return {
            k: {
                "feature": v.module.cached_feature_g,
                "action": v.module.cached_action_g
            }
            for k, v in self.feature_aggregators.items()
        }

    def debug_feature_aggregator_feature_summaries(self):
        return {
            k: {
                "feature_local": v.module.cached_feature_local_summary,
                "feature_fused": v.module.cached_feature_fused_summary,
                "action_local": v.module.cached_action_local_summary,
                "action_fused": v.module.cached_action_fused_summary,
            }
            for k, v in self.feature_aggregators.items()
        }
    
    def eval(self):
        for fa in self.feature_aggregators.values():
            fa.module.eval()

    def train(self):
        for fa in self.feature_aggregators.values():
            fa.module.train()
