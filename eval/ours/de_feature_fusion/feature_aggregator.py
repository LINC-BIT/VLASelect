import torch
from torch import nn
import torch.nn.functional as F
import random
import copy
from ours.utils.dl.common.model import LayerActivation3, get_module, get_model_device


class LowRankLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = None, bias: bool = True):
        super().__init__()
        if rank is None:
            rank = min(input_dim, output_dim, 512)
        rank = max(1, min(int(rank), input_dim, output_dim))
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = rank
        self.left = nn.Linear(self.input_dim, self.rank, bias=False)
        self.right = nn.Linear(self.rank, self.output_dim, bias=bias)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))

    def reset_parameters(self):
        nn.init.normal_(self.left.weight, mean=0.0, std=1e-2)
        nn.init.normal_(self.right.weight, mean=0.0, std=1e-3)
        if self.right.bias is not None:
            nn.init.zeros_(self.right.bias)


class FeatureAggregatorModule(nn.Module):
    """
    使用本地特征作为query、其它客户端发送过来的远程特征作为key/value，
    基于Attention机制去查询与本地特征最相关的远程特征，并将其与本地特征进行融合，
    以提高本地特征的质量
    """

    def __init__(self,
                 local_feature_dim: int,
                 remote_feature_dim: int,
                 remote_action_dim: int = 0,
                 remote_feature_proj_rank: int = None,
                 attention_num_heads: int = 1,
                 gate_type: str = 'single-layer',
                 gate_activation: str = 'relu',
                 norm_type: str = 'none',
                 feature_gate_open_max: float = 0.25,
                 action_gate_open_max: float = 0.10,
                 q_ret_weight: float = 0.85,
                 q_attn_weight: float = 0.15):

        super().__init__()

        self.local_feature_dim = local_feature_dim
        self.remote_feature_dim = remote_feature_dim
        self.remote_action_dim = remote_action_dim
        self.attention_num_heads = attention_num_heads
        self.gate_type = gate_type
        self.gate_activation = gate_activation
        self.norm_type = norm_type
        self.feature_gate_open_max = float(feature_gate_open_max)
        self.action_gate_open_max = float(action_gate_open_max)
        q_ret_weight = float(q_ret_weight)
        q_attn_weight = float(q_attn_weight)
        q_weight_sum = q_ret_weight + q_attn_weight
        if q_weight_sum <= 0:
            raise ValueError("q_ret_weight + q_attn_weight must be > 0")
        self.q_ret_weight = q_ret_weight / q_weight_sum
        self.q_attn_weight = q_attn_weight / q_weight_sum

        if local_feature_dim % attention_num_heads != 0:
            raise ValueError(
                f"local_feature_dim ({local_feature_dim}) must be divisible by "
                f"attention_num_heads ({attention_num_heads})"
            )

        self.remote_feature_proj = LowRankLinear(
            remote_feature_dim,
            local_feature_dim,
            rank=remote_feature_proj_rank,
        )
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=local_feature_dim,
            num_heads=attention_num_heads,
            bias=True,
            batch_first=True
        )
        self.feature_local_norm = self._build_norm(local_feature_dim)
        self.feature_remote_norm = self._build_norm(local_feature_dim)
        self.feature_attn_output_norm = self._build_norm(local_feature_dim)
        self.feature_gate = self._build_gate(local_feature_dim * 2, local_feature_dim)
        self.remote_action_proj = None
        self.action_attention = None
        self.action_gate = None
        self.action_output_proj = None
        self.action_local_norm = None
        self.action_remote_feature_norm = None
        self.action_remote_action_norm = None
        self.action_context_norm = None
        if remote_action_dim is not None and remote_action_dim > 0:
            self.remote_action_proj = nn.Linear(remote_action_dim, local_feature_dim)
            self.action_attention = nn.MultiheadAttention(
                embed_dim=local_feature_dim,
                num_heads=attention_num_heads,
                bias=True,
                batch_first=True
            )
            self.action_local_norm = self._build_norm(local_feature_dim)
            self.action_remote_feature_norm = self._build_norm(local_feature_dim)
            self.action_remote_action_norm = self._build_norm(local_feature_dim)
            self.action_context_norm = self._build_norm(local_feature_dim)
            self.action_gate = self._build_gate(local_feature_dim * 2, local_feature_dim)
            self.action_output_proj = nn.Linear(local_feature_dim, local_feature_dim)

        # Don't zero-init attention output - allow non-zero output so gradients
        # can flow to the gate. The gate controls the contribution magnitude.
        # nn.init.constant_(self.attention.out_proj.weight, 0)
        # nn.init.constant_(self.attention.out_proj.bias, 0)

        self._init_small_aggregator_weights()
        self._init_gate(self.feature_gate)
        if self.action_gate is not None:
            self._init_gate(self.action_gate)

        self.cached_g = None # 方便调试时观察门控权重的分布
        self.cached_feature_g = None
        self.cached_action_g = None
        self.gate_mean = None # 保留在计算图中，用于gate正则化损失
        self.gate_std = None
        self.feature_gate_mean = None
        self.feature_gate_std = None
        self.action_gate_mean = None
        self.action_gate_std = None
        self.feature_consistency_loss = None
        self.action_consistency_loss = None
        self.feature_gate_quality_loss = None
        self.action_gate_quality_loss = None
        self.feature_attn_entropy_loss = None
        self.action_attn_entropy_loss = None
        self.feature_attn_diversity_loss = None
        self.action_attn_diversity_loss = None
        self.feature_attn_im_loss = None
        self.action_attn_im_loss = None
        self.feature_q_mean = None
        self.action_q_mean = None
        self.feature_q_ret_mean = None
        self.action_q_ret_mean = None
        self.feature_q_attn_mean = None
        self.action_q_attn_mean = None
        self.feature_q_attn_token_mean = None
        self.action_q_attn_token_mean = None
        self.feature_q_attn_traj_mean = None
        self.action_q_attn_traj_mean = None
        self.cached_feature_local_summary = None
        self.cached_feature_fused_summary = None
        self.cached_action_local_summary = None
        self.cached_action_fused_summary = None

    @staticmethod
    def _runtime_scalar_field_names():
        return (
            "gate_mean",
            "gate_std",
            "feature_gate_mean",
            "feature_gate_std",
            "action_gate_mean",
            "action_gate_std",
            "feature_consistency_loss",
            "action_consistency_loss",
            "feature_gate_quality_loss",
            "action_gate_quality_loss",
            "feature_attn_entropy_loss",
            "action_attn_entropy_loss",
            "feature_attn_diversity_loss",
            "action_attn_diversity_loss",
            "feature_attn_im_loss",
            "action_attn_im_loss",
            "feature_q_mean",
            "action_q_mean",
            "feature_q_ret_mean",
            "action_q_ret_mean",
            "feature_q_attn_mean",
            "action_q_attn_mean",
            "feature_q_attn_token_mean",
            "action_q_attn_token_mean",
            "feature_q_attn_traj_mean",
            "action_q_attn_traj_mean",
        )

    @staticmethod
    def _runtime_summary_field_names():
        return (
            "cached_feature_local_summary",
            "cached_feature_fused_summary",
            "cached_action_local_summary",
            "cached_action_fused_summary",
        )

    @classmethod
    def _feature_runtime_field_names(cls):
        return (
            "cached_g",
            "cached_feature_g",
            "feature_gate_mean",
            "feature_gate_std",
            "feature_consistency_loss",
            "feature_gate_quality_loss",
            "feature_attn_entropy_loss",
            "feature_attn_diversity_loss",
            "feature_attn_im_loss",
            "feature_q_mean",
            "feature_q_ret_mean",
            "feature_q_attn_mean",
            "feature_q_attn_token_mean",
            "feature_q_attn_traj_mean",
            "cached_feature_local_summary",
            "cached_feature_fused_summary",
        )

    @classmethod
    def _action_runtime_field_names(cls):
        return (
            "cached_action_g",
            "gate_mean",
            "gate_std",
            "action_gate_mean",
            "action_gate_std",
            "action_consistency_loss",
            "action_gate_quality_loss",
            "action_attn_entropy_loss",
            "action_attn_diversity_loss",
            "action_attn_im_loss",
            "action_q_mean",
            "action_q_ret_mean",
            "action_q_attn_mean",
            "action_q_attn_token_mean",
            "action_q_attn_traj_mean",
            "cached_action_local_summary",
            "cached_action_fused_summary",
        )

    @staticmethod
    def _mean_tensor_list(values):
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return torch.stack(valid, dim=0).mean(dim=0)

    @staticmethod
    def _stack_tensor_list(values):
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return torch.stack(valid, dim=0)

    def export_runtime_state(self):
        state = {
            "cached_g": self.cached_g,
            "cached_feature_g": self.cached_feature_g,
            "cached_action_g": self.cached_action_g,
        }
        for field_name in self._runtime_scalar_field_names():
            state[field_name] = getattr(self, field_name)
        for field_name in self._runtime_summary_field_names():
            state[field_name] = getattr(self, field_name)
        return state

    def export_feature_runtime_state(self):
        state = {}
        for field_name in self._feature_runtime_field_names():
            state[field_name] = getattr(self, field_name)
        return state

    def export_action_runtime_state(self):
        state = {}
        for field_name in self._action_runtime_field_names():
            state[field_name] = getattr(self, field_name)
        return state

    def load_runtime_state(self, state):
        self.cached_g = state.get("cached_g")
        self.cached_feature_g = state.get("cached_feature_g")
        self.cached_action_g = state.get("cached_action_g")
        for field_name in self._runtime_scalar_field_names():
            setattr(self, field_name, state.get(field_name))
        for field_name in self._runtime_summary_field_names():
            setattr(self, field_name, state.get(field_name))

    def aggregate_runtime_states(self, states):
        if not states:
            self.reset_runtime_stats()
            return

        feature_gates = self._stack_tensor_list([state.get("cached_feature_g") for state in states])
        action_gates = self._stack_tensor_list([state.get("cached_action_g") for state in states])
        self.cached_feature_g = feature_gates
        self.cached_action_g = action_gates
        self.cached_g = feature_gates

        for field_name in self._runtime_scalar_field_names():
            setattr(
                self,
                field_name,
                self._mean_tensor_list([state.get(field_name) for state in states]),
            )
        for field_name in self._runtime_summary_field_names():
            setattr(
                self,
                field_name,
                self._mean_tensor_list([state.get(field_name) for state in states]),
            )

    def reset_runtime_stats(self):
        self.cached_g = None
        self.cached_feature_g = None
        self.cached_action_g = None
        self.gate_mean = None
        self.gate_std = None
        self.feature_gate_mean = None
        self.feature_gate_std = None
        self.action_gate_mean = None
        self.action_gate_std = None
        self.feature_consistency_loss = None
        self.action_consistency_loss = None
        self.feature_gate_quality_loss = None
        self.action_gate_quality_loss = None
        self.feature_attn_entropy_loss = None
        self.action_attn_entropy_loss = None
        self.feature_attn_diversity_loss = None
        self.action_attn_diversity_loss = None
        self.feature_attn_im_loss = None
        self.action_attn_im_loss = None
        self.feature_q_mean = None
        self.action_q_mean = None
        self.feature_q_ret_mean = None
        self.action_q_ret_mean = None
        self.feature_q_attn_mean = None
        self.action_q_attn_mean = None
        self.feature_q_attn_token_mean = None
        self.action_q_attn_token_mean = None
        self.feature_q_attn_traj_mean = None
        self.action_q_attn_traj_mean = None
        self.cached_feature_local_summary = None
        self.cached_feature_fused_summary = None
        self.cached_action_local_summary = None
        self.cached_action_fused_summary = None

    def _build_gate_activation(self):
        activation = self.gate_activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        if activation == 'gelu':
            return nn.GELU()
        if activation == 'silu':
            return nn.SiLU()
        if activation == 'tanh':
            return nn.Tanh()
        raise ValueError(
            f"Unsupported gate_activation: {self.gate_activation}. "
            f"Expected one of: relu, gelu, silu, tanh"
        )

    def _build_gate(self, input_dim: int, output_dim: int):
        if self.gate_type == 'single-layer':
            return nn.Linear(input_dim, output_dim)
        if self.gate_type == 'two-layers':
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                self._build_gate_activation(),
                nn.Linear(output_dim, output_dim),
            )
        raise ValueError(
            f"Unsupported gate_type: {self.gate_type}. "
            f"Expected one of: single-layer, two-layers"
        )

    def _build_norm(self, feature_dim: int):
        if self.norm_type == 'none':
            return nn.Identity()
        if self.norm_type == 'layernorm':
            return nn.LayerNorm(feature_dim)
        raise ValueError(
            f"Unsupported norm_type: {self.norm_type}. "
            f"Expected one of: none, layernorm"
        )

    def _init_gate(self, gate: nn.Module):
        if isinstance(gate, nn.Linear):
            nn.init.constant_(gate.weight, 0)
            nn.init.constant_(gate.bias, -3)
            return
        if isinstance(gate, nn.Sequential):
            linear_layers = [module for module in gate if isinstance(module, nn.Linear)]
            if len(linear_layers) != 2:
                raise ValueError("two-layers gate is expected to contain exactly two Linear layers")
            nn.init.kaiming_uniform_(linear_layers[0].weight, a=0, nonlinearity='relu')
            nn.init.zeros_(linear_layers[0].bias)
            nn.init.normal_(linear_layers[1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(linear_layers[1].bias, -3)
            return
        raise TypeError(f"Unsupported gate module type: {type(gate)}")

    @staticmethod
    def _init_small_linear(linear: nn.Linear, std: float = 1e-3):
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    def _init_small_aggregator_weights(self):
        if isinstance(self.remote_feature_proj, LowRankLinear):
            self.remote_feature_proj.reset_parameters()
        if self.remote_action_proj is not None:
            self._init_small_linear(self.remote_action_proj, std=1e-3)
        self._init_small_linear(self.feature_attention.out_proj, std=1e-3)
        if self.action_attention is not None:
            self._init_small_linear(self.action_attention.out_proj, std=1e-3)
        if self.action_output_proj is not None:
            self._init_small_linear(self.action_output_proj, std=1e-3)

    def _prepare_remote_tensor(self, remote_tensor, tensor_name: str):
        if remote_tensor is None:
            return None
        if remote_tensor.ndim == 1:
            return remote_tensor.unsqueeze(0).unsqueeze(0)
        if remote_tensor.ndim == 2:
            return remote_tensor.unsqueeze(0)
        if remote_tensor.ndim == 3:
            return remote_tensor
        raise ValueError(f"{tensor_name} should be 1D/2D/3D, got {remote_tensor.shape}")

    def _update_gate_stats(self):
        available_means = []
        available_stds = []
        if self.feature_gate_mean is not None:
            available_means.append(self.feature_gate_mean)
        if self.feature_gate_std is not None:
            available_stds.append(self.feature_gate_std)
        if self.action_gate_mean is not None:
            available_means.append(self.action_gate_mean)
        if self.action_gate_std is not None:
            available_stds.append(self.action_gate_std)
        self.gate_mean = None if len(available_means) == 0 else torch.stack(available_means).mean()
        self.gate_std = None if len(available_stds) == 0 else torch.stack(available_stds).mean()

    def _summarize_feature(self, feature: torch.Tensor):
        if feature is None:
            return None
        if feature.ndim == 1:
            summary = feature.detach()
        elif feature.ndim == 2:
            summary = feature.detach().mean(dim=0)
        else:
            raise ValueError(f"feature summary expects 1D/2D tensor, got {feature.shape}")
        return summary.cpu()

    def _attend(self, local_feature, key_tensor, value_tensor, valid_mask, attention):
        key = key_tensor.reshape(1, -1, self.local_feature_dim).expand(local_feature.size(0), -1, -1)
        value = value_tensor.reshape(1, -1, self.local_feature_dim).expand(local_feature.size(0), -1, -1)
        key_padding_mask = (~valid_mask.reshape(1, -1)).expand(local_feature.size(0), -1)
        query = local_feature.unsqueeze(1)
        attn_output, attn_weights = attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )
        return attn_output.squeeze(1), attn_weights.squeeze(1)

    def _compute_task_prior(self, remote_meta, valid_mask, attn_weights):
        if remote_meta is None:
            return None

        q_task = remote_meta.get("q_task")
        if q_task is None:
            return None

        if not torch.is_tensor(q_task):
            q_task = torch.tensor(q_task, dtype=attn_weights.dtype, device=attn_weights.device)
        else:
            q_task = q_task.to(device=attn_weights.device, dtype=attn_weights.dtype)
        if q_task.ndim != 1 or q_task.numel() != valid_mask.shape[0]:
            return None

        valid_flat = valid_mask.reshape(1, -1).expand(attn_weights.size(0), -1)
        attn_weights = attn_weights * valid_flat.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        traj_count, step_count = valid_mask.shape
        traj_attn_weights = attn_weights.reshape(attn_weights.size(0), traj_count, step_count)
        traj_attn_weights = traj_attn_weights * valid_mask.unsqueeze(0).to(dtype=traj_attn_weights.dtype)
        traj_attn_weights = traj_attn_weights.sum(dim=-1)
        valid_traj_mask = valid_mask.any(dim=1)
        traj_attn_weights = traj_attn_weights * valid_traj_mask.unsqueeze(0).to(dtype=traj_attn_weights.dtype)
        traj_attn_weights = traj_attn_weights / traj_attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        q_task = q_task.clamp(0.0, 1.0) * valid_traj_mask.to(dtype=q_task.dtype)
        return (traj_attn_weights * q_task.unsqueeze(0)).sum(dim=1).clamp(0.0, 1.0)

    def _compute_attention_quality(self, attn_weights, valid_mask):
        valid_flat = valid_mask.reshape(1, -1).expand(attn_weights.size(0), -1)
        attn_weights = attn_weights * valid_flat.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        entropy = -(attn_weights * torch.log(attn_weights.clamp_min(1e-8))).sum(dim=1)
        valid_counts = valid_flat.sum(dim=1).to(dtype=attn_weights.dtype)
        max_entropy = torch.log(valid_counts.clamp_min(2.0))
        normalized_entropy = torch.where(
            valid_counts > 1,
            entropy / max_entropy.clamp_min(1e-8),
            torch.zeros_like(entropy),
        )
        q_attn_token = (1.0 - normalized_entropy).clamp(0.0, 1.0)

        traj_count, step_count = valid_mask.shape
        traj_attn_weights = attn_weights.reshape(attn_weights.size(0), traj_count, step_count)
        valid_traj_mask = valid_mask.any(dim=1)
        traj_attn_weights = traj_attn_weights * valid_mask.unsqueeze(0).to(dtype=traj_attn_weights.dtype)
        traj_attn_weights = traj_attn_weights.sum(dim=-1)
        traj_attn_weights = traj_attn_weights * valid_traj_mask.unsqueeze(0).to(dtype=traj_attn_weights.dtype)
        traj_attn_weights = traj_attn_weights / traj_attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        traj_entropy = -(traj_attn_weights * torch.log(traj_attn_weights.clamp_min(1e-8))).sum(dim=1)
        valid_traj_counts = valid_traj_mask.reshape(1, -1).expand(attn_weights.size(0), -1).sum(dim=1).to(dtype=attn_weights.dtype)
        traj_max_entropy = torch.log(valid_traj_counts.clamp_min(2.0))
        traj_normalized_entropy = torch.where(
            valid_traj_counts > 1,
            traj_entropy / traj_max_entropy.clamp_min(1e-8),
            torch.zeros_like(traj_entropy),
        )
        q_attn_traj = (1.0 - traj_normalized_entropy).clamp(0.0, 1.0)
        return attn_weights, traj_attn_weights, q_attn_token, q_attn_traj

    def _compute_attention_im_losses(self, attn_weights):
        attn_weights = attn_weights.clamp_min(1e-8)
        entropy_loss = -(attn_weights * torch.log(attn_weights)).sum(dim=1).mean()

        marginal = attn_weights.mean(dim=0)
        marginal = marginal / marginal.sum().clamp_min(1e-8)
        diversity_loss = (marginal * torch.log(marginal.clamp_min(1e-8))).sum()

        return entropy_loss, diversity_loss, entropy_loss + diversity_loss

    def _compute_q_and_losses(self, attn_weights, valid_mask, gate, gate_open_max, remote_meta, stream_name: str):
        attn_weights, traj_attn_weights, q_attn, q_attn_traj = self._compute_attention_quality(attn_weights, valid_mask)
        q_ret = self._compute_task_prior(remote_meta, valid_mask, attn_weights)
        if q_ret is None:
            q_ret = torch.full(
                (attn_weights.size(0),),
                0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        q = (self.q_ret_weight * q_ret + self.q_attn_weight * q_attn).clamp(0.0, 1.0)
        q_detached = q.detach()

        gate_target = q_detached * gate_open_max
        gate_quality_loss = F.mse_loss(gate.mean(dim=1), gate_target)
        attn_entropy_loss, attn_diversity_loss, attn_im_loss = self._compute_attention_im_losses(
            attn_weights
        )

        return (
            q_detached,
            q_ret.detach(),
            q_attn.detach(),
            q_attn_traj.detach(),
            gate_quality_loss,
            attn_entropy_loss,
            attn_diversity_loss,
            attn_im_loss,
        )

    def _compute_feature_stream_outputs(
        self,
        local_feature,
        remote_feature_proj,
        valid_remote_steps,
    ):
        local_query = self.feature_local_norm(local_feature)
        attn_output_squeezed, feature_attn_weights = self._attend(
            local_query,
            remote_feature_proj,
            remote_feature_proj,
            valid_remote_steps,
            self.feature_attention,
        )
        attn_output_squeezed = self.feature_attn_output_norm(attn_output_squeezed)
        g = torch.sigmoid(self.feature_gate(torch.cat([local_feature, attn_output_squeezed], dim=1)))
        fused_feature = local_feature + g * attn_output_squeezed
        return fused_feature, g, feature_attn_weights

    def _compute_action_stream_outputs(
        self,
        local_feature,
        remote_feature_key,
        remote_action_value,
        valid_remote_steps,
    ):
        local_query = self.action_local_norm(local_feature)
        action_context, action_attn_weights = self._attend(
            local_query,
            remote_feature_key,
            remote_action_value,
            valid_remote_steps,
            self.action_attention,
        )
        action_context = self.action_output_proj(action_context)
        action_context = self.action_context_norm(action_context)
        g = torch.sigmoid(self.action_gate(torch.cat([local_feature, action_context], dim=1)))
        fused_feature = local_feature + g * action_context
        return fused_feature, g, action_attn_weights

    def fuse_feature_stream(
        self,
        local_feature,
        remote_feature,
        remote_meta=None,
        remote_feature_proj=None,
    ):
        """
        Args:
            local_feature: 尺寸为 (B, local_feature_dim)
            remote_feature: 优先使用尺寸为 (num_traj, num_steps, remote_feature_dim)
        """
        self.cached_action_g = None
        self.action_gate_mean = None
        self.action_gate_std = None
        self.action_consistency_loss = None
        self.action_gate_quality_loss = None
        self.action_attn_entropy_loss = None
        self.action_attn_diversity_loss = None
        self.action_attn_im_loss = None
        self.action_q_mean = None
        self.action_q_ret_mean = None
        self.action_q_attn_mean = None
        self.action_q_attn_token_mean = None
        self.action_q_attn_traj_mean = None
        self.cached_feature_local_summary = self._summarize_feature(local_feature)
        remote_feature = remote_feature.to(local_feature.device)
        remote_feature = self._prepare_remote_tensor(remote_feature, "remote_feature")

        valid_remote_steps = remote_feature.abs().sum(dim=-1) > 0
        if not valid_remote_steps.any():
            self.cached_g = None
            self.cached_feature_g = None
            self.gate_mean = None
            self.gate_std = None
            self.feature_gate_mean = None
            self.feature_gate_std = None
            self.feature_consistency_loss = None
            self.feature_gate_quality_loss = None
            self.feature_attn_entropy_loss = None
            self.feature_attn_diversity_loss = None
            self.feature_attn_im_loss = None
            self.feature_q_mean = None
            self.feature_q_ret_mean = None
            self.feature_q_attn_mean = None
            self.feature_q_attn_token_mean = None
            self.feature_q_attn_traj_mean = None
            self.cached_feature_fused_summary = self.cached_feature_local_summary
            return local_feature

        if remote_feature_proj is None:
            remote_feature_proj = self.remote_feature_proj(remote_feature.detach())
            remote_feature_proj = self.feature_remote_norm(remote_feature_proj)
        fused_feature, g, feature_attn_weights = self._compute_feature_stream_outputs(
            local_feature,
            remote_feature_proj,
            valid_remote_steps,
        )
        self.cached_g = g.detach().cpu() # 兼容旧调试逻辑
        self.cached_feature_g = self.cached_g
        self.feature_gate_mean = g.mean()
        self.feature_gate_std = g.std(unbiased=False)
        self._update_gate_stats()
        (
            feature_q,
            feature_q_ret,
            feature_q_attn,
            feature_q_attn_traj,
            feature_gate_quality_loss,
            feature_attn_entropy_loss,
            feature_attn_diversity_loss,
            feature_attn_im_loss,
        ) = self._compute_q_and_losses(
            feature_attn_weights,
            valid_remote_steps,
            g,
            gate_open_max=self.feature_gate_open_max,
            remote_meta=remote_meta,
            stream_name="feature",
        )
        self.feature_gate_quality_loss = feature_gate_quality_loss
        self.feature_attn_entropy_loss = feature_attn_entropy_loss
        self.feature_attn_diversity_loss = feature_attn_diversity_loss
        self.feature_attn_im_loss = feature_attn_im_loss
        self.feature_q_mean = feature_q.mean()
        self.feature_q_ret_mean = feature_q_ret.mean()
        self.feature_q_attn_mean = feature_q_attn.mean()
        self.feature_q_attn_token_mean = self.feature_q_attn_mean
        self.feature_q_attn_traj_mean = feature_q_attn_traj.mean()
        per_sample_feature_delta = (fused_feature - local_feature).pow(2).mean(dim=1)
        self.feature_consistency_loss = (per_sample_feature_delta * (1.0 - feature_q)).mean()
        self.cached_feature_fused_summary = self._summarize_feature(fused_feature)

        return fused_feature

    def fuse_action_stream(
        self,
        local_feature,
        remote_feature,
        remote_action,
        remote_meta=None,
        remote_feature_key=None,
        remote_action_value=None,
    ):
        if self.action_attention is None or remote_feature is None or remote_action is None:
            self.cached_action_g = None
            self.action_gate_mean = None
            self.action_gate_std = None
            self.action_consistency_loss = None
            self.action_gate_quality_loss = None
            self.action_attn_entropy_loss = None
            self.action_attn_diversity_loss = None
            self.action_attn_im_loss = None
            self.action_q_mean = None
            self.action_q_ret_mean = None
            self.action_q_attn_mean = None
            self.action_q_attn_token_mean = None
            self.action_q_attn_traj_mean = None
            self.cached_action_local_summary = self._summarize_feature(local_feature)
            self.cached_action_fused_summary = self.cached_action_local_summary
            self._update_gate_stats()
            return local_feature

        self.cached_action_local_summary = self._summarize_feature(local_feature)
        remote_feature = self._prepare_remote_tensor(remote_feature.to(local_feature.device), "remote_feature")
        remote_action = self._prepare_remote_tensor(remote_action.to(local_feature.device), "remote_action")
        if remote_feature.shape[:2] != remote_action.shape[:2]:
            raise ValueError(
                f"remote feature/action shape mismatch: {remote_feature.shape} vs {remote_action.shape}"
            )

        valid_remote_steps = (remote_feature.abs().sum(dim=-1) > 0) & (remote_action.abs().sum(dim=-1) > 0)
        if not valid_remote_steps.any():
            self.cached_action_g = None
            self.action_gate_mean = None
            self.action_gate_std = None
            self.action_consistency_loss = None
            self.action_gate_quality_loss = None
            self.action_attn_entropy_loss = None
            self.action_attn_diversity_loss = None
            self.action_attn_im_loss = None
            self.action_q_mean = None
            self.action_q_ret_mean = None
            self.action_q_attn_mean = None
            self.action_q_attn_token_mean = None
            self.action_q_attn_traj_mean = None
            self.cached_action_fused_summary = self.cached_action_local_summary
            self._update_gate_stats()
            return local_feature

        if remote_feature_key is None:
            remote_feature_key = self.remote_feature_proj(remote_feature.detach())
            remote_feature_key = self.action_remote_feature_norm(remote_feature_key)
        if remote_action_value is None:
            remote_action_value = self.remote_action_proj(remote_action.detach())
            remote_action_value = self.action_remote_action_norm(remote_action_value)
        fused_feature, g, action_attn_weights = self._compute_action_stream_outputs(
            local_feature,
            remote_feature_key,
            remote_action_value,
            valid_remote_steps,
        )
        self.cached_action_g = g.detach().cpu()
        self.action_gate_mean = g.mean()
        self.action_gate_std = g.std(unbiased=False)
        self._update_gate_stats()
        (
            action_q,
            action_q_ret,
            action_q_attn,
            action_q_attn_traj,
            action_gate_quality_loss,
            action_attn_entropy_loss,
            action_attn_diversity_loss,
            action_attn_im_loss,
        ) = self._compute_q_and_losses(
            action_attn_weights,
            valid_remote_steps,
            g,
            gate_open_max=self.action_gate_open_max,
            remote_meta=remote_meta,
            stream_name="action",
        )
        self.action_gate_quality_loss = action_gate_quality_loss
        self.action_attn_entropy_loss = action_attn_entropy_loss
        self.action_attn_diversity_loss = action_attn_diversity_loss
        self.action_attn_im_loss = action_attn_im_loss
        self.action_q_mean = action_q.mean()
        self.action_q_ret_mean = action_q_ret.mean()
        self.action_q_attn_mean = action_q_attn.mean()
        self.action_q_attn_token_mean = self.action_q_attn_mean
        self.action_q_attn_traj_mean = action_q_attn_traj.mean()
        per_sample_action_delta = (fused_feature - local_feature).pow(2).mean(dim=1)
        self.action_consistency_loss = (per_sample_action_delta * (1.0 - action_q)).mean()
        self.cached_action_fused_summary = self._summarize_feature(fused_feature)
        return fused_feature
    

class FeatureAggregator:
    def __init__(self,
                 local_model: nn.Module,
                 layer_name_of_output_features: str,
                 local_feature_dim: int,
                 remote_feature_dim: int,
                 remote_action_dim: int = 0,
                 actor_layer_name: str = 'actor_mean',
                 action_position_layer_names = None,
                 action_position_actor_layer_names = None,
                 remote_feature_proj_rank: int = None,
                 attention_num_heads: int = 1,
                 gate_type: str = 'single-layer',
                 gate_activation: str = 'relu',
                 norm_type: str = 'none',
                 feature_gate_open_max: float = 0.25,
                 action_gate_open_max: float = 0.10,
                 q_ret_weight: float = 0.85,
                 q_attn_weight: float = 0.15,
                 remote_dropout_prob: float = 0.0,
                 remote_noise_std: float = 0.0,
                 remote_stale_shift_max: int = 0):
        
        self.local_model = local_model
        self.layer_name_of_output_features = layer_name_of_output_features
        self.local_feature_dim = local_feature_dim
        self.remote_feature_dim = remote_feature_dim
        self.remote_action_dim = remote_action_dim

        self.module = FeatureAggregatorModule(
            local_feature_dim,
            remote_feature_dim,
            remote_action_dim,
            remote_feature_proj_rank=remote_feature_proj_rank,
            attention_num_heads=attention_num_heads,
            gate_type=gate_type,
            gate_activation=gate_activation,
            norm_type=norm_type,
            feature_gate_open_max=feature_gate_open_max,
            action_gate_open_max=action_gate_open_max,
            q_ret_weight=q_ret_weight,
            q_attn_weight=q_attn_weight,
        )

        feature_layer_names = action_position_layer_names
        if feature_layer_names is None:
            feature_layer_names = [layer_name_of_output_features]
        elif isinstance(feature_layer_names, str):
            feature_layer_names = [feature_layer_names]
        else:
            feature_layer_names = list(feature_layer_names)
        if len(feature_layer_names) == 0:
            raise ValueError("FeatureAggregator requires at least one feature layer name")

        actor_layer_names = action_position_actor_layer_names
        if actor_layer_names is None:
            actor_layer_names = [actor_layer_name] * len(feature_layer_names)
        elif isinstance(actor_layer_names, str):
            actor_layer_names = [actor_layer_names]
        else:
            actor_layer_names = list(actor_layer_names)
        if len(actor_layer_names) != len(feature_layer_names):
            raise ValueError(
                "action_position_actor_layer_names should have the same length as action_position_layer_names"
            )

        self.layer_name_of_output_features = feature_layer_names[0]
        self.layer_names_of_output_features = feature_layer_names
        self.actor_layer_names = actor_layer_names
        self.num_action_positions = len(feature_layer_names)
        self._feature_position_to_idx = {name: idx for idx, name in enumerate(feature_layer_names)}
        self._actor_position_to_idx = {name: idx for idx, name in enumerate(actor_layer_names)}

        for layer_name in feature_layer_names:
            layer_of_output_features = get_module(local_model, layer_name)
            setattr(layer_of_output_features, "_feature_aggregator_hook_name", layer_name)
            layer_of_output_features.register_forward_hook(self._run_feature_stream)
        for actor_name in actor_layer_names:
            actor_layer = get_module(local_model, actor_name)
            if actor_layer is not None:
                setattr(actor_layer, "_feature_aggregator_hook_name", actor_name)
                actor_layer.register_forward_pre_hook(self._run_action_stream)

        self.remote_features = None
        self.remote_actions = None
        self.remote_meta = None
        self.enable_random_drop = True # 是否启用随机丢弃远程特征的机制，以模拟通信不稳定的情况
        self.remote_dropout_prob = float(remote_dropout_prob)
        self.remote_noise_std = float(remote_noise_std)
        self.remote_stale_shift_max = int(remote_stale_shift_max)
        self._current_remote_features = None
        self._current_remote_actions = None
        self._current_remote_meta = None
        self._train_remote_features = None
        self._train_remote_actions = None
        self._train_remote_meta = None
        self._current_feature_states = {}
        self._current_action_states = {}
        self._current_remote_features_device_tensor = None
        self._current_remote_actions_device_tensor = None
        self._current_feature_remote_keys = None
        self._current_action_remote_keys = None
        self._current_action_remote_values = None
        self._current_remote_projection_device = None
        self._persistent_remote_features_device_tensor = None
        self._persistent_remote_actions_device_tensor = None
        self._persistent_feature_remote_keys = None
        self._persistent_action_remote_keys = None
        self._persistent_action_remote_values = None
        self._persistent_projection_device = None
        self._persistent_projection_source = None

    def _clone_message_item(self, item):
        if torch.is_tensor(item):
            return item.detach().clone()
        if isinstance(item, dict):
            return {k: self._clone_message_item(v) for k, v in item.items()}
        if isinstance(item, list):
            return [self._clone_message_item(v) for v in item]
        if isinstance(item, tuple):
            return tuple(self._clone_message_item(v) for v in item)
        return copy.deepcopy(item)

    def snapshot_remote_message(self):
        return {
            "remote_features": self._clone_message_item(self.remote_features),
            "remote_actions": self._clone_message_item(self.remote_actions),
            "remote_meta": self._clone_message_item(self.remote_meta),
            "train_remote_features": self._clone_message_item(self._train_remote_features),
            "train_remote_actions": self._clone_message_item(self._train_remote_actions),
            "train_remote_meta": self._clone_message_item(self._train_remote_meta),
        }

    def restore_remote_message(self, snapshot):
        if snapshot is None:
            return
        self.remote_features = self._clone_message_item(snapshot.get("remote_features"))
        self.remote_actions = self._clone_message_item(snapshot.get("remote_actions"))
        self.remote_meta = self._clone_message_item(snapshot.get("remote_meta"))
        self._train_remote_features = self._clone_message_item(snapshot.get("train_remote_features"))
        self._train_remote_actions = self._clone_message_item(snapshot.get("train_remote_actions"))
        self._train_remote_meta = self._clone_message_item(snapshot.get("train_remote_meta"))
        self._current_remote_features = None
        self._current_remote_actions = None
        self._current_remote_meta = None
        self._current_feature_states = {}
        self._current_action_states = {}
        self._current_remote_features_device_tensor = None
        self._current_remote_actions_device_tensor = None
        self._current_feature_remote_keys = None
        self._current_action_remote_keys = None
        self._current_action_remote_values = None
        self._current_remote_projection_device = None
        self._reset_persistent_forward_projection_cache()

    def set_remote_message(self, remote_message):
        if isinstance(remote_message, dict):
            remote_features = remote_message.get('feature')
            remote_actions = remote_message.get('action')
            remote_meta = remote_message.get('meta')
        else:
            remote_features = remote_message
            remote_actions = None
            remote_meta = None
        self.remote_features = None if remote_features is None else remote_features.detach()
        self.remote_actions = None if remote_actions is None else remote_actions.detach()
        self.remote_meta = remote_meta
        self._current_remote_features = None
        self._current_remote_actions = None
        self._current_remote_meta = None
        self._train_remote_features = None
        self._train_remote_actions = None
        self._train_remote_meta = None
        self._current_feature_states = {}
        self._current_action_states = {}
        self._current_remote_features_device_tensor = None
        self._current_remote_actions_device_tensor = None
        self._current_feature_remote_keys = None
        self._current_action_remote_keys = None
        self._current_action_remote_values = None
        self._current_remote_projection_device = None
        self._reset_persistent_forward_projection_cache()
        self._refresh_training_remote_message()

    def set_remote_features(self, remote_features):
        self.set_remote_message(remote_features)

    def _apply_stale_shift(self, tensor, shift: int):
        if tensor is None or shift <= 0 or tensor.ndim < 3 or tensor.shape[1] <= shift:
            return tensor
        shifted = torch.zeros_like(tensor)
        shifted[:, shift:] = tensor[:, :-shift]
        return shifted

    def _apply_noise(self, tensor):
        if tensor is None or self.remote_noise_std <= 0:
            return tensor
        valid_mask = (tensor.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=tensor.dtype)
        noise = torch.randn_like(tensor) * self.remote_noise_std
        return tensor + noise * valid_mask

    def _prepare_remote_message_for_training(self):
        remote_features = self.remote_features
        remote_actions = self.remote_actions

        if self.remote_dropout_prob > 0 and random.random() < self.remote_dropout_prob:
            return None, None, None

        stale_shift = 0
        if self.remote_stale_shift_max > 0:
            stale_shift = random.randint(0, self.remote_stale_shift_max)

        remote_features = self._apply_stale_shift(remote_features, stale_shift)
        remote_actions = self._apply_stale_shift(remote_actions, stale_shift)

        remote_features = self._apply_noise(remote_features)
        remote_actions = self._apply_noise(remote_actions)
        return remote_features, remote_actions, self.remote_meta

    def _refresh_training_remote_message(self):
        if self.remote_features is None:
            self._train_remote_features = None
            self._train_remote_actions = None
            self._train_remote_meta = None
            return

        if not self.module.training:
            self._train_remote_features = self.remote_features
            self._train_remote_actions = self.remote_actions
            self._train_remote_meta = self.remote_meta
            return

        self._train_remote_features, self._train_remote_actions, self._train_remote_meta = self._prepare_remote_message_for_training()

    def _reset_position_runtime_cache(self):
        self._current_feature_states = {}
        self._current_action_states = {}

    def _reset_forward_projection_cache(self):
        self._current_remote_features_device_tensor = None
        self._current_remote_actions_device_tensor = None
        self._current_feature_remote_keys = None
        self._current_action_remote_keys = None
        self._current_action_remote_values = None
        self._current_remote_projection_device = None

    def _reset_persistent_forward_projection_cache(self):
        self._persistent_remote_features_device_tensor = None
        self._persistent_remote_actions_device_tensor = None
        self._persistent_feature_remote_keys = None
        self._persistent_action_remote_keys = None
        self._persistent_action_remote_values = None
        self._persistent_projection_device = None
        self._persistent_projection_source = None

    def _can_persist_forward_projection_cache(self):
        return not any(param.requires_grad for param in self.module.parameters())

    def _projection_param_version_signature(self):
        signature = []
        for param in self.module.parameters():
            version = getattr(param, "_version", 0)
            signature.append(int(version))
        return tuple(signature)

    def _current_projection_source_key(self, device):
        return (
            id(self._current_remote_features),
            id(self._current_remote_actions),
            str(device),
            self._projection_param_version_signature(),
        )

    def _try_load_persistent_forward_projection_cache(self, device):
        if not self._can_persist_forward_projection_cache():
            return False
        source_key = self._current_projection_source_key(device)
        if (
            self._persistent_projection_source != source_key
            or self._persistent_feature_remote_keys is None
            or self._persistent_projection_device != device
        ):
            return False
        self._current_remote_features_device_tensor = self._persistent_remote_features_device_tensor
        self._current_remote_actions_device_tensor = self._persistent_remote_actions_device_tensor
        self._current_feature_remote_keys = self._persistent_feature_remote_keys
        self._current_action_remote_keys = self._persistent_action_remote_keys
        self._current_action_remote_values = self._persistent_action_remote_values
        self._current_remote_projection_device = self._persistent_projection_device
        return True

    def _save_persistent_forward_projection_cache(self, device):
        if not self._can_persist_forward_projection_cache():
            self._reset_persistent_forward_projection_cache()
            return
        self._persistent_remote_features_device_tensor = self._current_remote_features_device_tensor
        self._persistent_remote_actions_device_tensor = self._current_remote_actions_device_tensor
        self._persistent_feature_remote_keys = self._current_feature_remote_keys
        self._persistent_action_remote_keys = self._current_action_remote_keys
        self._persistent_action_remote_values = self._current_action_remote_values
        self._persistent_projection_device = device
        self._persistent_projection_source = self._current_projection_source_key(device)

    def _get_hook_name(self, hook_module):
        hook_name = getattr(hook_module, "_feature_aggregator_hook_name", None)
        if hook_name is None:
            hook_name = getattr(hook_module, "_get_name", lambda: hook_module.__class__.__name__)()
        return hook_name

    def _resolve_feature_position_index(self, hook_module):
        hook_name = self._get_hook_name(hook_module)
        if hook_name not in self._feature_position_to_idx:
            raise KeyError(f"Unknown feature hook module name: {hook_name}")
        return self._feature_position_to_idx[hook_name]

    def _resolve_actor_position_index(self, hook_module):
        hook_name = self._get_hook_name(hook_module)
        if hook_name not in self._actor_position_to_idx:
            raise KeyError(f"Unknown actor hook module name: {hook_name}")
        return self._actor_position_to_idx[hook_name]

    def _slice_remote_feature_for_position(self, position_idx: int):
        remote_feature = self._current_remote_features_device_tensor
        if remote_feature is None:
            remote_feature = self._current_remote_features
        if remote_feature is None:
            return None
        if remote_feature.ndim == 4:
            if position_idx >= remote_feature.shape[2]:
                raise IndexError(
                    f"remote feature action position index {position_idx} out of range for shape {remote_feature.shape}"
                )
            return remote_feature[:, :, position_idx, :]
        return remote_feature

    @staticmethod
    def _slice_projected_tensor_for_position(projected_tensor, position_idx: int):
        if projected_tensor is None:
            return None
        if projected_tensor.ndim == 4:
            if position_idx >= projected_tensor.shape[2]:
                raise IndexError(
                    f"projected remote feature action position index {position_idx} out of range for shape {projected_tensor.shape}"
                )
            return projected_tensor[:, :, position_idx, :]
        return projected_tensor

    def _ensure_forward_projection_cache(self, device):
        if self._current_remote_features is None:
            return
        if self._try_load_persistent_forward_projection_cache(device):
            return
        if self._current_feature_remote_keys is not None and self._current_remote_projection_device == device:
            return

        remote_features = self._current_remote_features.to(device)
        self._current_remote_features_device_tensor = remote_features
        remote_feature_proj = self.module.remote_feature_proj(remote_features)
        self._current_feature_remote_keys = self.module.feature_remote_norm(remote_feature_proj)

        if self.module.action_attention is not None:
            self._current_action_remote_keys = self.module.action_remote_feature_norm(remote_feature_proj)
            if self._current_remote_actions is not None:
                remote_actions = self._current_remote_actions.to(device)
                self._current_remote_actions_device_tensor = remote_actions
                remote_action_value = self.module.remote_action_proj(remote_actions)
                self._current_action_remote_values = self.module.action_remote_action_norm(remote_action_value)
            else:
                self._current_remote_actions_device_tensor = None
                self._current_action_remote_values = None
        else:
            self._current_action_remote_keys = None
            self._current_remote_actions_device_tensor = None
            self._current_action_remote_values = None

        self._current_remote_projection_device = device
        self._save_persistent_forward_projection_cache(device)

    def _aggregate_position_runtime_states(self):
        states = []
        position_indices = sorted(set(self._current_feature_states.keys()) | set(self._current_action_states.keys()))
        for position_idx in position_indices:
            state = {}
            if position_idx in self._current_feature_states:
                state.update(self._current_feature_states[position_idx].copy())
            if position_idx in self._current_action_states:
                state.update(self._current_action_states[position_idx].copy())
            states.append(state)
        self.module.aggregate_runtime_states(states)

    def _run_feature_stream(self, module, input, local_output_features):
        if self.remote_features is None:
            self._current_remote_features = None
            self._current_remote_actions = None
            self._current_remote_meta = None
            self._train_remote_features = None
            self._train_remote_actions = None
            self._train_remote_meta = None
            self._reset_position_runtime_cache()
            self._reset_forward_projection_cache()
            self.module.reset_runtime_stats()
            return None

        if self.module.training:
            if self._train_remote_features is None and self.remote_features is not None:
                self._refresh_training_remote_message()
            self._current_remote_features = self._train_remote_features
            self._current_remote_actions = self._train_remote_actions
            self._current_remote_meta = self._train_remote_meta
        else:
            self._current_remote_features = self.remote_features
            self._current_remote_actions = self.remote_actions
            self._current_remote_meta = self.remote_meta

        if self._current_remote_features is None:
            self._reset_position_runtime_cache()
            self._reset_forward_projection_cache()
            self.module.reset_runtime_stats()
            return None

        position_idx = self._resolve_feature_position_index(module)
        if position_idx == 0:
            self._reset_position_runtime_cache()
            self._reset_forward_projection_cache()
        self._ensure_forward_projection_cache(local_output_features.device)
        remote_feature = self._slice_remote_feature_for_position(position_idx)
        remote_feature_key = self._slice_projected_tensor_for_position(self._current_feature_remote_keys, position_idx)
        fused_output = self.module.fuse_feature_stream(
            local_output_features,
            remote_feature,
            remote_meta=self._current_remote_meta,
            remote_feature_proj=remote_feature_key,
        )
        self._current_feature_states[position_idx] = self.module.export_feature_runtime_state()
        self._aggregate_position_runtime_states()
        return fused_output

    def _run_action_stream(self, module, input):
        if self._current_remote_features is None or self._current_remote_actions is None:
            return None

        position_idx = self._resolve_actor_position_index(module)
        if position_idx not in self._current_feature_states:
            return None

        local_actor_input = input[0]
        self._ensure_forward_projection_cache(local_actor_input.device)
        remote_feature = self._slice_remote_feature_for_position(position_idx)
        remote_feature_key = self._slice_projected_tensor_for_position(self._current_action_remote_keys, position_idx)
        fused_actor_input = self.module.fuse_action_stream(
            local_actor_input,
            remote_feature,
            self._current_remote_actions_device_tensor,
            remote_meta=self._current_remote_meta,
            remote_feature_key=remote_feature_key,
            remote_action_value=self._current_action_remote_values,
        )
        state = self._current_feature_states[position_idx].copy()
        state.update(self.module.export_action_runtime_state())
        self._current_action_states[position_idx] = state
        self._aggregate_position_runtime_states()
        return (fused_actor_input,)
