import torch
import sys; sys.path.append('.')
from torch import nn 
from torch.distributions.normal import Normal
import numpy as np
from train.reinforcement_learning.utils import RunningMeanStd

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_dim=256,
        max_pooling=True,
        inactivated_output=False,  # False for ConvBody, True for CNN
        use_bn=False,
    ):
        super().__init__()

        if not use_bn:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [64, 64]
                nn.Conv2d(16, 16, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 128, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(128, 128, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [64, 64]
                nn.Conv2d(16, 16, 3, padding=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 128, 3, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(128, 128, 1, padding=0, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class PlainConvV2(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_dim=256,
        pooling=True,
        inactivated_output=False,
    ):
        super().__init__()

        def block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(4, out_c),   # ⭐ 关键替代 BN
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            block(in_channels, 16, stride=2),   # 64x64
            block(16, 32, stride=2),            # 32x32  ↓（替代 pooling）
            block(32, 64, stride=2),            # 16x16
            block(64, 128, stride=2),           # 8x8
            block(128, 128, stride=2),          # 4x4
            block(128, 128, stride=1),          # 保留空间信息
        )

        if pooling:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # ⭐ 改 max → avg（更稳）
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, camera_count=1, use_bn=False):
        super().__init__()

        # 输入 (B, 3, 128, 128)，输出 (B, 256)
        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False, use_bn=use_bn
        )

        # 输入 (B, 1, 128, 128)，输出 (B, 256)
        self.depth_encoder = PlainConv(
            in_channels=1 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False, use_bn=use_bn
        )

        # 输入 (B, state_dim)，输出 (B, state_dim)
        # self.state_encoder = make_mlp(
        #     state_dim, [state_dim, state_dim * 2, state_dim], last_act=False
        # )
        self.state_encoder = nn.Linear(state_dim, 256)
        
        self.decoder = make_mlp(
            256 * 3, [512, 256, action_dim], last_act=False
        )
        self.get_eval_action = self.get_action = self.forward

    def forward(self, rgb, depth=None, state=None):

        if depth is None and state is None:
            rgb, depth, state = rgb['rgb'], rgb['depth'], rgb['state']

        rgb, depth, state = self.rgb_encoder(rgb), self.depth_encoder(depth), self.state_encoder(state)
        # print(rgb.size(), depth.size(), state.size())
        x = torch.cat([rgb, depth, state], dim=1)
        return self.decoder(x)

# 正交初始化的 MLP
# 如果Linear为 y=Wx+b，则正交初始化是对W进行正交矩阵初始化，b初始化为0
# 则梯度 dy=W*dx 时，dy的范数会是dx的std ** 2 倍，使得dx无论方向如何，范数都不会发生剧烈变化
def make_mlp_with_orth_init(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True, is_actor=False):
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        if is_actor and idx == len(mlp_channels) - 1:
            module_list.append(layer_init(nn.Linear(c_in, c_out), std=0.01 * np.sqrt(2)))
        else:
            module_list.append(layer_init(nn.Linear(c_in, c_out)))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder(inplace=True))
        c_in = c_out
    return nn.Sequential(*module_list)

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, camera_count=1, normalize_state=True, use_depth=True):
        super().__init__()
        self.action_dim = action_dim
        # 输入 (B, 3, 128, 128)，输出 (B, 256  )
        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False
        )

        # 输入 (B, 1, 128, 128)，输出 (B, 256)
        if use_depth:
            self.depth_encoder = PlainConv(
                in_channels=1 * camera_count, out_dim=256, max_pooling=False, inactivated_output=False
            )
        else:
            self.depth_encoder = None

        # 输入 (B, state_dim)，输出 (B, state_dim)
        self.state_encoder = make_mlp(
            state_dim, [state_dim, 256], last_act=False
        )

        if use_depth:
            self.actor_mean = make_mlp_with_orth_init(
                256 * 3, [512, action_dim], last_act=False, is_actor=True
            )
            self.critic = make_mlp_with_orth_init(
                256 * 3, [512, 1], last_act=False
            )
        else:
            self.actor_mean = make_mlp_with_orth_init(
                256 * 2, [512, action_dim], last_act=False, is_actor=True
            )
            self.critic = make_mlp_with_orth_init(
                256 * 2, [512, 1], last_act=False
            )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        if normalize_state:
            self.state_rms = RunningMeanStd(shape=(state_dim,))
        else:
            self.state_rms = nn.Sequential()

    def get_feature(self, batch):
        if self.depth_encoder is not None:
            rgb, depth, state = batch['rgb'], batch['depth'], batch['state']
            state = self.state_rms(state)
            rgb, depth, state = self.rgb_encoder(rgb), self.depth_encoder(depth), self.state_encoder(state)
            x = torch.cat([rgb, depth, state], dim=1)
        else:
            try:
                rgb, state = batch['rgb'], batch['state']
            except Exception as e:
                print("🚨 batch 结构错误（缺少 rgb/state）")
                raise e

            # ---------- STEP 0: 原始输入 ----------
            _check_tensor(rgb, "rgb (raw)")
            _check_tensor(state, "state (raw)")

            # ---------- STEP 1: state_rms ----------
            try:
                state = self.state_rms(state)
                _check_tensor(state, "state after RMS")
            except Exception as e:
                print("🚨 state_rms 出问题（极可能是 std=0 或数值爆）")
                raise e

            # ---------- STEP 2: rgb encoder ----------
            try:
                rgb_feat = self.rgb_encoder(rgb)
                _check_tensor(rgb_feat, "rgb feature")
            except Exception as e:
                print("🚨 rgb_encoder 出问题")
                raise e

            # ---------- STEP 3: state encoder ----------
            try:
                state_feat = self.state_encoder(state)
                _check_tensor(state_feat, "state feature")
            except Exception as e:
                print("🚨 state_encoder 出问题")
                raise e

            # ---------- STEP 4: concat ----------
            try:
                if rgb_feat.shape[0] != state_feat.shape[0]:
                    print("❌ batch size 不一致:", rgb_feat.shape, state_feat.shape)
                    raise ValueError("Batch mismatch")

                x = torch.cat([rgb_feat, state_feat], dim=1)
                _check_tensor(x, "final feature")
            except Exception as e:
                print("🚨 concat 出问题")
                raise e
        return x

    def forward(self, batch, action=None):
        # ---------- Step 0: 检查 batch ----------
        try:
            if isinstance(batch, torch.Tensor):
                _check_finite(batch, "batch")
            elif isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        _check_finite(v, f"batch[{k}]")
        except Exception as e:
            print("🚨 Error at STEP 0: batch 本身就有问题")
            raise e

        # ---------- Step 1: feature ----------
        try:
            x = self.get_feature(batch)
            _check_finite(x, "feature(x)")
        except Exception as e:
            print("🚨 Error at STEP 1: get_feature 出问题")
            raise e

        # ---------- Step 2: actor mean ----------
        try:
            action_mean = self.actor_mean(x)
            _check_finite(action_mean, "action_mean")
        except Exception as e:
            print("🚨 Error at STEP 2: actor_mean 出问题")
            raise e

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        action = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=action_logstd.dtype, device=action_logstd.device)
        return (action.detach().cpu().numpy()), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x).squeeze(-1)

    def get_action(self, batch, deterministic=True):
        x = self.get_feature(batch)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_value(self, batch):
        x = self.get_feature(batch)
        return self.critic(x).squeeze(-1)

    @torch.no_grad()
    def reset_logstd(self, new_logstd=-0.5):
        self.actor_logstd.data.fill_(new_logstd)

    @torch.no_grad()
    def reset_value_head(self, use_depth=False):
        if use_depth:
            self.critic = make_mlp_with_orth_init(
                256 * 3, [512, 1], last_act=False
            )
        else:
            self.critic = make_mlp_with_orth_init(
                256 * 2, [512, 1], last_act=False
            )

    def load_actor(self, path, strict=True):
        actor_state = torch.load(path, map_location="cpu")['actor']

        # 1. encoder
        self.rgb_encoder.load_state_dict(
            {k.replace("rgb_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("rgb_encoder.")},
            strict=strict
        )

        self.depth_encoder.load_state_dict(
            {k.replace("depth_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("depth_encoder.")},
            strict=strict
        )

        self.state_encoder.load_state_dict(
            {k.replace("state_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("state_encoder.")},
            strict=strict
        )

        # 2. decoder → actor_mean
        self.actor_mean.load_state_dict(
            {k.replace("decoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("decoder.")},
            strict=strict
        )

    @torch.no_grad()
    def update_state_stats(self, obs):
        if isinstance(self.state_rms, RunningMeanStd):
            self.state_rms.update(obs['state'])

    def freeze_state_stats(self):
        self.state_rms.freeze()

    def unfreeze_state_stats(self):
        self.state_rms.unfreeze()

def _check_finite(x, name):
    if not torch.isfinite(x).all():
        print(f"❌ {name} contains NaN/Inf")
        try:
            print(f"{name} stats -> min: {x.min().item()}, max: {x.max().item()}")
        except:
            pass
        raise ValueError(f"{name} not finite")

def _check_tensor(x, name):
    if not torch.isfinite(x).all():
        print(f"❌ {name} has NaN/Inf")

        mask = ~torch.isfinite(x)
        idx = mask.nonzero(as_tuple=False)[0]
        b = idx[0].item()

        print(f"NaN/Inf 出现在 batch[{b}]")
        print(f"{name}[{b}] = {x[b]}")

        raise ValueError(f"{name} not finite")

    else:
        # ---- 每个 batch 的最大值 ----
        if x.dim() > 1:
            per_batch_max = x.abs().reshape(x.shape[0], -1).max(dim=1).values
        else:
            per_batch_max = x.abs()

        max_val, max_idx = per_batch_max.max(dim=0)

        if max_val.item() > 1e4:
            b = max_idx.item()
            print(f"⚠️ {name} value too large: {max_val.item()}")
            print(f"最大值出现在 batch[{b}]")

            print(f"{name}[{b}] = {x[b]}")

            # 🔥 再定位具体维度
            flat = x[b].reshape(-1).abs()
            val, idx = flat.max(dim=0)
            print(f"该样本最大元素 index: {idx.item()}, value: {val.item()}")

import torch
import torch.nn as nn
from torch.distributions import Normal


class MAPPOAgent(nn.Module):
    def __init__(
        self,
        agent_names,
        state_dim,
        global_state_dim,
        action_dim,
        camera_count=1,
        use_depth=False,
        normalize_state=True
    ):
        super().__init__()

        self.agent_names = agent_names
        self.action_dim = action_dim

        # =========================================================
        # Shared RGB Encoder
        # =========================================================
        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count,
            out_dim=256,
            max_pooling=False,
            inactivated_output=False
        )

        # =========================================================
        # Per-agent RMS (方案A核心)
        # =========================================================
        if normalize_state:
            self.actor_state_rms = nn.ModuleDict({
                name: RunningMeanStd(shape=(state_dim,))
                for name in agent_names
            })
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        # =========================================================
        # Actor Encoder
        # =========================================================
        self.actor_state_encoder = make_mlp(
            state_dim,
            [256, 256],
            last_act=False
        )

        # =========================================================
        # Critic Encoder
        # =========================================================
        self.critic_state_encoder = make_mlp(
            global_state_dim,
            [512, 512],
            last_act=False
        )

        # =========================================================
        # Actor Heads
        # =========================================================
        self.actor_heads = nn.ModuleDict()
        self.actor_logstd = nn.ParameterDict()
        self.actor_feature_placeholders = nn.ModuleDict()

        for name in agent_names:
            self.actor_heads[name] = make_mlp_with_orth_init(
                256 + 256,
                [512, action_dim],
                last_act=False,
                is_actor=True
            )

            self.actor_logstd[name] = nn.Parameter(
                torch.ones(1, action_dim) * -0.5
            )

            self.actor_feature_placeholders[name] = nn.Identity()
            

        # =========================================================
        # Critic
        # =========================================================
        self.critic = make_mlp_with_orth_init(
            256 + 512,
            [512, 1],
            last_act=False
        )

    # =========================================================
    # Actor feature
    # =========================================================
    def get_actor_feature(self, rgb_feat, agent_state):
        state_feat = self.actor_state_encoder(agent_state)
        return torch.cat([rgb_feat, state_feat], dim=1)

    # =========================================================
    # Critic feature
    # =========================================================
    def get_critic_feature(self, rgb_feat, global_state):
        global_feat = self.critic_state_encoder(global_state)
        return torch.cat([rgb_feat, global_feat], dim=1)

    # =========================================================
    # Forward
    # =========================================================
    def get_action_and_value(self, batch, actions_input=None):
        rgb = batch['rgb']
        global_state = batch['global_state']
        rgb_feat = self.rgb_encoder(rgb)
        # ---------------- critic RMS ----------------
        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)
        # ---------------- critic ----------------
        critic_feat = self.get_critic_feature(rgb_feat, global_state)
        value = self.critic(critic_feat).squeeze(-1)
        # ---------------- actor RMS (per-agent) ----------------
        agent_states = {}
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                agent_states[name] = self.actor_state_rms[name](batch[f'agent_states_{name}'])
        # ---------------- actor ----------------
        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            agent_state = agent_states[name]
            actor_feat = self.get_actor_feature(rgb_feat, agent_state)
            actor_feat = self.actor_feature_placeholders[name](actor_feat)  # ⭐ 占位模块，方便后续特征hook输出
            mean = self.actor_heads[name](actor_feat)
            logstd = self.actor_logstd[name].expand_as(mean)

            std = torch.exp(logstd)
            dist = Normal(mean, std)
            if actions_input is None:
                action = dist.sample()
            else:
                action = actions_input[name]
            action = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=logstd.dtype, device=logstd.device)
            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)

        return actions_out, log_probs, entropies, value
    
    def forward(self, batch):
        rgb = batch['rgb']
        global_state = batch['global_state']
        rgb_feat = self.rgb_encoder(rgb)
        # ---------------- critic RMS ----------------
        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)
        # ---------------- critic ----------------
        critic_feat = self.get_critic_feature(rgb_feat, global_state)
        value = self.critic(critic_feat).squeeze(-1)
        # ---------------- actor RMS (per-agent) ----------------
        agent_states = {}
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                agent_states[name] = self.actor_state_rms[name](batch[f'agent_states_{name}'])
        # ---------------- actor ----------------
        ret = [value.unsqueeze(-1)]
        for name in self.agent_names:
            agent_state = agent_states[name]
            actor_feat = self.get_actor_feature(rgb_feat, agent_state)
            actor_feat = self.actor_feature_placeholders[name](actor_feat)  # ⭐ 占位模块，方便后续特征hook输出
            mean = self.actor_heads[name](actor_feat)
            ret.append(mean)

        return torch.cat(ret, dim=-1)

    # =========================================================
    # Get action
    # =========================================================
    @torch.no_grad()
    def get_action(self, batch, deterministic=False):

        rgb = batch['rgb']

        rgb_feat = self.rgb_encoder(rgb)
        
        actions = {}

        for name in self.agent_names:

            state = batch[f'agent_states_{name}']   

            if self.actor_state_rms is not None:
                state = self.actor_state_rms[name](state)

            feat = self.get_actor_feature(rgb_feat, state)
            feat = self.actor_feature_placeholders[name](feat)  # ⭐ 占位模块，方便后续特征hook输出
            mean = self.actor_heads[name](feat)

            if deterministic:
                actions[name] = mean
            else:
                std = torch.exp(self.actor_logstd[name].expand_as(mean))
                dist = Normal(mean, std)
                actions[name] = dist.sample()

        return actions

    # =========================================================
    # Value
    # =========================================================
    def get_value(self, batch):

        rgb = batch['rgb']
        global_state = batch['global_state']

        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)

        rgb_feat = self.rgb_encoder(rgb)
        critic_feat = self.get_critic_feature(rgb_feat, global_state)

        return self.critic(critic_feat).squeeze(-1)

    # =========================================================
    # RMS update
    # =========================================================
    @torch.no_grad()
    def update_state_stats(self, obs):

        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    obs[f'agent_states_{name}']
                )

        if self.critic_state_rms is not None:
            self.critic_state_rms.update(
                obs['global_state']
            )

    # =========================================================
    # freeze / unfreeze
    # =========================================================
    def freeze_state_stats(self):

        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()

        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self):

        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()

        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def reset_logstd(self, new_logstd=-0.5):
        for agent_name in self.agent_names:
            self.actor_logstd[agent_name].data.fill_(new_logstd)

    @torch.no_grad()
    def reset_value_head(self, use_depth=False):
        device = self.critic[0].weight.device  # 获取当前 critic 模块所在设备
        if use_depth:
            self.critic = make_mlp_with_orth_init(
                256 * 2 + 512,
                [512, 1],
                last_act=False
            )
        else:
            self.critic = make_mlp_with_orth_init(
                256 + 512,
                [512, 1],
                last_act=False
            )
        self.critic.to(device)

    def load_actor(self, path):
        # 单 actor 参数复制到多个 head，actor的backbone直接使用
        actor_state = torch.load(path, map_location="cpu")['actor']
        self.rgb_encoder.load_state_dict(
            {k.replace("rgb_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("rgb_encoder.")}
        )
        self.state_encoder.load_state_dict(
            {k.replace("state_encoder.", ""): v
            for k, v in actor_state.items()
            if k.startswith("state_encoder.")}
        )
        for agent_name in self.agent_names:
            self.actor_heads[agent_name].load_state_dict(
                {k.replace("decoder.", ""): v
                for k, v in actor_state.items()
                if k.startswith("decoder.")}
            )

if __name__ == '__main__':
    actor = Actor(state_dim=29, action_dim=8, camera_count=1)
    print(actor)

    batch_size = 16
    rgb, depth, state = torch.rand((batch_size, 3, 128, 128)), torch.rand((batch_size, 1, 128, 128)), torch.rand((batch_size, 29))
    action = actor(rgb, depth, state)

    print(action.size()) # (batch_size, action_dim)

    from ours.utils.dl.common.model import get_model_size
    print('model size (MB): ', get_model_size(actor, True))