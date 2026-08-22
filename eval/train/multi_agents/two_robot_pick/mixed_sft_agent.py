from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.reinforcement_learning.utils import RunningMeanStd
from train.multi_agents.two_robot_pick.model import (
    build_batch_from_obs,
)
from train.multi_agents.two_robot_pick.tiny_vla import (
    MLPProjector,
    SharedTinyVLA4DActor,
    TinyVLABackbone,
)


class TinySmolVLAContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: int = 112,
        hidden_dim: int = 384,
        vision_layers: int = 6,
        attention_heads: int = 6,
        patch_size: int = 14,
        ffn_mult: int = 4,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.vla = TinyVLABackbone(
            image_size=self.image_size,
            patch_size=int(patch_size),
            hidden_dim=self.hidden_dim,
            vision_layers=int(vision_layers),
            decoder_layers=1,
            attention_heads=int(attention_heads),
            ffn_mult=int(ffn_mult),
            prompt_length=1,
            max_action_dim=self.action_dim,
            num_action_bins=2,
        )
        self.state_projector = MLPProjector(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        ).to(dtype=torch.float32)
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)
        self.action_mean_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        ).to(dtype=torch.float32)
        self.log_std_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        ).to(dtype=torch.float32)
        self._train_backbone = True

    @property
    def device(self) -> torch.device:
        return next(self.vla.parameters()).device

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self._train_backbone = bool(train_backbone)
        for parameter in self.vla.parameters():
            parameter.requires_grad = train_backbone
        for module in [
            self.state_projector,
            self.context_projector,
            self.action_mean_head,
            self.log_std_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _prepare_pixel_values(self, rgbs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach()
            rgb_batch = rgb_batch.to(device=self.device, dtype=torch.float32)
        else:
            rgb_batch = torch.as_tensor(np.asarray(rgbs)[..., :3], device=self.device, dtype=torch.float32)
        if rgb_batch.ndim != 4:
            raise ValueError(f"Expected RGB batch [B, H, W, 3], got {rgb_batch.shape}")
        rgb_batch = rgb_batch.permute(0, 3, 1, 2).contiguous() / 255.0
        if rgb_batch.shape[-2:] != (self.image_size, self.image_size):
            rgb_batch = F.interpolate(
                rgb_batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return rgb_batch

    def sample_actions(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        actions_input: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_position_placeholders: Optional[nn.ModuleList] = None,
        action_position_actor_placeholders: Optional[nn.ModuleList] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        del planner_subtasks
        pixel_values = self._prepare_pixel_values(rgbs)
        vision_tokens = self.vla.encode_vision(pixel_values)
        vision_feature = vision_tokens.mean(dim=1).to(torch.float32)

        if state_features is None:
            state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            state_feature = self.state_projector(state_tensor)
        else:
            state_feature = torch.as_tensor(state_features, device=self.device, dtype=torch.float32)
            state_feature = torch.nan_to_num(state_feature, nan=0.0, posinf=1e4, neginf=-1e4)
        context_feature = self.context_projector(torch.cat([vision_feature, state_feature], dim=-1))
        action_position_features = None
        policy_feature = context_feature
        if action_position_placeholders is not None:
            if len(action_position_placeholders) != self.action_dim:
                raise ValueError(
                    f"Expected {self.action_dim} action-position placeholders, got {len(action_position_placeholders)}"
                )
            if (
                action_position_actor_placeholders is not None
                and len(action_position_actor_placeholders) != self.action_dim
            ):
                raise ValueError(
                    "action_position_actor_placeholders must match action_dim when provided"
                )
            per_action_features = []
            for action_idx in range(self.action_dim):
                pos_feature = action_position_placeholders[action_idx](context_feature)
                if action_position_actor_placeholders is not None:
                    pos_feature = action_position_actor_placeholders[action_idx](pos_feature)
                per_action_features.append(pos_feature)
            action_position_features = torch.stack(per_action_features, dim=1)
            policy_feature = action_position_features.mean(dim=1)

        mean = torch.tanh(self.action_mean_head(policy_feature))
        log_std = torch.clamp(self.log_std_head(policy_feature), min=self.log_std_min, max=self.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if actions_input is not None:
            actions = torch.as_tensor(actions_input, device=self.device, dtype=torch.float32)
        elif deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        return {
            "actions": clipped_actions,
            "log_prob": dist.log_prob(clipped_actions).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "mean": mean,
            "std": std,
            "context_feature": context_feature,
            "policy_feature": policy_feature,
            "action_position_features": action_position_features,
        }


class MixedTinyVLAMultiAgentsSFTAgent(nn.Module):
    def __init__(
        self,
        agent_names: List[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        model_dir=None,
        normalize_state: bool = True,
        freeze_vla_backbone: bool = False,
        critic_hidden_dim: int = 512,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = None,
        use_vla_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        train_vision_backbone: bool = False,
        vision_token_pool_size: Optional[int] = None,
        policy_mode: str = "native",
        tiny_hidden_dim: int = 640,
        tiny_vision_layers: int = 7,
        tiny_decoder_layers: int = 8,
        tiny_attention_heads: int = 10,
        tiny_patch_size: int = 14,
        tiny_ffn_mult: int = 4,
        tiny_num_action_bins: int = 256,
        tiny_prompt_length: int = 24,
        smolvla_hidden_dim: int = 384,
        smolvla_vision_layers: int = 6,
        smolvla_attention_heads: int = 6,
        smolvla_patch_size: int = 14,
        smolvla_ffn_mult: int = 4,
    ):
        super().__init__()
        if len(agent_names) != 2:
            raise ValueError(f"Mixed SFT agent expects exactly 2 agents, got {agent_names}")
        self.agent_names = list(agent_names)
        self.state_dim = int(state_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)
        self.vla_agent_name = self.agent_names[0]
        self.smolvla_agent_name = self.agent_names[1]
        self.smolvla_num_action_bins = max(int(tiny_num_action_bins), 2)

        self.vla_actor = SharedTinyVLA4DActor(
            model_dir=model_dir,
            state_dim=self.state_dim,
            env_action_dim=self.action_dim,
            attention_implementation=attention_implementation,
            image_size=image_size,
            use_lora=use_vla_lora,
            use_vision_lora=use_vision_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            train_vision_backbone=train_vision_backbone,
            vision_token_pool_size=vision_token_pool_size,
            policy_mode=policy_mode,
            tiny_hidden_dim=tiny_hidden_dim,
            tiny_vision_layers=tiny_vision_layers,
            tiny_decoder_layers=tiny_decoder_layers,
            tiny_attention_heads=tiny_attention_heads,
            tiny_patch_size=tiny_patch_size,
            tiny_ffn_mult=tiny_ffn_mult,
            tiny_num_action_bins=tiny_num_action_bins,
            tiny_prompt_length=tiny_prompt_length,
        )
        self.vla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.smolvla_actor = TinySmolVLAContinuousActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            image_size=112 if image_size is None else int(image_size),
            hidden_dim=smolvla_hidden_dim,
            vision_layers=smolvla_vision_layers,
            attention_heads=smolvla_attention_heads,
            patch_size=smolvla_patch_size,
            ffn_mult=smolvla_ffn_mult,
        )
        self.smolvla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.vla_actor_action_position_placeholders = nn.ModuleList(
            [nn.Identity() for _ in range(self.action_dim)]
        )
        self.vla_actor_action_position_actor_placeholders = nn.ModuleList(
            [nn.Identity() for _ in range(self.action_dim)]
        )
        self.smolvla_actor_action_position_placeholders = nn.ModuleList(
            [nn.Identity() for _ in range(self.action_dim)]
        )
        self.smolvla_actor_action_position_actor_placeholders = nn.ModuleList(
            [nn.Identity() for _ in range(self.action_dim)]
        )

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(self.global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        critic_hidden = int(tiny_hidden_dim)
        critic_visual_input_dim = int(tiny_hidden_dim) + int(smolvla_hidden_dim)
        self.critic_state_encoder = nn.Sequential(
            nn.LayerNorm(self.global_state_dim),
            nn.Linear(self.global_state_dim, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
        ).to(dtype=torch.float32)
        self.critic_visual_encoder = nn.Sequential(
            nn.LayerNorm(critic_visual_input_dim),
            nn.Linear(critic_visual_input_dim, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
        ).to(dtype=torch.float32)
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_hidden * 2),
            nn.Linear(critic_hidden * 2, int(critic_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(critic_hidden_dim), 1),
        ).to(dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return self.vla_actor.device

    @property
    def actor(self):
        # Compatibility shim for legacy BC/SFT utilities that expect `agent.actor`.
        return self.vla_actor

    def requires_action_bins(self) -> Dict[str, bool]:
        return {name: False for name in self.agent_names}

    def _adapt_actor_state_features(
        self,
        vla_state_feature: torch.Tensor,
        smolvla_state_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return vla_state_feature, smolvla_state_feature

    def _normalize_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[agent_name](state)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is None:
            return state
        return self.critic_state_rms(state)

    def _smolvla_actions_to_bin_indices(self, actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        scaled = (action_tensor + 1.0) * 0.5
        bins = torch.round(scaled * float(self.smolvla_num_action_bins - 1))
        return bins.to(dtype=torch.long)

    def _smolvla_bin_indices_to_actions(self, bins: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        bin_tensor = torch.as_tensor(bins, device=self.device, dtype=torch.float32)
        if self.smolvla_num_action_bins <= 1:
            return torch.zeros_like(bin_tensor, dtype=torch.float32)
        scaled = bin_tensor / float(self.smolvla_num_action_bins - 1)
        return (scaled * 2.0 - 1.0).clamp(-1.0, 1.0)

    def _compute_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        global_state = batch["global_state"].to(device=self.device, dtype=torch.float32)
        global_state = self._normalize_global_state(global_state)
        global_feature = self.critic_state_encoder(global_state)

        rgb = batch["rgb"]
        with torch.no_grad():
            left_visual = self.vla_actor._project_vision_features(
                self.vla_actor._prepare_pixel_values(rgb)
            ).mean(dim=1).to(torch.float32)
            right_visual = self.smolvla_actor.vla.encode_vision(
                self.smolvla_actor._prepare_pixel_values(rgb)
            ).mean(dim=1).to(torch.float32)
        critic_visual = self.critic_visual_encoder(torch.cat([left_visual, right_visual], dim=-1))
        critic_input = torch.cat([global_feature, critic_visual], dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input=None,
        return_token_logits: bool = False,
        **kwargs,
    ):
        action_bins_input = kwargs.get("action_bins_input")
        deterministic = bool(kwargs.get("deterministic", False))
        rgb = batch["rgb"]
        vla_state = self._normalize_state(
            batch[f"agent_states_{self.vla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.vla_agent_name,
        )
        smol_state = self._normalize_state(
            batch[f"agent_states_{self.smolvla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.smolvla_agent_name,
        )
        vla_state_feature = self.vla_actor.state_projector(vla_state)
        smol_state_feature = self.smolvla_actor.state_projector(smol_state)
        vla_state_feature, smol_state_feature = self._adapt_actor_state_features(
            vla_state_feature,
            smol_state_feature,
        )

        vla_bins = None
        if action_bins_input is not None and self.vla_agent_name in action_bins_input:
            vla_bins = action_bins_input[self.vla_agent_name]
        elif actions_input is not None and self.vla_agent_name in actions_input:
            vla_bins = self.vla_actor.env_actions_to_bin_indices(actions_input[self.vla_agent_name])
        vla_out = self.vla_actor.get_action_and_stats(
            rgbs=rgb,
            states=vla_state,
            state_features=vla_state_feature,
            action_bins=vla_bins,
            prompt_role_ids=torch.zeros(vla_state.shape[0], device=self.device, dtype=torch.long),
            deterministic=deterministic,
        )
        if (
            vla_out.get("action_position_prompt_hidden") is not None
            and vla_out.get("action_position_context_feature") is not None
            and vla_out.get("state_feature") is not None
        ):
            vla_position_features = []
            local_prompt_hidden = vla_out["action_position_prompt_hidden"]
            local_context_feature = vla_out["action_position_context_feature"]
            local_state_feature = vla_out["state_feature"]
            for action_idx in range(local_prompt_hidden.shape[1]):
                position_feature = torch.cat(
                    [
                        local_prompt_hidden[:, action_idx, :],
                        local_context_feature[:, action_idx, :],
                        local_state_feature,
                    ],
                    dim=-1,
                )
                position_feature = self.vla_actor_action_position_placeholders[action_idx](position_feature)
                position_feature = self.vla_actor_action_position_actor_placeholders[action_idx](position_feature)
                vla_position_features.append(position_feature)
            vla_out["action_position_features"] = torch.stack(vla_position_features, dim=1)

        smol_actions_input = None
        if action_bins_input is not None and self.smolvla_agent_name in action_bins_input:
            smol_actions_input = self._smolvla_bin_indices_to_actions(action_bins_input[self.smolvla_agent_name])
        elif actions_input is not None and self.smolvla_agent_name in actions_input:
            smol_actions_input = actions_input[self.smolvla_agent_name]
        smol_out = self.smolvla_actor.sample_actions(
            rgbs=rgb,
            states=smol_state,
            state_features=smol_state_feature,
            actions_input=smol_actions_input,
            deterministic=deterministic,
            action_position_placeholders=self.smolvla_actor_action_position_placeholders,
            action_position_actor_placeholders=self.smolvla_actor_action_position_actor_placeholders,
        )

        actions_out = {
            self.vla_agent_name: vla_out["env_actions"].detach().cpu().numpy(),
            self.smolvla_agent_name: smol_out["actions"].detach().to(dtype=torch.float32).cpu().numpy(),
        }
        log_probs = {
            self.vla_agent_name: vla_out["log_prob"],
            self.smolvla_agent_name: smol_out["log_prob"],
        }
        entropies = {
            self.vla_agent_name: vla_out["entropy"],
            self.smolvla_agent_name: smol_out["entropy"],
        }
        value = self._compute_value(batch)

        if return_token_logits:
            return actions_out, log_probs, entropies, value, {self.vla_agent_name: vla_out["token_logits"]}
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False):
        rgb = batch["rgb"]
        vla_state = self._normalize_state(
            batch[f"agent_states_{self.vla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.vla_agent_name,
        )
        smol_state = self._normalize_state(
            batch[f"agent_states_{self.smolvla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.smolvla_agent_name,
        )
        vla_state_feature = self.vla_actor.state_projector(vla_state)
        smol_state_feature = self.smolvla_actor.state_projector(smol_state)
        vla_state_feature, smol_state_feature = self._adapt_actor_state_features(
            vla_state_feature,
            smol_state_feature,
        )
        vla_out = self.vla_actor.get_action_and_stats(
            rgbs=rgb,
            states=vla_state,
            state_features=vla_state_feature,
            prompt_role_ids=torch.zeros(vla_state.shape[0], device=self.device, dtype=torch.long),
            deterministic=deterministic,
        )
        if (
            vla_out.get("action_position_prompt_hidden") is not None
            and vla_out.get("action_position_context_feature") is not None
            and vla_out.get("state_feature") is not None
        ):
            for action_idx in range(vla_out["action_position_prompt_hidden"].shape[1]):
                position_feature = torch.cat(
                    [
                        vla_out["action_position_prompt_hidden"][:, action_idx, :],
                        vla_out["action_position_context_feature"][:, action_idx, :],
                        vla_out["state_feature"],
                    ],
                    dim=-1,
                )
                position_feature = self.vla_actor_action_position_placeholders[action_idx](position_feature)
                _ = self.vla_actor_action_position_actor_placeholders[action_idx](position_feature)
        smol_out = self.smolvla_actor.sample_actions(
            rgbs=rgb,
            states=smol_state,
            state_features=smol_state_feature,
            deterministic=deterministic,
            action_position_placeholders=self.smolvla_actor_action_position_placeholders,
            action_position_actor_placeholders=self.smolvla_actor_action_position_actor_placeholders,
        )
        return {
            self.vla_agent_name: vla_out["env_actions"].detach().cpu().numpy(),
            self.smolvla_agent_name: smol_out["actions"].detach().to(dtype=torch.float32).cpu().numpy(),
        }

    def get_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._compute_value(batch)

    def forward(self, batch: Dict[str, Any]):
        _, log_probs, _, values = self.get_action_and_value(batch)
        return log_probs, values

    @torch.no_grad()
    def update_state_stats(
        self,
        obs: Dict[str, Any],
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        parsed = build_batch_from_obs(obs, self.agent_names)
        if update_actor and self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    parsed[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
                )
        if update_critic and self.critic_state_rms is not None:
            self.critic_state_rms.update(parsed["global_state"].to(device=self.device, dtype=torch.float32))

    def freeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def checkpoint_state_dict(self):
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict):
        target_state = self.state_dict()
        matched = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in target_state or target_state[key].shape != value.shape:
                skipped.append(key)
                continue
            matched[key] = value
        if not matched:
            raise RuntimeError("No compatible parameters found for MixedTinyVLAMultiAgentsSFTAgent")
        merged_state = dict(target_state)
        merged_state.update(matched)
        missing = sorted(set(target_state.keys()) - set(matched.keys()))
        self.load_state_dict(merged_state, strict=True)
        print(
            "[Checkpoint] MixedTinyVLAMultiAgentsSFTAgent "
            f"matched={len(matched)} skipped={len(skipped)} missing={len(missing)}"
        )


def build_mixed_mappo_optimizer(args, agent: MixedTinyVLAMultiAgentsSFTAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": [p for p in agent.vla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "vla_actor_vla",
        },
        {
            "params": [p for p in agent.vla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "vla_actor_state_projector",
        },
        {
            "params": [p for p in agent.vla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.vla_actor.actor_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "vla_actor_heads",
        },
        {
            "params": [p for p in agent.smolvla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "smolvla_actor_vla",
        },
        {
            "params": [p for p in agent.smolvla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "smolvla_actor_state_projector",
        },
        {
            "params": [p for p in agent.smolvla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.action_mean_head.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.log_std_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "smolvla_actor_heads",
        },
        {
            "params": [p for p in agent.critic_state_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_state_encoder",
        },
        {
            "params": [p for p in agent.critic_visual_encoder.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic_visual_encoder",
        },
        {
            "params": [p for p in agent.critic.parameters() if p.requires_grad],
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)
