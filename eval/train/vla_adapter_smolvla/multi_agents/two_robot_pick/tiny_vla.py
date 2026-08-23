from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDiscreteActorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_bins = int(num_bins)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        action_features: torch.Tensor,
        state_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = action_features.shape
        action_features = self.context_encoder(
            action_features.reshape(batch_size * seq_len, 1, hidden_dim)
        ).reshape(batch_size, seq_len, hidden_dim)
        expanded_state = state_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_context = context_feature if context_feature.ndim == 3 else context_feature.unsqueeze(1)
        expanded_context = expanded_context.expand(-1, seq_len, -1)
        fused = torch.cat([action_features, expanded_state, expanded_context], dim=-1)
        return self.logit_head(fused) * self.residual_scale


class TinyVLABackbone(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        vision_layers: int,
        decoder_layers: int,
        attention_heads: int,
        ffn_mult: int,
        prompt_length: int,
        max_action_dim: int,
        num_action_bins: int,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by attention_heads={attention_heads}"
            )

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.llm_dim = int(hidden_dim)
        self.vocab_size = int(num_action_bins)
        self.prompt_length = int(prompt_length)
        self.max_action_dim = int(max_action_dim)
        self.num_action_bins = int(num_action_bins)
        self.bos_token_id = self.num_action_bins

        grid_size = self.image_size // self.patch_size
        self.num_patches = grid_size * grid_size

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.vision_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        self.memory_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))
        vision_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=attention_heads,
            dim_feedforward=self.hidden_dim * int(ffn_mult),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vision_encoder = nn.TransformerEncoder(
            encoder_layer=vision_layer,
            num_layers=int(vision_layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.role_prompt_embeddings = nn.Parameter(torch.zeros(2, self.prompt_length, self.hidden_dim))
        self.token_pos_embed = nn.Parameter(
            torch.zeros(1, self.prompt_length + self.max_action_dim, self.hidden_dim)
        )
        self.action_token_embeddings = nn.Embedding(self.num_action_bins + 1, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=attention_heads,
            dim_feedforward=self.hidden_dim * int(ffn_mult),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=int(decoder_layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.lm_head = nn.Linear(self.hidden_dim, self.num_action_bins, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init_std = 0.02
        nn.init.normal_(self.vision_pos_embed, mean=0.0, std=init_std)
        nn.init.normal_(self.memory_pos_embed, mean=0.0, std=init_std)
        nn.init.normal_(self.role_prompt_embeddings, mean=0.0, std=init_std)
        nn.init.normal_(self.token_pos_embed, mean=0.0, std=init_std)
        nn.init.normal_(self.action_token_embeddings.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(pixel_values)
        patches = patches.flatten(2).transpose(1, 2).contiguous()
        patches = patches + self.vision_pos_embed[:, : patches.shape[1]]
        return self.vision_encoder(patches)

    def build_prompt_embeddings(
        self,
        prompt_role_ids: torch.Tensor,
        planner_subtask_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_role_ids = prompt_role_ids.to(device=self.role_prompt_embeddings.device, dtype=torch.long)
        prompt_embeddings = self.role_prompt_embeddings[prompt_role_ids]
        if planner_subtask_embedding is not None:
            planner_subtask_embedding = planner_subtask_embedding.to(
                device=prompt_embeddings.device,
                dtype=prompt_embeddings.dtype,
            ).unsqueeze(1)
            prompt_embeddings = prompt_embeddings.clone()
            prompt_embeddings[:, :1, :] = prompt_embeddings[:, :1, :] + planner_subtask_embedding
        return prompt_embeddings

    def add_target_positional_embeddings(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return target_tokens + self.token_pos_embed[:, : target_tokens.shape[1]]

    def add_memory_positional_embeddings(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return memory_tokens + self.memory_pos_embed[:, : memory_tokens.shape[1]]

    def embed_action_bins(self, action_bins: torch.Tensor) -> torch.Tensor:
        action_bins = action_bins.to(device=self.action_token_embeddings.weight.device, dtype=torch.long)
        return self.action_token_embeddings(action_bins)

    def _decoder_layer_incremental_step(
        self,
        layer: nn.TransformerDecoderLayer,
        token_input: torch.Tensor,
        memory_tokens: torch.Tensor,
        cached_layer_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not getattr(layer, "norm_first", False):
            raise NotImplementedError("Incremental tiny VLA decoding requires norm_first=True decoder layers.")

        if cached_layer_tokens is None:
            cached_layer_tokens = token_input.new_zeros(
                (token_input.shape[0], 0, token_input.shape[-1]),
                dtype=token_input.dtype,
                device=token_input.device,
            )

        self_attn_tokens = torch.cat([cached_layer_tokens, token_input], dim=1)
        self_attn_tokens = layer.norm1(self_attn_tokens)
        self_attn_query = self_attn_tokens[:, -1:, :]
        self_attn_out, _ = layer.self_attn(
            self_attn_query,
            self_attn_tokens,
            self_attn_tokens,
            need_weights=False,
        )
        hidden = token_input + layer.dropout1(self_attn_out)

        cross_attn_query = layer.norm2(hidden)
        cross_attn_out, _ = layer.multihead_attn(
            cross_attn_query,
            memory_tokens,
            memory_tokens,
            need_weights=False,
        )
        hidden = hidden + layer.dropout2(cross_attn_out)

        ff_input = layer.norm3(hidden)
        ff_hidden = layer.linear1(ff_input)
        ff_hidden = layer.activation(ff_hidden)
        ff_hidden = layer.dropout(ff_hidden)
        ff_hidden = layer.linear2(ff_hidden)
        hidden = hidden + layer.dropout3(ff_hidden)
        return hidden

    def decode_next_token_incremental(
        self,
        memory_tokens: torch.Tensor,
        token_input: torch.Tensor,
        layer_caches: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if token_input.ndim != 3 or token_input.shape[1] != 1:
            raise ValueError(
                f"decode_next_token_incremental expects token_input [B, 1, H], got {token_input.shape}"
            )

        memory_with_pos = self.add_memory_positional_embeddings(memory_tokens)
        if layer_caches is None:
            layer_caches = [None] * len(self.decoder.layers)
        if len(layer_caches) != len(self.decoder.layers):
            raise ValueError(
                f"decode_next_token_incremental expects {len(self.decoder.layers)} layer caches, got {len(layer_caches)}"
            )

        hidden = token_input
        next_caches: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.decoder.layers):
            cached_tokens = layer_caches[layer_idx]
            layer_input = hidden
            hidden = self._decoder_layer_incremental_step(layer, layer_input, memory_with_pos, cached_tokens)
            if cached_tokens is None:
                next_caches.append(layer_input)
            else:
                next_caches.append(torch.cat([cached_tokens, layer_input], dim=1))

        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return hidden, next_caches

    def decode_tokens(
        self,
        memory_tokens: torch.Tensor,
        prompt_role_ids: torch.Tensor,
        action_prefix_bins: Optional[torch.Tensor] = None,
        planner_subtask_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeddings = self.build_prompt_embeddings(
            prompt_role_ids.to(device=memory_tokens.device, dtype=torch.long),
            planner_subtask_embedding=planner_subtask_embedding,
        ).to(device=memory_tokens.device, dtype=memory_tokens.dtype)
        if action_prefix_bins is not None and action_prefix_bins.numel() > 0:
            action_prefix_bins = action_prefix_bins.to(device=memory_tokens.device, dtype=torch.long)
            prefix_embeddings = self.embed_action_bins(action_prefix_bins).to(
                device=memory_tokens.device,
                dtype=memory_tokens.dtype,
            )
            target_tokens = torch.cat([prompt_embeddings, prefix_embeddings], dim=1)
        else:
            target_tokens = prompt_embeddings

        target_tokens = self.add_target_positional_embeddings(target_tokens)
        memory_tokens = self.add_memory_positional_embeddings(memory_tokens)
        target_mask = torch.triu(
            torch.full(
                (target_tokens.shape[1], target_tokens.shape[1]),
                float("-inf"),
                device=memory_tokens.device,
                dtype=memory_tokens.dtype,
            ),
            diagonal=1,
        )
        return self.decoder(
            tgt=target_tokens,
            memory=memory_tokens,
            tgt_mask=target_mask,
        )


class SharedTinyVLA4DActor(nn.Module):
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]],
        state_dim: int,
        env_action_dim: int = 4,
        prompt: str = "",
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = None,
        use_lora: bool = False,
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
        use_decode_cache: bool = False,
    ):
        super().__init__()
        del prompt, attention_implementation, lora_r, lora_alpha, lora_dropout
        self.model_dir = None if model_dir is None else Path(model_dir)
        self.state_dim = int(state_dim)
        self.env_action_dim = int(env_action_dim)
        self.use_lora = False
        self.use_vision_lora = False
        self.image_size = 112 if image_size is None else int(image_size)
        self.train_vision_backbone = bool(train_vision_backbone)
        self.vision_token_pool_size = vision_token_pool_size
        self.policy_mode = policy_mode
        self.eval_micro_batch_size = 32
        self._vla_trainable = True
        self.use_decode_cache = bool(use_decode_cache)

        self.vla = TinyVLABackbone(
            image_size=self.image_size,
            patch_size=int(tiny_patch_size),
            hidden_dim=int(tiny_hidden_dim),
            vision_layers=int(tiny_vision_layers),
            decoder_layers=int(tiny_decoder_layers),
            attention_heads=int(tiny_attention_heads),
            ffn_mult=int(tiny_ffn_mult),
            prompt_length=int(tiny_prompt_length),
            max_action_dim=self.env_action_dim,
            num_action_bins=int(tiny_num_action_bins),
        )

        self.full_vocab_size = int(self.vla.vocab_size)
        self.action_token_start_idx = 0
        self.action_token_end_idx = int(self.vla.vocab_size)
        self.num_action_bins = int(self.vla.num_action_bins)
        self.hidden_dim = int(self.vla.llm_dim)
        self.register_buffer(
            "action_bin_centers",
            torch.linspace(-1.0, 1.0, steps=self.num_action_bins, dtype=torch.float32),
            persistent=False,
        )

        self.state_projector = MLPProjector(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        ).to(dtype=torch.float32)
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)
        self.actor_head = ResidualDiscreteActorHead(
            hidden_dim=self.hidden_dim,
            action_dim=self.env_action_dim,
            num_bins=self.num_action_bins,
        ).to(dtype=torch.float32)
        self.max_planner_subtask_bytes = 64
        self.subtask_byte_embedding = nn.Embedding(257, self.hidden_dim)
        self.subtask_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)
        if self.policy_mode not in {"residual", "native"}:
            raise ValueError(f"Unsupported policy_mode={self.policy_mode}")

        self.configure_trainable_modules(train_backbone=True)

    @property
    def device(self) -> torch.device:
        return next(self.vla.parameters()).device

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self._vla_trainable = bool(train_backbone)
        for parameter in self.vla.parameters():
            parameter.requires_grad = train_backbone
        for module in [
            self.state_projector,
            self.context_projector,
            self.actor_head,
            self.subtask_byte_embedding,
            self.subtask_projector,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _vla_has_trainable_params(self) -> bool:
        return any(parameter.requires_grad for parameter in self.vla.parameters())

    def _prepare_pixel_values(self, rgbs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach()
            if rgb_batch.ndim != 4:
                raise ValueError(f"Expected RGB tensor to have shape [B, H, W, 3], got {rgb_batch.shape}")
            rgb_batch = rgb_batch.to(device=self.device, dtype=torch.float32)
            rgb_batch = rgb_batch.permute(0, 3, 1, 2).contiguous() / 255.0
        else:
            rgb_batch = torch.from_numpy(np.asarray(rgbs)[..., :3].astype(np.float32, copy=False))
            if rgb_batch.ndim != 4:
                raise ValueError(f"Expected RGB array to have shape [B, H, W, 3], got {rgb_batch.shape}")
            rgb_batch = rgb_batch.to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0

        if rgb_batch.shape[-2:] != (self.image_size, self.image_size):
            rgb_batch = F.interpolate(
                rgb_batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return rgb_batch

    def _resolve_prompt_inputs(
        self,
        batch_size: int,
        prompt_role_ids: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if prompt_role_ids is None:
            prompt_role_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        else:
            prompt_role_ids = prompt_role_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        planner_subtask_embeddings = self._encode_planner_subtasks(planner_subtasks, batch_size)
        return prompt_role_ids, planner_subtask_embeddings

    def _encode_planner_subtasks(
        self,
        planner_subtasks: Optional[Sequence[Optional[str]]],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if planner_subtasks is None:
            return None
        if len(planner_subtasks) != batch_size:
            raise ValueError(f"planner_subtasks expects {batch_size} items, got {len(planner_subtasks)}")

        byte_ids = torch.zeros(
            batch_size,
            self.max_planner_subtask_bytes,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(byte_ids, dtype=torch.bool)
        has_nonempty = False
        for row, value in enumerate(planner_subtasks):
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            encoded = text.encode("utf-8", errors="ignore")[: self.max_planner_subtask_bytes]
            if not encoded:
                continue
            ids = torch.as_tensor([byte + 1 for byte in encoded], dtype=torch.long, device=self.device)
            byte_ids[row, : ids.shape[0]] = ids
            mask[row, : ids.shape[0]] = True
            has_nonempty = True

        if not has_nonempty:
            return None

        embedded = self.subtask_byte_embedding(byte_ids)
        masked_embedded = embedded * mask.unsqueeze(-1).to(dtype=embedded.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=embedded.dtype)
        pooled = masked_embedded.sum(dim=1) / denom
        return self.subtask_projector(pooled.to(torch.float32))

    def env_actions_to_bin_indices(self, env_actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(env_actions, torch.Tensor):
            actions = env_actions.detach().to(dtype=torch.float32)
        else:
            actions = torch.as_tensor(np.asarray(env_actions, dtype=np.float32))
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.clamp(-1.0, 1.0)
        scaled = (actions + 1.0) * 0.5 * (self.num_action_bins - 1)
        return scaled.round().to(dtype=torch.long)

    def bin_indices_to_env_actions(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if bin_indices.ndim == 1:
            bin_indices = bin_indices.unsqueeze(0)
        bin_indices = bin_indices.to(device=self.device, dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, 0, self.num_action_bins - 1)
        env_actions = self.action_bin_centers.to(device=self.device)[bin_indices].to(torch.float32)
        return torch.nan_to_num(env_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

    def _project_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._vla_has_trainable_params():
            projected = self.vla.encode_vision(pixel_values)
        else:
            with torch.no_grad():
                projected = self.vla.encode_vision(pixel_values)
        return self._pool_vision_tokens(projected)

    def _pool_vision_tokens(self, projected_patch_embeddings: torch.Tensor) -> torch.Tensor:
        target_tokens = self.vision_token_pool_size
        if target_tokens is None:
            return projected_patch_embeddings
        if target_tokens <= 0:
            raise ValueError(f"vision_token_pool_size must be positive, got {target_tokens}")

        batch_size, num_tokens, hidden_dim = projected_patch_embeddings.shape
        if target_tokens >= num_tokens:
            return projected_patch_embeddings

        pooled = F.adaptive_avg_pool1d(
            projected_patch_embeddings.transpose(1, 2),
            output_size=target_tokens,
        )
        return pooled.transpose(1, 2).reshape(batch_size, target_tokens, hidden_dim)

    @staticmethod
    def _append_state_token_to_patches(
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        state_token = state_feature.to(dtype=projected_patch_embeddings.dtype).unsqueeze(1)
        return torch.cat([projected_patch_embeddings, state_token], dim=1)

    def _decode_autoregressive_actions(
        self,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
        prompt_role_ids: torch.Tensor,
        planner_subtask_embeddings: Optional[torch.Tensor],
        action_bins: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        prompt_hidden_out: Optional[torch.Tensor] = None
        context_feature_out: Optional[torch.Tensor] = None
        action_position_prompt_hidden: List[torch.Tensor] = []
        action_position_context_feature: List[torch.Tensor] = []
        generated_bins: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        generated_logits: List[torch.Tensor] = []

        if action_bins is not None:
            action_bins = action_bins.to(device=self.device, dtype=torch.long)
            if action_bins.ndim == 1:
                action_bins = action_bins.unsqueeze(1)

        if self.use_decode_cache:
            return self._decode_autoregressive_actions_with_cache(
                projected_patch_embeddings=projected_patch_embeddings,
                state_feature=state_feature,
                prompt_role_ids=prompt_role_ids,
                planner_subtask_embeddings=planner_subtask_embeddings,
                action_bins=action_bins,
                deterministic=deterministic,
            )

        for action_idx in range(self.env_action_dim):
            prefix_bins = None
            if generated_bins:
                prefix_bins = torch.stack(generated_bins, dim=1)

            decoded = self.vla.decode_tokens(
                memory_tokens=projected_patch_embeddings,
                prompt_role_ids=prompt_role_ids,
                action_prefix_bins=prefix_bins,
                planner_subtask_embedding=planner_subtask_embeddings,
            )
            prompt_hidden = decoded[:, -1, :].to(torch.float32)
            context_feature = self.context_projector(decoded.mean(dim=1).to(torch.float32))
            base_bin_logits = self.vla.lm_head(prompt_hidden).to(torch.float32)
            if self.policy_mode == "residual":
                residual_logits = self.actor_head(
                    prompt_hidden.unsqueeze(1),
                    state_feature,
                    context_feature,
                ).squeeze(1)
                logits = base_bin_logits + residual_logits
            else:
                logits = base_bin_logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

            if action_idx == 0:
                prompt_hidden_out = prompt_hidden
                context_feature_out = context_feature

            action_position_prompt_hidden.append(prompt_hidden)
            action_position_context_feature.append(context_feature)
            categorical = torch.distributions.Categorical(logits=logits)
            if action_bins is None:
                selected_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
            else:
                selected_bin = action_bins[:, action_idx]
            generated_bins.append(selected_bin)
            log_probs.append(categorical.log_prob(selected_bin))
            entropies.append(categorical.entropy())
            generated_logits.append(logits)

        selected_bins = torch.stack(generated_bins, dim=1)
        return {
            "env_actions": self.bin_indices_to_env_actions(selected_bins),
            "log_prob": torch.stack(log_probs, dim=1).sum(dim=-1),
            "entropy": torch.stack(entropies, dim=1).mean(dim=-1),
            "token_logits": torch.stack(generated_logits, dim=1),
            "action_bins": selected_bins,
            "prompt_hidden": prompt_hidden_out,
            "context_feature": context_feature_out,
            "state_feature": state_feature,
            "action_position_prompt_hidden": torch.stack(action_position_prompt_hidden, dim=1),
            "action_position_context_feature": torch.stack(action_position_context_feature, dim=1),
        }

    def _decode_autoregressive_actions_with_cache(
        self,
        projected_patch_embeddings: torch.Tensor,
        state_feature: torch.Tensor,
        prompt_role_ids: torch.Tensor,
        planner_subtask_embeddings: Optional[torch.Tensor],
        action_bins: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        prompt_embeddings = self.vla.build_prompt_embeddings(
            prompt_role_ids,
            planner_subtask_embedding=planner_subtask_embeddings,
        ).to(device=self.device, dtype=projected_patch_embeddings.dtype)
        prompt_tokens = self.vla.add_target_positional_embeddings(prompt_embeddings)

        layer_caches: List[Optional[torch.Tensor]] = [None] * len(self.vla.decoder.layers)
        decoded_tokens: List[torch.Tensor] = []
        decoded_sum: Optional[torch.Tensor] = None

        for prompt_idx in range(prompt_tokens.shape[1]):
            prompt_hidden, layer_caches = self.vla.decode_next_token_incremental(
                memory_tokens=projected_patch_embeddings,
                token_input=prompt_tokens[:, prompt_idx : prompt_idx + 1, :],
                layer_caches=layer_caches,
            )
            decoded_tokens.append(prompt_hidden)
            decoded_sum = prompt_hidden if decoded_sum is None else decoded_sum + prompt_hidden

        if decoded_sum is None:
            raise RuntimeError("Tiny VLA prompt decode produced no tokens.")

        prompt_hidden_out: Optional[torch.Tensor] = None
        context_feature_out: Optional[torch.Tensor] = None
        action_position_prompt_hidden: List[torch.Tensor] = []
        action_position_context_feature: List[torch.Tensor] = []
        generated_bins: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        generated_logits: List[torch.Tensor] = []

        decoded_length = prompt_tokens.shape[1]
        for action_idx in range(self.env_action_dim):
            prompt_hidden = decoded_tokens[-1].squeeze(1).to(torch.float32)
            context_hidden = (decoded_sum / float(decoded_length)).squeeze(1).to(torch.float32)
            context_feature = self.context_projector(context_hidden)
            base_bin_logits = self.vla.lm_head(prompt_hidden).to(torch.float32)
            if self.policy_mode == "residual":
                residual_logits = self.actor_head(
                    prompt_hidden.unsqueeze(1),
                    state_feature,
                    context_feature,
                ).squeeze(1)
                logits = base_bin_logits + residual_logits
            else:
                logits = base_bin_logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

            if action_idx == 0:
                prompt_hidden_out = prompt_hidden
                context_feature_out = context_feature

            action_position_prompt_hidden.append(prompt_hidden)
            action_position_context_feature.append(context_feature)
            categorical = torch.distributions.Categorical(logits=logits)
            if action_bins is None:
                selected_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
            else:
                selected_bin = action_bins[:, action_idx]
            generated_bins.append(selected_bin)
            log_probs.append(categorical.log_prob(selected_bin))
            entropies.append(categorical.entropy())
            generated_logits.append(logits)

            if action_idx + 1 >= self.env_action_dim:
                continue

            next_token = self.vla.embed_action_bins(selected_bin.unsqueeze(1)).to(
                device=self.device,
                dtype=projected_patch_embeddings.dtype,
            )
            next_token = next_token + self.vla.token_pos_embed[:, decoded_length : decoded_length + 1]
            decoded_hidden, layer_caches = self.vla.decode_next_token_incremental(
                memory_tokens=projected_patch_embeddings,
                token_input=next_token,
                layer_caches=layer_caches,
            )
            decoded_tokens.append(decoded_hidden)
            decoded_sum = decoded_sum + decoded_hidden
            decoded_length += 1

        selected_bins = torch.stack(generated_bins, dim=1)
        return {
            "env_actions": self.bin_indices_to_env_actions(selected_bins),
            "log_prob": torch.stack(log_probs, dim=1).sum(dim=-1),
            "entropy": torch.stack(entropies, dim=1).mean(dim=-1),
            "token_logits": torch.stack(generated_logits, dim=1),
            "action_bins": selected_bins,
            "prompt_hidden": prompt_hidden_out,
            "context_feature": context_feature_out,
            "state_feature": state_feature,
            "action_position_prompt_hidden": torch.stack(action_position_prompt_hidden, dim=1),
            "action_position_context_feature": torch.stack(action_position_context_feature, dim=1),
        }

    def get_action_and_stats(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        action_bins: Optional[torch.Tensor] = None,
        prompt_role_ids: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pixel_values = self._prepare_pixel_values(rgbs)
        projected_patch_embeddings = self._project_vision_features(pixel_values)
        input_ids, planner_subtask_embeddings = self._resolve_prompt_inputs(
            projected_patch_embeddings.shape[0],
            prompt_role_ids,
            planner_subtasks=planner_subtasks,
        )
        return self._get_action_and_stats_from_prepared_inputs(
            input_ids=input_ids,
            attention_mask=None,
            projected_patch_embeddings=projected_patch_embeddings,
            states=states,
            state_features=state_features,
            action_bins=action_bins,
            planner_subtask_embeddings=planner_subtask_embeddings,
            deterministic=deterministic,
        )

    def _get_action_and_stats_from_prepared_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        projected_patch_embeddings: torch.Tensor,
        states: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        action_bins: Optional[torch.Tensor] = None,
        planner_subtask_embeddings: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if (
            planner_subtask_embeddings is None
            and attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.shape[-1] == self.hidden_dim
        ):
            planner_subtask_embeddings = attention_mask.to(device=self.device, dtype=torch.float32)
        if state_features is None:
            state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            state_feature = self.state_projector(state_tensor)
        else:
            state_feature = torch.as_tensor(state_features, device=self.device, dtype=torch.float32)
            state_feature = torch.nan_to_num(state_feature, nan=0.0, posinf=1e4, neginf=-1e4)
        memory_tokens = self._append_state_token_to_patches(projected_patch_embeddings, state_feature)
        return self._decode_autoregressive_actions(
            projected_patch_embeddings=memory_tokens,
            state_feature=state_feature,
            prompt_role_ids=input_ids,
            planner_subtask_embeddings=planner_subtask_embeddings,
            action_bins=action_bins,
            deterministic=deterministic,
        )


__all__ = [
    "SharedTinyVLA4DActor",
]
