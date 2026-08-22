from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class RandomInitTokenizer:
    def __init__(self, vocab_size: int = 4096, pad_token_id: int = 0) -> None:
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)

    def _encode_text(self, text: str) -> list[int]:
        tokens = [abs(hash(piece)) % (self.vocab_size - 2) + 1 for piece in text.split()]
        return tokens or [1]

    def __call__(self, text, return_tensors: str | None = None, padding: bool | str = False, **_: Any):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode_text(item) for item in texts]
        max_len = max(len(item) for item in encoded)
        if not padding:
            max_len = None
        input_ids = []
        attention_mask = []
        for item in encoded:
            if max_len is None:
                padded = item
                mask = [1] * len(item)
            else:
                pad_len = max_len - len(item)
                padded = item + [self.pad_token_id] * pad_len
                mask = [1] * len(item) + [0] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        if return_tensors == 'pt':
            return {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor}
        return {'input_ids': input_ids_tensor.tolist(), 'attention_mask': attention_mask_tensor.tolist()}

    def decode(self, token_ids) -> str:
        ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
        return ' '.join(f'tok{int(idx)}' for idx in ids)

    def batch_decode(self, batch_token_ids) -> list[str]:
        return [self.decode(item) for item in batch_token_ids]


class RandomInitImageProcessor:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = int(image_size)

    def __call__(self, images, return_tensors: str = 'pt', **_: Any):
        pixel_values = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_image = image.convert('RGB').resize((self.image_size, self.image_size))
                array = np.asarray(pil_image, dtype=np.float32) / 255.0
            else:
                array = np.asarray(image, dtype=np.float32)
                if array.max() > 1.0:
                    array = array / 255.0
                if array.shape[:2] != (self.image_size, self.image_size):
                    pil_image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8)).resize((self.image_size, self.image_size))
                    array = np.asarray(pil_image, dtype=np.float32) / 255.0
            pixel_values.append(torch.from_numpy(array).permute(2, 0, 1))
        stacked = torch.stack(pixel_values, dim=0)
        if return_tensors == 'pt':
            return {'pixel_values': stacked}
        return {'pixel_values': stacked.numpy()}


class RandomInitProcessor:
    def __init__(self, vocab_size: int = 4096, image_size: int = 224) -> None:
        self.tokenizer = RandomInitTokenizer(vocab_size=vocab_size)
        self.image_processor = RandomInitImageProcessor(image_size=image_size)

    def __call__(self, text, images, padding: bool | str = True, return_tensors: str = 'pt', **_: Any):
        tokenized = self.tokenizer(text, return_tensors='pt', padding=padding)
        image_outputs = self.image_processor(images=images, return_tensors='pt')
        tokenized['pixel_values'] = image_outputs['pixel_values']
        return tokenized


class RandomInitAttention(nn.Module):
    """Small attention-shaped module with the module names used by the reference FBS code."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        # A lightweight content-aware attention approximation keeps the fallback fast
        # while every declared attention projection remains on the forward path.
        weights = torch.sigmoid((query * key).mean(dim=-1, keepdim=True))
        return self.proj(value * weights)


class RandomInitMLP(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.fc2 = nn.Linear(hidden_dim * expansion, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(hidden)))


class RandomInitVisionBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RandomInitAttention(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = RandomInitMLP(hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden))
        return hidden + self.mlp(self.norm2(hidden))


class RandomInitVisionFeaturizer(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int, grid_height: int, grid_width: int) -> None:
        super().__init__()
        self.patch_embed = SimpleNamespace(num_patches=grid_height * grid_width)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embed_dim = hidden_dim
        self.patch_projection = nn.Linear(3, hidden_dim)
        self.blocks = nn.ModuleList([RandomInitVisionBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"expected [batch, channels, height, width] pixels, got {tuple(pixel_values.shape)}")
        pixels = pixel_values.to(dtype=self.patch_projection.weight.dtype)
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, (self.grid_height, self.grid_width))
        hidden = pooled.permute(0, 2, 3, 1).reshape(pooled.shape[0], -1, 3)
        hidden = self.patch_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


class RandomInitVisionBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        num_patches: int = 8,
        num_images_in_input: int = 1,
    ) -> None:
        super().__init__()
        if num_patches != 8:
            raise ValueError("the declared random-init VLA architecture uses an 2x4 vision patch grid")
        self._num_patches = int(num_patches)
        self._num_images_in_input = int(num_images_in_input)
        self.featurizer = RandomInitVisionFeaturizer(hidden_dim, num_blocks, grid_height=2, grid_width=4)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.featurizer(pixel_values)

    def get_num_patches(self) -> int:
        return self._num_patches

    def get_num_images_in_input(self) -> int:
        return self._num_images_in_input


class RandomInitDecoderAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        weights = torch.sigmoid((query * key).mean(dim=-1, keepdim=True))
        return self.o_proj(value * weights)


class RandomInitDecoderMLP(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 2) -> None:
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.gate_proj = nn.Linear(hidden_dim, expanded_dim)
        self.up_proj = nn.Linear(hidden_dim, expanded_dim)
        self.down_proj = nn.Linear(expanded_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class RandomInitDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = RandomInitDecoderAttention(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = RandomInitDecoderMLP(hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class RandomInitLanguageCore(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([RandomInitDecoderLayer(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class RandomInitLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int) -> None:
        super().__init__()
        self.model = RandomInitLanguageCore(hidden_dim, vocab_size, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=False, output_hidden_states=True, return_dict=True):
        del attention_mask, position_ids, past_key_values, labels, use_cache, output_attentions
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        parameter_dtype = next(self.model.parameters()).dtype
        hidden = self.model(inputs_embeds.to(parameter_dtype))
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype))
        if return_dict:
            # Returning None intentionally exercises the reference runner's uncached
            # autoregressive branch, which is valid for this random-init architecture.
            return SimpleNamespace(hidden_states=[hidden], logits=logits, past_key_values=None)
        return (hidden, logits)


class RandomInitOpenVLA(nn.Module):
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 4096, num_patch_tokens: int = 8, num_action_tokens: int = 8, action_stats_dim: int = 7, vision_num_layers: int = 2, language_num_layers: int = 2) -> None:
        super().__init__()
        self.llm_dim = int(hidden_dim)
        self.vocab_size = int(vocab_size)
        self.num_action_tokens = int(num_action_tokens)
        self.language_model = RandomInitLanguageModel(self.llm_dim, self.vocab_size, language_num_layers)
        self.embed = self.language_model.get_input_embeddings()
        self.action_queries = nn.Embedding(self.num_action_tokens, self.llm_dim)
        self.vision_backbone = RandomInitVisionBackbone(
            hidden_dim=self.llm_dim,
            num_blocks=vision_num_layers,
            num_patches=num_patch_tokens,
            num_images_in_input=1,
        )
        self.norm_stats = {
            'fallback_random_init': {
                'q99': [1.0] * action_stats_dim,
                'q01': [-1.0] * action_stats_dim,
                'max': [1.0] * action_stats_dim,
                'min': [-1.0] * action_stats_dim,
                'mask': [True] * action_stats_dim,
            }
        }

    def set_version(self, *_args, **_kwargs) -> None:
        return None

    def get_input_embeddings(self):
        return self.embed

    def _process_vision_features(self, pixel_values: torch.Tensor, language_embeddings=None, use_film: bool = False):
        del language_embeddings, use_film
        return self.vision_backbone(pixel_values)

    def _build_multimodal_attention(self, input_embeddings: torch.Tensor, projected_patch_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        multimodal_embeddings = torch.cat([projected_patch_embeddings.to(input_embeddings.dtype), input_embeddings], dim=1)
        patch_mask = torch.ones((attention_mask.shape[0], projected_patch_embeddings.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
        return multimodal_embeddings, torch.cat([patch_mask, attention_mask], dim=1)

    def _prepare_input_for_action_prediction(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        extra_ids = torch.zeros((input_ids.shape[0], self.num_action_tokens), device=input_ids.device, dtype=input_ids.dtype)
        extra_mask = torch.ones((attention_mask.shape[0], self.num_action_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
        return torch.cat([input_ids, extra_ids], dim=1), torch.cat([attention_mask, extra_mask], dim=1)

    def _prepare_labels_for_action_prediction(self, labels: torch.Tensor, input_ids: torch.Tensor):
        if labels.shape[1] >= input_ids.shape[1]:
            return labels
        extra = torch.full((labels.shape[0], input_ids.shape[1] - labels.shape[1]), -100, device=labels.device, dtype=labels.dtype)
        return torch.cat([labels, extra], dim=1)

    def _process_action_masks(self, labels: torch.Tensor):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        if self.num_action_tokens > 0:
            mask[:, -self.num_action_tokens:] = True
        return mask

    def _replace_input_embeddings(self, input_embeddings: torch.Tensor, all_actions_mask: torch.Tensor, action_queries: torch.Tensor):
        replaced = input_embeddings.clone()
        num_tokens = min(self.num_action_tokens, action_queries.shape[1])
        if num_tokens > 0:
            replaced[:, -num_tokens:, :] = action_queries[:, :num_tokens, :].to(replaced.dtype)
        return replaced

    def get_action_stats(self, _key: str):
        return self.norm_stats['fallback_random_init']


def maybe_build_random_init_vla_bundle(model_dir: Path, prompt: str, device: torch.device, num_action_tokens: int, action_stats_dim: int = 7, hidden_dim: int = 256, vocab_size: int = 4096, num_patch_tokens: int = 8, vision_num_layers: int = 2, language_num_layers: int = 2):
    del prompt, device, num_action_tokens, action_stats_dim, hidden_dim, vocab_size
    del num_patch_tokens, vision_num_layers, language_num_layers
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"pretrained VLA model directory does not exist: {model_dir}; "
            "random model fallback is disabled in API examples"
        )
    return None
