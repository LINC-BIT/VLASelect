import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

def action_to_action_tokens(
    actions: torch.Tensor,          # [B, action_dim]
    action_stats: dict,              # q01, q99, mask
    bin_centers: torch.Tensor,       # [num_bins]
    vocab_size: int,
):
    """
    Return:
        action_token_ids: LongTensor [B, action_dim]
    """
    dtype = actions.dtype

    q01 = torch.tensor(action_stats["q01"], device=actions.device, dtype=dtype)
    q99 = torch.tensor(action_stats["q99"], device=actions.device, dtype=dtype)
    bin_centers = torch.tensor(bin_centers, device=actions.device, dtype=dtype)
    mask = torch.tensor(
        action_stats.get("mask", [1] * actions.shape[-1]),
        device=actions.device,
        dtype=torch.bool,
    )

    # ---- normalize to [-1, 1] ----
    normalized = actions.clone()
    normalized[:, mask] = 2 * (actions[:, mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1
    normalized = torch.clamp(normalized, -1, 1)

    # ---- nearest bin ----
    # [B, action_dim, num_bins]
    dist = torch.abs(
        normalized.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)
    )
    bin_index = dist.argmin(dim=-1)

    # ---- bin -> token id ----
    action_token_ids = vocab_size - bin_index - 1

    return action_token_ids.long()

class OpenVLAActor(nn.Module):
    def __init__(
        self,
        base_vla,
        use_single_lora,
    ):
        super().__init__()
        self.vla = base_vla
        self.lm_head = self.vla.get_output_embeddings()
        self.use_single_lora = use_single_lora

    # ==========================================================
    # forward (BC 用)
    # ==========================================================
    def forward(self, batch):
        """
        batch: dict
            {
                "input_ids": [B, seq_len],
                "pixel_values": [B, C, H, W],
                "attention_mask": [B, seq_len],
                "labels": [B, seq_len]
            }
        """
        outputs = self.vla(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def predict_action_batch(self, batch_obs, action_target=None):
        """
        batch_obs:
            input_ids      [B, T]
            pixel_values   [B, C, H, W]
            attention_mask [B, T]
        """
        if not self.use_single_lora:
            self.vla.set_adapter("actor")
        B = batch_obs["input_ids"].shape[0]

        actions = []
        logprobs = []
        entropies = []
        last_hiddens = []

        for i in range(B):
            single_obs = {
                k: v[i:i+1]    # 保留 batch 维度
                for k, v in batch_obs.items()
            }
            single_action_target = None
            if action_target is not None:
                single_action_target = action_target[i][None, :]

            action, logprob, entropy, last_hidden = self.predict_action_with_logprob(single_obs, action_target=single_action_target)  # [K, D]
            actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            last_hiddens.append(last_hidden)


        return np.concatenate(actions, axis=0), torch.cat(logprobs, dim=0).squeeze(-1), torch.cat(entropies, dim=0), torch.cat(last_hiddens, dim=0)   # [B, K, D]

    def predict_action_with_logprob(self, batch, action_target=None):
        """
        Predict actions and their log-probabilities for PPO.

        Args:
            input_ids: Input token ids [B, seq_len]
            kwargs: passed to predict_action (pixel_values, attention_mask, etc.)

        Returns:
            actions: [B, action_dim] continuous actions (already unnormalized)
            logprob: [B] log-probabilities of the selected bin
            entropy: [B] entropy of the action

        """
        vocab_size = self.vla.vocab_size
        bin_centers = self.vla.bin_centers

        # 1️⃣ forward to get hidden_states and actions
        actions, hidden_states = self.vla.predict_action(**batch, unnorm_key="maniskill", do_sample=False)
        # actions: [B, action_dim], hidden_states: [B, action_dim, hid_size]
        action_raw_shape = actions.shape
        last_hidden_state = hidden_states[:, 0]
        hidden_states = hidden_states.view(1, action_raw_shape[0], action_raw_shape[1], -1)
        hidden_states = hidden_states[:, 0, :, :]
        hidden_states = self.lm_head(hidden_states)
        actions = actions[0, :][None, :]

        B, action_dim, _ = hidden_states.shape
        num_bins = bin_centers.shape[0]

        # 2️⃣ softmax over vocab
        vocab_probs = torch.softmax(hidden_states, dim=-1)  # [B, action_dim, vocab_size]

        # 3️⃣ bin -> token_id mapping
        # token_id = vocab_size - bin_index - 1
        bin_token_ids = vocab_size - torch.arange(num_bins, device=hidden_states.device) - 1  # [num_bins]

        # 4️⃣ gather bin token probs
        gather_idx = bin_token_ids.view(1, 1, -1).expand(B, action_dim, -1)  # [B, action_dim, num_bins]
        bin_probs = torch.gather(vocab_probs, dim=-1, index=gather_idx)  # [B, action_dim, num_bins]

        # 5️⃣ left edge: all tokens < min(bin_token_ids) -> accumulate to first bin
        left_mask = torch.arange(vocab_probs.shape[-1], device=hidden_states.device) < bin_token_ids.min()
        if left_mask.any():
            bin_probs[:, :, 0] += vocab_probs[:, :, left_mask].sum(dim=-1)

        # 6️⃣ right edge: all tokens > max(bin_token_ids) -> accumulate to last bin
        right_mask = torch.arange(vocab_probs.shape[-1], device=hidden_states.device) > bin_token_ids.max()
        if right_mask.any():
            bin_probs[:, :, -1] += vocab_probs[:, :, right_mask].sum(dim=-1)

        # 7️⃣ selected bin index for each action
        # Find the nearest bin for each action
        if action_target is not None:
            actions = action_target

        normalized = actions[:, :, None] - bin_centers.reshape(1, 1, -1)  # [B, action_dim, num_bins]
        bin_index = torch.LongTensor(np.abs(normalized).argmin(axis=-1)).unsqueeze(-1).to(vocab_probs.device)  # [B, action_dim]

        # 8️⃣ compute log-probabilities
        logprob = torch.log(torch.gather(bin_probs, dim=-1, index=bin_index)).sum(1)  # [B]

        entropy = -torch.sum(
            bin_probs * torch.log(bin_probs + 1e-8),
            dim=-1
        ).sum(1)

        return actions, logprob, entropy, last_hidden_state

    # ==========================================================
    # 保存 / 加载（只保存 LoRA）
    # ==========================================================
    def save_lora(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 只保存 LoRA 参数并转换为 bfloat16
        lora_state = {
            k: v.to(torch.bfloat16) 
            for k, v in self.vla.state_dict().items() 
            if "lora" in k and "actor" in k
        }

        torch.save(lora_state, path)
        print(f"[OpenVLAActor] LoRA saved to {path} (bfloat16)")

    def load_lora(self, path):
        lora_state = torch.load(path, map_location=self.device)

        # 加载后可以根据模型需要转换 dtype
        self.vla.load_state_dict(lora_state, strict=False)
        print(f"[OpenVLAActor] LoRA loaded from {path}")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class OpenVLACritic(nn.Module):
    def __init__(
        self,
        base_vla,
        use_single_lora
    ):
        super().__init__()
        self.vla = base_vla
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(np.array(self.vla.llm_dim).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.use_single_lora = use_single_lora

    def get_value(self, batch):
        """
        Predict actions and their log-probabilities for PPO.

        Args:
            input_ids: Input token ids [B, seq_len]
            kwargs: passed to predict_action (pixel_values, attention_mask, etc.)

        Returns:
            values: [B] value of the obs
        """
        if not self.use_single_lora:
            self.vla.set_adapter("critic")
        hidden_states = self.vla.language_model(output_hidden_states=True, return_dict=True, **batch)['hidden_states'][-1]
        # 计算 value
        state_latent = hidden_states[:, -1]
        values = self.value_head(state_latent).squeeze(-1)
        return values

    # ==========================================================
    # 保存 / 加载（只保存 LoRA）
    # ==========================================================
    def save_lora(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 只保存 LoRA 参数并转换为 bfloat16
        lora_state = {
            k: v.to(torch.bfloat16) 
            for k, v in self.vla.state_dict().items() 
            if "lora" in k and "critic" in k
        }

        torch.save(lora_state, path)
        print(f"[OpenVLAActor] LoRA saved to {path} (bfloat16)")

    def load_lora(self, path):
        lora_state = torch.load(path, map_location=self.device)

        # 加载后可以根据模型需要转换 dtype
        self.vla.load_state_dict(lora_state, strict=False)
        print(f"[OpenVLAActor] LoRA loaded from {path}")

class OpenVLA(nn.Module):
    def __init__(
        self,
        model_path,
        data_stat,
        device="cuda",
        use_single_lora=False,
        pretrained_lora_ckpt=None,  # 预训练 LoRA
        lora_ckpt=None,
        actor_lora_ckpt=None,
        critic_lora_ckpt=None,
        latest_critic_head=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.use_single_lora = use_single_lora
        self.actor_lora_ckpt = actor_lora_ckpt
        self.critic_lora_ckpt = critic_lora_ckpt

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ---------- 1️⃣ 加载基础 VLA ----------
        self.base_vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)

        # ---------- 2️⃣ 归一化 stats ----------
        self.base_vla.norm_stats["maniskill"] = {
            "action": {k: np.array(v) for k, v in data_stat.items()}
        }

        # ---------- 3️⃣ 如果有预训练 LoRA, 加载 ----------
        if pretrained_lora_ckpt is not None:
            print(f"[OpenVLA] Loading pretrained LoRA from {pretrained_lora_ckpt}")
            lora_state = torch.load(pretrained_lora_ckpt, map_location=device)
            self.base_vla = get_peft_model(self.base_vla, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj","k_proj","v_proj","o_proj","gate_proj",
                    "up_proj","down_proj","qkv","proj","fc1","fc2","fc3","q","kv"
                ],
            ))
            self.base_vla.load_state_dict(lora_state, strict=False)
            self.base_vla.merge_and_unload()

        self.base_vla.requires_grad_(False)

        if not use_single_lora:
            actor_lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "qkv",
                    "proj",
                    "fc1",
                    "fc2",
                    "fc3",
                    "q",
                    "kv"
                ],
            )

            self.base_vla = get_peft_model(self.base_vla, actor_lora_cfg, adapter_name='actor')

            if actor_lora_ckpt is not None and os.path.exists(actor_lora_ckpt):
                print(f"[OpenVLAActor] Loading LoRA checkpoint from {actor_lora_ckpt}")
                self.base_vla.load_state_dict(
                    torch.load(actor_lora_ckpt, map_location=device),
                    strict=False,
                )

            critic_lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "qkv",
                    "proj",
                    "fc1",
                    "fc2",
                    "fc3",
                    "q",
                    "kv"
                ],
            )

            self.base_vla.add_adapter(peft_config=critic_lora_cfg, adapter_name='critic')

            if critic_lora_ckpt is not None and os.path.exists(critic_lora_ckpt):
                print(f"[OpenVLACritic] Loading LoRA checkpoint from {critic_lora_ckpt}")
                self.base_vla.load_state_dict(
                    torch.load(critic_lora_ckpt, map_location=device),
                    strict=False,
                )

            self.base_vla.print_trainable_parameters()
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "qkv",
                    "proj",
                    "fc1",
                    "fc2",
                    "fc3",
                    "q",
                    "kv"
                ],
            )

            self.base_vla = get_peft_model(self.base_vla, lora_cfg)

            if lora_ckpt is not None and os.path.exists(lora_ckpt):
                print(f"[OpenVLA] Loading LoRA checkpoint from {lora_ckpt}")
                self.base_vla.load_state_dict(
                    torch.load(lora_ckpt, map_location=device),
                    strict=False,
                )

        # 2️⃣ actor
        self.actor = OpenVLAActor(self.base_vla, use_single_lora=use_single_lora)

        # 3️⃣ critic
        self.critic = OpenVLACritic(self.base_vla, use_single_lora=use_single_lora)

        if latest_critic_head is not None and os.path.exists(latest_critic_head):
            print(f"[OpenVLACritic] Loading Head checkpoint from {latest_critic_head}")
            self.critic.value_head.load_state_dict(
                torch.load(latest_critic_head, map_location=device),
                strict=False,
            )

    def forward(self, batch, action_target=None):
        """
        actor 和 critic 完全分开 forward
        Returns:
            actions, logprobs, values
        """
        # actor forward
        actions, logprob, entropy, last_hiddens = self.actor.predict_action_batch(batch, action_target)

        if not self.use_single_lora:
            # critic forward (独立 LoRA)
            value = self.critic.get_value(batch)
        else:
            value = self.critic.value_head(last_hiddens).squeeze(-1)

        return actions, logprob, entropy, value

    def save_lora(self, value_head_path=None, lora_path=None, actor_path=None, critic_path=None):
        if not self.use_single_lora:
            assert actor_path is not None and critic_path is not None, "Path is none"
            self.actor.save_lora(actor_path)
            self.critic.save_lora(critic_path)
        else:
            assert lora_path is not None
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)

            # 只保存 LoRA 参数并转换为 bfloat16
            lora_state = {
                k: v.to(torch.bfloat16) 
                for k, v in self.base_vla.state_dict().items() 
                if "lora" in k
            }

            torch.save(lora_state, lora_path)
            print(f"[OpenVLAActor] LoRA saved to {lora_path} (bfloat16)")

        head_state_dict = {
            k: v.to(torch.bfloat16) 
            for k, v in self.critic.value_head.state_dict().items() 
        }
        torch.save(head_state_dict, value_head_path)
        print(f"[OpenVLACritic] Value Head saved to {value_head_path} (bfloat16)")