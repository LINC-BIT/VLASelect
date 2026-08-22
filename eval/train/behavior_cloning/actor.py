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
        model_path,
        data_stat,
        device,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_ckpt_path=None,   # ← 用于断点恢复
    ):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.use_lora = use_lora

        # ---------- processor ----------
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ---------- base model ----------
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)

        # ---------- norm stats ----------
        self.vla.norm_stats["maniskill"] = {
            "action": {
                k: np.array(v)
                for k, v in data_stat.items()
            }
        }

        # ---------- LoRA ----------
        if use_lora:
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

            self.vla = get_peft_model(self.vla, lora_cfg)

            if lora_ckpt_path is not None and os.path.exists(lora_ckpt_path):
                print(f"[OpenVLAActor] Loading LoRA checkpoint from {lora_ckpt_path}")
                self.vla.load_state_dict(
                    torch.load(lora_ckpt_path, map_location=device),
                    strict=False,
                )

            self.vla.print_trainable_parameters()

        self.vla.train()

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

    # ==========================================================
    # 保存 / 加载（只保存 LoRA）
    # ==========================================================
    def save_lora(self, path):
        assert self.use_lora, "LoRA is not enabled"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 只保存 LoRA 参数并转换为 bfloat16
        lora_state = {
            k: v.to(torch.bfloat16) 
            for k, v in self.vla.state_dict().items() 
            if "lora" in k
        }

        torch.save(lora_state, path)
        print(f"[OpenVLAActor] LoRA saved to {path} (bfloat16)")

    def load_lora(self, path):
        assert self.use_lora, "LoRA is not enabled"
        lora_state = torch.load(path, map_location=self.device)

        # 加载后可以根据模型需要转换 dtype
        self.vla.load_state_dict(lora_state, strict=False)
        print(f"[OpenVLAActor] LoRA loaded from {path}")
