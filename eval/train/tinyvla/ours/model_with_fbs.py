import contextlib
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.append(".")

import torch

from ours.libs.train_with_fbs.lib_cnn import get_model_size
from ours.pretrain_fbs_model.main import add_FBS_into_transformer


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _vision_transformer_layers(actor) -> Tuple[List[str], List[str], List[str], List[str]]:
    qkv_layers: List[str] = []
    proj_layers: List[str] = []
    ff1_layers: List[str] = []
    ff2_layers: List[str] = []

    vision_backbone = actor.vla.vision_backbone
    for featurizer_name in ("featurizer", "fused_featurizer"):
        featurizer = getattr(vision_backbone, featurizer_name, None)
        blocks = getattr(featurizer, "blocks", None)
        if blocks is None:
            continue
        for block_i in range(len(blocks)):
            prefix = f"vla.vision_backbone.{featurizer_name}.blocks.{block_i}"
            qkv_layers.append(f"{prefix}.attn.qkv")
            proj_layers.append(f"{prefix}.attn.proj")
            ff1_layers.append(f"{prefix}.mlp.fc1")
            ff2_layers.append(f"{prefix}.mlp.fc2")

    return qkv_layers, proj_layers, ff1_layers, ff2_layers


def _language_model_layers(actor):
    language_model_core = getattr(actor.vla.language_model, "model", None)
    layers = getattr(language_model_core, "layers", None)
    if layers is None:
        return [], [], [], []
    qkv_layers = []
    proj_layers = []
    ff1_layers = []
    ff2_layers = []
    for block_i in range(len(layers)):
        prefix = f"model.layers.{block_i}"
        qkv_layers.extend([[f"{prefix}.self_attn.{name}_proj"] for name in ("q", "k", "v")])
        proj_layers.append(f"{prefix}.self_attn.o_proj")
        ff1_layers.append([f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"])
        ff2_layers.append(f"{prefix}.mlp.down_proj")
    return qkv_layers, proj_layers, ff1_layers, ff2_layers


def convert_to_fbs_model(
    actor,
    device: torch.device,
    max_sparsity: float = 0.9,
    fbs_r: int = 16,
    dtype: torch.dtype = torch.bfloat16,
):
    print(f"original model: {get_model_size(actor, True):.3f}MB")

    sample_batch = {
        "rgbs": torch.randint(0, 256, (1, 224, 448, 3), dtype=torch.uint8),
        "states": torch.randn(1, actor.state_dim, device=device, dtype=torch.float32),
        "input_ids": torch.randint(0, min(actor.full_vocab_size, 1000), (1, 16), device=device),
    }

    qkv_layers, proj_layers, ff1_layers, ff2_layers = _vision_transformer_layers(actor)
    if qkv_layers:
        with _autocast_context(device, dtype):
            actor = add_FBS_into_transformer(
                actor.to(device),
                qkv_layers,
                proj_layers,
                ff1_layers,
                ff2_layers,
                sample_batch,
                max_sparsity,
                fbs_r,
                lambda model, batch: model(
                    rgbs=batch["rgbs"],
                    states=batch["states"],
                    mode="policy",
                )[0].float().sum(),
            ).cpu()
    else:
        print("[setup] skipping FBS vision conversion because the current actor has no compatible transformer blocks")

    qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2 = _language_model_layers(actor)
    if qkv_layers2:
        with _autocast_context(device, dtype):
            actor.vla.language_model = add_FBS_into_transformer(
                actor.vla.language_model.to(device),
                qkv_layers2,
                proj_layers2,
                ff1_layers2,
                ff2_layers2,
                sample_batch,
                max_sparsity,
                fbs_r,
                lambda model, batch: model(
                    input_ids=batch["input_ids"].to(device),
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1].float().mean(),
            ).cpu()
    else:
        print("[setup] skipping FBS language conversion because the current actor has no compatible decoder blocks")

    actor.vla.to(dtype=dtype)
    return actor


if __name__ == "__main__":
    from train.tinyvla.model_impl.online_rl_open_cabinet_drawer import EdgeVLAActorCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = EdgeVLAActorCritic(
        Path("eval/ckpt/vla_adapter_new/LIBERO-Object"),
        device=device,
    )
    convert_to_fbs_model(policy, device)
