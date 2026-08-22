import sys

sys.path.append('./')  # 添加当前目录到sys.path，以便导入ours.pretrain_fbs_model.main
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ours.libs.gen_scaling_law_data_points import generate_small_model_v2
from ours.libs.train_with_fbs.lib_transformer import StaticFBS
from ours.libs.train_with_fbs.lib_cnn import get_model_size
from ours.pretrain_fbs_model.main import add_FBS_into_transformer
from train.vla_adapter_new.model_impl.online_rl_hold_cube_in_hand import HandVLAAdapterActorCritic


actor = HandVLAAdapterActorCritic(Path('eval/ckpt/vla_adapter_new/LIBERO-Object'), 'cuda')

device = 'cuda'
dtype = torch.bfloat16


actor.to(device=device, dtype=dtype)

print(f'original model: {get_model_size(actor, True):.3f}MB')

sample_batch = {
    "rgbs": torch.randn(1, 6, 224, 224, device=device, dtype=dtype),
    "states": torch.randn(1, 105, device=device, dtype=dtype),
    'input_ids': torch.randint(0, 1000, (1, 16), device=device),
    "action_bins": torch.randint(
        0,
        actor.num_action_bins,
        (1, actor.env_action_dim),
        device=device,
        dtype=torch.long,
    ),
}


# -------- 添加FBS ---------------
qkv_layers, proj_layers, ff1_layers, ff2_layers = [], [], [], []
for block_i in range(len(actor.vla.vision_backbone.featurizer.blocks)):
    qkv_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.qkv']
    proj_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.proj']
    ff1_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc1']
    ff2_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc2']
for block_i in range(len(actor.vla.vision_backbone.fused_featurizer.blocks)):
    qkv_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.qkv']
    proj_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.proj']
    ff1_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc1']
    ff2_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc2']

from ours.pretrain_fbs_model.main import add_FBS_into_transformer
with torch.autocast(device_type='cuda', dtype=dtype):
    actor = add_FBS_into_transformer(
        actor.to(device),
        qkv_layers, proj_layers, ff1_layers, ff2_layers,
        sample_batch, 0.9, 16, 
        lambda model, batch: model(
            rgbs=batch['rgbs'],
            states=batch['states'],
            mode='action_and_value',
            action_bins=batch['action_bins'],
        )[1].sum()
    ).cpu()

# 语言模型的qkv/FFN结构比较独特，单独添加fbs模块
qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2 = [], [], [], []
for block_i in range(len(actor.vla.language_model.model.layers)):
    qkv_layers2 += [[f'model.layers.{block_i}.self_attn.{k}_proj'] for k in ['q', 'k', 'v']]
    proj_layers2 += [f'model.layers.{block_i}.self_attn.o_proj']
    ff1_layers2 += [[f'model.layers.{block_i}.mlp.gate_proj', f'model.layers.{block_i}.mlp.up_proj']]
    ff2_layers2 += [f'model.layers.{block_i}.mlp.down_proj']

with torch.autocast(device_type='cuda', dtype=dtype):
    actor.vla.language_model = add_FBS_into_transformer(
        actor.vla.language_model.to(device),
        qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2,
        sample_batch, 0.9, 16, 
        lambda model, batch: model(
            input_ids=batch['input_ids'].to(device),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1].mean()
    ).cpu()

def prepare_model_for_rollout(model: HandVLAAdapterActorCritic) -> HandVLAAdapterActorCritic:
    target_device = torch.device(device)
    model.to(device=device)
    model.vla.to(device=device, dtype=dtype)
    model.state_projector.to(device=device, dtype=torch.float32)
    model.context_projector.to(device=device, dtype=torch.float32)
    model.actor_head.to(device=device, dtype=torch.float32)
    model.value_head.to(device=device, dtype=torch.float32)

    moved_tensors = 0
    for _, parameter in model.named_parameters():
        if parameter.device != target_device:
            parameter.data = parameter.data.to(device=target_device)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(device=target_device)
            moved_tensors += 1
    for _, buffer in model.named_buffers():
        if buffer.device != target_device:
            buffer.data = buffer.data.to(device=target_device)
            moved_tensors += 1
    if moved_tensors > 0:
        print(f"[prepare] moved {moved_tensors} tensors to {target_device}")
    return model


def replace_static_fbs_with_identity(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, StaticFBS):
            setattr(module, name, nn.Identity())
            replaced += 1
            continue
        replaced += replace_static_fbs_with_identity(child)
    return replaced


actor = prepare_model_for_rollout(actor)
small_model = generate_small_model_v2(actor, qkv_layers, proj_layers, ff1_layers, ff2_layers)
small_model.vla.language_model = generate_small_model_v2(small_model.vla.language_model, qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2)
num_static_fbs_replaced = replace_static_fbs_with_identity(small_model)
print(f'replaced StaticFBS with Identity in small_model: {num_static_fbs_replaced}')
print(f'small model: {get_model_size(small_model, True):.3f}MB')
small_model = prepare_model_for_rollout(small_model)


def build_rollout_batch(batch_size: int):
    return {
        "rgbs": torch.randint(0, 256, (batch_size, 6, 224, 224), dtype=torch.uint8),
        "states": np.random.randn(batch_size, 105).astype(np.float32),
    }


def benchmark_rollout_time(
    model: HandVLAAdapterActorCritic,
    model_name: str,
    batch_sizes,
    warmup_iters: int = 3,
    measure_iters: int = 10,
):
    model.eval()
    results = []

    with torch.inference_mode():
        for batch_size in batch_sizes:
            print(f"[benchmark] model={model_name} batch_size={batch_size} start")
            batch = build_rollout_batch(batch_size)

            for _ in range(warmup_iters):
                model.get_action_and_value(
                    rgbs=batch["rgbs"],
                    states=batch["states"],
                    deterministic=False,
                )
            torch.cuda.synchronize()

            elapsed_ms = []
            for _ in range(measure_iters):
                start = time.perf_counter()
                model.get_action_and_value(
                    rgbs=batch["rgbs"],
                    states=batch["states"],
                    deterministic=False,
                )
                torch.cuda.synchronize()
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

            results.append(
                {
                    "batch_size": batch_size,
                    "mean_ms": float(np.mean(elapsed_ms)),
                    "std_ms": float(np.std(elapsed_ms)),
                }
            )
            print(
                f"[benchmark] model={model_name} batch_size={batch_size} "
                f"mean={results[-1]['mean_ms']:.2f}ms std={results[-1]['std_ms']:.2f}ms"
            )

    return results


def plot_results(actor_results, small_results, output_path: Path, small_model_label: str) -> None:
    batch_sizes = [item["batch_size"] for item in actor_results]
    actor_times = [item["mean_ms"] for item in actor_results]
    small_times = [item["mean_ms"] for item in small_results]

    plt.figure(figsize=(9, 6))
    plt.plot(batch_sizes, actor_times, marker="o", linewidth=2, label="actor")
    plt.plot(batch_sizes, small_times, marker="o", linewidth=2, label=small_model_label)
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.title("Rollout Time vs Batch Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


batch_sizes = [1, 16, 32, 64, 128, 256]
output_dir = Path(__file__).resolve().parent / "perf_test_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

small_model_label = "small_model_no_static_fbs"

actor_results = benchmark_rollout_time(actor, "actor", batch_sizes)
small_results = benchmark_rollout_time(small_model, small_model_label, batch_sizes)

results = {
    "batch_sizes": batch_sizes,
    "actor": actor_results,
    small_model_label: small_results,
    "num_static_fbs_replaced": num_static_fbs_replaced,
}

results_path = output_dir / "rollout_speed_results_no_static_fbs.json"
figure_path = output_dir / "rollout_speed_comparison_no_static_fbs.png"

with results_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

plot_results(actor_results, small_results, figure_path, small_model_label)

print(f"Saved benchmark results to {results_path}")
print(f"Saved comparison figure to {figure_path}")
for name, rows in [("actor", actor_results), (small_model_label, small_results)]:
    print(name)
    for row in rows:
        print(
            f"  batch_size={row['batch_size']:>3d} "
            f"time={row['mean_ms']:.2f}ms "
            f"+/- {row['std_ms']:.2f}ms"
        )
