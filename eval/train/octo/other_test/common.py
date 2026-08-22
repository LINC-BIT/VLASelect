from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from tensorboard.backend.event_processing import event_accumulator

from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn
from ours.utils.dl.common.model import get_module, set_module
from train.octo.model import Actor
from train.octo.other_test.peft_layers import freeze_non_lora_parameters, inject_lora_modules


TOY_CNN_CONV_LAYER_NAMES = [f"rgb_encoder.cnn.{i}" for i in [0, 6, 12]] + [f"depth_encoder.cnn.{i}" for i in [0, 6, 12]]
TOY_CNN_ACTOR_FC_LAYER_NAMES = ["decoder.0", "rgb_encoder.fc.0.0", "depth_encoder.fc.0.0"]
TOY_CNN_AGENT_FC_LAYER_NAMES = ["actor_mean.0", "critic.0"]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_run_metadata(
    run_name: str,
    args: Any,
    script_path: Path,
    *,
    extra_files: Iterable[Path] = (),
) -> None:
    code_dir = Path("ckpt") / run_name / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(script_path, code_dir / "script.py")
    for extra_file in extra_files:
        target_name = extra_file.name
        if target_name == script_path.name:
            target_name = f"extra_{target_name}"
        shutil.copyfile(extra_file, code_dir / target_name)

    if is_dataclass(args):
        args_payload = asdict(args)
    else:
        args_payload = dict(vars(args))
    with (code_dir / "args.txt").open("w", encoding="utf-8") as f:
        for key, value in args_payload.items():
            f.write(f"{key}: {value}\n")
    save_json(Path("ckpt") / run_name / "args.json", args_payload)


def _disable_inplace_relu(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.ReLU):
            submodule.inplace = False


def build_checkpoint_compatible_agent(reference_module, args, device: torch.device) -> nn.Module:
    actor = Actor(42, 4, 1, False).to(device=device)
    set_module(actor, "rgb_encoder.fc.0", svd_decompose_linear(get_module(actor, "rgb_encoder.fc.0")))
    set_module(actor, "depth_encoder.fc.0", svd_decompose_linear(get_module(actor, "depth_encoder.fc.0")))

    actor_example = {
        "rgb": torch.rand((1, 3, 128, 128), device=device),
        "depth": torch.rand((1, 1, 128, 128), device=device),
        "state": torch.rand((1, 42), device=device),
    }
    add_FBS_into_cnn(
        actor,
        TOY_CNN_CONV_LAYER_NAMES,
        TOY_CNN_ACTOR_FC_LAYER_NAMES,
        actor_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample["rgb"], sample["depth"], sample["state"]),
    )

    state_max, state_min = torch.load(args.state_norm_stats_path, map_location="cpu")
    state_max = state_max.to(device)
    state_min = state_min.to(device)
    agent = reference_module.Agent(actor, 256 * 3, state_max, state_min, args.normalize_states, args.actor_logstd).to(device)
    actor.decoder = nn.Identity()

    agent_example = {
        "rgb": torch.rand((1, 128, 128, 3), device=device),
        "depth": torch.rand((1, 128, 128, 1), device=device),
        "state": torch.rand((1, 42), device=device),
    }
    add_FBS_into_cnn(
        agent,
        [],
        TOY_CNN_AGENT_FC_LAYER_NAMES,
        agent_example,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(agent.load_state_dict(checkpoint["agent"], strict=True))
    _disable_inplace_relu(agent)
    return agent.to(device)


def build_original_agent(reference_module, args, device: torch.device, env_kwargs: dict[str, Any]) -> nn.Module:
    del env_kwargs
    agent = build_checkpoint_compatible_agent(reference_module, args, device)
    return agent.to(device)


def build_peft_agent(
    reference_module,
    args,
    device: torch.device,
    env_kwargs: dict[str, Any],
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> nn.Module:
    agent = build_checkpoint_compatible_agent(reference_module, args, device)
    replaced_linear, replaced_conv = inject_lora_modules(agent, rank=rank, alpha=alpha, dropout=dropout)
    freeze_non_lora_parameters(agent, train_actor_logstd=True)
    print(
        f"Injected LoRA adapters: linear_layers={replaced_linear}, conv_layers={replaced_conv}, "
        f"trainable_params={count_trainable_parameters(agent)}, total_params={count_parameters(agent)}"
    )
    return agent.to(device)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def load_tb_scalars(run_dir: Path, tag: str) -> list[Any]:
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return []
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    try:
        accumulator.Reload()
    except Exception:
        return []
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def load_gpu_metrics_rows(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "analysis" / "gpu_metrics.csv"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []
