"""Compare gradients on neurons shared by the FBS model and its small model.

This script follows the model loading and small-model generation path used by
``online_rl_ours_single_agent_cl.sh``.  It stops immediately after one backward
pass through each model and reports cosine similarity only for parameters that
belong to retained neurons (and their corresponding input columns).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class GradientPair:
    name: str
    large: torch.Tensor
    small: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute gradient cosine similarity on retained neurons."
    )
    parser.add_argument(
        "--env-id", default="PickCubeObjectScaleUp1p2-v1"
    )
    parser.add_argument(
        "--env-config-path",
        default=(
            "datasets/PickCube-v1/motionplanning/"
            "trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json"
        ),
    )
    parser.add_argument(
        "--state-norm-stats-path",
        default="ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/"
            "20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt"
        ),
    )
    parser.add_argument("--max-sparsity", type=float, default=0.8)
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument("--num-eval-steps", type=int, default=50)
    parser.add_argument(
        "--num-gradient-samples",
        type=int,
        default=5,
        help="Number of independently collected samples evaluated after generation.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample-strategy",
        choices=("target-batch", "target-single", "target-single-traj"),
        default="target-single-traj",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for model inference and backward.",
    )
    return parser.parse_args()


def _find_next_param_module(model: nn.Module, layer_name: str):
    found_current = False
    for candidate_name, candidate_module in model.named_modules():
        if candidate_name == layer_name:
            found_current = True
            continue
        if not found_current:
            continue
        if candidate_name.startswith(layer_name + "."):
            continue
        if isinstance(candidate_module, (nn.Conv2d, nn.Linear)):
            return candidate_name, candidate_module
    return None, None


def _parameter_gradient(parameter: nn.Parameter, name: str) -> torch.Tensor:
    if parameter.grad is None:
        raise RuntimeError(f"Parameter {name!r} did not receive a gradient")
    return parameter.grad.detach()


def collect_intersection_gradient_pairs(
    large_model: nn.Module,
    small_model: nn.Module,
    pruning_info: dict,
) -> tuple[list[GradientPair], list[str]]:
    """Align small-model gradients with matching retained large-model entries."""
    from ours.utils.dl.common.model import get_module

    pairs: list[GradientPair] = []
    mapped_small_parameter_ids: set[int] = set()
    selected_indices = pruning_info["selected_indices"]

    def add_pair(name: str, large_grad: torch.Tensor, small_parameter: nn.Parameter):
        small_grad = _parameter_gradient(small_parameter, name)
        if large_grad.shape != small_grad.shape:
            raise RuntimeError(
                f"Gradient mapping for {name} has different shapes: "
                f"{tuple(large_grad.shape)} vs {tuple(small_grad.shape)}"
            )
        pairs.append(GradientPair(name, large_grad.reshape(-1), small_grad.reshape(-1)))
        mapped_small_parameter_ids.add(id(small_parameter))

    for layer_name, indices in selected_indices.items():
        indices = torch.as_tensor(indices, dtype=torch.long)
        large_layer = get_module(large_model, layer_name)
        small_layer = get_module(small_model, layer_name)
        if not isinstance(small_layer, nn.Sequential) or len(small_layer) == 0:
            raise RuntimeError(f"Unexpected generated layer at {layer_name}: {small_layer}")

        if hasattr(large_layer, "raw_conv2d"):
            large_core = large_layer.raw_conv2d
            small_core = small_layer[0]
            is_conv = True
        elif hasattr(large_layer, "raw_linear"):
            large_core = large_layer.raw_linear
            small_core = small_layer[0]
            is_conv = False
        else:
            raise RuntimeError(f"No raw Conv2d/Linear found at FBS layer {layer_name}")

        large_weight_grad = _parameter_gradient(
            large_core.weight, f"{layer_name}.large_weight"
        )
        device_indices = indices.to(large_weight_grad.device)
        add_pair(
            f"{layer_name}.weight[retained_outputs]",
            large_weight_grad.index_select(0, device_indices),
            small_core.weight,
        )
        if large_core.bias is not None:
            add_pair(
                f"{layer_name}.bias[retained_outputs]",
                _parameter_gradient(
                    large_core.bias, f"{layer_name}.large_bias"
                ).index_select(0, device_indices),
                small_core.bias,
            )

        large_next_name, large_next = _find_next_param_module(
            large_model, layer_name
        )
        small_next_name, small_next = _find_next_param_module(
            small_model, layer_name
        )
        if (large_next_name, type(large_next)) != (small_next_name, type(small_next)):
            raise RuntimeError(
                f"Next-layer mismatch after {layer_name}: "
                f"large={large_next_name}/{type(large_next).__name__}, "
                f"small={small_next_name}/{type(small_next).__name__}"
            )
        if large_next is None:
            continue

        large_next_grad = _parameter_gradient(
            large_next.weight, f"{large_next_name}.large_weight"
        )
        if isinstance(large_next, nn.Conv2d):
            aligned_large_next_grad = large_next_grad.index_select(1, device_indices)
        elif is_conv:
            block_size = small_next.weight.shape[1] // len(indices)
            column_indices = torch.cat(
                [
                    torch.arange(
                        int(index) * block_size,
                        (int(index) + 1) * block_size,
                        device=large_next_grad.device,
                    )
                    for index in indices
                ]
            )
            aligned_large_next_grad = large_next_grad.index_select(1, column_indices)
        else:
            aligned_large_next_grad = large_next_grad.index_select(1, device_indices)
        add_pair(
            f"{large_next_name}.weight[retained_inputs]",
            aligned_large_next_grad,
            small_next.weight,
        )

    large_named_parameters = dict(large_model.named_parameters())
    for name, small_parameter in small_model.named_parameters():
        if not small_parameter.requires_grad or id(small_parameter) in mapped_small_parameter_ids:
            continue
        large_parameter = large_named_parameters.get(name)
        if large_parameter is None or large_parameter.shape != small_parameter.shape:
            continue
        # For example, Agent.forward() does not use actor_logstd. Such a tensor
        # has no gradient in either model and is not part of the gradient set.
        if large_parameter.grad is None and small_parameter.grad is None:
            mapped_small_parameter_ids.add(id(small_parameter))
            continue
        if large_parameter.grad is None or small_parameter.grad is None:
            continue
        add_pair(
            name,
            _parameter_gradient(large_parameter, name),
            small_parameter,
        )

    unmatched = [
        name
        for name, parameter in small_model.named_parameters()
        if parameter.requires_grad and id(parameter) not in mapped_small_parameter_ids
    ]
    return pairs, unmatched


def cosine_similarity(pairs: Iterable[GradientPair]) -> tuple[float, int]:
    pairs = list(pairs)
    if not pairs:
        raise RuntimeError("No intersecting trainable gradients were found")
    large = torch.cat([pair.large.float() for pair in pairs])
    small = torch.cat([pair.small.float() for pair in pairs])
    if large.norm() == 0 or small.norm() == 0:
        raise RuntimeError("The intersecting gradient vector has zero norm")
    return F.cosine_similarity(large, small, dim=0).item(), large.numel()


def main() -> None:
    cli_args = parse_args()
    if cli_args.num_gradient_samples < 1:
        raise ValueError("--num-gradient-samples must be at least 1")
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    device = torch.device(cli_args.device)

    from train.octo.ours_single_agent import online_rl_cl as online
    from ours.pretrain_fbs_model.main import generate_small_cnn_with_verify

    args = online.Args()
    args.seed = cli_args.seed
    args.cuda = device.type == "cuda"
    args.env_id = cli_args.env_id
    args.env_config_path = cli_args.env_config_path
    args.state_norm_stats_path = cli_args.state_norm_stats_path
    args.checkpoint = cli_args.checkpoint
    args.max_sparsity = cli_args.max_sparsity
    args.num_envs = 1
    args.num_eval_envs = cli_args.num_eval_envs
    args.num_eval_steps = cli_args.num_eval_steps
    args.capture_video = False
    args.small_model_generation_strategy = cli_args.sample_strategy
    args.small_model_generation_policy = "large"

    # load_agent follows the training entry point and reads these two globals.
    online.args = args
    online.device = device
    large_model = online.load_agent()

    with open(args.env_config_path, "r", encoding="utf-8") as file:
        env_kwargs = json.load(file)["env_info"]["env_kwargs"]
    env_kwargs["sim_backend"] = "physx_cuda" if device.type == "cuda" else "physx_cpu"
    env_kwargs.pop("num_envs", None)
    env_kwargs.pop("reward_mode", None)

    envs, eval_envs = online.make_envs_for_env_id(
        args, args.env_id, env_kwargs, "gradient-similarity", 0
    )
    try:
        generation_sample = online.collect_sample_for_small_model_generation(
            args=args,
            large_agent=large_model,
            small_agent=None,
            eval_envs=eval_envs,
            env_kwargs=env_kwargs,
            device=device,
        )
        small_model, pruning_info = generate_small_cnn_with_verify(
            large_model,
            args.max_sparsity,
            generation_sample,
            lambda model, data: model(data),
            return_pruning_info=True,
        )
        for model in (large_model, small_model):
            model.eval()
            for module in model.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False

        similarities = []
        tensor_similarities: dict[str, list[float]] = {}
        element_count = None
        matched_tensor_count = None
        unmatched: list[str] = []

        print("\nGradient similarity on retained-neuron intersection")
        for sample_index in range(cli_args.num_gradient_samples):
            sample = online.collect_sample_for_small_model_generation(
                args=args,
                large_agent=large_model,
                small_agent=small_model,
                eval_envs=eval_envs,
                env_kwargs=env_kwargs,
                device=device,
            )
            large_model.zero_grad(set_to_none=True)
            small_model.zero_grad(set_to_none=True)

            large_loss = large_model(sample)
            small_loss = small_model(sample)
            large_loss.backward()
            small_loss.backward()

            pairs, current_unmatched = collect_intersection_gradient_pairs(
                large_model, small_model, pruning_info
            )
            similarity, current_element_count = cosine_similarity(pairs)
            similarities.append(similarity)
            unmatched = current_unmatched

            if element_count is None:
                element_count = current_element_count
                matched_tensor_count = len(pairs)
            elif element_count != current_element_count or matched_tensor_count != len(pairs):
                raise RuntimeError("The matched gradient set changed between samples")

            for pair in pairs:
                if pair.large.norm() == 0 or pair.small.norm() == 0:
                    pair_similarity = float("nan")
                else:
                    pair_similarity = F.cosine_similarity(
                        pair.large.float(), pair.small.float(), dim=0
                    ).item()
                tensor_similarities.setdefault(pair.name, []).append(pair_similarity)

            print(
                f"sample {sample_index + 1}/{cli_args.num_gradient_samples}: "
                f"cosine={similarity:.10f}, "
                f"large_loss={large_loss.item():.10f}, "
                f"small_loss={small_loss.item():.10f}"
            )

        similarity_values = np.asarray(similarities, dtype=np.float64)
        print("\nAverage over independently collected post-generation samples")
        print(f"samples: {len(similarities)}")
        print(f"matched tensors per sample: {matched_tensor_count}")
        print(f"matched gradient elements per sample: {element_count}")
        print(f"mean cosine similarity: {similarity_values.mean():.10f}")
        print(f"std cosine similarity: {similarity_values.std():.10f}")
        print(f"min cosine similarity: {similarity_values.min():.10f}")
        print(f"max cosine similarity: {similarity_values.max():.10f}")
        print("per-tensor mean cosine similarity:")
        for name, values in tensor_similarities.items():
            tensor_values = np.asarray(values, dtype=np.float64)
            print(
                f"  {name}: mean={np.nanmean(tensor_values):.10f}, "
                f"std={np.nanstd(tensor_values):.10f}"
            )
        if unmatched:
            print("unmatched trainable small-model parameters (excluded):")
            for name in unmatched:
                print(f"  {name}")
    finally:
        envs.close()
        eval_envs.close()


if __name__ == "__main__":
    main()
