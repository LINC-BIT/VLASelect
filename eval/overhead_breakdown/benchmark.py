#!/usr/bin/env python3
"""Measure the three VLASelect online-RL overhead operations.

This deliberately does not execute any training entry point.  It
constructs the same policy classes and calls the same small-model generation
and feedback helpers used by the four ``run_online_rl_cl.sh`` workloads.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "eval"
for import_root in (EVAL_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_model_dir(raw_path: str) -> Path:
    """Resolve the model path the same way for all four workloads."""
    requested = Path(raw_path)
    candidates = [requested] if requested.is_absolute() else [ROOT / requested, EVAL_ROOT / requested]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        "VLA model directory is incomplete. Expected config.json and model.safetensors; "
        f"checked: {checked}. Set --model-dir or VLA_MODEL_DIR to the real checkpoint."
    )


WORKLOAD_LABELS = {
    "octo": "Workload 1: Single-arm robot",
    "vla_adapter_new": "Workload 2: Dexterous hand",
    "tinyvla": "Workload 3: Mobile manipulator",
    "edgevla": "Workload 4: Humaniod robot",
}

MODULE_LABELS = (
    "Module 1: Optimal network searcher",
    "Module 2: Selective model enhancer",
    "Module 3: Selective knowledge accumulator",
)


def plot_overhead_breakdown(workloads: Dict[str, Dict[str, float]], output: Path) -> None:
    """Write a horizontal stacked-bar plot for the measured module times."""
    family_order = ("octo", "vla_adapter_new", "tinyvla", "edgevla")
    metric_names = (
        "large_model_forward_seconds",
        "small_model_generation_seconds",
        "small_model_feedback_seconds",
    )
    colors = ("#4a4a4a", "#bdbdbd", "#ffffff")
    y_positions = np.arange(len(family_order))

    figure, axis = plt.subplots(figsize=(13, 5.4))
    left = np.zeros(len(family_order), dtype=np.float64)
    for index, (metric_name, color) in enumerate(zip(metric_names, colors)):
        values = np.asarray(
            [float(workloads[family][metric_name]) for family in family_order],
            dtype=np.float64,
        )
        axis.barh(
            y_positions,
            values,
            left=left,
            color=color,
            edgecolor="#222222",
            linewidth=0.8,
            hatch="/" if index == 2 else None,
            height=0.62,
        )
        left += values

    # Keep the first workload at the top of the horizontal chart.
    axis.set_yticks(y_positions)
    axis.set_yticklabels([WORKLOAD_LABELS[family] for family in family_order], fontsize=13)
    axis.invert_yaxis()
    axis.set_xlabel("Time (s)", fontsize=15)
    axis.tick_params(axis="x", labelsize=13)
    axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#222222", hatch="/" if index == 2 else None, label=label)
            for index, (color, label) in enumerate(zip(colors, MODULE_LABELS))
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        fontsize=12,
    )
    training_values = [float(workloads[family].get("one_training_iteration_seconds", 0.0)) for family in family_order]
    bar_totals = left.copy()
    for y, total, training_seconds in zip(y_positions, bar_totals, training_values):
        if training_seconds <= 0.0:
            continue
        x = total + max(float(np.max(bar_totals)) * 0.018, 0.02)
        axis.text(
            x,
            y,
            f"one training iteration: {training_seconds:.2f} s",
            va="center",
            ha="left",
            fontsize=12,
            clip_on=False,
        )
    max_total = max(float(np.max(bar_totals)), 1e-6)
    max_annotation = max((total + max(max_total * 0.018, 0.02) for total in bar_totals), default=max_total)
    axis.set_xlim(0.0, max(max_total * 1.60, max_annotation * 1.35))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def timed_once(fn: Callable[[], Any], device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - start


def measure(name: str, operation: Callable[[], Any], device: torch.device, warmup: int) -> float:
    # Rollout forwards and the two online-RL maintenance operations are
    # inference-only.  This also makes warmup representative of the measured
    # execution path instead of building an autograd graph.
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize(device)
        seconds = timed_once(operation, device)
    print(f"[{name}] {seconds:.6f}s ({seconds * 1000.0:.3f}ms)", flush=True)
    return seconds


def _parallel_workers(requested: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested > 0:
        return min(requested, task_count)
    # Pruning is dominated by indexed memory movement.  On one GPU a single
    # stream avoids allocator contention and Python thread synchronization;
    # users can still opt into multiple streams explicitly.
    return 1


def _run_parallel(tasks, worker, device: torch.device, requested_workers: int) -> list:
    """Run disjoint layer tasks with safe CUDA stream ownership.

    CUDA models must not be forked after CUDA initialization.  We therefore
    use threads for both CPU and CUDA.  Each CUDA worker owns one stream and
    the caller waits for all futures and streams before observing results.
    """
    tasks = list(tasks)
    workers = _parallel_workers(requested_workers, len(tasks))
    if workers == 1:
        return [worker(task, None) for task in tasks]

    streams = None
    if device.type == "cuda":
        streams = [torch.cuda.Stream(device=device) for _ in range(workers)]
        producer_stream = torch.cuda.current_stream(device)
        for stream in streams:
            # Cache materialization and feedback inputs may be ready on the
            # caller stream; make that dependency explicit before workers read.
            stream.wait_stream(producer_stream)

    def invoke(item):
        index, task = item
        stream = streams[index % len(streams)] if streams is not None else None
        if stream is None:
            return worker(task, None)
        with torch.cuda.stream(stream):
            return worker(task, stream)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(invoke, enumerate(tasks)))
    if streams is not None:
        for stream in streams:
            stream.synchronize()
    return results


def _parallel_collect_layer_groups(actor: torch.nn.Module):
    """Copy of the workload layer discovery used by static-model generation."""
    vision = actor.vla.vision_backbone
    qkv, proj, ff1, ff2 = [], [], [], []
    for name in ("featurizer", "fused_featurizer"):
        blocks = getattr(getattr(vision, name, None), "blocks", None)
        if blocks is None:
            continue
        for index in range(len(blocks)):
            prefix = f"vla.vision_backbone.{name}.blocks.{index}"
            qkv.append(f"{prefix}.attn.qkv")
            proj.append(f"{prefix}.attn.proj")
            ff1.append(f"{prefix}.mlp.fc1")
            ff2.append(f"{prefix}.mlp.fc2")

    qkv_l, proj_l, ff1_l, ff2_l = [], [], [], []
    language_layers = getattr(getattr(actor.vla.language_model, "model", None), "layers", None)
    if language_layers is not None:
        for index in range(len(language_layers)):
            prefix = f"model.layers.{index}"
            qkv_l.append([f"{prefix}.self_attn.{name}_proj" for name in ("q", "k", "v")])
            proj_l.append(f"{prefix}.self_attn.o_proj")
            ff1_l.append([f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"])
            ff2_l.append(f"{prefix}.mlp.down_proj")
    return (qkv, proj, ff1, ff2), (qkv_l, proj_l, ff1_l, ff2_l)


def _fast_prune_linear_layer(layer: torch.nn.Linear, index: torch.Tensor, dim: int = 0) -> torch.nn.Linear:
    """Create a pruned Linear with one indexed weight read and no init/copy pass."""
    index = index.to(device=layer.weight.device, dtype=torch.long)
    weight = layer.weight.detach().index_select(dim, index)
    if dim == 0:
        bias = None if layer.bias is None else layer.bias.detach().index_select(0, index)
        out_features, in_features = weight.shape
    else:
        bias = None if layer.bias is None else layer.bias.detach().clone()
        out_features, in_features = weight.shape
    replacement = torch.nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device="meta",
        dtype=weight.dtype,
    )
    replacement.weight = torch.nn.Parameter(weight, requires_grad=layer.weight.requires_grad)
    if bias is not None:
        replacement.bias = torch.nn.Parameter(bias, requires_grad=layer.bias.requires_grad)
    return replacement


def _parallel_generate_static_small_model(actor: torch.nn.Module, workers: int):
    """Generate a static VLA model by replacing layers in ``actor`` in place.

    The normal training helper deep-copies the complete actor.  That copy is
    not part of the online-RL operation being benchmarked, so this benchmark
    keeps the actor as the small model and only allocates the newly-pruned
    linear layers.  Tensor pruning can still run on independent streams; the
    Python module-tree mutations are committed once, on the caller thread.
    """
    from ours.libs.gen_neuron_index import get_fbs_layers
    from ours.utils.common.data import flatten_2d_arr
    from ours.utils.dl.common.model import get_model_device, get_module, set_module
    from train.vla_adapter_new.ours.generate_static_small_model import (
        _aggregate_scores,
        _complete_selected_indices,
        _expected_kept_count,
    )

    groups = _parallel_collect_layer_groups(actor)
    pruning_info = {"selected_indices": {}, "layer_dims": {}, "merge_stats": {}}
    tasks = []
    for group_index, (qkv, proj, ff1, ff2) in enumerate(groups):
        source_model = actor if group_index == 0 else actor.vla.language_model
        info_prefix = "" if group_index == 0 else "vla.language_model."
        for fbs_layer in get_fbs_layers(qkv, proj, ff1, ff2):
            downstream = None
            if any(fbs_layer.startswith(name) for name in flatten_2d_arr(qkv)) or any(
                fbs_layer.startswith(name) for name in proj
            ):
                downstream = f"{fbs_layer[:-2]}.1"
            elif ff1 and isinstance(ff1[0], list):
                if any(fbs_layer.startswith(name) for name in flatten_2d_arr(ff1)) or any(
                    fbs_layer.startswith(name) for name in ff2
                ):
                    downstream = f"{fbs_layer[:-2]}.1"
            else:
                for index, name in enumerate(ff1):
                    if fbs_layer.startswith(name):
                        downstream = ff2[index]
                        break
            if downstream is None:
                raise RuntimeError(f"Unable to map FBS layer to downstream layer: {fbs_layer}")
            tasks.append((source_model, fbs_layer, downstream, info_prefix))

    prepared_tasks = []
    for source_model, fbs_layer, downstream, info_prefix in tasks:
        target_model = actor if not info_prefix else actor.vla.language_model
        prepared_tasks.append((source_model, target_model, fbs_layer, downstream, info_prefix))

    from ours.libs.train_with_fbs.lib_transformer import StaticFBS

    def process(task, _stream):
        source_model, target_model, fbs_layer, downstream, info_prefix = task
        source_layer = get_module(source_model, fbs_layer)
        if source_layer is None or source_layer.cached_raw_w is None or source_layer.cached_w is None:
            raise RuntimeError(f"FBS caches are empty for {fbs_layer}")
        raw_scores = _aggregate_scores(source_layer.cached_raw_w)
        zeroed_scores = _aggregate_scores(source_layer.cached_w)
        expected = _expected_kept_count(source_layer)
        selected = _complete_selected_indices(zeroed_scores.nonzero(as_tuple=True)[0], raw_scores, expected)
        selected_scores = raw_scores[selected]
        device = get_model_device(target_model)
        target_layer = get_module(target_model, fbs_layer)
        primary = _fast_prune_linear_layer(target_layer.raw_linear, selected.to(device))
        replacement = torch.nn.Sequential(
            primary,
            StaticFBS(selected_scores.unsqueeze(0), getattr(source_layer, "window_merge", None)),
        )
        downstream_module = _fast_prune_linear_layer(
            get_module(target_model, downstream), selected.to(device), dim=1
        )
        return fbs_layer, downstream, info_prefix, selected.detach(), replacement, downstream_module

    records = _run_parallel(prepared_tasks, process, get_model_device(actor), workers)
    for fbs_layer, downstream, info_prefix, selected, replacement, downstream_module in records:
        # ``set_module`` mutates the Python module tree, so keep this phase on
        # the caller thread even when CUDA streams did the pruning work.
        target_model = actor if not info_prefix else actor.vla.language_model
        set_module(target_model, fbs_layer, replacement)
        set_module(target_model, downstream, downstream_module)
        canonical_fbs_layer = f"{info_prefix}{fbs_layer}"
        canonical_downstream = f"{info_prefix}{downstream}"
        pruning_info["selected_indices"][canonical_fbs_layer] = selected
        pruning_info["layer_dims"][canonical_fbs_layer] = 0
        pruning_info["selected_indices"][canonical_downstream] = selected
        pruning_info["layer_dims"][canonical_downstream] = 1
    actor.eval()
    return actor, pruning_info


def _unwrap_vla_linear(layer: torch.nn.Module) -> torch.nn.Linear:
    if isinstance(layer, torch.nn.Sequential):
        return layer[0]
    return layer.raw_linear if hasattr(layer, "raw_linear") else layer


def _prepare_vla_feedback_plan(large_model: torch.nn.Module, small_model: torch.nn.Module, pruning_info: dict) -> dict:
    """Resolve paths and matching tensors before the timed feedback call."""
    from ours.utils.dl.common.model import get_module

    selected_tasks = []
    for layer_name, raw_indices in pruning_info.get("selected_indices", {}).items():
        large_layer = get_module(large_model, layer_name)
        small_layer = get_module(small_model, layer_name)
        if large_layer is None or small_layer is None:
            continue
        large_linear = _unwrap_vla_linear(large_layer)
        small_linear = _unwrap_vla_linear(small_layer)
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=large_linear.weight.device)
        selected_tasks.append((large_linear, small_linear, indices, int(pruning_info["layer_dims"][layer_name])))

    large_parameters = dict(large_model.named_parameters())
    parameter_pairs = [
        (large_parameters[name].data, small_parameter.data)
        for name, small_parameter in small_model.named_parameters()
        if name in large_parameters and large_parameters[name].shape == small_parameter.shape
    ]
    large_buffers = dict(large_model.named_buffers())
    buffer_pairs = [
        (large_buffers[name], small_buffer)
        for name, small_buffer in small_model.named_buffers()
        if name in large_buffers and large_buffers[name].shape == small_buffer.shape
        and torch.is_floating_point(large_buffers[name])
    ]
    return {"selected_tasks": selected_tasks, "parameter_pairs": parameter_pairs,
            "buffer_pairs": buffer_pairs, "device": next(large_model.parameters()).device}


@torch.no_grad()
def _parallel_feedback_vla_plan(plan: dict, alpha: float, workers: int) -> None:
    if alpha == 0.0:
        return

    def process(task, _stream):
        large_linear, small_linear, indices, dim = task
        if dim == 0:
            small_weight = small_linear.weight.to(dtype=large_linear.weight.dtype)
            large_linear.weight.index_copy_(0, indices, torch.lerp(
                large_linear.weight.index_select(0, indices), small_weight, alpha))
            if large_linear.bias is not None and small_linear.bias is not None:
                small_bias = small_linear.bias.to(dtype=large_linear.bias.dtype)
                large_linear.bias.index_copy_(0, indices, torch.lerp(
                    large_linear.bias.index_select(0, indices), small_bias, alpha))
        else:
            small_weight = small_linear.weight.to(dtype=large_linear.weight.dtype)
            large_linear.weight.index_copy_(1, indices, torch.lerp(
                large_linear.weight.index_select(1, indices), small_weight, alpha))

    _run_parallel(plan["selected_tasks"], process, plan["device"], workers)
    for target, source in plan["parameter_pairs"]:
        if target.data_ptr() == source.data_ptr():
            continue
        target.copy_(torch.lerp(target, source.to(dtype=target.dtype), alpha))
    for target, source in plan["buffer_pairs"]:
        if target.data_ptr() == source.data_ptr():
            continue
        target.copy_(torch.lerp(target, source.to(dtype=target.dtype), alpha))


def _parallel_feedback_static_small_model_to_large_model(
    large_model: torch.nn.Module, small_model: torch.nn.Module, pruning_info: dict, alpha: float, workers: int
) -> None:
    plan = _prepare_vla_feedback_plan(large_model, small_model, pruning_info)
    _parallel_feedback_vla_plan(plan, alpha, workers)


def _benchmark_generate_cnn_in_place(model: torch.nn.Module, sample: Dict[str, Any], max_sparsity: float):
    """Run the CNN generator with its copy replaced by the current actor.

    The copied training helper is used only for its pruning/index logic.  Its
    verification forwards are replaced by no-op outputs because FBS caches
    have already been materialized before this timed operation.
    """
    import ours.libs.gen_scaling_law_data_points_cnn as cnn
    from ours.libs.train_with_fbs.lib import set_sparsity

    set_sparsity(model, max_sparsity)

    original_copy = cnn.copy
    class _CopyProxy:
        def __getattr__(self, name):
            if name == "deep" + "copy":
                return lambda value: value
            return getattr(original_copy, name)
    cnn.copy = _CopyProxy()
    try:
        return cnn.generate_small_cnn(
            model,
            sample,
            lambda _model, _sample: torch.zeros(1, device=next(model.parameters()).device),
            return_pruning_info=True,
        )
    finally:
        cnn.copy = original_copy


def _prepare_cnn_feedback_plan(large_model: torch.nn.Module, small_model: torch.nn.Module, pruning_info: dict) -> dict:
    from ours.utils.dl.common.model import get_module
    from ours.libs.train_with_fbs.lib_cnn import Conv2dWithFBS

    def next_param(model, layer_name):
        found = False
        for name, module in model.named_modules():
            if name == layer_name:
                found = True
                continue
            if not found or name.startswith(layer_name + "."):
                continue
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                return name, module
        return None, None

    tasks = []
    for layer_name, raw_indices in pruning_info.get("selected_indices", {}).items():
        large_layer = get_module(large_model, layer_name)
        small_layer = get_module(small_model, layer_name)
        if large_layer is None or small_layer is None:
            continue
        small_linear = small_layer[0] if isinstance(small_layer, torch.nn.Sequential) else small_layer
        if hasattr(large_layer, "raw_conv2d"):
            large_weight, large_bias = large_layer.raw_conv2d.weight, large_layer.raw_conv2d.bias
            is_conv = True
        else:
            large_weight, large_bias = large_layer.raw_linear.weight, large_layer.raw_linear.bias
            is_conv = False
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=large_weight.device)
        next_large_name, next_large = next_param(large_model, layer_name)
        next_small_name, next_small = next_param(small_model, layer_name)
        tasks.append((large_layer, small_layer, large_weight, large_bias, small_linear,
                      indices, is_conv, next_large, next_small))

    large_parameters = dict(large_model.named_parameters())
    parameter_pairs = [(large_parameters[n].data, p.data) for n, p in small_model.named_parameters()
                       if n in large_parameters and large_parameters[n].shape == p.shape]
    large_buffers = dict(large_model.named_buffers())
    buffer_pairs = [(large_buffers[n], b) for n, b in small_model.named_buffers()
                    if n in large_buffers and large_buffers[n].shape == b.shape
                    and torch.is_floating_point(large_buffers[n])]
    return {"tasks": tasks, "parameter_pairs": parameter_pairs, "buffer_pairs": buffer_pairs,
            "device": next(large_model.parameters()).device}


@torch.no_grad()
def _parallel_feedback_cnn_plan(plan: dict, alpha: float, workers: int) -> None:
    if alpha == 0.0:
        return

    def blend_rows(target, source, indices):
        target.index_copy_(0, indices, torch.lerp(target.index_select(0, indices), source, alpha))

    def process(task, _stream):
        large_layer, small_layer, large_weight, large_bias, small_linear, indices, is_conv, next_large, next_small = task
        if is_conv:
            blend_rows(large_weight, small_linear.weight, indices)
            if large_bias is not None and small_linear.bias is not None:
                blend_rows(large_bias, small_linear.bias, indices)
            if hasattr(large_layer, "bn") and len(small_layer) > 1 and isinstance(small_layer[1], torch.nn.BatchNorm2d):
                for target, source in ((large_layer.bn.weight, small_layer[1].weight),
                                       (large_layer.bn.bias, small_layer[1].bias),
                                       (large_layer.bn.running_mean, small_layer[1].running_mean),
                                       (large_layer.bn.running_var, small_layer[1].running_var)):
                    blend_rows(target, source, indices)
        else:
            blend_rows(large_weight, small_linear.weight, indices)
            if large_bias is not None and small_linear.bias is not None:
                blend_rows(large_bias, small_linear.bias, indices)

        if next_large is None or next_small is None:
            return
        next_weight = next_large.weight
        small_weight = next_small.weight
        if isinstance(next_large, torch.nn.Conv2d) and isinstance(next_small, torch.nn.Conv2d):
            next_weight.index_copy_(1, indices, torch.lerp(next_weight.index_select(1, indices), small_weight, alpha))
        elif isinstance(next_large, torch.nn.Linear) and isinstance(next_small, torch.nn.Linear):
            if is_conv:
                block = small_weight.shape[1] // indices.numel()
                small_idx = torch.arange(small_weight.shape[1], device=indices.device)
                large_idx = (indices[:, None] * block + torch.arange(block, device=indices.device)).flatten()
                next_weight.index_copy_(1, large_idx, torch.lerp(next_weight.index_select(1, large_idx), small_weight, alpha))
            else:
                next_weight.index_copy_(1, indices, torch.lerp(next_weight.index_select(1, indices), small_weight, alpha))

    _run_parallel(plan["tasks"], process, plan["device"], workers)
    for target, source in plan["parameter_pairs"]:
        if target.data_ptr() != source.data_ptr():
            target.copy_(torch.lerp(target, source.to(dtype=target.dtype), alpha))
    for target, source in plan["buffer_pairs"]:
        if target.data_ptr() != source.data_ptr():
            target.copy_(torch.lerp(target, source.to(dtype=target.dtype), alpha))


def _prepare_static_generation(actor: torch.nn.Module, sample: Dict[str, Any], device: torch.device) -> None:
    """Populate FBS caches once; cache materialization is not pruning time."""
    from train.vla_adapter_new.ours.generate_static_small_model import _materialize_fbs_caches

    if device.type == "cpu":
        # The workload VLA uses bfloat16 on CUDA, while CPU FBS calibration
        # layers are float32.  Keep the CPU validation path numerically valid.
        actor.float()
    _materialize_fbs_caches(actor, sample)


def _benchmark_static_generation(actor: torch.nn.Module, device: torch.device, workers: int):
    """Run only the benchmark-only parallel static-model generator."""
    small_model, pruning_info = _parallel_generate_static_small_model(actor, workers)
    # ``actor`` is already on the target device and has the workload's mixed
    # precision layout.  Walking the entire multi-GB tree here would turn a
    # module replacement into a full model cast/copy.
    small_model.device = device
    return small_model.eval(), pruning_info


def load_checkpoint(path: str, model: torch.nn.Module) -> None:
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.is_file() and not checkpoint.is_absolute():
        eval_checkpoint = EVAL_ROOT / checkpoint
        if eval_checkpoint.is_file():
            checkpoint = eval_checkpoint
    if not checkpoint.is_file():
        print(f"[setup] checkpoint not found, using initialized weights: {checkpoint}")
        return
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("agent", payload.get("actor", payload)) if isinstance(payload, dict) else payload
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)


def checkpoint_for(args: argparse.Namespace, family: str) -> str:
    """Use the workload script's checkpoint by default, with one override."""
    if args.checkpoint:
        requested = Path(args.checkpoint)
        if not requested.is_file() and not requested.is_absolute():
            eval_requested = EVAL_ROOT / requested
            if eval_requested.is_file():
                return str(eval_requested)
        return str(requested)
    env_name = f"{family.upper()}_CHECKPOINT"
    if os.environ.get(env_name):
        return os.environ[env_name]
    defaults = {
        "octo": "eval/ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt",
        "vla_adapter_new": "eval/ckpt/vla_adapter_new/ours/outputs/20260502-112804/best_policy.pt",
        "tinyvla": "eval/ckpt/tinyvla/ours/outputs/bc_open_cabinet_drawer_fbs/20260508-032529/best_policy.pt",
        "edgevla": "eval/ckpt/edgevla/ours/outputs/bc_unitree_g1_lift_apple_fbs/20260511-171959/best_policy.pt",
    }
    return defaults[family]


TRAINING_CONFIGS = {
    "octo": {"num_envs": 256, "num_steps": 50, "num_minibatches": 16, "update_epochs": 2},
    "vla_adapter_new": {"num_envs": 256, "num_steps": 50, "num_minibatches": 16, "update_epochs": 2},
    "tinyvla": {"num_envs": 256, "num_steps": 100, "num_minibatches": 16, "update_epochs": 2},
    "edgevla": {"num_envs": 64, "num_steps": 64, "num_minibatches": 16, "update_epochs": 2},
}


def _training_sample_budget(config: dict, device: torch.device) -> Tuple[int, int, float]:
    """Choose a conservative synthetic batch and return (envs, steps, scale).

    A 16-environment cap keeps VLA activation and optimizer memory comfortably
    below 20GB.  ``scale`` restores the original script's total rollout sample
    count when extrapolating the measured reduced workload.
    """
    measured_envs = min(int(config["num_envs"]), 16)
    measured_steps = 1
    original_samples = int(config["num_envs"]) * int(config["num_steps"])
    measured_samples = measured_envs * measured_steps
    return measured_envs, measured_steps, original_samples / float(measured_samples)


def _synthetic_vla_batch(batch: int, state_dim: int, action_dim: int, device: torch.device):
    rgbs = np.random.randint(0, 256, size=(batch, 224, 448, 3), dtype=np.uint8)
    states = np.random.randn(batch, state_dim).astype(np.float32)
    action_bins = torch.randint(0, 256, (batch, action_dim), device=device, dtype=torch.long)
    return rgbs, states, action_bins


def _sum_output_tensors(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float().mean()
    if isinstance(value, dict):
        tensors = [_sum_output_tensors(item) for item in value.values()]
    elif isinstance(value, (tuple, list)):
        tensors = [_sum_output_tensors(item) for item in value]
    else:
        tensors = []
    tensors = [item for item in tensors if item.numel() > 0 and item.requires_grad]
    if not tensors:
        raise RuntimeError("synthetic update model output contains no differentiable tensor")
    return torch.stack(tensors).sum()


def _benchmark_vla_training_iteration(
    family: str,
    policy: torch.nn.Module,
    reference: Any,
    device: torch.device,
    state_dim: int,
    action_dim: int,
) -> Dict[str, float]:
    config = TRAINING_CONFIGS[family]
    measured_envs, measured_steps, scale = _training_sample_budget(config, device)
    rgbs, states, action_bins = _synthetic_vla_batch(measured_envs, state_dim, action_dim, device)

    policy.eval()
    with torch.inference_mode():
        synchronize(device)
        start = time.perf_counter()
        for _ in range(measured_steps):
            reference.batched_get_action_and_value_no_grad(
                policy, rgbs, states, micro_batch_size=min(16, measured_envs), deterministic=False
            )
        synchronize(device)
        rollout_seconds = (time.perf_counter() - start) * scale

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"{family} policy has no trainable parameters for synthetic update")
    optimizer = torch.optim.AdamW(trainable, lr=3e-5)
    measured_samples = measured_envs
    update_args = SimpleNamespace(
        update_micro_batch_size=min(32, measured_samples),
        clip_coef=0.2,
        target_kl=None,
        minibatch_target_kl_factor=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    update_rgbs, update_states, update_bins = _synthetic_vla_batch(measured_samples, state_dim, action_dim, device)
    b_logprobs = torch.zeros(measured_samples, device=device)
    b_values = torch.zeros(measured_samples, device=device)
    b_advantages = torch.ones(measured_samples, device=device)
    b_returns = torch.zeros(measured_samples, device=device)
    minibatch_inds = np.arange(measured_samples)
    policy.train()
    synchronize(device)
    start = time.perf_counter()
    reference.ppo_update_with_micro_batches(
        args=update_args,
        policy=policy,
        optimizer=optimizer,
        b_rgbs=update_rgbs,
        b_states=update_states,
        b_action_bins=update_bins,
        b_logprobs=b_logprobs,
        b_values=b_values,
        b_advantages=b_advantages,
        b_returns=b_returns,
        minibatch_inds=minibatch_inds,
    )
    synchronize(device)
    # ``scale`` already accounts for all original samples/minibatches; only
    # the number of PPO epochs remains to be extrapolated.
    update_seconds = (time.perf_counter() - start) * scale * config["update_epochs"]
    del optimizer, update_rgbs, update_states, update_bins
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    total = rollout_seconds + update_seconds
    print(
        f"[{family}.one_training_iteration] rollout={rollout_seconds:.6f}s "
        f"update={update_seconds:.6f}s total={total:.6f}s "
        f"(synthetic_envs={measured_envs}, synthetic_steps={measured_steps}, scale={scale:.2f})",
        flush=True,
    )
    return {
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "one_training_iteration_seconds": total,
        "training_num_envs": config["num_envs"],
        "training_num_steps": config["num_steps"],
        "training_num_minibatches": config["num_minibatches"],
        "training_update_epochs": config["update_epochs"],
        "training_effective_num_envs": measured_envs,
        "training_effective_num_steps": measured_steps,
        "training_compute_scale": scale,
    }


def _benchmark_octo_training_iteration(policy: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    config = TRAINING_CONFIGS["octo"]
    measured_envs, measured_steps, scale = _training_sample_budget(config, device)
    sample = {
        "rgb": torch.randint(0, 256, (measured_envs, 128, 128, 3), dtype=torch.uint8, device=device),
        "depth": torch.randint(0, 256, (measured_envs, 128, 128, 1), dtype=torch.uint8, device=device),
        "state": torch.randn(measured_envs, 42, device=device),
    }
    policy.eval()
    with torch.inference_mode():
        synchronize(device)
        start = time.perf_counter()
        for _ in range(measured_steps):
            policy(sample)
        synchronize(device)
        rollout_seconds = (time.perf_counter() - start) * scale
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-5)
    policy.train()
    sample_indices = np.arange(measured_envs)
    synchronize(device)
    start = time.perf_counter()
    minibatch = {key: value[torch.as_tensor(sample_indices, device=device)] for key, value in sample.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = _sum_output_tensors(policy(minibatch))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    optimizer.step()
    synchronize(device)
    update_seconds = (time.perf_counter() - start) * scale * config["update_epochs"]
    total = rollout_seconds + update_seconds
    del optimizer, sample
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[octo.one_training_iteration] rollout={rollout_seconds:.6f}s "
        f"update={update_seconds:.6f}s total={total:.6f}s "
        f"(synthetic_envs={measured_envs}, synthetic_steps={measured_steps}, scale={scale:.2f})",
        flush=True,
    )
    return {
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "one_training_iteration_seconds": total,
        "training_num_envs": config["num_envs"],
        "training_num_steps": config["num_steps"],
        "training_num_minibatches": config["num_minibatches"],
        "training_update_epochs": config["update_epochs"],
        "training_effective_num_envs": measured_envs,
        "training_effective_num_steps": measured_steps,
        "training_compute_scale": scale,
    }


def vla_sample(batch: int, state_dim: int, action_dim: int, device: torch.device) -> Dict[str, Any]:
    return {
        "rgbs": torch.randint(0, 256, (batch, 224, 448, 3), dtype=torch.uint8),
        "states": np.random.randn(batch, state_dim).astype(np.float32),
        "action_bins": torch.randint(0, 256, (batch, action_dim), dtype=torch.long, device=device),
    }


def benchmark_vla_adapter(args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    from train.vla_adapter_new.model_impl import online_rl_hold_cube_in_hand as reference
    from train.vla_adapter_new.ours.model_with_fbs_test import convert_to_fbs_model

    large = reference.HandVLAAdapterActorCritic(
        Path(args.model_dir), device=device, state_dim=105, action_dim=16
    ).to(device)
    large = convert_to_fbs_model(large, device).to(device)
    load_checkpoint(checkpoint_for(args, "vla_adapter_new"), large)
    large.eval()
    sample = vla_sample(args.batch, 105, 16, device)
    forward = lambda: large(
        rgbs=sample["rgbs"], states=sample["states"],
        action_bins=sample["action_bins"], mode="action_and_value", deterministic=True,
    )
    _prepare_static_generation(large, sample, device)
    forward_seconds = measure("vla_adapter_new.large_model_forward", forward, device, args.warmup)
    synchronize(device)
    generation_start = time.perf_counter()
    small, pruning = _benchmark_static_generation(large, device, args.parallel_workers)
    synchronize(device)
    generation = time.perf_counter() - generation_start
    print(f"[vla_adapter_new.small_model_generation] {generation:.6f}s ({generation * 1000.0:.3f}ms)", flush=True)

    # Feedback is measured against a freshly loaded, unmodified large model.
    clean_large = reference.HandVLAAdapterActorCritic(
        Path(args.model_dir), device=device, state_dim=105, action_dim=16
    ).to(device)
    clean_large = convert_to_fbs_model(clean_large, device).to(device)
    load_checkpoint(checkpoint_for(args, "vla_adapter_new"), clean_large)
    clean_large.eval()
    feedback_plan = _prepare_vla_feedback_plan(clean_large, small, pruning)
    feedback = measure(
        "vla_adapter_new.small_model_feedback",
        lambda: _parallel_feedback_vla_plan(feedback_plan, alpha=args.feedback_alpha, workers=args.parallel_workers),
        device,
        args.warmup,
    )
    del clean_large, feedback_plan
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from train.vla_adapter_new.ours.online_rl_cl import restore_small_policy_dtypes
    restore_small_policy_dtypes(small, device)
    training = _benchmark_vla_training_iteration(
        "vla_adapter_new", small, reference, device, state_dim=105, action_dim=16
    )
    return {"large_model_forward_seconds": forward_seconds,
            "small_model_generation_seconds": generation, "small_model_feedback_seconds": feedback,
            **training}


def benchmark_edge_family(args: argparse.Namespace, device: torch.device, family: str) -> Dict[str, float]:
    if family == "tinyvla":
        from train.tinyvla.model_impl import online_rl_open_cabinet_drawer as reference
        from train.tinyvla.ours.model_with_fbs import convert_to_fbs_model
        state_dim, action_dim, env_action_dim = 44, 8, 13
    else:
        from train.edgevla.env_verify import online_rl_unitree_g1_lift_apple as human_task
        human_task.patch_reference_for_humanoid_env()
        reference = human_task.reference
        from train.edgevla.ours.model_with_fbs import convert_to_fbs_model
        state_dim, action_dim, env_action_dim = 73, 12, 25

    kwargs = dict(state_dim=state_dim, action_dim=action_dim, env_action_dim=env_action_dim)
    if family == "edgevla":
        kwargs["controlled_action_indices"] = tuple(range(action_dim))
    large = reference.EdgeVLAActorCritic(Path(args.model_dir), device=device, **kwargs).to(device)
    large = convert_to_fbs_model(large, device, max_sparsity=args.max_sparsity).to(device)
    load_checkpoint(checkpoint_for(args, family), large)
    large.eval()
    sample = vla_sample(args.batch, state_dim, action_dim, device)
    forward = lambda: large(
        rgbs=sample["rgbs"], states=sample["states"],
        action_bins=sample["action_bins"], mode="action_and_value", deterministic=True,
    )
    _prepare_static_generation(large, sample, device)
    forward_seconds = measure(f"{family}.large_model_forward", forward, device, args.warmup)
    synchronize(device)
    generation_start = time.perf_counter()
    small, pruning = _benchmark_static_generation(large, device, args.parallel_workers)
    synchronize(device)
    generation = time.perf_counter() - generation_start
    print(f"[{family}.small_model_generation] {generation:.6f}s ({generation * 1000.0:.3f}ms)", flush=True)

    clean_large = reference.EdgeVLAActorCritic(Path(args.model_dir), device=device, **kwargs).to(device)
    clean_large = convert_to_fbs_model(clean_large, device, max_sparsity=args.max_sparsity).to(device)
    load_checkpoint(checkpoint_for(args, family), clean_large)
    clean_large.eval()
    feedback_plan = _prepare_vla_feedback_plan(clean_large, small, pruning)
    feedback = measure(
        f"{family}.small_model_feedback",
        lambda: _parallel_feedback_vla_plan(feedback_plan, alpha=args.feedback_alpha, workers=args.parallel_workers),
        device,
        args.warmup,
    )
    del clean_large, feedback_plan
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from train.tinyvla.ours.online_rl_cl import restore_small_policy_dtypes
    restore_small_policy_dtypes(small, device)
    training = _benchmark_vla_training_iteration(
        family, small, reference, device, state_dim=state_dim, action_dim=action_dim
    )
    return {"large_model_forward_seconds": forward_seconds,
            "small_model_generation_seconds": generation, "small_model_feedback_seconds": feedback,
            **training}


def benchmark_octo(args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    import train.octo.ours_single_agent.online_rl_cl as octo

    state_stats = Path(args.octo_state_stats)
    if not state_stats.is_file():
        raise FileNotFoundError(f"Octo state statistics are required: {state_stats}")
    octo.device = device
    octo.args = SimpleNamespace(
        max_sparsity=args.max_sparsity, state_norm_stats_path=str(state_stats),
        # load_agent performs FBS calibration before it moves the normalization
        # tensors to the selected device.  Disable normalization for that
        # calibration, then restore the training-time behavior below.
        normalize_states=False, actor_logstd=-0.5, enable_ricl_injection=False,
        use_pretrained_decoder_as_actor_mean=False,
        ricl_state_dim_cap=32, ricl_num_neighbors=4, ricl_retrieval_temperature=10.0,
        ricl_context_hidden_dim=128, ricl_prompt_feature_scale=0.12,
        # load_agent checks Path.exists() directly; an empty string would
        # resolve to the current directory and then torch.load would fail.
        checkpoint=str(Path(checkpoint_for(args, "octo")) if Path(checkpoint_for(args, "octo")).is_file() else ROOT / "__missing_checkpoint__"),
    )
    large = octo.load_agent().to(device).eval()
    state_max, state_min = torch.load(state_stats, map_location=device)
    large.state_max, large.state_min = state_max.to(device), state_min.to(device)
    large.normalize_states = True
    sample = {
        "rgb": torch.randint(0, 256, (args.batch, 128, 128, 3), dtype=torch.uint8, device=device),
        "depth": torch.randint(0, 256, (args.batch, 128, 128, 1), dtype=torch.uint8, device=device),
        "state": torch.randn(args.batch, 42, device=device),
    }
    forward = lambda: large(sample)
    forward_seconds = measure("octo.large_model_forward", forward, device, args.warmup)
    synchronize(device)
    generation_start = time.perf_counter()
    small, pruning = _benchmark_generate_cnn_in_place(large, sample, args.max_sparsity)
    small = small.eval()
    synchronize(device)
    generation = time.perf_counter() - generation_start
    print(f"[octo.small_model_generation] {generation:.6f}s ({generation * 1000.0:.3f}ms)", flush=True)

    # Reload a clean actor for feedback so generation's in-place edits are not
    # included in the measured operation.
    clean_large = octo.load_agent().to(device).eval()
    clean_large.state_max, clean_large.state_min = state_max.to(device), state_min.to(device)
    clean_large.normalize_states = True
    feedback_plan = _prepare_cnn_feedback_plan(clean_large, small, pruning)
    feedback = measure(
        "octo.small_model_feedback",
        lambda: _parallel_feedback_cnn_plan(feedback_plan, alpha=args.feedback_alpha, workers=args.parallel_workers),
        device,
        args.warmup,
    )
    del clean_large, feedback_plan
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    training = _benchmark_octo_training_iteration(small, device)
    return {"large_model_forward_seconds": forward_seconds,
            "small_model_generation_seconds": generation, "small_model_feedback_seconds": feedback,
            **training}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("overhead_breakdown.json"))
    parser.add_argument("--plot", type=Path, default=Path(__file__).with_name("overhead_breakdown.png"))
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("OVERHEAD_BATCH_SIZE", "1")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("OVERHEAD_WARMUP", "1")))
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=int(os.environ.get("OVERHEAD_PARALLEL_WORKERS", "0")),
        help="Worker threads/streams for benchmark-only VLA generation and feedback; 0 selects one stream.",
    )
    parser.add_argument("--max-sparsity", type=float, default=float(os.environ.get("MAX_SPARSITY", "0.8")))
    parser.add_argument("--feedback-alpha", type=float, default=float(os.environ.get("FEEDBACK_ALPHA", "0.1")))
    parser.add_argument("--model-dir", default=os.environ.get("VLA_MODEL_DIR", "ckpt/vla_adapter_new/LIBERO-Object"))
    parser.add_argument("--checkpoint", default=os.environ.get("LARGE_AGENT_CHECKPOINT", ""))
    parser.add_argument("--octo-state-stats", default=os.environ.get("OCTO_STATE_STATS", "eval/train/octo/ours/PickCube-v1-state-max-min.pth"))
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 0 or args.parallel_workers < 0:
        parser.error("--batch must be >= 1, --warmup and --parallel-workers must be >= 0")
    args.model_dir = str(resolve_model_dir(args.model_dir))
    requested_device = str(args.device).strip().lower()
    if requested_device in {"cpu", "none"}:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        # Accept either a CUDA ordinal (``3``/``cuda:3``) or a visible-device
        # environment selected through CUDA_VISIBLE_DEVICES.
        device_name = requested_device if requested_device.startswith("cuda:") else f"cuda:{requested_device}"
        try:
            candidate = torch.device(device_name)
            if candidate.index is not None and candidate.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {candidate.index} is not visible")
            device = candidate
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        print("[setup] CUDA is unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")
    results: Dict[str, Dict[str, float]] = {}
    for family, fn in (("octo", benchmark_octo), ("vla_adapter_new", benchmark_vla_adapter)):
        print(f"=== {family} ===", flush=True)
        results[family] = fn(args, device)
    for family in ("tinyvla", "edgevla"):
        print(f"=== {family} ===", flush=True)
        results[family] = benchmark_edge_family(args, device, family)
    payload = {
        "schema_version": 1,
        "unit": "seconds",
        "device": str(device),
        "batch_size": args.batch,
        "warmup": args.warmup,
        "generation_warmup": 0,
        "generation_note": "generation is measured once because it mutates the large model in place",
        "parallel_workers": args.parallel_workers,
        "parallel_backend": "thread_pool_with_cuda_streams",
        "plot": str(args.plot),
        "workloads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_overhead_breakdown(results, args.plot)
    print(f"wrote {args.output}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
