import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

sys.path.append("./")

from ours.libs.gen_neuron_index import get_fbs_layers
from ours.libs.gen_scaling_law_data_points import prune_linear_layer_and_its_after_layer
from ours.libs.train_with_fbs.lib_cnn import get_model_size
from ours.utils.common.data import flatten_2d_arr
from ours.utils.dl.common.model import get_model_device, get_module, set_module


def _collect_transformer_layer_groups(actor: nn.Module):
    qkv_layers, proj_layers, ff1_layers, ff2_layers = [], [], [], []
    vision_backbone = actor.vla.vision_backbone
    featurizer = getattr(vision_backbone, "featurizer", None)
    fused_featurizer = getattr(vision_backbone, "fused_featurizer", None)
    featurizer_blocks = getattr(featurizer, "blocks", None)
    fused_blocks = getattr(fused_featurizer, "blocks", None)
    if featurizer_blocks is not None:
        for block_i in range(len(featurizer_blocks)):
            qkv_layers.append(f"vla.vision_backbone.featurizer.blocks.{block_i}.attn.qkv")
            proj_layers.append(f"vla.vision_backbone.featurizer.blocks.{block_i}.attn.proj")
            ff1_layers.append(f"vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc1")
            ff2_layers.append(f"vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc2")
    if fused_blocks is not None:
        for block_i in range(len(fused_blocks)):
            qkv_layers.append(f"vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.qkv")
            proj_layers.append(f"vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.proj")
            ff1_layers.append(f"vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc1")
            ff2_layers.append(f"vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc2")

    qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2 = [], [], [], []
    language_model_core = getattr(actor.vla.language_model, "model", None)
    language_layers = getattr(language_model_core, "layers", None)
    if language_layers is not None:
        for block_i in range(len(language_layers)):
            qkv_layers2.append([f"model.layers.{block_i}.self_attn.{k}_proj" for k in ["q", "k", "v"]])
            proj_layers2.append(f"model.layers.{block_i}.self_attn.o_proj")
            ff1_layers2.append(
                [
                    f"model.layers.{block_i}.mlp.gate_proj",
                    f"model.layers.{block_i}.mlp.up_proj",
                ]
            )
            ff2_layers2.append(f"model.layers.{block_i}.mlp.down_proj")

    return (qkv_layers, proj_layers, ff1_layers, ff2_layers), (qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2)


def _prepare_sample_batch(sample_batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    rgbs = sample_batch["rgbs"]
    states = sample_batch["states"]
    action_bins = sample_batch.get("action_bins")

    if isinstance(rgbs, torch.Tensor):
        rgbs = rgbs.detach().cpu().contiguous()
    else:
        rgbs = torch.as_tensor(rgbs).detach().cpu().contiguous()

    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    else:
        states = torch.as_tensor(states).detach().cpu().numpy()

    if action_bins is not None:
        if isinstance(action_bins, torch.Tensor):
            action_bins = action_bins.detach().to(device=device, dtype=torch.long)
        else:
            action_bins = torch.as_tensor(action_bins, device=device, dtype=torch.long)

    return {
        "rgbs": rgbs,
        "states": states,
        "action_bins": action_bins,
    }


@torch.no_grad()
def _materialize_fbs_caches(actor: nn.Module, sample_batch: Dict[str, Any]) -> Dict[str, Any]:
    device = get_model_device(actor)
    prepared = _prepare_sample_batch(sample_batch, device)
    actor.eval()
    actor.get_action_and_value(
        rgbs=prepared["rgbs"],
        states=prepared["states"],
        action_bins=prepared["action_bins"],
        deterministic=True,
    )
    return prepared


def _aggregate_scores(score_tensor: torch.Tensor) -> torch.Tensor:
    if score_tensor.size(0) == 1:
        return score_tensor[0].detach().to(torch.float32)
    return score_tensor.mean(0).detach().to(torch.float32)


def _expected_kept_count(module: nn.Module) -> int:
    if not hasattr(module, "raw_linear") or not isinstance(module.raw_linear, nn.Linear):
        raise TypeError(f"Unsupported FBS layer type: {type(module).__name__}")
    total = int(module.raw_linear.out_features)
    pruned = int(total * float(module.k_takes_all.k))
    return total - pruned


def _complete_selected_indices(indices: torch.Tensor, raw_scores: torch.Tensor, expected_kept_count: int) -> torch.Tensor:
    indices = torch.unique(indices.to(dtype=torch.long), sorted=True)
    current_count = int(indices.numel())
    if current_count == expected_kept_count:
        return indices
    if current_count > expected_kept_count:
        ranked = torch.argsort(raw_scores[indices], descending=True, stable=True)
        return indices[ranked[:expected_kept_count]].sort().values

    candidate_mask = torch.ones(raw_scores.size(0), dtype=torch.bool, device=raw_scores.device)
    candidate_mask[indices] = False
    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    ranked_candidates = torch.argsort(raw_scores[candidate_indices], descending=True, stable=True)
    fill_count = expected_kept_count - current_count
    filled_indices = candidate_indices[ranked_candidates[:fill_count]]
    return torch.cat([indices, filled_indices], dim=0).sort().values


def _sanitize_previous_indices(previous_indices: torch.Tensor, total_count: int) -> torch.Tensor:
    if previous_indices.numel() == 0:
        return previous_indices.to(dtype=torch.long)
    previous_indices = previous_indices.to(dtype=torch.long)
    valid_mask = (previous_indices >= 0) & (previous_indices < total_count)
    previous_indices = previous_indices[valid_mask]
    if previous_indices.numel() == 0:
        return previous_indices
    return torch.unique(previous_indices, sorted=True)


def _rank_indices(indices: torch.Tensor, raw_scores: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or indices.numel() == 0:
        return indices[:0]
    ranked = torch.argsort(raw_scores[indices], descending=True, stable=True)
    return indices[ranked[:limit]]


def _merge_selected_indices(
    layer_name: str,
    current_indices: torch.Tensor,
    raw_scores: torch.Tensor,
    expected_kept_count: int,
    previous_pruning_info: Optional[dict],
    regeneration_increment_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    current_indices = current_indices.sort().values
    if previous_pruning_info is None or regeneration_increment_ratio >= 1.0:
        return current_indices, {}

    previous_selected_indices = previous_pruning_info.get("selected_indices", {})
    previous_indices = previous_selected_indices.get(layer_name)
    if previous_indices is None:
        return current_indices, {}

    if not isinstance(previous_indices, torch.Tensor):
        previous_indices = torch.as_tensor(previous_indices, device=raw_scores.device)
    else:
        previous_indices = previous_indices.to(raw_scores.device)

    previous_indices = _sanitize_previous_indices(previous_indices, raw_scores.size(0))
    if previous_indices.numel() == 0:
        return current_indices, {}

    previous_indices = _complete_selected_indices(previous_indices, raw_scores, expected_kept_count)
    if regeneration_increment_ratio <= 0.0:
        merged_indices = previous_indices
    else:
        num_replace = int(round(expected_kept_count * regeneration_increment_ratio))
        num_replace = max(0, min(expected_kept_count, num_replace))
        num_keep_old = expected_kept_count - num_replace

        kept_old = _rank_indices(previous_indices, raw_scores, num_keep_old)
        keep_mask = torch.ones(raw_scores.size(0), dtype=torch.bool, device=raw_scores.device)
        keep_mask[kept_old] = False
        candidate_new = current_indices[keep_mask[current_indices]]
        added_new = _rank_indices(candidate_new, raw_scores, num_replace)
        merged_indices = torch.cat([kept_old, added_new], dim=0)

        if merged_indices.numel() < expected_kept_count:
            fallback_mask = torch.ones(raw_scores.size(0), dtype=torch.bool, device=raw_scores.device)
            fallback_mask[merged_indices] = False
            fallback_candidates = fallback_mask.nonzero(as_tuple=True)[0]
            fallback_added = _rank_indices(
                fallback_candidates,
                raw_scores,
                expected_kept_count - merged_indices.numel(),
            )
            merged_indices = torch.cat([merged_indices, fallback_added], dim=0)

    merged_indices = torch.unique(merged_indices, sorted=True)
    if merged_indices.numel() != expected_kept_count:
        merged_indices = _complete_selected_indices(merged_indices, raw_scores, expected_kept_count)

    overlap_with_current = int(torch.isin(merged_indices, current_indices).sum().item())
    overlap_with_previous = int(torch.isin(merged_indices, previous_indices).sum().item())
    merge_stats = {
        "previous_count": int(previous_indices.numel()),
        "current_count": int(current_indices.numel()),
        "merged_count": int(merged_indices.numel()),
        "overlap_with_current": overlap_with_current,
        "retained_from_previous": overlap_with_previous,
        "replaced_count": int(expected_kept_count - overlap_with_previous),
    }
    return merged_indices.sort().values, merge_stats


def _record_pruning_info(
    pruning_info: Dict[str, Any],
    layer_name: str,
    selected_indices: torch.Tensor,
    dim: int,
) -> None:
    pruning_info["selected_indices"][layer_name] = selected_indices.detach().cpu()
    pruning_info["layer_dims"][layer_name] = int(dim)


def _generate_static_small_model_internal(
    actor: nn.Module,
    previous_pruning_info: Optional[dict],
    regeneration_increment_ratio: float,
):
    if not 0.0 <= regeneration_increment_ratio <= 1.0:
        raise ValueError(
            f"regeneration_increment_ratio must be in [0, 1], got {regeneration_increment_ratio}"
        )

    (qkv_layers, proj_layers, ff1_layers, ff2_layers), (qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2) = (
        _collect_transformer_layer_groups(actor)
    )
    if not qkv_layers and not qkv_layers2:
        print("[setup] skipping static small-model generation because the current actor has no compatible transformer blocks")
        return copy.deepcopy(actor), {"selected_indices": {}, "layer_dims": {}, "merge_stats": {}}
    layer_groups = [
        (actor, qkv_layers, proj_layers, ff1_layers, ff2_layers),
        (actor.vla.language_model, qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2),
    ]

    small_model = copy.deepcopy(actor)
    model_copies = [
        (small_model, qkv_layers, proj_layers, ff1_layers, ff2_layers),
        (small_model.vla.language_model, qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2),
    ]

    pruning_info: Dict[str, Any] = {
        "selected_indices": {},
        "layer_dims": {},
        "merge_stats": {},
    }

    for (source_model, qkv_names, proj_names, ff1_names, ff2_names), (
        target_model,
        _,
        _,
        _,
        _,
    ) in zip(layer_groups, model_copies):
        fbs_layers = get_fbs_layers(qkv_names, proj_names, ff1_names, ff2_names)
        flat_qkv_names = flatten_2d_arr(qkv_names)
        for fbs_layer in fbs_layers:
            source_layer = get_module(source_model, fbs_layer)
            if source_layer is None:
                raise KeyError(f"FBS layer not found: {fbs_layer}")
            if source_layer.cached_raw_w is None or source_layer.cached_w is None:
                raise RuntimeError(
                    f"FBS caches are empty for {fbs_layer}; run a forward pass before generating the static model"
                )

            raw_scores = _aggregate_scores(source_layer.cached_raw_w)
            zeroed_scores = _aggregate_scores(source_layer.cached_w)
            window_merge = getattr(source_layer, "window_merge", None)
            sparsity = float(source_layer.k_takes_all.k)
            expected_kept = _expected_kept_count(source_layer)

            current_indices = zeroed_scores.nonzero(as_tuple=True)[0]
            current_indices = _complete_selected_indices(current_indices, raw_scores, expected_kept)
            selected_indices, merge_stats = _merge_selected_indices(
                fbs_layer,
                current_indices,
                raw_scores,
                expected_kept,
                previous_pruning_info=previous_pruning_info,
                regeneration_increment_ratio=regeneration_increment_ratio,
            )
            if merge_stats:
                pruning_info["merge_stats"][fbs_layer] = merge_stats

            selected_scores = raw_scores[selected_indices]
            target_layer = get_module(target_model, fbs_layer)
            set_module(target_model, fbs_layer, target_layer.raw_linear)

            handled = False
            for qkv_layer_name in flat_qkv_names:
                if not fbs_layer.startswith(qkv_layer_name):
                    continue
                downstream_layer = f"{fbs_layer[:-2]}.1"
                prune_linear_layer_and_its_after_layer(
                    target_model,
                    fbs_layer,
                    downstream_layer,
                    selected_indices,
                    selected_scores,
                    sparsity,
                    get_model_device(target_model),
                    window_merge,
                )
                _record_pruning_info(pruning_info, fbs_layer, selected_indices, 0)
                _record_pruning_info(pruning_info, downstream_layer, selected_indices, 1)
                handled = True
                break
            if handled:
                continue

            for proj_layer_name in proj_names:
                if not fbs_layer.startswith(proj_layer_name):
                    continue
                downstream_layer = f"{fbs_layer[:-2]}.1"
                prune_linear_layer_and_its_after_layer(
                    target_model,
                    fbs_layer,
                    downstream_layer,
                    selected_indices,
                    selected_scores,
                    sparsity,
                    get_model_device(target_model),
                    window_merge,
                )
                _record_pruning_info(pruning_info, fbs_layer, selected_indices, 0)
                _record_pruning_info(pruning_info, downstream_layer, selected_indices, 1)
                handled = True
                break
            if handled:
                continue

            if isinstance(ff1_names[0], list):
                for ff1_layer_name in flatten_2d_arr(ff1_names):
                    if not fbs_layer.startswith(ff1_layer_name):
                        continue
                    downstream_layer = f"{fbs_layer[:-2]}.1"
                    prune_linear_layer_and_its_after_layer(
                        target_model,
                        fbs_layer,
                        downstream_layer,
                        selected_indices,
                        selected_scores,
                        sparsity,
                        get_model_device(target_model),
                        window_merge,
                    )
                    _record_pruning_info(pruning_info, fbs_layer, selected_indices, 0)
                    _record_pruning_info(pruning_info, downstream_layer, selected_indices, 1)
                    handled = True
                    break
            else:
                for ff1_idx, ff1_layer_name in enumerate(ff1_names):
                    if not fbs_layer.startswith(ff1_layer_name):
                        continue
                    downstream_layer = ff2_names[ff1_idx]
                    prune_linear_layer_and_its_after_layer(
                        target_model,
                        fbs_layer,
                        downstream_layer,
                        selected_indices,
                        selected_scores,
                        sparsity,
                        get_model_device(target_model),
                        window_merge,
                    )
                    _record_pruning_info(pruning_info, fbs_layer, selected_indices, 0)
                    _record_pruning_info(pruning_info, downstream_layer, selected_indices, 1)
                    handled = True
                    break
            if handled:
                continue

            if isinstance(ff1_names[0], list):
                for ff2_layer_name in ff2_names:
                    if not fbs_layer.startswith(ff2_layer_name):
                        continue
                    downstream_layer = f"{fbs_layer[:-2]}.1"
                    prune_linear_layer_and_its_after_layer(
                        target_model,
                        fbs_layer,
                        downstream_layer,
                        selected_indices,
                        selected_scores,
                        sparsity,
                        get_model_device(target_model),
                        window_merge,
                    )
                    _record_pruning_info(pruning_info, fbs_layer, selected_indices, 0)
                    _record_pruning_info(pruning_info, downstream_layer, selected_indices, 1)
                    handled = True
                    break
            if not handled:
                raise RuntimeError(f"Unable to map FBS layer to downstream layer: {fbs_layer}")

    return small_model, pruning_info


def _verify_static_small_model(actor: nn.Module, small_model: nn.Module, sample_batch: Optional[Dict[str, Any]]) -> None:
    if sample_batch is None:
        return
    device = get_model_device(actor)
    prepared = _prepare_sample_batch(sample_batch, device)
    actor.eval()
    small_model.eval()
    with torch.no_grad():
        _, large_logprob, _, large_value, _ = actor.get_action_and_value(
            rgbs=prepared["rgbs"],
            states=prepared["states"],
            action_bins=prepared["action_bins"],
            deterministic=True,
        )
        _, small_logprob, _, small_value, _ = small_model.get_action_and_value(
            rgbs=prepared["rgbs"],
            states=prepared["states"],
            action_bins=prepared["action_bins"],
            deterministic=True,
        )
        diff = ((large_logprob - small_logprob) ** 2).sum() + ((large_value - small_value) ** 2).sum()
    print(
        f"FBS verify passed (kb size: {get_model_size(actor, True):.3f}MB, "
        f"proxy model size: {get_model_size(small_model, True):.3f}MB, diff: {float(diff.item()):.6f})"
    )


def generate_static_small_model(
    actor: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    sample_batch: Optional[Dict[str, Any]] = None,
):
    small_model, _ = generate_static_small_model_with_returning_pruning_info(
        actor,
        sample_batch=sample_batch,
        device=device,
        dtype=dtype,
        previous_pruning_info=None,
        regeneration_increment_ratio=1.0,
        verify=sample_batch is not None,
    )
    return small_model


def generate_static_small_model_with_returning_pruning_info(
    actor: nn.Module,
    sample_batch: Optional[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    previous_pruning_info: Optional[dict] = None,
    regeneration_increment_ratio: float = 1.0,
    verify: bool = True,
):
    prepared_sample = None
    if sample_batch is not None:
        prepared_sample = _materialize_fbs_caches(actor, sample_batch)
    small_model, pruning_info = _generate_static_small_model_internal(
        actor,
        previous_pruning_info=previous_pruning_info,
        regeneration_increment_ratio=regeneration_increment_ratio,
    )
    small_model.to(device=device, dtype=dtype)
    small_model.device = device
    if hasattr(small_model, "vla"):
        small_model.vla.to(device=device, dtype=dtype)
    for module_name in ("state_projector", "context_projector", "actor_head", "value_head"):
        if hasattr(small_model, module_name):
            getattr(small_model, module_name).to(device=device, dtype=torch.float32)
    if hasattr(small_model, "action_bin_centers") and "action_bin_centers" in small_model._buffers:
        small_model._buffers["action_bin_centers"] = small_model.action_bin_centers.to(
            device=device,
            dtype=torch.float32,
        )
    if verify:
        _verify_static_small_model(actor, small_model, prepared_sample)
    return small_model, pruning_info


def _unwrap_small_linear(layer: nn.Module) -> nn.Linear:
    if isinstance(layer, nn.Sequential):
        if len(layer) == 0 or not isinstance(layer[0], nn.Linear):
            raise TypeError(f"Unexpected small layer structure: {layer}")
        return layer[0]
    if isinstance(layer, nn.Linear):
        return layer
    raise TypeError(f"Unsupported small layer type: {type(layer).__name__}")


def _unwrap_large_linear(layer: nn.Module) -> nn.Linear:
    if hasattr(layer, "raw_linear") and isinstance(layer.raw_linear, nn.Linear):
        return layer.raw_linear
    if isinstance(layer, nn.Linear):
        return layer
    raise TypeError(f"Unsupported large layer type: {type(layer).__name__}")


def _blend_(target: torch.Tensor, source: torch.Tensor, alpha: float) -> None:
    if target.shape != source.shape:
        raise ValueError(f"shape mismatch during feedback: {tuple(target.shape)} vs {tuple(source.shape)}")
    target.copy_((1.0 - alpha) * target + alpha * source)


@torch.no_grad()
def feedback_static_small_model_to_large_model(
    large_model: nn.Module,
    small_model: nn.Module,
    pruning_info: dict,
    alpha: float,
) -> None:
    if alpha == 0.0:
        return

    selected_indices = pruning_info.get("selected_indices", {})
    layer_dims = pruning_info.get("layer_dims", {})

    for layer_name, indices in selected_indices.items():
        dim = int(layer_dims[layer_name])
        indices = torch.as_tensor(indices, dtype=torch.long)
        large_layer = get_module(large_model, layer_name)
        small_layer = get_module(small_model, layer_name)
        if large_layer is None or small_layer is None:
            continue

        large_linear = _unwrap_large_linear(large_layer)
        small_linear = _unwrap_small_linear(small_layer)
        if dim == 0:
            for small_idx, large_idx in enumerate(indices.tolist()):
                _blend_(large_linear.weight.data[large_idx], small_linear.weight.data[small_idx], alpha)
                if large_linear.bias is not None and small_linear.bias is not None:
                    _blend_(large_linear.bias.data[large_idx], small_linear.bias.data[small_idx], alpha)
        elif dim == 1:
            for small_idx, large_idx in enumerate(indices.tolist()):
                _blend_(large_linear.weight.data[:, large_idx], small_linear.weight.data[:, small_idx], alpha)
        else:
            raise ValueError(f"Unsupported pruning dim: {dim}")

    large_named_parameters = dict(large_model.named_parameters())
    for small_name, small_parameter in small_model.named_parameters():
        large_parameter = large_named_parameters.get(small_name)
        if large_parameter is None or large_parameter.shape != small_parameter.shape:
            continue
        _blend_(large_parameter.data, small_parameter.data, alpha)

    large_named_buffers = dict(large_model.named_buffers())
    for small_name, small_buffer in small_model.named_buffers():
        large_buffer = large_named_buffers.get(small_name)
        if large_buffer is None or large_buffer.shape != small_buffer.shape:
            continue
        if not torch.is_floating_point(large_buffer):
            continue
        _blend_(large_buffer.data, small_buffer.data, alpha)


def _retained_positions(
    new_indices: torch.Tensor,
    previous_indices: torch.Tensor,
) -> Tuple[list, list]:
    new_indices = torch.as_tensor(new_indices, dtype=torch.long).cpu()
    previous_indices = torch.as_tensor(previous_indices, dtype=torch.long).cpu()
    previous_pos = {int(value): position for position, value in enumerate(previous_indices.tolist())}
    new_positions, previous_positions = [], []
    for new_position, value in enumerate(new_indices.tolist()):
        if value in previous_pos:
            new_positions.append(new_position)
            previous_positions.append(previous_pos[value])
    return new_positions, previous_positions


@torch.no_grad()
def inherit_static_small_model_retained_channels(
    new_small_model: nn.Module,
    previous_small_model: nn.Module,
    new_pruning_info: dict,
    previous_pruning_info: dict,
) -> None:
    previous_named_parameters = dict(previous_small_model.named_parameters())
    for new_name, new_parameter in new_small_model.named_parameters():
        previous_parameter = previous_named_parameters.get(new_name)
        if previous_parameter is None or previous_parameter.shape != new_parameter.shape:
            continue
        new_parameter.data.copy_(previous_parameter.data)

    previous_named_buffers = dict(previous_small_model.named_buffers())
    for new_name, new_buffer in new_small_model.named_buffers():
        previous_buffer = previous_named_buffers.get(new_name)
        if previous_buffer is None or previous_buffer.shape != new_buffer.shape:
            continue
        new_buffer.data.copy_(previous_buffer.data)

    new_selected_indices = new_pruning_info.get("selected_indices", {})
    previous_selected_indices = previous_pruning_info.get("selected_indices", {})
    layer_dims = new_pruning_info.get("layer_dims", {})

    for layer_name, new_indices in new_selected_indices.items():
        previous_indices = previous_selected_indices.get(layer_name)
        if previous_indices is None:
            continue
        dim = int(layer_dims[layer_name])
        new_positions, previous_positions = _retained_positions(new_indices, previous_indices)
        if not new_positions:
            continue

        new_layer = get_module(new_small_model, layer_name)
        previous_layer = get_module(previous_small_model, layer_name)
        if new_layer is None or previous_layer is None:
            continue

        new_linear = _unwrap_small_linear(new_layer)
        previous_linear = _unwrap_small_linear(previous_layer)
        if dim == 0:
            for new_pos, previous_pos in zip(new_positions, previous_positions):
                new_linear.weight.data[new_pos].copy_(previous_linear.weight.data[previous_pos])
                if new_linear.bias is not None and previous_linear.bias is not None:
                    new_linear.bias.data[new_pos].copy_(previous_linear.bias.data[previous_pos])
            if (
                isinstance(new_layer, nn.Sequential)
                and len(new_layer) > 1
                and hasattr(new_layer[1], "w")
                and isinstance(previous_layer, nn.Sequential)
                and len(previous_layer) > 1
                and hasattr(previous_layer[1], "w")
            ):
                for new_pos, previous_pos in zip(new_positions, previous_positions):
                    new_layer[1].w.data[:, new_pos].copy_(previous_layer[1].w.data[:, previous_pos])
        elif dim == 1:
            for new_pos, previous_pos in zip(new_positions, previous_positions):
                new_linear.weight.data[:, new_pos].copy_(previous_linear.weight.data[:, previous_pos])
        else:
            raise ValueError(f"Unsupported pruning dim: {dim}")
