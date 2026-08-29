import torch
from torch import nn
import copy
from ours.utils.common.others import get_cur_time_str
from ours.utils.dl.common.model import get_model_size, get_module, set_module, get_model_device
from transformers.pytorch_utils import prune_linear_layer
from ours.libs.train_with_fbs.lib_transformer import StaticFBS
from ours.utils.common.log import logger

import os
from ours.libs.train_with_fbs.lib_cnn import Conv2dWithFBS, Linear2DWithFBS
# from ours.libs.train_with_fbs.lib_transformer import LinearWithFBS
from ours.utils.third_party.nni_new.compression.pytorch.speedup import ModelSpeedup


class FeatureBoosting(nn.Module):
    def __init__(self, w: torch.Tensor, is_conv):
        super(FeatureBoosting, self).__init__()
        assert w.dim() == 1, w.size()
        if is_conv:
            self.w = nn.Parameter(w.unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)
        else:
            self.w = nn.Parameter(w.unsqueeze(0), requires_grad=False)
        
    def forward(self, x):
        # print(x.size(), self.w.size())
        return x * self.w


@torch.no_grad()
def generate_small_cnn(fbs_model: nn.Module,
                       a_sample,
                       model_forward_fn,
                       return_pruning_info=False,
                       previous_pruning_info=None,
                       regeneration_increment_ratio=1.0,
                       ab_strategy=None):
    feature_boosting_info = {}
    pruning_masks = {}
    merge_stats = {}

    if not 0.0 <= regeneration_increment_ratio <= 1.0:
        raise ValueError(
            f'regeneration_increment_ratio must be in [0, 1], got {regeneration_increment_ratio}'
        )

    previous_selected_indices_dict = None
    if previous_pruning_info is not None:
        previous_selected_indices_dict = previous_pruning_info.get('selected_indices')

    def _aggregate_scores(score_tensor: torch.Tensor):
        if score_tensor.size(0) == 1:
            return score_tensor.squeeze()
        return score_tensor.mean(0).squeeze()

    def _get_expected_kept_count(layer):
        if isinstance(layer, Conv2dWithFBS):
            total_count = layer.raw_conv2d.out_channels
            sparsity = layer.k_takes_all.k
        elif isinstance(layer, Linear2DWithFBS):
            total_count = layer.raw_linear.out_features
            sparsity = layer.k_takes_all.k
        else:
            raise TypeError(f'Unsupported FBS layer type: {type(layer).__name__}')
        return total_count - int(total_count * sparsity)

    def _complete_unpruned_filters_index(unpruned_filters_index, raw_scores, expected_kept_count):
        current_count = len(unpruned_filters_index)
        if current_count == expected_kept_count:
            return unpruned_filters_index

        if current_count > expected_kept_count:
            ranked_existing = torch.argsort(raw_scores[unpruned_filters_index], descending=True, stable=True)
            selected = unpruned_filters_index[ranked_existing[:expected_kept_count]]
            return selected.sort().values

        candidate_mask = torch.ones(raw_scores.size(0), dtype=torch.bool, device=raw_scores.device)
        candidate_mask[unpruned_filters_index] = False
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        ranked_candidates = torch.argsort(raw_scores[candidate_indices], descending=True, stable=True)
        fill_count = expected_kept_count - current_count
        filled_indices = candidate_indices[ranked_candidates[:fill_count]]
        completed = torch.cat([unpruned_filters_index, filled_indices], dim=0)
        return completed.sort().values

    def _sanitize_previous_indices(previous_indices: torch.Tensor, total_count: int):
        if previous_indices.numel() == 0:
            return previous_indices.to(dtype=torch.long)

        previous_indices = previous_indices.to(dtype=torch.long)
        valid_mask = (previous_indices >= 0) & (previous_indices < total_count)
        previous_indices = previous_indices[valid_mask]
        if previous_indices.numel() == 0:
            return previous_indices
        return torch.unique(previous_indices, sorted=True)

    def _rank_indices(indices: torch.Tensor, raw_scores: torch.Tensor, limit: int):
        if limit <= 0 or indices.numel() == 0:
            return indices[:0]
        ranked = torch.argsort(raw_scores[indices], descending=True, stable=True)
        return indices[ranked[:limit]]

    def _merge_selected_indices(layer_name: str,
                                current_indices: torch.Tensor,
                                raw_scores: torch.Tensor,
                                expected_kept_count: int):
        if (
            previous_selected_indices_dict is None
            or layer_name not in previous_selected_indices_dict
            or regeneration_increment_ratio >= 1.0
        ):
            return current_indices.sort().values

        previous_indices = previous_selected_indices_dict[layer_name]
        if not isinstance(previous_indices, torch.Tensor):
            previous_indices = torch.as_tensor(previous_indices, device=raw_scores.device)
        else:
            previous_indices = previous_indices.to(raw_scores.device)

        previous_indices = _sanitize_previous_indices(previous_indices, raw_scores.size(0))
        if previous_indices.numel() == 0:
            return current_indices.sort().values

        previous_indices = _complete_unpruned_filters_index(
            previous_indices,
            raw_scores,
            expected_kept_count,
        )

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
            merged_indices = _complete_unpruned_filters_index(
                merged_indices,
                raw_scores,
                expected_kept_count,
            )

        overlap_count = torch.isin(merged_indices, current_indices).sum().item()
        merge_stats[layer_name] = {
            'previous_count': int(previous_indices.numel()),
            'current_count': int(current_indices.numel()),
            'merged_count': int(merged_indices.numel()),
            'overlap_with_current': int(overlap_count),
            'replaced_count': int(expected_kept_count - torch.isin(merged_indices, previous_indices).sum().item()),
        }
        return merged_indices.sort().values

    fbs_model.eval()
    o1 = model_forward_fn(fbs_model, a_sample)

    
    unpruned_filters_index_dict = {}
    
    for layer_name, layer in fbs_model.named_modules():
        if isinstance(layer, Conv2dWithFBS):
            cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_conv2d.weight.data)}
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_conv2d.bias.data)
            
            w = _aggregate_scores(get_module(fbs_model, layer_name).cached_w)
            raw_w = _aggregate_scores(get_module(fbs_model, layer_name).cached_raw_w)
            expected_kept_count = _get_expected_kept_count(layer)

            if ab_strategy == 'random':
                unpruned_filters_index = torch.randperm(raw_w.size(0), device=raw_w.device)[:expected_kept_count]
            elif ab_strategy == 'inverse':
                unpruned_filters_index = torch.argsort(raw_w, descending=False, stable=True)[:expected_kept_count]
            else:
                unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            unpruned_filters_index = _complete_unpruned_filters_index(
                unpruned_filters_index,
                raw_w,
                expected_kept_count,
            )
            unpruned_filters_index = _merge_selected_indices(
                layer_name,
                unpruned_filters_index,
                raw_w,
                expected_kept_count,
            )
            feature_boosting_info[layer_name] = raw_w
            
            cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            pruning_masks[layer_name + '.0'] = cur_pruning_mask

            unpruned_filters_index_dict[layer_name] = unpruned_filters_index

        elif isinstance(layer, Linear2DWithFBS):
            # cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_linear.weight.data)}
            # if layer.raw_linear.bias is not None:
            #     cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_linear.bias.data)
            
            w = _aggregate_scores(layer.cached_w)
            raw_w = _aggregate_scores(layer.cached_raw_w)
            expected_kept_count = _get_expected_kept_count(layer)

            if ab_strategy == 'random':
                unpruned_filters_index = torch.randperm(raw_w.size(0), device=raw_w.device)[:expected_kept_count]
            elif ab_strategy == 'inverse':
                unpruned_filters_index = torch.argsort(raw_w, descending=False, stable=True)[:expected_kept_count]
            else:
                unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            unpruned_filters_index = _complete_unpruned_filters_index(
                unpruned_filters_index,
                raw_w,
                expected_kept_count,
            )
            unpruned_filters_index = _merge_selected_indices(
                layer_name,
                unpruned_filters_index,
                raw_w,
                expected_kept_count,
            )
            feature_boosting_info[layer_name] = raw_w

            

            unpruned_filters_index_dict[layer_name] = unpruned_filters_index
            
            # cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            # if layer.raw_linear.bias is not None:
            #     cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            # pruning_masks[layer_name] = cur_pruning_mask
    
    no_gate_model = copy.deepcopy(fbs_model)
    for name, layer in no_gate_model.named_modules():
        if isinstance(layer, Conv2dWithFBS):
            set_module(no_gate_model, name, nn.Sequential(layer.raw_conv2d, layer.bn, nn.Identity()))
        elif isinstance(layer, Linear2DWithFBS):
            set_module(no_gate_model, name, layer.raw_linear)

    last_linear = None
    last_linear_map = {}
    for name, layer in no_gate_model.named_modules():
        if isinstance(layer, nn.Linear):
            if last_linear is not None:
                last_linear_map[last_linear] = name
            last_linear = name

    # print(no_gate_model)
    # print(pruning_masks)
        
    # fixed_pruning_masks = fix_mask_conflict(pruning_masks, fbs_model, sample.size(), None, True, True, True)
    tmp_mask_path = f'tmp_mask_{get_cur_time_str()}_{os.getpid()}.pth'
    torch.save(pruning_masks, tmp_mask_path)
    pruned_model = no_gate_model
    pruned_model.eval()
    from functools import partial
    # print(last_linear_map)
    model_speedup = ModelSpeedup(pruned_model, a_sample, tmp_mask_path)
    model_speedup.speedup_model()
    os.remove(tmp_mask_path)

    for layer_name, w in unpruned_filters_index_dict.items():
        if isinstance(get_module(pruned_model, layer_name), nn.Linear):
            # print('prune linear')
            set_module(pruned_model, layer_name, prune_linear_layer(get_module(pruned_model, layer_name), w, dim=0))
            set_module(pruned_model, last_linear_map[layer_name], 
                       prune_linear_layer(get_module(pruned_model, last_linear_map[layer_name]), w, dim=1))
    # print(pruned_model)
    
    # add feature boosting module
    for layer_name, feature_boosting_w in feature_boosting_info.items():
        feature_boosting_w = feature_boosting_w[unpruned_filters_index_dict[layer_name]]
        # print(layer_name, feature_boosting_w.size())
        if not isinstance(get_module(pruned_model, layer_name), nn.Linear):
            set_module(pruned_model, layer_name + '.2', FeatureBoosting(feature_boosting_w, True))
        else:
            set_module(pruned_model, layer_name, nn.Sequential(get_module(pruned_model, layer_name), 
                                                               FeatureBoosting(feature_boosting_w, False)))
    
    pruned_model_size = get_model_size(pruned_model, True)
    pruned_model.eval()
    o2 = model_forward_fn(pruned_model, a_sample)
    diff = ((o1 - o2) ** 2).sum()
    print(f'pruned model size: {pruned_model_size:.3f}MB, diff: {diff}')
    
    if return_pruning_info:
        return pruned_model, {
            'selected_indices': unpruned_filters_index_dict,
            'merge_stats': merge_stats,
        }

    return pruned_model


def _find_next_param_module(model: nn.Module, layer_name: str):
    found_current = False
    for candidate_name, candidate_module in model.named_modules():
        if candidate_name == layer_name:
            found_current = True
            continue
        if not found_current:
            continue
        if candidate_name.startswith(layer_name + '.'):
            continue
        if isinstance(candidate_module, (nn.Conv2d, nn.Linear)):
            return candidate_name, candidate_module
    return None, None


def get_small_cnn_channel_change_info(new_pruning_info: dict,
                                      previous_pruning_info: dict):
    new_selected_indices = new_pruning_info.get('selected_indices', {})
    previous_selected_indices = previous_pruning_info.get('selected_indices', {})
    channel_change_info = {}

    for layer_name, new_indices in new_selected_indices.items():
        previous_indices = previous_selected_indices.get(layer_name, [])

        if not isinstance(new_indices, torch.Tensor):
            new_indices = torch.as_tensor(new_indices, dtype=torch.long)
        else:
            new_indices = new_indices.detach().to(dtype=torch.long).cpu()

        if not isinstance(previous_indices, torch.Tensor):
            previous_indices = torch.as_tensor(previous_indices, dtype=torch.long)
        else:
            previous_indices = previous_indices.detach().to(dtype=torch.long).cpu()

        new_large_indices = [int(index) for index in new_indices.tolist()]
        previous_large_indices = [int(index) for index in previous_indices.tolist()]
        previous_large_index_set = set(previous_large_indices)

        new_positions_by_large_index = {
            large_index: position
            for position, large_index in enumerate(new_large_indices)
        }
        previous_positions_by_large_index = {
            large_index: position
            for position, large_index in enumerate(previous_large_indices)
        }

        retained_large_indices = [
            large_index
            for large_index in new_large_indices
            if large_index in previous_large_index_set
        ]
        replaced_large_indices = [
            large_index
            for large_index in new_large_indices
            if large_index not in previous_large_index_set
        ]

        channel_change_info[layer_name] = {
            'new_large_indices': new_large_indices,
            'previous_large_indices': previous_large_indices,
            'retained_large_indices': retained_large_indices,
            'replaced_large_indices': replaced_large_indices,
            'retained_new_positions': [
                new_positions_by_large_index[index] for index in retained_large_indices
            ],
            'retained_previous_positions': [
                previous_positions_by_large_index[index] for index in retained_large_indices
            ],
            'replaced_new_positions': [
                new_positions_by_large_index[index] for index in replaced_large_indices
            ],
            'new_selected_count': len(new_large_indices),
            'previous_selected_count': len(previous_large_indices),
        }

    return channel_change_info


@torch.no_grad()
def remap_small_cnn_optimizer_state(optimizer,
                                    small_model: nn.Module,
                                    new_pruning_info: dict,
                                    previous_pruning_info: dict):
    channel_change_info = get_small_cnn_channel_change_info(
        new_pruning_info,
        previous_pruning_info,
    )

    def _remap_matching_state_tensors(param: torch.nn.Parameter, copy_fn):
        param_state = optimizer.state.get(param)
        if not param_state:
            return

        for state_value in param_state.values():
            if not isinstance(state_value, torch.Tensor):
                continue
            if state_value.shape != param.shape:
                continue

            remapped_state = torch.zeros_like(state_value)
            copy_fn(remapped_state, state_value)
            state_value.copy_(remapped_state)

    def _copy_output_rows(remapped_state: torch.Tensor,
                          previous_state: torch.Tensor,
                          previous_positions,
                          new_positions):
        for previous_position, new_position in zip(previous_positions, new_positions):
            remapped_state[new_position].copy_(previous_state[previous_position])

    def _copy_conv_input_columns(remapped_state: torch.Tensor,
                                 previous_state: torch.Tensor,
                                 previous_positions,
                                 new_positions):
        for previous_position, new_position in zip(previous_positions, new_positions):
            remapped_state[:, new_position, ...].copy_(previous_state[:, previous_position, ...])

    def _copy_linear_input_columns(remapped_state: torch.Tensor,
                                   previous_state: torch.Tensor,
                                   previous_positions,
                                   new_positions):
        for previous_position, new_position in zip(previous_positions, new_positions):
            remapped_state[:, new_position].copy_(previous_state[:, previous_position])

    def _copy_flattened_linear_input_blocks(remapped_state: torch.Tensor,
                                            previous_state: torch.Tensor,
                                            previous_positions,
                                            new_positions,
                                            previous_selected_count: int,
                                            new_selected_count: int):
        if previous_selected_count <= 0 or new_selected_count <= 0:
            return

        previous_channel_block = previous_state.size(1) // previous_selected_count
        new_channel_block = remapped_state.size(1) // new_selected_count
        if previous_channel_block != new_channel_block:
            raise ValueError(
                f'channel block mismatch during optimizer state remap: '
                f'{previous_channel_block} vs {new_channel_block}'
            )

        for previous_position, new_position in zip(previous_positions, new_positions):
            previous_slice = slice(
                previous_position * previous_channel_block,
                (previous_position + 1) * previous_channel_block,
            )
            new_slice = slice(
                new_position * new_channel_block,
                (new_position + 1) * new_channel_block,
            )
            remapped_state[:, new_slice].copy_(previous_state[:, previous_slice])

    for layer_name, layer_change in channel_change_info.items():
        layer = get_module(small_model, layer_name)
        if layer is None:
            continue

        retained_previous_positions = layer_change['retained_previous_positions']
        retained_new_positions = layer_change['retained_new_positions']
        new_selected_count = layer_change['new_selected_count']
        previous_selected_count = layer_change['previous_selected_count']

        if len(layer) == 0:
            continue

        if isinstance(layer[0], nn.Conv2d):
            _remap_matching_state_tensors(
                layer[0].weight,
                lambda remapped_state, previous_state: _copy_output_rows(
                    remapped_state,
                    previous_state,
                    retained_previous_positions,
                    retained_new_positions,
                ),
            )
            if layer[0].bias is not None:
                _remap_matching_state_tensors(
                    layer[0].bias,
                    lambda remapped_state, previous_state: _copy_output_rows(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                    ),
                )
            if len(layer) > 1 and isinstance(layer[1], nn.BatchNorm2d):
                _remap_matching_state_tensors(
                    layer[1].weight,
                    lambda remapped_state, previous_state: _copy_output_rows(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                    ),
                )
                _remap_matching_state_tensors(
                    layer[1].bias,
                    lambda remapped_state, previous_state: _copy_output_rows(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                    ),
                )
            is_current_layer_conv = True
        elif isinstance(layer[0], nn.Linear):
            _remap_matching_state_tensors(
                layer[0].weight,
                lambda remapped_state, previous_state: _copy_output_rows(
                    remapped_state,
                    previous_state,
                    retained_previous_positions,
                    retained_new_positions,
                ),
            )
            if layer[0].bias is not None:
                _remap_matching_state_tensors(
                    layer[0].bias,
                    lambda remapped_state, previous_state: _copy_output_rows(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                    ),
                )
            is_current_layer_conv = False
        else:
            raise ValueError(
                f'unsupported layer type during optimizer state remap: '
                f'{type(layer[0]).__name__}'
            )

        _, next_layer = _find_next_param_module(small_model, layer_name)
        if next_layer is None:
            continue

        if isinstance(next_layer, nn.Conv2d):
            _remap_matching_state_tensors(
                next_layer.weight,
                lambda remapped_state, previous_state: _copy_conv_input_columns(
                    remapped_state,
                    previous_state,
                    retained_previous_positions,
                    retained_new_positions,
                ),
            )
        elif isinstance(next_layer, nn.Linear):
            if is_current_layer_conv:
                _remap_matching_state_tensors(
                    next_layer.weight,
                    lambda remapped_state, previous_state: _copy_flattened_linear_input_blocks(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                        previous_selected_count,
                        new_selected_count,
                    ),
                )
            else:
                _remap_matching_state_tensors(
                    next_layer.weight,
                    lambda remapped_state, previous_state: _copy_linear_input_columns(
                        remapped_state,
                        previous_state,
                        retained_previous_positions,
                        retained_new_positions,
                    ),
                )
        else:
            raise ValueError(
                f'unsupported next layer during optimizer state remap: '
                f'{type(next_layer).__name__}'
            )


@torch.no_grad()
def inherit_small_cnn_retained_channels(new_small_model: nn.Module,
                                        previous_small_model: nn.Module,
                                        new_pruning_info: dict,
                                        previous_pruning_info: dict):
    channel_change_info = get_small_cnn_channel_change_info(
        new_pruning_info,
        previous_pruning_info,
    )
    regenerated_state_dict = {
        key: value.detach().clone()
        for key, value in new_small_model.state_dict().items()
    }

    def _copy_same_name_same_shape_tensors():
        previous_named_parameters = dict(previous_small_model.named_parameters())
        for new_name, new_parameter in new_small_model.named_parameters():
            previous_parameter = previous_named_parameters.get(new_name)
            if previous_parameter is None:
                continue
            if previous_parameter.shape != new_parameter.shape:
                continue
            new_parameter.data.copy_(previous_parameter.data)

        previous_named_buffers = dict(previous_small_model.named_buffers())
        for new_name, new_buffer in new_small_model.named_buffers():
            previous_buffer = previous_named_buffers.get(new_name)
            if previous_buffer is None:
                continue
            if previous_buffer.shape != new_buffer.shape:
                continue
            new_buffer.data.copy_(previous_buffer.data)

    def _rebuild_output_rows(previous_tensor: torch.Tensor,
                             regenerated_tensor: torch.Tensor,
                             previous_positions,
                             new_positions):
        rebuilt_tensor = regenerated_tensor.clone()
        for previous_position, new_position in zip(previous_positions, new_positions):
            rebuilt_tensor[new_position].copy_(previous_tensor[previous_position])
        return rebuilt_tensor

    def _rebuild_conv_input_columns(previous_tensor: torch.Tensor,
                                    regenerated_tensor: torch.Tensor,
                                    previous_positions,
                                    new_positions):
        rebuilt_tensor = regenerated_tensor.clone()
        for previous_position, new_position in zip(previous_positions, new_positions):
            rebuilt_tensor[:, new_position, ...].copy_(previous_tensor[:, previous_position, ...])
        return rebuilt_tensor

    def _rebuild_linear_input_columns(previous_tensor: torch.Tensor,
                                      regenerated_tensor: torch.Tensor,
                                      previous_positions,
                                      new_positions):
        rebuilt_tensor = regenerated_tensor.clone()
        for previous_position, new_position in zip(previous_positions, new_positions):
            rebuilt_tensor[:, new_position].copy_(previous_tensor[:, previous_position])
        return rebuilt_tensor

    def _rebuild_flattened_linear_input_blocks(previous_tensor: torch.Tensor,
                                               regenerated_tensor: torch.Tensor,
                                               previous_positions,
                                               new_positions,
                                               previous_selected_count: int,
                                               new_selected_count: int):
        rebuilt_tensor = regenerated_tensor.clone()
        if len(previous_positions) == 0:
            return rebuilt_tensor

        previous_channel_block = previous_tensor.size(1) // previous_selected_count
        new_channel_block = regenerated_tensor.size(1) // new_selected_count
        if previous_channel_block != new_channel_block:
            raise ValueError(
                f'channel block mismatch during retained channel inheritance: '
                f'{previous_channel_block} vs {new_channel_block}'
            )

        for previous_position, new_position in zip(previous_positions, new_positions):
            previous_slice = slice(
                previous_position * previous_channel_block,
                (previous_position + 1) * previous_channel_block,
            )
            new_slice = slice(
                new_position * new_channel_block,
                (new_position + 1) * new_channel_block,
            )
            rebuilt_tensor[:, new_slice].copy_(previous_tensor[:, previous_slice])
        return rebuilt_tensor

    def _copy_next_layer_input_columns_by_positions(
        new_next_module: nn.Module,
        previous_next_module: nn.Module,
        next_layer_name: str,
        previous_positions,
        new_positions,
        is_current_layer_conv: bool,
    ):
        if new_next_module is None or previous_next_module is None:
            return

        if isinstance(new_next_module, nn.Conv2d) and isinstance(previous_next_module, nn.Conv2d):
            new_next_module.weight.data.copy_(
                _rebuild_conv_input_columns(
                    previous_next_module.weight.data,
                    regenerated_state_dict[f'{next_layer_name}.weight'],
                    previous_positions,
                    new_positions,
                )
            )
            return

        if isinstance(new_next_module, nn.Linear) and isinstance(previous_next_module, nn.Linear):
            if is_current_layer_conv:
                new_next_module.weight.data.copy_(
                    _rebuild_flattened_linear_input_blocks(
                        previous_next_module.weight.data,
                        regenerated_state_dict[f'{next_layer_name}.weight'],
                        previous_positions,
                        new_positions,
                        previous_selected_count,
                        new_selected_count,
                    )
                )
                return

            new_next_module.weight.data.copy_(
                _rebuild_linear_input_columns(
                    previous_next_module.weight.data,
                    regenerated_state_dict[f'{next_layer_name}.weight'],
                    previous_positions,
                    new_positions,
                )
            )
            return

        raise ValueError(
            f'unsupported next-layer pair during retained channel inheritance: '
            f'{type(previous_next_module).__name__} vs {type(new_next_module).__name__}'
        )

    def _find_feature_boosting_module(layer_module: nn.Module):
        if not isinstance(layer_module, nn.Sequential):
            return None, None
        for module_index, sub_module in enumerate(layer_module):
            if isinstance(sub_module, FeatureBoosting):
                return str(module_index), sub_module
        return None, None

    def _copy_feature_boosting_channels_by_positions(
        layer_name: str,
        new_feature_boosting: FeatureBoosting,
        previous_feature_boosting: FeatureBoosting,
        previous_positions,
        new_positions,
    ):
        if new_feature_boosting is None or previous_feature_boosting is None:
            return

        new_weight = new_feature_boosting.w.data
        previous_weight = previous_feature_boosting.w.data
        if new_weight.shape != previous_weight.shape:
            raise ValueError(
                f'feature boosting shape mismatch during retained channel inheritance: '
                f'{tuple(previous_weight.shape)} vs {tuple(new_weight.shape)}'
            )
        if new_weight.dim() < 2 or new_weight.size(0) != 1:
            raise ValueError(
                f'unexpected feature boosting weight shape during retained channel inheritance: '
                f'{tuple(new_weight.shape)}'
            )

        new_weight.copy_(
            _rebuild_output_rows(
                previous_weight.transpose(0, 1),
                regenerated_state_dict[layer_name].transpose(0, 1),
                previous_positions,
                new_positions,
            ).transpose(0, 1)
        )

    _copy_same_name_same_shape_tensors()

    for layer_name, layer_change in channel_change_info.items():
        retained_new_positions = layer_change['retained_new_positions']
        retained_previous_positions = layer_change['retained_previous_positions']
        if not retained_new_positions:
            continue

        new_layer = get_module(new_small_model, layer_name)
        previous_layer = get_module(previous_small_model, layer_name)
        if new_layer is None or previous_layer is None:
            continue

        is_current_layer_conv = isinstance(new_layer[0], nn.Conv2d) and isinstance(previous_layer[0], nn.Conv2d)
        is_current_layer_linear = isinstance(new_layer[0], nn.Linear) and isinstance(previous_layer[0], nn.Linear)

        if is_current_layer_conv or is_current_layer_linear:
            new_layer[0].weight.data.copy_(
                _rebuild_output_rows(
                    previous_layer[0].weight.data,
                    regenerated_state_dict[f'{layer_name}.0.weight'],
                    retained_previous_positions,
                    retained_new_positions,
                )
            )
            if new_layer[0].bias is not None and previous_layer[0].bias is not None:
                new_layer[0].bias.data.copy_(
                    _rebuild_output_rows(
                        previous_layer[0].bias.data,
                        regenerated_state_dict[f'{layer_name}.0.bias'],
                        retained_previous_positions,
                        retained_new_positions,
                    )
                )
            if is_current_layer_conv and len(new_layer) > 1 and isinstance(new_layer[1], nn.BatchNorm2d) and isinstance(previous_layer[1], nn.BatchNorm2d):
                for bn_tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    getattr(new_layer[1], bn_tensor_name).data.copy_(
                        _rebuild_output_rows(
                            getattr(previous_layer[1], bn_tensor_name).data,
                            regenerated_state_dict[f'{layer_name}.1.{bn_tensor_name}'],
                            retained_previous_positions,
                            retained_new_positions,
                        )
                    )
        else:
            raise ValueError(
                f'unsupported layer pair during retained channel inheritance: '
                f'{type(previous_layer[0]).__name__} vs {type(new_layer[0]).__name__}'
            )

        new_feature_boosting_index, new_feature_boosting = _find_feature_boosting_module(new_layer)
        previous_feature_boosting_index, previous_feature_boosting = _find_feature_boosting_module(previous_layer)
        if new_feature_boosting_index != previous_feature_boosting_index:
            raise ValueError(
                f'feature boosting index mismatch during retained channel inheritance: '
                f'{previous_feature_boosting_index} vs {new_feature_boosting_index}'
            )
        if new_feature_boosting is not None and previous_feature_boosting is not None:
            _copy_feature_boosting_channels_by_positions(
                f'{layer_name}.{new_feature_boosting_index}.w',
                new_feature_boosting,
                previous_feature_boosting,
                retained_previous_positions,
                retained_new_positions,
            )

        previous_selected_count = layer_change['previous_selected_count']
        new_selected_count = layer_change['new_selected_count']
        previous_next_layer_name, previous_next_layer = _find_next_param_module(previous_small_model, layer_name)
        new_next_layer_name, new_next_layer = _find_next_param_module(new_small_model, layer_name)
        if previous_next_layer_name is None or new_next_layer_name is None:
            continue

        _copy_next_layer_input_columns_by_positions(
            new_next_layer,
            previous_next_layer,
            new_next_layer_name,
            retained_previous_positions,
            retained_new_positions,
            is_current_layer_conv=is_current_layer_conv,
        )


def small_cnn_feedback(large_model: nn.Module,
                       small_model: nn.Module,
                       pruning_info: dict,
                       alpha: float):
    
    if alpha == 0.:
        return

    if 'selected_indices' in pruning_info:
        selected_indices_dict = pruning_info['selected_indices']
    else:
        selected_indices_dict = {
            layer_name: feature_boosting_w.nonzero(as_tuple=True)[0]
            for layer_name, feature_boosting_w in pruning_info.items()
        }

    def _blend_(target: torch.Tensor, source: torch.Tensor):
        if target.shape != source.shape:
            raise ValueError(f'shape mismatch during small_cnn_feedback: {target.shape} vs {source.shape}')
        target.copy_((1. - alpha) * target + alpha * source)

    def _copy_same_name_same_shape_tensors():
        large_named_parameters = dict(large_model.named_parameters())
        for small_name, small_parameter in small_model.named_parameters():
            large_parameter = large_named_parameters.get(small_name)
            if large_parameter is None or large_parameter.shape != small_parameter.shape:
                continue
            _blend_(large_parameter.data, small_parameter.data)

        large_named_buffers = dict(large_model.named_buffers())
        for small_name, small_buffer in small_model.named_buffers():
            large_buffer = large_named_buffers.get(small_name)
            if large_buffer is None or large_buffer.shape != small_buffer.shape:
                continue
            if not torch.is_floating_point(large_buffer):
                continue
            _blend_(large_buffer.data, small_buffer.data)

    def _find_next_param_module(model: nn.Module, layer_name: str):
        found_current = False
        for candidate_name, candidate_module in model.named_modules():
            if candidate_name == layer_name:
                found_current = True
                continue
            if not found_current:
                continue
            if candidate_name.startswith(layer_name + '.'):
                continue
            if isinstance(candidate_module, (nn.Conv2d, nn.Linear)):
                return candidate_name, candidate_module
        return None, None

    def _copy_next_layer_input_columns(
        large_next_module: nn.Module,
        small_next_module: nn.Module,
        unpruned_filters_index: torch.Tensor,
        is_current_layer_conv: bool,
    ):
        if large_next_module is None or small_next_module is None:
            return

        if isinstance(large_next_module, nn.Conv2d) and isinstance(small_next_module, nn.Conv2d):
            if small_next_module.weight.size(1) != len(unpruned_filters_index):
                raise ValueError(
                    f'conv input size mismatch during small_cnn_feedback: '
                    f'{small_next_module.weight.size(1)} vs {len(unpruned_filters_index)}'
                )
            for small_in_idx, large_in_idx in enumerate(unpruned_filters_index.tolist()):
                _blend_(
                    large_next_module.weight.data[:, large_in_idx, ...],
                    small_next_module.weight.data[:, small_in_idx, ...],
                )
            return

        if isinstance(large_next_module, nn.Linear) and isinstance(small_next_module, nn.Linear):
            if is_current_layer_conv:
                if len(unpruned_filters_index) == 0:
                    return
                if small_next_module.weight.size(1) % len(unpruned_filters_index) != 0:
                    raise ValueError(
                        f'flattened linear input is not divisible by selected channels: '
                        f'{small_next_module.weight.size(1)} vs {len(unpruned_filters_index)}'
                    )
                channel_block = small_next_module.weight.size(1) // len(unpruned_filters_index)
                for small_in_idx, large_in_idx in enumerate(unpruned_filters_index.tolist()):
                    small_slice = slice(small_in_idx * channel_block, (small_in_idx + 1) * channel_block)
                    large_slice = slice(large_in_idx * channel_block, (large_in_idx + 1) * channel_block)
                    _blend_(
                        large_next_module.weight.data[:, large_slice],
                        small_next_module.weight.data[:, small_slice],
                    )
                return

            if small_next_module.weight.size(1) != len(unpruned_filters_index):
                raise ValueError(
                    f'linear input size mismatch during small_cnn_feedback: '
                    f'{small_next_module.weight.size(1)} vs {len(unpruned_filters_index)}'
                )
            for small_in_idx, large_in_idx in enumerate(unpruned_filters_index.tolist()):
                _blend_(
                    large_next_module.weight.data[:, large_in_idx],
                    small_next_module.weight.data[:, small_in_idx],
                )
            return

        raise ValueError(
            f'unsupported next-layer pair during small_cnn_feedback: '
            f'{type(large_next_module).__name__} vs {type(small_next_module).__name__}'
        )
    
    for layer_name, unpruned_filters_index in selected_indices_dict.items():
        
        fbs_layer = get_module(large_model, layer_name)
        sub_model_layer = get_module(small_model, layer_name)
        next_large_layer_name, next_large_layer = _find_next_param_module(large_model, layer_name)
        next_small_layer_name, next_small_layer = _find_next_param_module(small_model, layer_name)
        is_current_layer_conv = hasattr(fbs_layer, 'raw_conv2d') and isinstance(fbs_layer.raw_conv2d, nn.Conv2d)

        for fi_in_sub_layer, fi_in_fbs_layer in enumerate(unpruned_filters_index):
            if is_current_layer_conv:
                _blend_(
                    fbs_layer.raw_conv2d.weight.data[fi_in_fbs_layer],
                    sub_model_layer[0].weight.data[fi_in_sub_layer],
                )

                if fbs_layer.raw_conv2d.bias is not None:
                    if sub_model_layer[0].bias is None:
                        raise ValueError(f'{layer_name} bias structure mismatch during small_cnn_feedback')
                    _blend_(
                        fbs_layer.raw_conv2d.bias.data[fi_in_fbs_layer],
                        sub_model_layer[0].bias.data[fi_in_sub_layer],
                    )
                
            if hasattr(fbs_layer, 'raw_linear') and isinstance(fbs_layer.raw_linear, nn.Linear):
                _blend_(
                    fbs_layer.raw_linear.weight.data[fi_in_fbs_layer],
                    sub_model_layer[0].weight.data[fi_in_sub_layer],
                )
                
                if fbs_layer.raw_linear.bias is not None:
                    if sub_model_layer[0].bias is None:
                        raise ValueError(f'{layer_name} bias structure mismatch during small_cnn_feedback')
                    _blend_(
                        fbs_layer.raw_linear.bias.data[fi_in_fbs_layer],
                        sub_model_layer[0].bias.data[fi_in_sub_layer],
                    )

            if is_current_layer_conv and hasattr(fbs_layer, 'bn') and isinstance(fbs_layer.bn, nn.BatchNorm2d):
                sub_bn = sub_model_layer[1]
                _blend_(
                    fbs_layer.bn.weight.data[fi_in_fbs_layer],
                    sub_bn.weight.data[fi_in_sub_layer],
                )
                _blend_(
                    fbs_layer.bn.bias.data[fi_in_fbs_layer],
                    sub_bn.bias.data[fi_in_sub_layer],
                )
                _blend_(
                    fbs_layer.bn.running_mean.data[fi_in_fbs_layer],
                    sub_bn.running_mean.data[fi_in_sub_layer],
                )
                _blend_(
                    fbs_layer.bn.running_var.data[fi_in_fbs_layer],
                    sub_bn.running_var.data[fi_in_sub_layer],
                )

        if next_large_layer_name is not None or next_small_layer_name is not None:
            if next_large_layer_name is None or next_small_layer_name is None:
                raise ValueError(
                    f'next-layer mapping mismatch for {layer_name}: '
                    f'large={next_large_layer_name}, small={next_small_layer_name}'
                )
            _copy_next_layer_input_columns(
                next_large_layer,
                next_small_layer,
                unpruned_filters_index,
                is_current_layer_conv=is_current_layer_conv,
            )

    _copy_same_name_same_shape_tensors()
