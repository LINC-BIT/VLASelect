import random
from copy import deepcopy
import sys; sys.path.append('.')
import numpy as np
import torch

from ours.libs.gen_scaling_law_data_points_cnn import (
    FeatureBoosting,
    get_small_cnn_channel_change_info,
    inherit_small_cnn_retained_channels,
    remap_small_cnn_optimizer_state,
    small_cnn_feedback,
)
from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn, generate_small_cnn_with_verify
from ours.utils.dl.common.model import get_module, set_module
from train.octo.model import Actor


def build_large_model(device, max_sparsity):
    last_error = None
    for attempt in range(5):
        large_model = Actor(42, 4, 1, False).to(device)
        set_module(large_model, 'rgb_encoder.fc.0', svd_decompose_linear(get_module(large_model, 'rgb_encoder.fc.0')))
        set_module(large_model, 'depth_encoder.fc.0', svd_decompose_linear(get_module(large_model, 'depth_encoder.fc.0')))

        sample = {
            'rgb': torch.rand((2, 3, 128, 128), device=device),
            'depth': torch.rand((2, 1, 128, 128), device=device),
            'state': torch.rand((2, 42), device=device),
        }
        try:
            add_FBS_into_cnn(
                large_model,
                [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
                ['decoder.0', 'rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
                sample,
                max_sparsity,
                8,
                lambda model, batch: model(batch),
            )
            return large_model
        except AssertionError as error:
            last_error = error
            print(f'build_large_model retry {attempt + 1}/5 due to add_FBS_into_cnn verification diff={error}')

    raise last_error


def compare_state_dicts(lhs, rhs):
    lhs_keys = list(lhs.keys())
    rhs_keys = list(rhs.keys())
    if lhs_keys != rhs_keys:
        print('FAIL: state_dict keys are different')
        missing_in_rhs = sorted(set(lhs_keys) - set(rhs_keys))
        missing_in_lhs = sorted(set(rhs_keys) - set(lhs_keys))
        if missing_in_rhs:
            print(f'  missing in regenerated model: {missing_in_rhs[:10]}')
        if missing_in_lhs:
            print(f'  missing in original model: {missing_in_lhs[:10]}')
        return False

    all_equal = True
    for key in lhs_keys:
        left_tensor = lhs[key]
        right_tensor = rhs[key]
        if left_tensor.shape != right_tensor.shape:
            print(f'FAIL: shape mismatch at {key}: {tuple(left_tensor.shape)} vs {tuple(right_tensor.shape)}')
            all_equal = False
            continue
        if not torch.equal(left_tensor, right_tensor):
            max_abs_diff = (left_tensor - right_tensor).abs().max().item()
            print(f'FAIL: value mismatch at {key}, max_abs_diff={max_abs_diff}')
            all_equal = False

    return all_equal


def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def clone_module_structure(model):
    return [
        (name, type(module).__name__)
        for name, module in model.named_modules()
    ]


def clone_optimizer_state(optimizer, model):
    cloned_state = {}
    for name, parameter in model.named_parameters():
        parameter_state = optimizer.state.get(parameter)
        if not parameter_state:
            continue
        cloned_state[name] = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
            for key, value in parameter_state.items()
        }
    return cloned_state


def compare_optimizer_states(lhs, rhs):
    lhs_keys = list(lhs.keys())
    rhs_keys = list(rhs.keys())
    if lhs_keys != rhs_keys:
        print('FAIL: optimizer state parameter keys are different')
        missing_in_rhs = sorted(set(lhs_keys) - set(rhs_keys))
        missing_in_lhs = sorted(set(rhs_keys) - set(lhs_keys))
        if missing_in_rhs:
            print(f'  missing in regenerated optimizer state: {missing_in_rhs[:10]}')
        if missing_in_lhs:
            print(f'  missing in original optimizer state: {missing_in_lhs[:10]}')
        return False

    all_equal = True
    for parameter_name in lhs_keys:
        lhs_state = lhs[parameter_name]
        rhs_state = rhs[parameter_name]
        lhs_state_keys = list(lhs_state.keys())
        rhs_state_keys = list(rhs_state.keys())
        if lhs_state_keys != rhs_state_keys:
            print(f'FAIL: optimizer state keys mismatch at {parameter_name}')
            all_equal = False
            continue

        for state_key in lhs_state_keys:
            left_value = lhs_state[state_key]
            right_value = rhs_state[state_key]
            if torch.is_tensor(left_value) and torch.is_tensor(right_value):
                if left_value.shape != right_value.shape:
                    print(
                        f'FAIL: optimizer tensor shape mismatch at {parameter_name}.{state_key}: '
                        f'{tuple(left_value.shape)} vs {tuple(right_value.shape)}'
                    )
                    all_equal = False
                    continue
                if not torch.equal(left_value, right_value):
                    max_abs_diff = (left_value - right_value).abs().max().item()
                    print(
                        f'FAIL: optimizer tensor mismatch at {parameter_name}.{state_key}, '
                        f'max_abs_diff={max_abs_diff}'
                    )
                    all_equal = False
                continue

            if left_value != right_value:
                print(
                    f'FAIL: optimizer scalar mismatch at {parameter_name}.{state_key}: '
                    f'{left_value} vs {right_value}'
                )
                all_equal = False

    return all_equal


def seed_optimizer_state(optimizer, model):
    for index, (_, parameter) in enumerate(model.named_parameters()):
        parameter_state = optimizer.state[parameter]
        parameter_state['step'] = torch.tensor(float(index + 1), device=parameter.device)
        parameter_state['exp_avg'] = (
            torch.arange(parameter.numel(), dtype=parameter.dtype, device=parameter.device)
            .reshape(parameter.shape)
            .add_((index + 1) * 1000.0)
        )
        parameter_state['exp_avg_sq'] = (
            torch.arange(parameter.numel(), dtype=parameter.dtype, device=parameter.device)
            .reshape(parameter.shape)
            .add_((index + 1) * 2000.0)
        )


def remap_rows_for_expected_state(original_tensor, previous_positions, new_positions):
    expected = torch.zeros_like(original_tensor)
    for previous_position, new_position in zip(previous_positions, new_positions):
        expected[new_position].copy_(original_tensor[previous_position])
    return expected


def remap_conv_columns_for_expected_state(original_tensor, previous_positions, new_positions):
    expected = torch.zeros_like(original_tensor)
    for previous_position, new_position in zip(previous_positions, new_positions):
        expected[:, new_position, ...].copy_(original_tensor[:, previous_position, ...])
    return expected


def remap_linear_columns_for_expected_state(original_tensor, previous_positions, new_positions):
    expected = torch.zeros_like(original_tensor)
    for previous_position, new_position in zip(previous_positions, new_positions):
        expected[:, new_position].copy_(original_tensor[:, previous_position])
    return expected


def remap_flattened_linear_blocks_for_expected_state(
    original_tensor,
    previous_positions,
    new_positions,
    previous_selected_count,
    new_selected_count,
):
    expected = torch.zeros_like(original_tensor)
    previous_channel_block = original_tensor.size(1) // previous_selected_count
    new_channel_block = expected.size(1) // new_selected_count
    for previous_position, new_position in zip(previous_positions, new_positions):
        previous_slice = slice(
            previous_position * previous_channel_block,
            (previous_position + 1) * previous_channel_block,
        )
        new_slice = slice(
            new_position * new_channel_block,
            (new_position + 1) * new_channel_block,
        )
        expected[:, new_slice].copy_(original_tensor[:, previous_slice])
    return expected


def build_shifted_selected_indices(previous_indices, total_count):
    if not isinstance(previous_indices, torch.Tensor):
        previous_indices = torch.as_tensor(previous_indices, dtype=torch.long)
    previous_indices = previous_indices.detach().cpu().to(dtype=torch.long)
    previous_list = [int(index) for index in previous_indices.tolist()]
    previous_set = set(previous_list)

    for remove_position in range(len(previous_list)):
        retained_indices = previous_list[:remove_position] + previous_list[remove_position + 1:]
        for candidate_index in range(total_count):
            if candidate_index in previous_set:
                continue
            new_indices = sorted(retained_indices + [candidate_index])
            layer_change = get_small_cnn_channel_change_info(
                {'selected_indices': {'layer': torch.as_tensor(new_indices, dtype=torch.long)}},
                {'selected_indices': {'layer': previous_indices}},
            )['layer']
            if layer_change['retained_new_positions'] != layer_change['retained_previous_positions']:
                return torch.as_tensor(new_indices, dtype=torch.long)

    raise AssertionError(
        f'cannot build shifted selected indices from {previous_list} with total_count={total_count}'
    )


def generate_small_model(large_model, sample, max_sparsity):
    return generate_small_cnn_with_verify(
        large_model,
        max_sparsity,
        sample,
        lambda model, batch: model(batch),
        return_pruning_info=True,
    )


def find_next_param_module(model, layer_name):
    found_current = False
    for candidate_name, candidate_module in model.named_modules():
        if candidate_name == layer_name:
            found_current = True
            continue
        if not found_current:
            continue
        if candidate_name.startswith(layer_name + '.'):
            continue
        if isinstance(candidate_module, (torch.nn.Conv2d, torch.nn.Linear)):
            return candidate_name, candidate_module
    return None, None


def find_feature_boosting_module(layer_module):
    if not isinstance(layer_module, torch.nn.Sequential):
        return None
    for sub_module in layer_module:
        if isinstance(sub_module, FeatureBoosting):
            return sub_module
    return None


def perturb_trainable_parameters(model):
    updated_parameter_names = []
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if not parameter.requires_grad:
                continue
            parameter.add_((index + 1) * 1e-5)
            updated_parameter_names.append(name)
    return updated_parameter_names


def perturb_all_floating_tensors(model):
    touched_tensor_names = []
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if not torch.is_floating_point(parameter):
                continue
            parameter.add_((index + 1) * 1e-5)
            touched_tensor_names.append(name)

        for index, (name, buffer) in enumerate(model.named_buffers()):
            if not torch.is_floating_point(buffer):
                continue
            buffer.add_((index + 1) * 1e-6)
            touched_tensor_names.append(name)

    return touched_tensor_names


def run_identity_feedback_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 1] feedback without updating small model')
    large_model = deepcopy(reference_large_model)

    print('Step 1: generate original small model from large model')
    original_small_model, pruning_info = generate_small_model(large_model, sample, max_sparsity)
    original_state_dict = clone_state_dict(original_small_model)

    print('Step 2: feedback original small model into large model without any update')
    small_cnn_feedback(large_model, original_small_model, pruning_info, alpha=1.0)

    print('Step 3: regenerate small model from same large model and same sample')
    regenerated_small_model, _ = generate_small_model(large_model, sample, max_sparsity)
    regenerated_state_dict = clone_state_dict(regenerated_small_model)

    print('Step 4: compare state_dict of original and regenerated small models')
    if compare_state_dicts(original_state_dict, regenerated_state_dict):
        print('PASS: regenerated small model is exactly identical to the original small model')
        return True

    print('FAIL: regenerated small model is not identical to the original small model')
    return False


def run_downstream_input_column_feedback_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 2] direct feedback mapping check')
    large_model = deepcopy(reference_large_model)

    print('Step 1: generate original small model from large model')
    edited_small_model, pruning_info = generate_small_model(large_model, sample, max_sparsity)

    print('Step 2: manually modify small-model weights on both sides of mapped connections')
    conv_current = get_module(edited_small_model, 'rgb_encoder.cnn.0.0')
    conv_next = get_module(edited_small_model, 'rgb_encoder.cnn.3')
    linear_current = get_module(edited_small_model, 'decoder.0.0')
    linear_next = get_module(edited_small_model, 'decoder.2')
    if conv_current is None or not isinstance(conv_current, torch.nn.Conv2d):
        print('FAIL: cannot find target module rgb_encoder.cnn.0.0 in small model')
        return False
    if conv_next is None or not isinstance(conv_next, torch.nn.Conv2d):
        print('FAIL: cannot find target module rgb_encoder.cnn.3 in small model')
        return False
    if linear_current is None or not isinstance(linear_current, torch.nn.Linear):
        print('FAIL: cannot find target module decoder.0.0 in small model')
        return False
    if linear_next is None or not isinstance(linear_next, torch.nn.Linear):
        print('FAIL: cannot find target module decoder.2 in small model')
        return False
    with torch.no_grad():
        conv_current.weight[0].add_(0.111111)
        if conv_current.bias is not None:
            conv_current.bias[0].add_(0.222222)
        conv_next.weight[:, 0, :, :].add_(0.123456)
        if conv_next.weight.size(1) > 1:
            conv_next.weight[:, 1, :, :].sub_(0.234567)

        linear_current.weight[0].add_(0.345678)
        if linear_current.bias is not None:
            linear_current.bias[0].sub_(0.456789)
        linear_next.weight[:, 0].add_(0.567891)
        if linear_next.weight.size(1) > 1:
            linear_next.weight[:, 1].sub_(0.678912)

    print('Step 3: feedback edited small model into large model')
    small_cnn_feedback(large_model, edited_small_model, pruning_info, alpha=1.0)

    print('Step 4: directly compare mapped weights in large model and small model')
    passed = True

    conv_indices = pruning_info['selected_indices']['rgb_encoder.cnn.0'].tolist()
    large_conv_fbs = get_module(large_model, 'rgb_encoder.cnn.0')
    large_conv_next = get_module(large_model, 'rgb_encoder.cnn.3')
    for small_idx, large_idx in enumerate(conv_indices):
        if not torch.equal(large_conv_fbs.raw_conv2d.weight.data[large_idx], conv_current.weight.data[small_idx]):
            print(f'FAIL: conv output-row mismatch at small={small_idx}, large={large_idx}')
            passed = False
        if conv_current.bias is not None and not torch.equal(large_conv_fbs.raw_conv2d.bias.data[large_idx], conv_current.bias.data[small_idx]):
            print(f'FAIL: conv bias mismatch at small={small_idx}, large={large_idx}')
            passed = False
        if isinstance(large_conv_fbs.bn, torch.nn.BatchNorm2d):
            small_conv_bn = get_module(edited_small_model, 'rgb_encoder.cnn.0.1')
            if small_conv_bn is None or not isinstance(small_conv_bn, torch.nn.BatchNorm2d):
                print('FAIL: expected BatchNorm2d at rgb_encoder.cnn.0.1 in small model')
                passed = False
            else:
                if not torch.equal(large_conv_fbs.bn.weight.data[large_idx], small_conv_bn.weight.data[small_idx]):
                    print(f'FAIL: conv bn.weight mismatch at small={small_idx}, large={large_idx}')
                    passed = False
                if not torch.equal(large_conv_fbs.bn.bias.data[large_idx], small_conv_bn.bias.data[small_idx]):
                    print(f'FAIL: conv bn.bias mismatch at small={small_idx}, large={large_idx}')
                    passed = False
        if not torch.equal(large_conv_next.weight.data[:, large_idx, :, :], conv_next.weight.data[:, small_idx, :, :]):
            print(f'FAIL: next conv input-column mismatch at small={small_idx}, large={large_idx}')
            passed = False

    linear_indices = pruning_info['selected_indices']['decoder.0'].tolist()
    large_linear_fbs = get_module(large_model, 'decoder.0')
    large_linear_next = get_module(large_model, 'decoder.2')
    for small_idx, large_idx in enumerate(linear_indices):
        if not torch.equal(large_linear_fbs.raw_linear.weight.data[large_idx], linear_current.weight.data[small_idx]):
            print(f'FAIL: linear output-row mismatch at small={small_idx}, large={large_idx}')
            passed = False
        if linear_current.bias is not None and not torch.equal(large_linear_fbs.raw_linear.bias.data[large_idx], linear_current.bias.data[small_idx]):
            print(f'FAIL: linear bias mismatch at small={small_idx}, large={large_idx}')
            passed = False
        if not torch.equal(large_linear_next.weight.data[:, large_idx], linear_next.weight.data[:, small_idx]):
            print(f'FAIL: next linear input-column mismatch at small={small_idx}, large={large_idx}')
            passed = False

    if passed:
        print('PASS: feedback copies both current-layer outputs and downstream input columns correctly')
        return True

    print('FAIL: feedback mapping is incorrect')
    return False


def run_full_parameter_feedback_coverage_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 3] full-parameter feedback coverage check')
    large_model = deepcopy(reference_large_model)

    print('Step 1: generate original small model from large model')
    edited_small_model, pruning_info = generate_small_model(large_model, sample, max_sparsity)

    print('Step 2: deterministically update every trainable parameter in the small model')
    updated_parameter_names = perturb_trainable_parameters(edited_small_model)
    if not updated_parameter_names:
        print('FAIL: no trainable parameters were updated in the small model')
        return False

    print('Step 3: feedback edited small model into large model')
    small_cnn_feedback(large_model, edited_small_model, pruning_info, alpha=1.0)

    print('Step 4: verify every updated small-model parameter is absorbed by the large model')
    passed = True
    covered_small_parameter_names = set()
    large_named_parameters = dict(large_model.named_parameters())

    for layer_name, unpruned_filters_index in pruning_info['selected_indices'].items():
        fbs_layer = get_module(large_model, layer_name)
        sub_model_layer = get_module(edited_small_model, layer_name)
        next_large_layer_name, next_large_layer = find_next_param_module(large_model, layer_name)
        next_small_layer_name, next_small_layer = find_next_param_module(edited_small_model, layer_name)
        is_current_layer_conv = hasattr(fbs_layer, 'raw_conv2d') and isinstance(fbs_layer.raw_conv2d, torch.nn.Conv2d)

        for small_idx, large_idx in enumerate(unpruned_filters_index.tolist()):
            if is_current_layer_conv:
                if not torch.equal(fbs_layer.raw_conv2d.weight.data[large_idx], sub_model_layer[0].weight.data[small_idx]):
                    print(f'FAIL: selected conv weight mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                    passed = False
                covered_small_parameter_names.add(f'{layer_name}.0.weight')

                if fbs_layer.raw_conv2d.bias is not None:
                    if not torch.equal(fbs_layer.raw_conv2d.bias.data[large_idx], sub_model_layer[0].bias.data[small_idx]):
                        print(f'FAIL: selected conv bias mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
                    covered_small_parameter_names.add(f'{layer_name}.0.bias')

                if hasattr(fbs_layer, 'bn') and isinstance(fbs_layer.bn, torch.nn.BatchNorm2d):
                    if not torch.equal(fbs_layer.bn.weight.data[large_idx], sub_model_layer[1].weight.data[small_idx]):
                        print(f'FAIL: selected conv bn.weight mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
                    if not torch.equal(fbs_layer.bn.bias.data[large_idx], sub_model_layer[1].bias.data[small_idx]):
                        print(f'FAIL: selected conv bn.bias mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
                    covered_small_parameter_names.add(f'{layer_name}.1.weight')
                    covered_small_parameter_names.add(f'{layer_name}.1.bias')

            if hasattr(fbs_layer, 'raw_linear') and isinstance(fbs_layer.raw_linear, torch.nn.Linear):
                if not torch.equal(fbs_layer.raw_linear.weight.data[large_idx], sub_model_layer[0].weight.data[small_idx]):
                    print(f'FAIL: selected linear weight mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                    passed = False
                covered_small_parameter_names.add(f'{layer_name}.0.weight')

                if fbs_layer.raw_linear.bias is not None:
                    if not torch.equal(fbs_layer.raw_linear.bias.data[large_idx], sub_model_layer[0].bias.data[small_idx]):
                        print(f'FAIL: selected linear bias mismatch at {layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
                    covered_small_parameter_names.add(f'{layer_name}.0.bias')

        if next_large_layer_name is None or next_small_layer_name is None:
            print(f'FAIL: next-layer mapping mismatch for {layer_name}: large={next_large_layer_name}, small={next_small_layer_name}')
            passed = False
            continue

        if isinstance(next_large_layer, torch.nn.Conv2d) and isinstance(next_small_layer, torch.nn.Conv2d):
            for small_idx, large_idx in enumerate(unpruned_filters_index.tolist()):
                if not torch.equal(next_large_layer.weight.data[:, large_idx, :, :], next_small_layer.weight.data[:, small_idx, :, :]):
                    print(f'FAIL: next conv input-column mismatch at {next_small_layer_name}, small={small_idx}, large={large_idx}')
                    passed = False
            covered_small_parameter_names.add(f'{next_small_layer_name}.weight')
        elif isinstance(next_large_layer, torch.nn.Linear) and isinstance(next_small_layer, torch.nn.Linear):
            if is_current_layer_conv:
                channel_block = next_small_layer.weight.size(1) // len(unpruned_filters_index)
                for small_idx, large_idx in enumerate(unpruned_filters_index.tolist()):
                    small_slice = slice(small_idx * channel_block, (small_idx + 1) * channel_block)
                    large_slice = slice(large_idx * channel_block, (large_idx + 1) * channel_block)
                    if not torch.equal(next_large_layer.weight.data[:, large_slice], next_small_layer.weight.data[:, small_slice]):
                        print(f'FAIL: flattened next linear input-column mismatch at {next_small_layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
            else:
                for small_idx, large_idx in enumerate(unpruned_filters_index.tolist()):
                    if not torch.equal(next_large_layer.weight.data[:, large_idx], next_small_layer.weight.data[:, small_idx]):
                        print(f'FAIL: next linear input-column mismatch at {next_small_layer_name}, small={small_idx}, large={large_idx}')
                        passed = False
            covered_small_parameter_names.add(f'{next_small_layer_name}.weight')
        else:
            print(
                f'FAIL: unsupported next-layer pair for {layer_name}: '
                f'{type(next_large_layer).__name__} vs {type(next_small_layer).__name__}'
            )
            passed = False

        if next_small_layer.bias is not None:
            if not torch.equal(next_large_layer.bias.data, next_small_layer.bias.data):
                print(f'FAIL: next-layer bias mismatch at {next_small_layer_name}')
                passed = False
            covered_small_parameter_names.add(f'{next_small_layer_name}.bias')

    for name in updated_parameter_names:
        if name in covered_small_parameter_names:
            continue

        large_parameter = large_named_parameters.get(name)
        small_parameter = dict(edited_small_model.named_parameters())[name]
        if large_parameter is None:
            print(f'FAIL: uncovered small parameter has no direct large counterpart: {name}')
            passed = False
            continue
        if large_parameter.shape != small_parameter.shape:
            print(
                f'FAIL: uncovered parameter shape mismatch at {name}: '
                f'{tuple(large_parameter.shape)} vs {tuple(small_parameter.shape)}'
            )
            passed = False
            continue
        if not torch.equal(large_parameter.data, small_parameter.data):
            print(f'FAIL: direct parameter mismatch at {name}')
            passed = False
            continue

        covered_small_parameter_names.add(name)

    uncovered_parameter_names = sorted(set(updated_parameter_names) - covered_small_parameter_names)
    if uncovered_parameter_names:
        print(f'FAIL: some updated small-model parameters were not verified: {uncovered_parameter_names}')
        passed = False

    if passed:
        print('PASS: every updated small-model parameter is absorbed by a corresponding large-model layer')
        return True

    print('FAIL: at least one updated small-model parameter was not absorbed by the large model')
    return False


def run_retained_channel_inheritance_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 4] retained channels inherit directly from previous small model')
    large_model = deepcopy(reference_large_model)

    print('Step 1: generate previous small model from large model')
    previous_small_model, previous_pruning_info = generate_small_model(large_model, sample, max_sparsity)

    print('Step 2: manually modify retained channels in the previous small model only')
    conv_layer = get_module(previous_small_model, 'rgb_encoder.cnn.0')
    conv_next_layer = get_module(previous_small_model, 'rgb_encoder.cnn.3')
    linear_layer = get_module(previous_small_model, 'decoder.0')
    linear_next_layer = get_module(previous_small_model, 'decoder.2')
    conv_feature_boosting = find_feature_boosting_module(conv_layer)
    linear_feature_boosting = find_feature_boosting_module(linear_layer)
    if (
        conv_layer is None
        or conv_next_layer is None
        or linear_layer is None
        or linear_next_layer is None
        or conv_feature_boosting is None
        or linear_feature_boosting is None
    ):
        print('FAIL: cannot find target layers in previous small model')
        return False

    with torch.no_grad():
        conv_layer[0].weight[0].add_(1.2345)
        if conv_layer[0].bias is not None:
            conv_layer[0].bias[0].sub_(0.3456)
        if isinstance(conv_layer[1], torch.nn.BatchNorm2d):
            conv_layer[1].weight[0].add_(0.2222)
            conv_layer[1].running_mean[0].sub_(0.1111)
        conv_feature_boosting.w[:, 0, ...].add_(0.3333)
        conv_next_layer.weight[:, 0, :, :].add_(0.4567)

        linear_layer[0].weight[0].sub_(0.5678)
        if linear_layer[0].bias is not None:
            linear_layer[0].bias[0].add_(0.6789)
        linear_feature_boosting.w[:, 0].sub_(0.4321)
        linear_next_layer.weight[:, 0].add_(0.7891)

    print('Step 3: regenerate a new small model from the unchanged large model')
    regenerated_small_model, regenerated_pruning_info = generate_small_cnn_with_verify(
        large_model,
        max_sparsity,
        sample,
        lambda model, batch: model(batch),
        return_pruning_info=True,
        previous_pruning_info=previous_pruning_info,
        regeneration_increment_ratio=0.0,
    )

    print('Step 4: explicitly graft retained channels from the previous small model')
    inherit_small_cnn_retained_channels(
        regenerated_small_model,
        previous_small_model,
        regenerated_pruning_info,
        previous_pruning_info,
    )

    print('Step 5: verify retained channels match the previous small model exactly')
    passed = True

    regenerated_conv_layer = get_module(regenerated_small_model, 'rgb_encoder.cnn.0')
    regenerated_conv_next_layer = get_module(regenerated_small_model, 'rgb_encoder.cnn.3')
    regenerated_linear_layer = get_module(regenerated_small_model, 'decoder.0')
    regenerated_linear_next_layer = get_module(regenerated_small_model, 'decoder.2')
    regenerated_conv_feature_boosting = find_feature_boosting_module(regenerated_conv_layer)
    regenerated_linear_feature_boosting = find_feature_boosting_module(regenerated_linear_layer)

    if not torch.equal(regenerated_conv_layer[0].weight[0], conv_layer[0].weight[0]):
        print('FAIL: retained conv channel weight did not inherit from previous small model')
        passed = False
    if conv_layer[0].bias is not None and not torch.equal(regenerated_conv_layer[0].bias[0], conv_layer[0].bias[0]):
        print('FAIL: retained conv channel bias did not inherit from previous small model')
        passed = False
    if isinstance(conv_layer[1], torch.nn.BatchNorm2d):
        if not torch.equal(regenerated_conv_layer[1].weight[0], conv_layer[1].weight[0]):
            print('FAIL: retained conv BN weight did not inherit from previous small model')
            passed = False
        if not torch.equal(regenerated_conv_layer[1].running_mean[0], conv_layer[1].running_mean[0]):
            print('FAIL: retained conv BN running_mean did not inherit from previous small model')
            passed = False
    if not torch.equal(regenerated_conv_feature_boosting.w[:, 0, ...], conv_feature_boosting.w[:, 0, ...]):
        print('FAIL: retained conv FeatureBoosting coefficient did not inherit from previous small model')
        passed = False
    if not torch.equal(regenerated_conv_next_layer.weight[:, 0, :, :], conv_next_layer.weight[:, 0, :, :]):
        print('FAIL: downstream conv input column did not inherit from previous small model')
        passed = False

    if not torch.equal(regenerated_linear_layer[0].weight[0], linear_layer[0].weight[0]):
        print('FAIL: retained linear channel weight did not inherit from previous small model')
        passed = False
    if linear_layer[0].bias is not None and not torch.equal(regenerated_linear_layer[0].bias[0], linear_layer[0].bias[0]):
        print('FAIL: retained linear channel bias did not inherit from previous small model')
        passed = False
    if not torch.equal(regenerated_linear_feature_boosting.w[:, 0], linear_feature_boosting.w[:, 0]):
        print('FAIL: retained linear FeatureBoosting coefficient did not inherit from previous small model')
        passed = False
    if not torch.equal(regenerated_linear_next_layer.weight[:, 0], linear_next_layer.weight[:, 0]):
        print('FAIL: downstream linear input column did not inherit from previous small model')
        passed = False

    if passed:
        print('PASS: retained channels are copied directly from the previous small model')
        return True

    print('FAIL: retained channels were not copied directly from the previous small model')
    return False


def run_optimizer_state_remap_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 5] optimizer state follows retained channels')
    small_model, previous_pruning_info = generate_small_model(reference_large_model, sample, max_sparsity)
    new_pruning_info = deepcopy(previous_pruning_info)

    for layer_name in ['rgb_encoder.cnn.0', 'decoder.0']:
        large_layer = get_module(reference_large_model, layer_name)
        if hasattr(large_layer, 'raw_conv2d'):
            total_count = large_layer.raw_conv2d.out_channels
        elif hasattr(large_layer, 'raw_linear'):
            total_count = large_layer.raw_linear.out_features
        else:
            print(f'FAIL: unsupported large layer type for {layer_name}')
            return False

        previous_indices = previous_pruning_info['selected_indices'][layer_name]
        if len(previous_indices) < 2:
            print(f'FAIL: not enough selected channels to test remap for {layer_name}')
            return False
        new_pruning_info['selected_indices'][layer_name] = build_shifted_selected_indices(
            previous_indices,
            total_count,
        )

    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
    seed_optimizer_state(optimizer, small_model)
    original_optimizer_state = clone_optimizer_state(optimizer, small_model)

    remap_small_cnn_optimizer_state(
        optimizer,
        small_model,
        new_pruning_info,
        previous_pruning_info,
    )
    remapped_optimizer_state = clone_optimizer_state(optimizer, small_model)
    channel_change_info = get_small_cnn_channel_change_info(
        new_pruning_info,
        previous_pruning_info,
    )

    passed = True
    touched_parameter_names = set()

    def _check_state_tensor(parameter_name, expected_tensor, state_key):
        nonlocal passed
        actual_tensor = remapped_optimizer_state[parameter_name][state_key]
        if not torch.equal(actual_tensor, expected_tensor):
            max_abs_diff = (actual_tensor - expected_tensor).abs().max().item()
            print(
                f'FAIL: optimizer state mismatch at {parameter_name}.{state_key}, '
                f'max_abs_diff={max_abs_diff}'
            )
            passed = False
        touched_parameter_names.add(parameter_name)

    def _check_step_unchanged(parameter_name):
        nonlocal passed
        original_step = original_optimizer_state[parameter_name]['step']
        remapped_step = remapped_optimizer_state[parameter_name]['step']
        if not torch.equal(remapped_step, original_step):
            print(f'FAIL: optimizer step should stay unchanged at {parameter_name}')
            passed = False

    for layer_name in ['rgb_encoder.cnn.0', 'decoder.0']:
        layer_change = channel_change_info[layer_name]
        previous_positions = layer_change['retained_previous_positions']
        new_positions = layer_change['retained_new_positions']
        previous_selected_count = layer_change['previous_selected_count']
        new_selected_count = layer_change['new_selected_count']

        layer = get_module(small_model, layer_name)
        next_layer_name, next_layer = find_next_param_module(small_model, layer_name)
        if layer is None or next_layer_name is None or next_layer is None:
            print(f'FAIL: cannot find layer mapping for {layer_name}')
            return False

        current_weight_name = f'{layer_name}.0.weight'
        for state_key in ['exp_avg', 'exp_avg_sq']:
            _check_state_tensor(
                current_weight_name,
                remap_rows_for_expected_state(
                    original_optimizer_state[current_weight_name][state_key],
                    previous_positions,
                    new_positions,
                ),
                state_key,
            )
        _check_step_unchanged(current_weight_name)

        current_bias_name = f'{layer_name}.0.bias'
        if current_bias_name in original_optimizer_state:
            for state_key in ['exp_avg', 'exp_avg_sq']:
                _check_state_tensor(
                    current_bias_name,
                    remap_rows_for_expected_state(
                        original_optimizer_state[current_bias_name][state_key],
                        previous_positions,
                        new_positions,
                    ),
                    state_key,
                )
            _check_step_unchanged(current_bias_name)

        if len(layer) > 1 and isinstance(layer[1], torch.nn.BatchNorm2d):
            for suffix in ['weight', 'bias']:
                bn_parameter_name = f'{layer_name}.1.{suffix}'
                for state_key in ['exp_avg', 'exp_avg_sq']:
                    _check_state_tensor(
                        bn_parameter_name,
                        remap_rows_for_expected_state(
                            original_optimizer_state[bn_parameter_name][state_key],
                            previous_positions,
                            new_positions,
                        ),
                        state_key,
                    )
                _check_step_unchanged(bn_parameter_name)

        next_weight_name = f'{next_layer_name}.weight'
        if isinstance(next_layer, torch.nn.Conv2d):
            expected_fn = remap_conv_columns_for_expected_state
            expected_kwargs = {}
        elif isinstance(next_layer, torch.nn.Linear):
            if isinstance(layer[0], torch.nn.Conv2d):
                expected_fn = remap_flattened_linear_blocks_for_expected_state
                expected_kwargs = {
                    'previous_selected_count': previous_selected_count,
                    'new_selected_count': new_selected_count,
                }
            else:
                expected_fn = remap_linear_columns_for_expected_state
                expected_kwargs = {}
        else:
            print(f'FAIL: unsupported next layer type {type(next_layer).__name__} for {layer_name}')
            return False

        for state_key in ['exp_avg', 'exp_avg_sq']:
            _check_state_tensor(
                next_weight_name,
                expected_fn(
                    original_optimizer_state[next_weight_name][state_key],
                    previous_positions,
                    new_positions,
                    **expected_kwargs,
                ),
                state_key,
            )
        _check_step_unchanged(next_weight_name)

    untouched_parameter_name = next(
        (name for name in original_optimizer_state.keys() if name not in touched_parameter_names),
        None,
    )
    if untouched_parameter_name is not None:
        for state_key in ['exp_avg', 'exp_avg_sq', 'step']:
            if not torch.equal(
                remapped_optimizer_state[untouched_parameter_name][state_key],
                original_optimizer_state[untouched_parameter_name][state_key],
            ):
                print(f'FAIL: untouched parameter state changed at {untouched_parameter_name}.{state_key}')
                passed = False

    if passed:
        print('PASS: retained channels keep optimizer state and replaced channels are reset')
        return True

    print('FAIL: optimizer state remap is incorrect')
    return False


def run_feature_boosting_retained_channel_inheritance_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 6] retained FeatureBoosting coefficients follow retained channels')
    previous_small_model, previous_pruning_info = generate_small_model(reference_large_model, sample, max_sparsity)
    regenerated_small_model, new_pruning_info = generate_small_model(reference_large_model, sample, max_sparsity)
    new_pruning_info = deepcopy(new_pruning_info)

    for layer_name in ['rgb_encoder.cnn.0', 'decoder.0']:
        large_layer = get_module(reference_large_model, layer_name)
        if hasattr(large_layer, 'raw_conv2d'):
            total_count = large_layer.raw_conv2d.out_channels
        elif hasattr(large_layer, 'raw_linear'):
            total_count = large_layer.raw_linear.out_features
        else:
            print(f'FAIL: unsupported large layer type for {layer_name}')
            return False

        previous_indices = previous_pruning_info['selected_indices'][layer_name]
        if len(previous_indices) < 2:
            print(f'FAIL: not enough selected channels to test retained FeatureBoosting copy for {layer_name}')
            return False
        new_pruning_info['selected_indices'][layer_name] = build_shifted_selected_indices(
            previous_indices,
            total_count,
        )

    channel_change_info = get_small_cnn_channel_change_info(
        new_pruning_info,
        previous_pruning_info,
    )

    tracked_layers = {}
    with torch.no_grad():
        for layer_name in ['rgb_encoder.cnn.0', 'decoder.0']:
            previous_layer = get_module(previous_small_model, layer_name)
            regenerated_layer = get_module(regenerated_small_model, layer_name)
            previous_feature_boosting = find_feature_boosting_module(previous_layer)
            regenerated_feature_boosting = find_feature_boosting_module(regenerated_layer)
            if previous_feature_boosting is None or regenerated_feature_boosting is None:
                print(f'FAIL: cannot find FeatureBoosting module for {layer_name}')
                return False

            previous_feature_boosting.w.copy_(
                torch.arange(
                    previous_feature_boosting.w.numel(),
                    dtype=previous_feature_boosting.w.dtype,
                    device=previous_feature_boosting.w.device,
                ).reshape_as(previous_feature_boosting.w).add_(123.0 if 'rgb_encoder' in layer_name else 456.0)
            )
            tracked_layers[layer_name] = {
                'previous_feature_boosting_before': previous_feature_boosting.w.detach().clone(),
                'regenerated_feature_boosting_before': regenerated_feature_boosting.w.detach().clone(),
            }

    inherit_small_cnn_retained_channels(
        regenerated_small_model,
        previous_small_model,
        new_pruning_info,
        previous_pruning_info,
    )

    passed = True
    for layer_name in ['rgb_encoder.cnn.0', 'decoder.0']:
        regenerated_layer = get_module(regenerated_small_model, layer_name)
        regenerated_feature_boosting = find_feature_boosting_module(regenerated_layer)
        layer_change = channel_change_info[layer_name]
        previous_positions = layer_change['retained_previous_positions']
        new_positions = layer_change['retained_new_positions']
        previous_before = tracked_layers[layer_name]['previous_feature_boosting_before']
        regenerated_before = tracked_layers[layer_name]['regenerated_feature_boosting_before']

        for previous_position, new_position in zip(previous_positions, new_positions):
            if not torch.equal(
                regenerated_feature_boosting.w[:, new_position, ...],
                previous_before[:, previous_position, ...],
            ):
                print(
                    f'FAIL: retained FeatureBoosting coefficient mismatch at {layer_name}, '
                    f'previous_position={previous_position}, new_position={new_position}'
                )
                passed = False

        replaced_positions = layer_change['replaced_new_positions']
        for replaced_position in replaced_positions:
            if not torch.equal(
                regenerated_feature_boosting.w[:, replaced_position, ...],
                regenerated_before[:, replaced_position, ...],
            ):
                print(
                    f'FAIL: replaced FeatureBoosting coefficient should keep regenerated value at '
                    f'{layer_name}, new_position={replaced_position}'
                )
                passed = False

    if passed:
        print('PASS: retained FeatureBoosting coefficients are copied by retained-channel mapping')
        return True

    print('FAIL: retained FeatureBoosting coefficient inheritance is incorrect')
    return False


def run_zero_increment_regeneration_identity_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 7] zero-increment regeneration is identical for arbitrary alpha')
    alpha_values = [0.0, 0.37, 1.0]

    for alpha in alpha_values:
        print(f'Subtest alpha={alpha}')
        large_model = deepcopy(reference_large_model)

        previous_small_model, previous_pruning_info = generate_small_model(large_model, sample, max_sparsity)
        touched_tensor_names = perturb_all_floating_tensors(previous_small_model)
        if not touched_tensor_names:
            print(f'FAIL: no floating tensors were updated for alpha={alpha}')
            return False

        expected_state_dict = clone_state_dict(previous_small_model)
        expected_module_structure = clone_module_structure(previous_small_model)

        small_cnn_feedback(
            large_model,
            previous_small_model,
            previous_pruning_info,
            alpha=alpha,
        )

        regenerated_small_model, regenerated_pruning_info = generate_small_cnn_with_verify(
            large_model,
            max_sparsity,
            sample,
            lambda model, batch: model(batch),
            return_pruning_info=True,
            previous_pruning_info=previous_pruning_info,
            regeneration_increment_ratio=0.0,
        )
        inherit_small_cnn_retained_channels(
            regenerated_small_model,
            previous_small_model,
            regenerated_pruning_info,
            previous_pruning_info,
        )

        actual_module_structure = clone_module_structure(regenerated_small_model)
        if actual_module_structure != expected_module_structure:
            print(f'FAIL: module structure changed for alpha={alpha}')
            print(f'  expected head: {expected_module_structure[:10]}')
            print(f'  actual head: {actual_module_structure[:10]}')
            return False

        if regenerated_pruning_info['selected_indices'].keys() != previous_pruning_info['selected_indices'].keys():
            print(f'FAIL: selected layer set changed for alpha={alpha}')
            return False

        for layer_name, previous_indices in previous_pruning_info['selected_indices'].items():
            regenerated_indices = regenerated_pruning_info['selected_indices'][layer_name]
            if not torch.equal(
                torch.as_tensor(regenerated_indices).cpu(),
                torch.as_tensor(previous_indices).cpu(),
            ):
                print(f'FAIL: selected indices changed at {layer_name} for alpha={alpha}')
                return False

        actual_state_dict = clone_state_dict(regenerated_small_model)
        if not compare_state_dicts(expected_state_dict, actual_state_dict):
            print(f'FAIL: regenerated small model is not exactly identical for alpha={alpha}')
            return False

    print('PASS: zero-increment regeneration keeps structure and all tensors identical for every tested alpha')
    return True


def run_zero_increment_optimizer_reset_identity_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 8] zero-increment optimizer reset keeps state identical for arbitrary alpha')
    alpha_values = [0.0, 0.37, 1.0]

    for alpha in alpha_values:
        print(f'Subtest alpha={alpha}')
        large_model = deepcopy(reference_large_model)

        small_model, previous_pruning_info = generate_small_model(large_model, sample, max_sparsity)
        touched_tensor_names = perturb_all_floating_tensors(small_model)
        if not touched_tensor_names:
            print(f'FAIL: no floating tensors were updated for alpha={alpha}')
            return False

        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        seed_optimizer_state(optimizer, small_model)
        expected_optimizer_state = clone_optimizer_state(optimizer, small_model)

        small_cnn_feedback(
            large_model,
            small_model,
            previous_pruning_info,
            alpha=alpha,
        )

        regenerated_small_model, regenerated_pruning_info = generate_small_cnn_with_verify(
            large_model,
            max_sparsity,
            sample,
            lambda model, batch: model(batch),
            return_pruning_info=True,
            previous_pruning_info=previous_pruning_info,
            regeneration_increment_ratio=0.0,
        )
        inherit_small_cnn_retained_channels(
            regenerated_small_model,
            small_model,
            regenerated_pruning_info,
            previous_pruning_info,
        )

        small_model.load_state_dict(regenerated_small_model.state_dict(), strict=True)
        optimizer.zero_grad(set_to_none=True)
        remap_small_cnn_optimizer_state(
            optimizer,
            small_model,
            regenerated_pruning_info,
            previous_pruning_info,
        )
        actual_optimizer_state = clone_optimizer_state(optimizer, small_model)

        if not compare_optimizer_states(expected_optimizer_state, actual_optimizer_state):
            print(f'FAIL: optimizer state changed after zero-increment reset for alpha={alpha}')
            return False

    print('PASS: zero-increment optimizer reset keeps optimizer state identical for every tested alpha')
    return True


def run_small_increment_regeneration_stays_close_test(reference_large_model, max_sparsity, sample):
    print('\n[Test 9] alpha=0.1 with increment_ratio=0.01 only changes a few neurons per layer')
    alpha = 0.1
    increment_ratio = 0.5
    large_model = deepcopy(reference_large_model)

    previous_small_model, previous_pruning_info = generate_small_model(large_model, sample, max_sparsity)
    touched_tensor_names = perturb_all_floating_tensors(previous_small_model)
    if not touched_tensor_names:
        print('FAIL: no floating tensors were updated in the previous small model')
        return False

    small_cnn_feedback(
        large_model,
        previous_small_model,
        previous_pruning_info,
        alpha=alpha,
    )

    regenerated_small_model, regenerated_pruning_info = generate_small_cnn_with_verify(
        large_model,
        max_sparsity,
        sample,
        lambda model, batch: model(batch),
        return_pruning_info=True,
        previous_pruning_info=previous_pruning_info,
        regeneration_increment_ratio=increment_ratio,
    )
    channel_change_info = get_small_cnn_channel_change_info(
        regenerated_pruning_info,
        previous_pruning_info,
    )
    if not channel_change_info:
        print('FAIL: no channel change information was produced')
        return False

    inherit_small_cnn_retained_channels(
        regenerated_small_model,
        previous_small_model,
        regenerated_pruning_info,
        previous_pruning_info,
    )

    merge_stats = regenerated_pruning_info.get('merge_stats', {})
    passed = True

    for layer_name, layer_change in channel_change_info.items():
        new_selected_count = layer_change['new_selected_count']
        previous_selected_count = layer_change['previous_selected_count']
        retained_previous_positions = layer_change['retained_previous_positions']
        retained_new_positions = layer_change['retained_new_positions']
        replaced_count = len(layer_change['replaced_large_indices'])
        allowed_replaced_count = int(round(new_selected_count * increment_ratio))

        print(
            f'  layer={layer_name}: selected={new_selected_count}, '
            f'retained={len(retained_new_positions)}, replaced={replaced_count}'
        )

        if new_selected_count != previous_selected_count:
            print(
                f'FAIL: selected neuron count changed at {layer_name}: '
                f'{previous_selected_count} vs {new_selected_count}'
            )
            passed = False

        print(f'{layer_name}, {replaced_count}')

        if replaced_count > allowed_replaced_count:
            print(
                f'FAIL: too many replaced neurons at {layer_name}: '
                f'{replaced_count} > {allowed_replaced_count}'
            )
            passed = False

        layer_merge_stats = merge_stats.get(layer_name)
        if layer_merge_stats is not None and layer_merge_stats.get('replaced_count') != replaced_count:
            print(
                f'FAIL: merge_stats replaced_count mismatch at {layer_name}: '
                f"{layer_merge_stats.get('replaced_count')} vs {replaced_count}"
            )
            passed = False

        new_layer = get_module(regenerated_small_model, layer_name)
        previous_layer = get_module(previous_small_model, layer_name)
        if new_layer is None or previous_layer is None:
            print(f'FAIL: cannot find layer {layer_name} in regenerated or previous small model')
            passed = False
            continue

        if not isinstance(new_layer, torch.nn.Sequential) or not isinstance(previous_layer, torch.nn.Sequential):
            print(f'FAIL: expected sequential layer at {layer_name}')
            passed = False
            continue

        new_next_layer_name, new_next_layer = find_next_param_module(regenerated_small_model, layer_name)
        previous_next_layer_name, previous_next_layer = find_next_param_module(previous_small_model, layer_name)
        if new_next_layer_name != previous_next_layer_name or new_next_layer is None or previous_next_layer is None:
            print(
                f'FAIL: next-layer mapping mismatch at {layer_name}: '
                f'{previous_next_layer_name} vs {new_next_layer_name}'
            )
            passed = False
            continue

        new_feature_boosting = find_feature_boosting_module(new_layer)
        previous_feature_boosting = find_feature_boosting_module(previous_layer)
        if (new_feature_boosting is None) != (previous_feature_boosting is None):
            print(f'FAIL: FeatureBoosting presence mismatch at {layer_name}')
            passed = False
            continue

        for previous_position, new_position in zip(retained_previous_positions, retained_new_positions):
            if isinstance(new_layer[0], torch.nn.Conv2d) and isinstance(previous_layer[0], torch.nn.Conv2d):
                if not torch.equal(new_layer[0].weight[new_position], previous_layer[0].weight[previous_position]):
                    print(
                        f'FAIL: retained conv weight mismatch at {layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False
                if (
                    new_layer[0].bias is not None
                    and previous_layer[0].bias is not None
                    and not torch.equal(new_layer[0].bias[new_position], previous_layer[0].bias[previous_position])
                ):
                    print(
                        f'FAIL: retained conv bias mismatch at {layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False
                if (
                    len(new_layer) > 1
                    and len(previous_layer) > 1
                    and isinstance(new_layer[1], torch.nn.BatchNorm2d)
                    and isinstance(previous_layer[1], torch.nn.BatchNorm2d)
                ):
                    for bn_tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                        if not torch.equal(
                            getattr(new_layer[1], bn_tensor_name)[new_position],
                            getattr(previous_layer[1], bn_tensor_name)[previous_position],
                        ):
                            print(
                                f'FAIL: retained conv BN {bn_tensor_name} mismatch at {layer_name}, '
                                f'previous_position={previous_position}, new_position={new_position}'
                            )
                            passed = False
            elif isinstance(new_layer[0], torch.nn.Linear) and isinstance(previous_layer[0], torch.nn.Linear):
                if not torch.equal(new_layer[0].weight[new_position], previous_layer[0].weight[previous_position]):
                    print(
                        f'FAIL: retained linear weight mismatch at {layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False
                if (
                    new_layer[0].bias is not None
                    and previous_layer[0].bias is not None
                    and not torch.equal(new_layer[0].bias[new_position], previous_layer[0].bias[previous_position])
                ):
                    print(
                        f'FAIL: retained linear bias mismatch at {layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False
            else:
                print(
                    f'FAIL: unsupported layer pair at {layer_name}: '
                    f'{type(previous_layer[0]).__name__} vs {type(new_layer[0]).__name__}'
                )
                passed = False
                break

            if new_feature_boosting is not None and previous_feature_boosting is not None:
                if not torch.equal(
                    new_feature_boosting.w[:, new_position, ...],
                    previous_feature_boosting.w[:, previous_position, ...],
                ):
                    print(
                        f'FAIL: retained FeatureBoosting coefficient mismatch at {layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False

            if isinstance(new_next_layer, torch.nn.Conv2d) and isinstance(previous_next_layer, torch.nn.Conv2d):
                if not torch.equal(
                    new_next_layer.weight[:, new_position, ...],
                    previous_next_layer.weight[:, previous_position, ...],
                ):
                    print(
                        f'FAIL: retained downstream conv input mismatch at {new_next_layer_name}, '
                        f'previous_position={previous_position}, new_position={new_position}'
                    )
                    passed = False
            elif isinstance(new_next_layer, torch.nn.Linear) and isinstance(previous_next_layer, torch.nn.Linear):
                if isinstance(new_layer[0], torch.nn.Conv2d):
                    previous_channel_block = previous_next_layer.weight.size(1) // previous_selected_count
                    new_channel_block = new_next_layer.weight.size(1) // new_selected_count
                    if previous_channel_block != new_channel_block:
                        print(
                            f'FAIL: flattened downstream block mismatch at {new_next_layer_name}: '
                            f'{previous_channel_block} vs {new_channel_block}'
                        )
                        passed = False
                        continue
                    previous_slice = slice(
                        previous_position * previous_channel_block,
                        (previous_position + 1) * previous_channel_block,
                    )
                    new_slice = slice(
                        new_position * new_channel_block,
                        (new_position + 1) * new_channel_block,
                    )
                    if not torch.equal(
                        new_next_layer.weight[:, new_slice],
                        previous_next_layer.weight[:, previous_slice],
                    ):
                        print(
                            f'FAIL: retained downstream flattened linear input mismatch at '
                            f'{new_next_layer_name}, previous_position={previous_position}, '
                            f'new_position={new_position}'
                        )
                        passed = False
                else:
                    if not torch.equal(
                        new_next_layer.weight[:, new_position],
                        previous_next_layer.weight[:, previous_position],
                    ):
                        print(
                            f'FAIL: retained downstream linear input mismatch at {new_next_layer_name}, '
                            f'previous_position={previous_position}, new_position={new_position}'
                        )
                        passed = False
            else:
                print(
                    f'FAIL: unsupported next-layer pair at {layer_name}: '
                    f'{type(previous_next_layer).__name__} vs {type(new_next_layer).__name__}'
                )
                passed = False
                break

    if passed:
        print('PASS: alpha=0.1 with increment_ratio=0.01 keeps every layer very close to the previous small model')
        return True

    print('FAIL: small incremental regeneration changed too many neurons or broke retained-channel inheritance')
    return False


def main():
    seed = 20260410
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cpu')
    max_sparsity = 0.8
    print(f'Build large model on {device} with seed={seed}, max_sparsity={max_sparsity}')
    reference_large_model = build_large_model(device, max_sparsity)

    sample = {
        'rgb': torch.rand((3, 3, 128, 128), device=device),
        'depth': torch.rand((3, 1, 128, 128), device=device),
        'state': torch.rand((3, 42), device=device),
    }

    # test1_passed = run_identity_feedback_test(reference_large_model, max_sparsity, sample)
    # test2_passed = run_downstream_input_column_feedback_test(reference_large_model, max_sparsity, sample)
    # test3_passed = run_full_parameter_feedback_coverage_test(reference_large_model, max_sparsity, sample)
    # test4_passed = run_retained_channel_inheritance_test(reference_large_model, max_sparsity, sample)
    # test5_passed = run_optimizer_state_remap_test(reference_large_model, max_sparsity, sample)
    # test6_passed = run_feature_boosting_retained_channel_inheritance_test(
    #     reference_large_model,
    #     max_sparsity,
    #     sample,
    # )
    # test7_passed = run_zero_increment_regeneration_identity_test(
    #     reference_large_model,
    #     max_sparsity,
    #     sample,
    # )
    # test8_passed = run_zero_increment_optimizer_reset_identity_test(
    #     reference_large_model,
    #     max_sparsity,
    #     sample,
    # )
    test9_passed = run_small_increment_regeneration_stays_close_test(
        reference_large_model,
        max_sparsity,
        sample,
    )

    print('\nSummary')
    # print(f'Test 1 passed: {test1_passed}')
    # print(f'Test 2 passed: {test2_passed}')
    # print(f'Test 3 passed: {test3_passed}')
    # print(f'Test 4 passed: {test4_passed}')
    # print(f'Test 5 passed: {test5_passed}')
    # print(f'Test 6 passed: {test6_passed}')
    # print(f'Test 7 passed: {test7_passed}')
    # print(f'Test 8 passed: {test8_passed}')
    # print(f'Test 9 passed: {test9_passed}')
    # if test1_passed and test2_passed and test3_passed and test4_passed and test5_passed and test6_passed and test7_passed and test8_passed and test9_passed:
    #     print('ALL TESTS PASSED')
    # else:
    #     print('SOME TESTS FAILED')


if __name__ == '__main__':
    main()
