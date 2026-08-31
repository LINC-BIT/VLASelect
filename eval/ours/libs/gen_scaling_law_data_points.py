import torch
from torch import nn
import copy
from functools import partial
from ours.utils.dl.common.model import get_module, set_module, get_model_device
from transformers.pytorch_utils import prune_linear_layer
from ours.libs.train_with_fbs.lib_transformer import StaticFBS
from ours.utils.common.log import logger


def _rebind_intermediate_featurizer_forward(featurizer: nn.Module) -> None:
    blocks = getattr(featurizer, 'blocks', None)
    get_intermediate_layers = getattr(featurizer, 'get_intermediate_layers', None)
    if blocks is None or not callable(get_intermediate_layers):
        return
    num_blocks = len(blocks)
    featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))


def _refresh_copied_featurizer_forward_bindings(module: nn.Module) -> None:
    for submodule in module.modules():
        for attr_name in ('featurizer', 'fused_featurizer'):
            featurizer = getattr(submodule, attr_name, None)
            if isinstance(featurizer, nn.Module):
                _rebind_intermediate_featurizer_forward(featurizer)


def unpack_tuple(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper


def prune_linear_layer_and_its_after_layer(model, layer_name, after_layer_name, unpruned_neurons_idx, attention_value, sparsity, device, window_merge):
    set_module(model, layer_name, nn.Sequential(
        prune_linear_layer(
            get_module(model, layer_name),
            unpruned_neurons_idx.to(device)
        ),
        StaticFBS(attention_value.unsqueeze(0), window_merge)
    ))
    set_module(model, after_layer_name, prune_linear_layer(
        get_module(model, after_layer_name),
        unpruned_neurons_idx.to(device),
        dim=1
    ))


def generate_small_model(large_model: nn.Module, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv=False,
                         return_detail=False):
    large_model = copy.deepcopy(large_model)
    _refresh_copied_featurizer_forward_bindings(large_model)
    device = get_model_device(large_model)
    
    from ..libs.gen_neuron_index import get_fbs_layers
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv)
    unpruned_neurons_idx_of_layers = {}
    attention_value_of_layers = {}
    
    for fbs_layer in fbs_layers:
        attention_value = get_module(large_model, fbs_layer).cached_raw_w # original code: .cached_w
        window_merge = getattr(get_module(large_model, fbs_layer), 'window_merge', None)
        
        assert attention_value.size(0) == 1
        attention_value = attention_value[0]
        attention_value_of_layers[fbs_layer] = attention_value
        
        sparsity = get_module(large_model, fbs_layer).k_takes_all.k 
        
        # unpruned_neurons_idx = attention_value.nonzero(as_tuple=True)[0]
        pruned_neurons_idx = get_module(large_model, fbs_layer).k_takes_all.cached_i[0].sort()[0]
        unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
        attention_value = attention_value[unpruned_neurons_idx]
        
        unpruned_neurons_idx_of_layers[fbs_layer + '.raw_linear'] = unpruned_neurons_idx

        set_module(large_model, fbs_layer, get_module(large_model, fbs_layer).raw_linear)
        
        from ours.utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
    
    logger.debug(f'Generated small model: {large_model}')
    
    if return_detail:
        return large_model, unpruned_neurons_idx_of_layers, attention_value_of_layers
    
    return large_model



def generate_small_model_v2(large_model: nn.Module, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv=False,
                            return_detail=False, model_for_attention_value=None):
    large_model = copy.deepcopy(large_model)
    _refresh_copied_featurizer_forward_bindings(large_model)
    device = get_model_device(large_model)

    if model_for_attention_value is None:
        model_for_attention_value = large_model
    
    from ..libs.gen_neuron_index import get_fbs_layers
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv)
    unpruned_neurons_idx_of_layers = {}
    attention_value_of_layers = {}
    attention_value_of_layers_after_zeroing = {}
    
    for fbs_layer in fbs_layers:
        attention_value = get_module(model_for_attention_value, fbs_layer).cached_raw_w # original code: .cached_w
        attention_value_after_zeroing = get_module(model_for_attention_value, fbs_layer).cached_w
        window_merge = getattr(get_module(model_for_attention_value, fbs_layer), 'window_merge', None)
        
        # assert attention_value.size(0) == 1

        if attention_value.size(0) == 1:
            attention_value = attention_value[0]
            attention_value_after_zeroing = attention_value_after_zeroing[0]
            # attention_value = attention_value.mean(0)
            # attention_value_of_layers[fbs_layer] = attention_value
            
            sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 
            
            # unpruned_neurons_idx = attention_value.nonzero(as_tuple=True)[0]
            pruned_neurons_idx = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0].sort()[0]
            unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value.clone()
            attention_value = attention_value[unpruned_neurons_idx]
        else:
            attention_value = attention_value.mean(0)
            attention_value_after_zeroing = attention_value_after_zeroing.mean(0)
            sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 

            num_pruned_neurons = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0].sort()[0].size(0)
            pruned_neurons_idx = attention_value.sort()[1][:num_pruned_neurons]
            unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value.clone()
            attention_value = attention_value[unpruned_neurons_idx]
        
        # unpruned_neurons_idx_of_layers[fbs_layer + '.raw_linear'] = unpruned_neurons_idx

        set_module(large_model, fbs_layer, get_module(large_model, fbs_layer).raw_linear)
        
        from ours.utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[ff2_layers_name[i]] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[ff2_layers_name[i]] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[ff2_layers_name[i]] = attention_value_after_zeroing
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
    
    logger.debug(f'Generated small model: {large_model}')
    
    if return_detail:
        return large_model, unpruned_neurons_idx_of_layers, attention_value_of_layers, attention_value_of_layers_after_zeroing
    
    return large_model




def generate_small_model_for_ab_study(large_model: nn.Module, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, ab_strategy,
                                       only_add_fbs_in_qkv=False,
                            return_detail=False, model_for_attention_value=None):
    large_model = copy.deepcopy(large_model)
    device = get_model_device(large_model)

    if model_for_attention_value is None:
        model_for_attention_value = large_model
    
    from ..gen_neuron_index.lib_transformer import get_fbs_layers
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv)
    unpruned_neurons_idx_of_layers = {}
    attention_value_of_layers = {}
    attention_value_of_layers_after_zeroing = {}
    
    for fbs_layer in fbs_layers:
        attention_value = get_module(model_for_attention_value, fbs_layer).cached_raw_w # original code: .cached_w
        attention_value_after_zeroing = get_module(model_for_attention_value, fbs_layer).cached_w
        window_merge = getattr(get_module(model_for_attention_value, fbs_layer), 'window_merge', None)
        
        # assert attention_value.size(0) == 1

        if attention_value.size(0) == 1:
            attention_value = attention_value[0]
            attention_value_after_zeroing = attention_value_after_zeroing[0]
            # attention_value = attention_value.mean(0)
            # attention_value_of_layers[fbs_layer] = attention_value
            
            sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 
            
            # unpruned_neurons_idx = attention_value.nonzero(as_tuple=True)[0]

            pruned_neurons_idx = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0].sort()[0]

            if ab_strategy == 'random':
                pruned_neurons_idx = torch.randperm(len(attention_value))[:len(pruned_neurons_idx)]
            elif ab_strategy == 'inverse':
                pruned_neurons_idx = attention_value.sort(descending=True)[1][:len(pruned_neurons_idx)]

            
            unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value.clone()
            attention_value = attention_value[unpruned_neurons_idx]
        else:
            attention_value = attention_value.mean(0)
            attention_value_after_zeroing = attention_value_after_zeroing.mean(0)
            sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 

            num_pruned_neurons = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0].sort()[0].size(0)
            pruned_neurons_idx = attention_value.sort()[1][:num_pruned_neurons]


            if ab_strategy == 'random':
                pruned_neurons_idx = torch.randperm(len(attention_value))[:len(pruned_neurons_idx)]
            elif ab_strategy == 'inverse':
                pruned_neurons_idx = attention_value.sort(descending=True)[1][:len(pruned_neurons_idx)]


            unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value.clone()
            attention_value = attention_value[unpruned_neurons_idx]
        
        # unpruned_neurons_idx_of_layers[fbs_layer + '.raw_linear'] = unpruned_neurons_idx

        set_module(large_model, fbs_layer, get_module(large_model, fbs_layer).raw_linear)
        
        from utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
                                                   unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[ff2_layers_name[i]] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[ff2_layers_name[i]] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[ff2_layers_name[i]] = attention_value_after_zeroing
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                                                       unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
    
    logger.debug(f'Generated small model: {large_model}')
    
    if return_detail:
        return large_model, unpruned_neurons_idx_of_layers, attention_value_of_layers, attention_value_of_layers_after_zeroing
    
    return large_model





def generate_small_model_v2_only_need_attention_value(large_model: nn.Module, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv=False,
                            return_detail=False, model_for_attention_value=None):
    # large_model = copy.deepcopy(large_model)
    device = get_model_device(large_model)

    if model_for_attention_value is None:
        model_for_attention_value = large_model
    
    from ..gen_neuron_index.lib_transformer import get_fbs_layers
    
    fbs_layers = get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_add_fbs_in_qkv)
    unpruned_neurons_idx_of_layers = {}
    attention_value_of_layers = {}
    attention_value_of_layers_after_zeroing = {}

    # logger.debug('111')
    
    for fbs_layer in fbs_layers:
        # logger.debug('111')

        attention_value = get_module(model_for_attention_value, fbs_layer).cached_raw_w # original code: .cached_w
        attention_value_after_zeroing = get_module(model_for_attention_value, fbs_layer).cached_w
        window_merge = getattr(get_module(model_for_attention_value, fbs_layer), 'window_merge', None)
        
        # assert attention_value.size(0) == 1
        # logger.debug('111')

        if attention_value.size(0) == 1:
            attention_value = attention_value[0]
            attention_value_after_zeroing = attention_value_after_zeroing[0]
            # attention_value = attention_value.mean(0)
            # attention_value_of_layers[fbs_layer] = attention_value
            
            # sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 
            
            # unpruned_neurons_idx = attention_value.nonzero(as_tuple=True)[0]
            # pruned_neurons_idx = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0] # WARN: modified
            # unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value
            # attention_value = attention_value[unpruned_neurons_idx]
        else:
            attention_value = attention_value.mean(0)
            attention_value_after_zeroing = attention_value_after_zeroing.mean(0)
            # sparsity = get_module(model_for_attention_value, fbs_layer).k_takes_all.k 

            # num_pruned_neurons = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0].size(0) # WARN: modified
            # pruned_neurons_idx = get_module(model_for_attention_value, fbs_layer).k_takes_all.cached_i[0] # WARN: modified
            # unpruned_neurons_idx = torch.LongTensor([ni for ni in range(len(attention_value)) if ni not in pruned_neurons_idx])
            full_attention_value = attention_value
            # attention_value = attention_value[unpruned_neurons_idx]
        
        pruned_neurons_idx, unpruned_neurons_idx = [], []
        # logger.debug('111')
        # unpruned_neurons_idx_of_layers[fbs_layer + '.raw_linear'] = unpruned_neurons_idx

        # set_module(large_model, fbs_layer, get_module(large_model, fbs_layer).raw_linear)
        
        from utils.common.data import flatten_2d_arr
        qkv_layers_name = flatten_2d_arr(qkv_layers_name)
        for qkv_layer_name in qkv_layers_name:
            if not fbs_layer.startswith(qkv_layer_name):
                continue
            
            # prune [qkv].0 and [qkv].1
            # prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
            #                                        unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        for proj_layer_name in proj_layers_name:
            if not fbs_layer.startswith(proj_layer_name):
                continue
            
            # prune [proj].0 and [proj].1
            # prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1', 
            #                                        unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
            unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
            unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
            attention_value_of_layers[fbs_layer] = full_attention_value
            attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
            attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
            attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
            break
        
        if isinstance(ff1_layers_name[0], list):
            for i, ff1_layer_name in enumerate(flatten_2d_arr(ff1_layers_name)):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                # prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                #                                        unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing
                break
        else:
            for i, ff1_layer_name in enumerate(ff1_layers_name):
                if not fbs_layer.startswith(ff1_layer_name):
                    continue
                # prune ff1 and ff2
                # prune_linear_layer_and_its_after_layer(large_model, fbs_layer, ff2_layers_name[i], 
                #                                        unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[ff2_layers_name[i]] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[ff2_layers_name[i]] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[ff2_layers_name[i]] = attention_value_after_zeroing
                break
            
        if isinstance(ff1_layers_name[0], list):
            for i, ff2_layer_name in enumerate(ff2_layers_name):
                if not fbs_layer.startswith(ff2_layer_name):
                    continue
                # prune [ff1].0 and [ff2].0
                # prune_linear_layer_and_its_after_layer(large_model, fbs_layer, fbs_layer[0: -2] + '.1',
                #                                        unpruned_neurons_idx, attention_value, sparsity, device, window_merge)
                unpruned_neurons_idx_of_layers[fbs_layer] = (unpruned_neurons_idx, 0)
                unpruned_neurons_idx_of_layers[fbs_layer[0: -2] + '.1'] = (unpruned_neurons_idx, 1)
                attention_value_of_layers[fbs_layer] = full_attention_value
                attention_value_of_layers[fbs_layer[0: -2] + '.1'] = full_attention_value
                attention_value_of_layers_after_zeroing[fbs_layer] = attention_value_after_zeroing
                attention_value_of_layers_after_zeroing[fbs_layer[0: -2] + '.1'] = attention_value_after_zeroing

        # logger.debug('111')

    logger.debug(f'Generated small model: {large_model}')
    
    if return_detail:
        return large_model, unpruned_neurons_idx_of_layers, attention_value_of_layers, attention_value_of_layers_after_zeroing
    
    return large_model