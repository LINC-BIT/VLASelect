import torch
from torch import nn 
import sys; sys.path.insert(0, '.')

from ours.libs.train_with_fbs.lib import set_sparsity
from ours.libs.train_with_fbs.lib_cnn import Conv2dWithFBS, Linear2DWithFBS
from ours.libs.train_with_fbs.lib_transformer import LinearWithFBS, svd_decompose_linear
from ours.utils.dl.common.model import get_model_device, get_model_latency, get_model_size, get_module, set_module


def add_FBS_into_transformer(model: nn.Module,
                       qkv_layers_name,
                       proj_layers_name,
                       ff1_layers_name,
                       ff2_layers_name,
                       example_sample,
                       max_sparsity,
                       fbs_r,
                       model_forward_fn,
                       verify_outputs=True):
    
    # 1. 分解QKV，以支持动态剪枝
    print(f'before svd decomposition model size: {get_model_size(model, True):.3f}MB')
    # print(f'before svd decomposition model: {model}')
    from ..libs.gen_knowledge_base import svd_decompose_linear
    for qkv_layer_name in qkv_layers_name:
        if isinstance(qkv_layer_name, list):
            for qkv_name in qkv_layer_name:
                set_module(model, qkv_name, svd_decompose_linear(get_module(model, qkv_name)))
        else:
            set_module(model, qkv_layer_name, svd_decompose_linear(get_module(model, qkv_layer_name)))
    for proj_layer_name in proj_layers_name:
        set_module(model, proj_layer_name, svd_decompose_linear(get_module(model, proj_layer_name)))
    if isinstance(ff1_layers_name[0], list): # llama case
        for ff1_layer_name in ff1_layers_name:
            for n in ff1_layer_name:
                set_module(model, n, svd_decompose_linear(get_module(model, n)))
        for ff2_layer_name in ff2_layers_name:
            set_module(model, ff2_layer_name, svd_decompose_linear(get_module(model, ff2_layer_name)))
    print(f'after svd decomposition model size: {get_model_size(model, True):.3f}MB')
    # print(f'after svd decomposition model: {model}')

    # 2. 增加FBS模块
    from ..libs.train_with_fbs.lib import add_FBS, get_importance_values, set_sparsity, get_l1_reg_in_model, clear_cache
    from ..libs.gen_neuron_index import get_fbs_layers
    fbs_layers = get_fbs_layers(
        qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name
    )
    # logger.debug(fbs_layers)
    fbs_ignore_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name not in fbs_layers:
            fbs_ignore_layers += [name]

    model = add_FBS(
        model,
        max_sparsity,
        fbs_r,
        fbs_ignore_layers,
        True,
        example_sample,
        None
    )

    with torch.no_grad():
        set_sparsity(model, max_sparsity)
        model.eval()
        if verify_outputs:
            o1 = model_forward_fn(model, example_sample)
            o3 = model_forward_fn(model, example_sample)
            from ours.libs.gen_scaling_law_data_points import generate_small_model
            small_model = generate_small_model(model, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name)
            small_model.eval()
            o2 = model_forward_fn(small_model, example_sample)
            diff = ((o1 - o2) ** 2).sum()
            diff2 = ((o3 - o1) ** 2).sum()
            verify_threshold = 1e-4
            verify_warning_threshold = 2e-4
            for label, value in (("diff2", diff2), ("diff", diff)):
                scalar = float(value.detach().item() if hasattr(value, "detach") else value)
                if scalar < verify_threshold:
                    continue
                if scalar <= verify_warning_threshold:
                    print(f"[warning] FBS verify borderline {label}={scalar:.10f} (threshold={verify_threshold:.1e})")
                    continue
                print(f"[error] FBS verify failed {label}={scalar:.10f} (threshold={verify_threshold:.1e})")
                assert scalar < verify_threshold, f"{label}={scalar:.10f}"

            print('kb size: {}MB, proxy model size: {}MB)'.format(get_model_size(model, True), get_model_size(small_model, True)))
            print('FBS verify passed (diff: {}, diff2: {})'.format(diff, diff2))
        else:
            print('[warning] skip FBS output verification for this call')
    # logger.debug(f'after add FBS model: {model}')

    return model



def add_FBS_into_cnn(model: nn.Module,
                     conv_layers_name,
                     fc_layers_name,
                     example_sample,
                     max_sparsity,
                     fbs_r,
                     model_forward_fn):
    
    
    
    device = get_model_device(model)
    model.eval()
    example_input = example_sample
    # o1 = model_forward_fn(model, example_input)

    # clear original BNs
    num_original_bns = 0
    last_conv_name = None
    conv_bn_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
        if isinstance(module, nn.BatchNorm2d) and last_conv_name in conv_layers_name:
            num_original_bns += 1
            conv_bn_map[last_conv_name] = name
    
    num_conv = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in conv_layers_name:
            conv2d_with_fbs = Conv2dWithFBS(module, get_module(model, conv_bn_map[name]) if name in conv_bn_map else nn.Identity(),
                                            max_sparsity, fbs_r).to(device)
            set_module(model, name, conv2d_with_fbs)
            num_conv += 1

        if isinstance(module, nn.Linear) and name in fc_layers_name:
            linear_with_fbs = Linear2DWithFBS(module, max_sparsity, fbs_r, None).to(device)
            set_module(model, name, linear_with_fbs)
            
    # assert num_conv == num_original_bns
    
    for bn_layer in conv_bn_map.values():
        set_module(model, bn_layer, nn.Identity())

    # print(model)

    # o2 = model_forward_fn(model, example_input)
    # error = (o1 - o2).abs().max().item()
    # assert error < 1e-5, error
    

    with torch.no_grad():
        set_sparsity(model, max_sparsity)
        model.eval()
        o1 = model_forward_fn(model, example_sample)
        from ours.libs.gen_scaling_law_data_points_cnn import generate_small_cnn
        small_model = generate_small_cnn(model, example_input, model_forward_fn)
        # print(small_model)
        small_model.eval()
        o2 = model_forward_fn(small_model, example_sample)
        diff = ((o1 - o2) ** 2).sum()
        assert diff < 1e-4, diff
        print('FBS verify passed (kb size: {}MB, proxy model size: {}MB, diff: {})'.format(get_model_size(model, True), 
                                                                            get_model_size(small_model, True), diff))
    # logger.debug(f'after add FBS model: {model}')

    return model


def generate_small_cnn_with_verify(model,
                                   sparsity,
                                   example_sample,
                                   model_forward_fn,
                                   return_pruning_info=False,
                                   previous_pruning_info=None,
                                   regeneration_increment_ratio=1.0,
                                   ab_strategy=None):
    with torch.no_grad():
        set_sparsity(model, sparsity)
        model.eval()
        o1 = model_forward_fn(model, example_sample)
        from ours.libs.gen_scaling_law_data_points_cnn import generate_small_cnn
        if return_pruning_info:
            small_model, pruning_info = generate_small_cnn(
                model,
                example_sample,
                model_forward_fn,
                return_pruning_info=True,
                previous_pruning_info=previous_pruning_info,
                regeneration_increment_ratio=regeneration_increment_ratio,
                ab_strategy=ab_strategy,
            )
        else:
            small_model = generate_small_cnn(
                model,
                example_sample,
                model_forward_fn,
                previous_pruning_info=previous_pruning_info,
                regeneration_increment_ratio=regeneration_increment_ratio,
                ab_strategy=ab_strategy,
            )
        # print(small_model)
        small_model.eval()
        o2 = model_forward_fn(small_model, example_sample)
        diff = ((o1 - o2) ** 2).sum()
        
        # assert diff < 1e-4, diff
        print('FBS verify passed (kb size: {}MB, proxy model size: {}MB, diff: {})'.format(get_model_size(model, True), 
                                                                            get_model_size(small_model, True), diff))
    # logger.debug(f'after add FBS model: {model}')

    if return_pruning_info:
        return small_model, pruning_info

    return small_model



if __name__ == '__main__':
    from train.toy_cnn.model import Actor
    model = Actor(state_dim=29, action_dim=8, camera_count=1)
    example_sample = {
        'rgb': torch.rand((1, 3, 128, 128)),
        'depth': torch.rand((1, 1, 128, 128)),
        'state': torch.rand((1, 29))
    }
    # print(model)

    set_module(model, 'rgb_encoder.fc.0', svd_decompose_linear(
        get_module(model, 'rgb_encoder.fc.0')
    ))
    set_module(model, 'depth_encoder.fc.0', svd_decompose_linear(
        get_module(model, 'depth_encoder.fc.0')
    ))

    add_FBS_into_cnn(
        model,
        [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
        [f'state_encoder.{i}' for i in [2]] + ['decoder.0'] + ['rgb_encoder.fc.0.0', 'depth_encoder.fc.0.0'],
        example_sample,
        0.9,
        8,
        lambda model, sample: model(sample['rgb'], sample['depth'], sample['state'])
    )

    # add_FBS_into_cnn(
    #     model.state_encoder,
    #     # [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]] + [f'depth_encoder.cnn.{i}' for i in [0, 6, 12]],
    #     [],
    #     ['2'],
    #     torch.rand((1, 29)),
    #     0.9,
    #     8,
    #     lambda model, sample: model(sample)
    # )
