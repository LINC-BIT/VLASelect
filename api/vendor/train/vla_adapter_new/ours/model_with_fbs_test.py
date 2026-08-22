import sys

sys.path.append('./')  # 添加当前目录到sys.path，以便导入ours.pretrain_fbs_model.main
from ours.libs.gen_scaling_law_data_points import generate_small_model_v2
from ours.libs.train_with_fbs.lib_cnn import get_model_size
from ours.pretrain_fbs_model.main import add_FBS_into_transformer
import torch
from train.vla_adapter_new.model_impl.online_rl_hold_cube_in_hand import HandVLAAdapterActorCritic
from pathlib import Path



# with open('./vla-adapter.txt', 'w') as f:
#     f.write(str(actor))
# exit()

def _module_at(root, path):
    current = root
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _validate_declared_layers(actor, layer_groups):
    if layer_groups is None:
        return
    paths = list(layer_groups.vision_qkv) + list(layer_groups.vision_proj)
    paths += list(layer_groups.vision_ff1) + list(layer_groups.vision_ff2)
    for group in layer_groups.language_qkv + layer_groups.language_ff1:
        paths.extend(group)
    paths += list(layer_groups.language_proj) + list(layer_groups.language_ff2)
    for path in paths:
        root = actor if path.startswith("vla.") else actor.vla.language_model
        try:
            module = _module_at(root, path)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(f"declared FBS layer `{path}` is absent from the constructed policy") from exc
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"declared FBS layer `{path}` must be torch.nn.Linear, got {type(module).__name__}")


def convert_to_fbs_model(actor, device, max_sparsity=0.9, layer_groups=None):

    print(f'original model: {get_model_size(actor, True):.3f}MB')

    sample_batch = {
        "rgbs": torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8),
        "states": torch.randn(1, 105, device=device),
        'input_ids': torch.randint(0, 1000, (1, 16), device=device)
    }


    # -------- 添加FBS ---------------
    qkv_layers, proj_layers, ff1_layers, ff2_layers = [], [], [], []
    if layer_groups is not None:
        qkv_layers = list(layer_groups.vision_qkv)
        proj_layers = list(layer_groups.vision_proj)
        ff1_layers = list(layer_groups.vision_ff1)
        ff2_layers = list(layer_groups.vision_ff2)
        _validate_declared_layers(actor, layer_groups)
    else:
        featurizer = getattr(actor.vla.vision_backbone, 'featurizer', None)
        fused_featurizer = getattr(actor.vla.vision_backbone, 'fused_featurizer', None)
        featurizer_blocks = getattr(featurizer, 'blocks', None)
        fused_blocks = getattr(fused_featurizer, 'blocks', None)
        if featurizer_blocks is not None:
            for block_i in range(len(featurizer_blocks)):
                qkv_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.qkv']
                proj_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.proj']
                ff1_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc1']
                ff2_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc2']
        if fused_blocks is not None:
            for block_i in range(len(fused_blocks)):
                qkv_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.qkv']
                proj_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.proj']
                ff1_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc1']
                ff2_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc2']

    from ours.pretrain_fbs_model.main import add_FBS_into_transformer
    if qkv_layers:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            actor = add_FBS_into_transformer(
                actor.to(device),
                qkv_layers, proj_layers, ff1_layers, ff2_layers,
                sample_batch, max_sparsity, 16, 
                lambda model, batch: model(
                    rgbs=batch['rgbs'],
                    states=batch['states'],
                    mode='policy'
                )[0].sum()
            ).cpu()
    else:
        print('[setup] skipping FBS vision conversion because the current actor has no compatible transformer blocks')

    # 语言模型的qkv/FFN结构比较独特，单独添加fbs模块
    qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2 = [], [], [], []
    if layer_groups is not None:
        qkv_layers2 = [list(group) for group in layer_groups.language_qkv]
        proj_layers2 = list(layer_groups.language_proj)
        ff1_layers2 = [list(group) for group in layer_groups.language_ff1]
        ff2_layers2 = list(layer_groups.language_ff2)
    else:
        language_model_core = getattr(actor.vla.language_model, 'model', None)
        language_layers = getattr(language_model_core, 'layers', None)
        if language_layers is not None:
            for block_i in range(len(language_layers)):
                qkv_layers2 += [[f'model.layers.{block_i}.self_attn.{k}_proj'] for k in ['q', 'k', 'v']]
                proj_layers2 += [f'model.layers.{block_i}.self_attn.o_proj']
                ff1_layers2 += [[f'model.layers.{block_i}.mlp.gate_proj', f'model.layers.{block_i}.mlp.up_proj']]
                ff2_layers2 += [f'model.layers.{block_i}.mlp.down_proj']

    if qkv_layers2:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            actor.vla.language_model = add_FBS_into_transformer(
                actor.vla.language_model.to(device),
                qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2,
                sample_batch, max_sparsity, 16, 
                lambda model, batch: model(
                    input_ids=batch['input_ids'].to(device),
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1].mean()
            ).cpu()
    else:
        print('[setup] skipping FBS language conversion because the current actor has no compatible decoder blocks')


    # small_model = generate_small_model_v2(actor, qkv_layers + qkv_layers2, proj_layers + proj_layers2, ff1_layers + ff1_layers2, ff2_layers + ff2_layers2)
    # print(f'small model: {get_model_size(small_model, True):.3f}MB')
    actor.vla.to(dtype=torch.bfloat16)

    return actor


def convert_to_fbs_model2(actor, device, dtype=torch.bfloat16):

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
    featurizer = getattr(actor.vla.vision_backbone, 'featurizer', None)
    fused_featurizer = getattr(actor.vla.vision_backbone, 'fused_featurizer', None)
    featurizer_blocks = getattr(featurizer, 'blocks', None)
    fused_blocks = getattr(fused_featurizer, 'blocks', None)
    if featurizer_blocks is not None:
        for block_i in range(len(featurizer_blocks)):
            qkv_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.qkv']
            proj_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.attn.proj']
            ff1_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc1']
            ff2_layers += [f'vla.vision_backbone.featurizer.blocks.{block_i}.mlp.fc2']
    if fused_blocks is not None:
        for block_i in range(len(fused_blocks)):
            qkv_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.qkv']
            proj_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.attn.proj']
            ff1_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc1']
            ff2_layers += [f'vla.vision_backbone.fused_featurizer.blocks.{block_i}.mlp.fc2']

    from ours.pretrain_fbs_model.main import add_FBS_into_transformer
    if qkv_layers:
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
    else:
        print('[setup] skipping FBS vision conversion because the current actor has no compatible transformer blocks')

    # 语言模型的qkv/FFN结构比较独特，单独添加fbs模块
    qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2 = [], [], [], []
    language_model_core = getattr(actor.vla.language_model, 'model', None)
    language_layers = getattr(language_model_core, 'layers', None)
    if language_layers is not None:
        for block_i in range(len(language_layers)):
            qkv_layers2 += [[f'model.layers.{block_i}.self_attn.{k}_proj'] for k in ['q', 'k', 'v']]
            proj_layers2 += [f'model.layers.{block_i}.self_attn.o_proj']
            ff1_layers2 += [[f'model.layers.{block_i}.mlp.gate_proj', f'model.layers.{block_i}.mlp.up_proj']]
            ff2_layers2 += [f'model.layers.{block_i}.mlp.down_proj']

    if qkv_layers2:
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
    else:
        print('[setup] skipping FBS language conversion because the current actor has no compatible decoder blocks')


    small_model = generate_small_model_v2(actor, qkv_layers, proj_layers, ff1_layers, ff2_layers) if qkv_layers else actor
    if qkv_layers2 and hasattr(small_model.vla, 'language_model'):
        small_model.vla.language_model = generate_small_model_v2(small_model.vla.language_model, qkv_layers2, proj_layers2, ff1_layers2, ff2_layers2)
    print(f'small model: {get_model_size(small_model, True):.3f}MB')
    small_model.to(dtype=dtype)

    return actor





if __name__ == '__main__':
    # actor = HandVLAAdapterActorCritic(Path('eval/ckpt/vla_adapter_new/LIBERO-Object'), 'cuda')
    # convert_to_fbs_model2(actor, 'cuda', dtype=torch.bfloat16)
    
    # actor = HandVLAAdapterActorCritic(Path('eval/ckpt/vla_adapter_new/LIBERO-Object'), 'cuda')
    # convert_to_fbs_model2(actor, 'cuda', dtype=torch.bfloat16)

    actor = HandVLAAdapterActorCritic(Path('eval/ckpt/vla_adapter_new/LIBERO-Object'), 'cuda')
