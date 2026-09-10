from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import os
os.environ['SVD_SIM'] = '1'
import torch

THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from discussion import sweep_model_size as sweep
from ours.pretrain_fbs_model import main as fbs_main
from ours.utils.dl.common.model import get_model_device, get_module, set_module
from transformers.pytorch_utils import prune_linear_layer


def tensor_storage_size_mb(module: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in module.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)


def patch_expensive_size_logging() -> None:
    # add_FBS_into_transformer prints model sizes before/after SVD.  For an
    # 11.3GB temporary model, serializing state_dicts just for logging is
    # unnecessarily slow and disk-heavy, so use tensor storage size there.
    def fast_model_size(model: torch.nn.Module, return_MB: bool = False, *_args, **_kwargs):
        size_mb = tensor_storage_size_mb(model)
        return size_mb if return_MB else int(size_mb * 1024.0 * 1024.0)

    fbs_main.get_model_size = fast_model_size


def patch_linear_with_fbs_dtype_alignment() -> None:
    from ours.libs.train_with_fbs import lib_transformer

    if getattr(lib_transformer.LinearWithFBS, '_direct_test_dtype_patch', False):
        return
    original_init = lib_transformer.LinearWithFBS.__init__

    def patched_init(self, raw_linear, k, r, window_merge=None):
        original_init(self, raw_linear, k, r, window_merge)
        try:
            ref_param = next(raw_linear.parameters())
        except StopIteration:
            return
        self.fbs = self.fbs.to(device=ref_param.device, dtype=ref_param.dtype)

    lib_transformer.LinearWithFBS.__init__ = patched_init
    lib_transformer.LinearWithFBS._direct_test_dtype_patch = True


def get_lm_layers(actor: torch.nn.Module) -> torch.nn.ModuleList:
    vla = getattr(actor, 'vla', None)
    language_model = getattr(vla, 'language_model', None)
    language_model_core = getattr(language_model, 'model', None)
    layers = getattr(language_model_core, 'layers', None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
        raise RuntimeError('language_model.model.layers is unavailable; cannot build target-size original model')
    return layers


def get_lm_layers_from_language_model(language_model: torch.nn.Module) -> torch.nn.ModuleList:
    language_model_core = getattr(language_model, 'model', None)
    layers = getattr(language_model_core, 'layers', None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
        raise RuntimeError('language_model.model.layers is unavailable')
    return layers


def sync_lm_layer_count_config(actor: torch.nn.Module) -> None:
    layers = get_lm_layers(actor)
    num_layers = len(layers)
    language_model = actor.vla.language_model
    for owner in (language_model, getattr(language_model, 'model', None), actor.vla, getattr(actor.vla, 'config', None)):
        config = getattr(owner, 'config', None)
        if config is not None and hasattr(config, 'num_hidden_layers'):
            try:
                config.num_hidden_layers = num_layers
            except Exception:
                pass
        text_config = getattr(config, 'text_config', None)
        if text_config is not None and hasattr(text_config, 'num_hidden_layers'):
            try:
                text_config.num_hidden_layers = num_layers
            except Exception:
                pass
        if owner is not None and hasattr(owner, 'num_hidden_layers'):
            try:
                owner.num_hidden_layers = num_layers
            except Exception:
                pass


def inflate_original_model_to_target(
    actor: torch.nn.Module,
    *,
    target_original_model_mb: float,
) -> Dict[str, Any]:
    actor = actor.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    layers = get_lm_layers(actor)
    base_language_layers = len(layers)
    base_original_model_mb = tensor_storage_size_mb(actor)
    template = layers[-1]
    language_layer_mb = tensor_storage_size_mb(template)
    if language_layer_mb <= 0:
        raise RuntimeError('last language layer has zero measured size')

    if target_original_model_mb <= base_original_model_mb:
        added_layers = 0
    else:
        raw_layers = (target_original_model_mb - base_original_model_mb) / language_layer_mb
        added_layers = max(0, int(round(raw_layers)))

    for _ in range(added_layers):
        layers.append(copy.deepcopy(template))
    sync_lm_layer_count_config(actor)

    actual_original_model_mb = tensor_storage_size_mb(actor)
    return {
        'base_original_model_mb': base_original_model_mb,
        'target_original_model_mb': target_original_model_mb,
        'actual_original_model_mb': actual_original_model_mb,
        'language_layer_mb': language_layer_mb,
        'base_language_layers': base_language_layers,
        'total_language_layers': len(layers),
        'added_language_layers': added_layers,
        'target_error_mb': actual_original_model_mb - target_original_model_mb,
    }


def flatten_2d_arr(values):
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_apply_qkv=False):
    qkv_layers_name = flatten_2d_arr(qkv_layers_name)
    fbs_layers = [qkv_layer_name + '.0' for qkv_layer_name in qkv_layers_name]
    if only_apply_qkv:
        return fbs_layers
    fbs_layers.extend(proj_layer_name + '.0' for proj_layer_name in proj_layers_name)
    for ff1_layer_name in ff1_layers_name:
        if isinstance(ff1_layer_name, list):
            fbs_layers.extend(name + '.0' for name in ff1_layer_name)
        else:
            fbs_layers.append(ff1_layer_name)
    if ff1_layers_name and isinstance(ff1_layers_name[0], list):
        fbs_layers.extend(ff2_layer_name + '.0' for ff2_layer_name in ff2_layers_name)
    return fbs_layers


def unpruned_indices_from_fbs_layer(fbs_module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_value = fbs_module.cached_raw_w
    attention_value_after_zeroing = fbs_module.cached_w
    if attention_value is None or attention_value_after_zeroing is None:
        raise RuntimeError('FBS cache is empty; run the FBS calibration forward before generating the proxy model')

    if attention_value.size(0) == 1:
        attention_value = attention_value[0]
        attention_value_after_zeroing = attention_value_after_zeroing[0]
        pruned_neurons_idx = fbs_module.k_takes_all.cached_i[0].sort()[0]
    else:
        attention_value = attention_value.mean(0)
        attention_value_after_zeroing = attention_value_after_zeroing.mean(0)
        num_pruned_neurons = fbs_module.k_takes_all.cached_i[0].sort()[0].size(0)
        pruned_neurons_idx = attention_value.sort()[1][:num_pruned_neurons]

    pruned_mask = torch.zeros(len(attention_value), dtype=torch.bool, device=pruned_neurons_idx.device)
    pruned_mask[pruned_neurons_idx] = True
    unpruned_neurons_idx = (~pruned_mask).nonzero(as_tuple=True)[0].cpu()
    return unpruned_neurons_idx, attention_value, attention_value_after_zeroing


def prune_linear_layer_and_its_after_layer(
    model: torch.nn.Module,
    layer_name: str,
    after_layer_name: str,
    unpruned_neurons_idx: torch.Tensor,
    attention_value: torch.Tensor,
    device: torch.device,
    window_merge,
) -> None:
    from ours.libs.train_with_fbs.lib_transformer import StaticFBS

    set_module(
        model,
        layer_name,
        torch.nn.Sequential(
            prune_linear_layer(get_module(model, layer_name), unpruned_neurons_idx.to(device)),
            StaticFBS(attention_value[unpruned_neurons_idx.to(attention_value.device)].unsqueeze(0), window_merge),
        ),
    )
    set_module(
        model,
        after_layer_name,
        prune_linear_layer(get_module(model, after_layer_name), unpruned_neurons_idx.to(device), dim=1),
    )


def compress_one_fbs_layer(
    model: torch.nn.Module,
    model_for_attention_value: torch.nn.Module,
    fbs_layer: str,
    *,
    qkv_layers_name,
    proj_layers_name,
    ff1_layers_name,
    ff2_layers_name,
    device: torch.device,
) -> None:
    fbs_module = get_module(model_for_attention_value, fbs_layer)
    unpruned_neurons_idx, attention_value, _attention_value_after_zeroing = unpruned_indices_from_fbs_layer(fbs_module)
    window_merge = getattr(fbs_module, 'window_merge', None)

    set_module(model, fbs_layer, get_module(model, fbs_layer).raw_linear)

    qkv_flat = flatten_2d_arr(qkv_layers_name)
    ff1_flat = flatten_2d_arr(ff1_layers_name)
    if any(fbs_layer.startswith(qkv_layer_name) for qkv_layer_name in qkv_flat):
        prune_linear_layer_and_its_after_layer(
            model, fbs_layer, fbs_layer[0:-2] + '.1', unpruned_neurons_idx, attention_value, device, window_merge
        )
        return

    if any(fbs_layer.startswith(proj_layer_name) for proj_layer_name in proj_layers_name):
        prune_linear_layer_and_its_after_layer(
            model, fbs_layer, fbs_layer[0:-2] + '.1', unpruned_neurons_idx, attention_value, device, window_merge
        )
        return

    if any(fbs_layer.startswith(ff1_layer_name) for ff1_layer_name in ff1_flat):
        if ff1_layers_name and isinstance(ff1_layers_name[0], list):
            after_layer_name = fbs_layer[0:-2] + '.1'
        else:
            after_layer_name = next(
                ff2_layers_name[i]
                for i, ff1_layer_name in enumerate(ff1_layers_name)
                if fbs_layer.startswith(ff1_layer_name)
            )
        prune_linear_layer_and_its_after_layer(
            model, fbs_layer, after_layer_name, unpruned_neurons_idx, attention_value, device, window_merge
        )
        return

    if ff1_layers_name and isinstance(ff1_layers_name[0], list):
        if any(fbs_layer.startswith(ff2_layer_name) for ff2_layer_name in ff2_layers_name):
            prune_linear_layer_and_its_after_layer(
                model, fbs_layer, fbs_layer[0:-2] + '.1', unpruned_neurons_idx, attention_value, device, window_merge
            )
            return

    raise RuntimeError(f'unhandled FBS layer during proxy generation: {fbs_layer}')


def generate_small_model_in_place(
    model: torch.nn.Module,
    qkv_layers_name,
    proj_layers_name,
    ff1_layers_name,
    ff2_layers_name,
) -> torch.nn.Module:
    device = get_model_device(model)
    for fbs_layer in get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name):
        compress_one_fbs_layer(
            model,
            model,
            fbs_layer,
            qkv_layers_name=qkv_layers_name,
            proj_layers_name=proj_layers_name,
            ff1_layers_name=ff1_layers_name,
            ff2_layers_name=ff2_layers_name,
            device=device,
        )
    return model


def generate_small_language_model_reusing_added_layers(
    language_model: torch.nn.Module,
    qkv_layers_name,
    proj_layers_name,
    ff1_layers_name,
    ff2_layers_name,
    *,
    base_language_layers: int,
    added_language_layers: int,
) -> torch.nn.Module:
    device = get_model_device(language_model)
    compress_layer_count = base_language_layers + (1 if added_language_layers > 0 else 0)

    qkv_subset = qkv_layers_name[:compress_layer_count]
    proj_subset = proj_layers_name[:compress_layer_count]
    ff1_subset = ff1_layers_name[:compress_layer_count]
    ff2_subset = ff2_layers_name[:compress_layer_count]
    for fbs_layer in get_fbs_layers(qkv_subset, proj_subset, ff1_subset, ff2_subset):
        compress_one_fbs_layer(
            language_model,
            language_model,
            fbs_layer,
            qkv_layers_name=qkv_subset,
            proj_layers_name=proj_subset,
            ff1_layers_name=ff1_subset,
            ff2_layers_name=ff2_subset,
            device=device,
        )

    if added_language_layers > 1:
        layers = get_lm_layers_from_language_model(language_model)
        compressed_added_layer = layers[base_language_layers]
        for layer_idx in range(base_language_layers + 1, base_language_layers + added_language_layers):
            layers[layer_idx] = copy.deepcopy(compressed_added_layer)

    return language_model


def build_small_actor_reusing_added_layers(
    actor: torch.nn.Module,
    layer_info: Dict[str, Any],
    inflation_info: Dict[str, Any],
) -> torch.nn.Module:
    small_actor = actor
    if layer_info['vision_qkv']:
        small_actor = generate_small_model_in_place(
            small_actor,
            layer_info['vision_qkv'],
            layer_info['vision_proj'],
            layer_info['vision_ff1'],
            layer_info['vision_ff2'],
        )
    if layer_info['lm_qkv'] and hasattr(small_actor, 'vla') and hasattr(small_actor.vla, 'language_model'):
        print(
            '[direct-test] optimized language proxy generation: '
            f"compress_base_layers={inflation_info['base_language_layers']} "
            f"compress_added_layers={1 if inflation_info['added_language_layers'] > 0 else 0} "
            f"copy_compressed_added_layers={max(0, inflation_info['added_language_layers'] - 1)}"
        )
        small_actor.vla.language_model = generate_small_language_model_reusing_added_layers(
            small_actor.vla.language_model,
            layer_info['lm_qkv'],
            layer_info['lm_proj'],
            layer_info['lm_ff1'],
            layer_info['lm_ff2'],
            base_language_layers=int(inflation_info['base_language_layers']),
            added_language_layers=int(inflation_info['added_language_layers']),
        )
    return small_actor


def set_actor_runtime_device(actor: torch.nn.Module, device: torch.device) -> None:
    if hasattr(actor, 'device'):
        actor.device = device


def empty_cuda_cache_for_device(device: torch.device) -> None:
    if device.type != 'cuda' or not torch.cuda.is_available():
        return
    try:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    except Exception:
        torch.cuda.empty_cache()


def release_original_model_before_training(
    proxy_actor: torch.nn.Module,
    *,
    build_device: torch.device,
    train_device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    # The direct-test proxy builder converts the FBS/original actor in-place into
    # the proxy.  Keep only that proxy object, move it off GPU, and clear allocator
    # state before measuring proxy training memory.
    proxy_actor = proxy_actor.cpu()
    set_actor_runtime_device(proxy_actor, torch.device('cpu'))
    gc.collect()
    empty_cuda_cache_for_device(build_device)
    empty_cuda_cache_for_device(train_device)

    cleanup_info: Dict[str, Any] = {
        'large_model_released_before_training': True,
        'proxy_device_before_training': 'cpu',
        'gpu_allocated_mb_after_large_model_release': None,
        'gpu_reserved_mb_after_large_model_release': None,
    }
    if train_device.type == 'cuda' and torch.cuda.is_available():
        cleanup_info['gpu_allocated_mb_after_large_model_release'] = (
            torch.cuda.memory_allocated(train_device) / (1024.0 * 1024.0)
        )
        cleanup_info['gpu_reserved_mb_after_large_model_release'] = (
            torch.cuda.memory_reserved(train_device) / (1024.0 * 1024.0)
        )
    return proxy_actor, cleanup_info


def make_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output_dir = base_dir / f'model_size_direct_test_{stamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_direct_test(args: argparse.Namespace) -> Dict[str, Any]:
    patch_expensive_size_logging()
    patch_linear_with_fbs_dtype_alignment()

    family = sweep.FAMILY_CONFIGS[args.family]
    requested_model_dir = Path(args.model_dir) if args.model_dir else Path(family.default_model_dir)
    model_dir = sweep.resolve_model_dir_path(requested_model_dir)
    build_device = torch.device(args.build_device)
    train_device = torch.device(args.train_device)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    target_original_model_mb = args.target_original_gb * 1024.0
    budget_mb = args.budget_gb * 1024.0

    resource_info: Dict[str, Any] = {}
    resource_info.update(sweep.read_host_memory_mb())
    resource_info.update(sweep.read_gpu_memory_mb(train_device))
    resource_info.update(
        {
            'family': family.name,
            'model_dir': str(model_dir),
            'build_device': str(build_device),
            'train_device': str(train_device),
            'dtype': args.dtype,
            'target_original_model_gb': args.target_original_gb,
            'target_original_model_mb': target_original_model_mb,
            'budget_gb': args.budget_gb,
            'budget_mb': budget_mb,
            'max_sparsity': args.sparsity,
            'train_batch_size': args.train_batch_size,
            'fbs_r': args.fbs_r,
        }
    )

    print(
        '[direct-test] '
        f'target_original_model_mb={target_original_model_mb:.2f} '
        f'budget_mb={budget_mb:.2f} '
        f'sparsity={args.sparsity:.2f} '
        f'build_device={build_device} '
        f'train_device={train_device}'
    )

    actor = family.actor_builder(model_dir, build_device)
    inflation_info = inflate_original_model_to_target(
        actor,
        target_original_model_mb=target_original_model_mb,
    )
    print(
        '[direct-test] '
        f"base_original_model_mb={inflation_info['base_original_model_mb']:.2f} "
        f"actual_original_model_mb={inflation_info['actual_original_model_mb']:.2f} "
        f"added_language_layers={inflation_info['added_language_layers']} "
        f"target_error_mb={inflation_info['target_error_mb']:.2f}"
    )

    actor = actor.to(build_device)
    set_actor_runtime_device(actor, build_device)
    actor, layer_info = sweep.convert_actor_to_fbs(
        actor,
        family,
        build_device,
        max_sparsity=args.sparsity,
        fbs_r=args.fbs_r,
        dtype=dtype,
    )
    print('building small model...')
    proxy_actor = build_small_actor_reusing_added_layers(actor, layer_info, inflation_info)
    del actor
    proxy_actor, cleanup_info = release_original_model_before_training(
        proxy_actor,
        build_device=build_device,
        train_device=train_device,
    )
    print(
        '[direct-test] released original model before proxy training: '
        f"gpu_allocated_mb={sweep.format_mb(cleanup_info.get('gpu_allocated_mb_after_large_model_release'))} "
        f"gpu_reserved_mb={sweep.format_mb(cleanup_info.get('gpu_reserved_mb_after_large_model_release'))}"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    proxy_model_mb = tensor_storage_size_mb(proxy_actor)
    compression_ratio = (
        inflation_info['actual_original_model_mb'] / proxy_model_mb
        if proxy_model_mb > 0
        else None
    )

    set_actor_runtime_device(proxy_actor, train_device)
    proxy_actor = sweep.prepare_actor_for_training(proxy_actor, train_device, dtype)
    train_metrics = sweep.run_peak_training_step(proxy_actor, family, train_device, args.train_batch_size)
    actual_peak_train_memory_mb = train_metrics.get('peak_train_memory_mb')
    within_budget = (
        actual_peak_train_memory_mb is not None
        and train_metrics.get('status') == 'completed'
        and actual_peak_train_memory_mb <= budget_mb
    )

    result = {
        **resource_info,
        **inflation_info,
        'proxy_model_mb': proxy_model_mb,
        'compression_ratio': compression_ratio,
        'compatible_layer_count': layer_info['compatible_layer_count'],
        'model_resident_mb_before_train': train_metrics.get('model_resident_mb_before_train'),
        'actual_peak_train_memory_mb': actual_peak_train_memory_mb,
        'train_overhead_mb': train_metrics.get('train_overhead_mb'),
        **cleanup_info,
        'within_budget': within_budget,
        'status': train_metrics.get('status'),
        'error': train_metrics.get('error'),
    }

    del proxy_actor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_result(result: Dict[str, Any], output_dir: Path) -> None:
    print('')
    print('Direct proxy training validation')
    print(f"max_memory_budget_gb={result['budget_gb']:.2f}")
    print(f"max_memory_budget_mb={result['budget_mb']:.2f}")
    print(f"target_original_model_gb={result['target_original_model_gb']:.2f}")
    print(f"target_original_model_mb={result['target_original_model_mb']:.2f}")
    print(f"actual_original_model_mb={result['actual_original_model_mb']:.2f}")
    print(f"proxy_model_mb={result['proxy_model_mb']:.2f}")
    print(f"large_model_released_before_training={result['large_model_released_before_training']}")
    print(f"gpu_allocated_mb_after_large_model_release={sweep.format_mb(result.get('gpu_allocated_mb_after_large_model_release'))}")
    print(f"actual_peak_train_memory_mb={sweep.format_mb(result.get('actual_peak_train_memory_mb'))}")
    print(f"within_budget={result['within_budget']}")
    print(f"status={result['status']}")
    if result.get('error'):
        print(f"error={result['error']}")

    print(
        '[direct-test-result] '
        f"max_memory_budget_mb={result['budget_mb']:.2f} "
        f"target_original_model_mb={result['target_original_model_mb']:.2f} "
        f"actual_original_model_mb={result['actual_original_model_mb']:.2f} "
        f"proxy_model_mb={result['proxy_model_mb']:.2f} "
        f"large_model_released_before_training={result['large_model_released_before_training']} "
        f"gpu_allocated_mb_after_large_model_release={sweep.format_mb(result.get('gpu_allocated_mb_after_large_model_release'))} "
        f"actual_peak_train_memory_mb={sweep.format_mb(result.get('actual_peak_train_memory_mb'))} "
        f"within_budget={result['within_budget']} "
        f"status={result['status']}"
    )
    print(f'[summary] result_json={output_dir / "direct_test_summary.json"}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Directly test whether a target-size original model proxy can train within a memory budget.'
    )
    parser.add_argument('--family', choices=sorted(sweep.FAMILY_CONFIGS.keys()), default='tinyvla')
    parser.add_argument('--model-dir', default='')
    parser.add_argument('--target-original-gb', type=float, default=11.3)
    parser.add_argument('--budget-gb', type=float, default=32.0)
    parser.add_argument('--sparsity', type=float, default=0.90)
    parser.add_argument('--train-batch-size', type=int, default=2)
    parser.add_argument('--fbs-r', type=int, default=16)
    parser.add_argument('--build-device', default='cpu')
    parser.add_argument('--train-device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--output-dir', default='discussion/results')
    args = parser.parse_args()

    output_dir = make_output_dir(Path(args.output_dir))
    result = run_direct_test(args)
    (output_dir / 'direct_test_summary.json').write_text(
        json.dumps(result, indent=2) + '\n',
        encoding='utf-8',
    )
    print_result(result, output_dir)


if __name__ == '__main__':
    main()
