from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parent
REPO_ROOT = EVAL_ROOT.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from ours.libs.gen_scaling_law_data_points import generate_small_model_v2
from ours.libs.train_with_fbs.lib_cnn import get_model_size
from ours.pretrain_fbs_model.main import add_FBS_into_transformer
from train.tinyvla.ours.model_with_fbs import _language_model_layers, _vision_transformer_layers


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    default_model_dir: str
    rgb_shape: tuple[int, int, int, int]
    actor_builder: Callable[[Path, torch.device], torch.nn.Module]


def _build_tinyvla_actor(model_dir: Path, device: torch.device) -> torch.nn.Module:
    from train.tinyvla.model_impl.online_rl_open_cabinet_drawer import DEFAULT_MODEL_DIR, EdgeVLAActorCritic

    resolved = model_dir if str(model_dir) else Path(DEFAULT_MODEL_DIR)
    return EdgeVLAActorCritic(resolved, device=device)


def _build_vla_adapter_actor(model_dir: Path, device: torch.device) -> torch.nn.Module:
    from train.vla_adapter_new.model_impl.online_rl_hold_cube_in_hand import (
        DEFAULT_MODEL_DIR,
        HandVLAAdapterActorCritic,
    )

    resolved = model_dir if str(model_dir) else Path(DEFAULT_MODEL_DIR)
    return HandVLAAdapterActorCritic(resolved, device=device)


FAMILY_CONFIGS: Dict[str, FamilyConfig] = {
    'tinyvla': FamilyConfig(
        name='tinyvla',
        default_model_dir='eval/ckpt/vla_adapter_new/LIBERO-Object',
        rgb_shape=(1, 224, 448, 3),
        actor_builder=_build_tinyvla_actor,
    ),
    'edgevla': FamilyConfig(
        name='edgevla',
        default_model_dir='eval/ckpt/vla_adapter_new/LIBERO-Object',
        rgb_shape=(1, 224, 448, 3),
        actor_builder=_build_tinyvla_actor,
    ),
    'vla_adapter_new': FamilyConfig(
        name='vla_adapter_new',
        default_model_dir='eval/ckpt/vla_adapter_new/LIBERO-Object',
        rgb_shape=(1, 224, 224, 3),
        actor_builder=_build_vla_adapter_actor,
    ),
}


def resolve_model_dir_path(model_dir: Path) -> Path:
    if model_dir.is_absolute():
        return model_dir
    candidates = [
        Path.cwd() / model_dir,
        REPO_ROOT / model_dir,
        EVAL_ROOT / model_dir,
    ]
    model_dir_str = model_dir.as_posix()
    if model_dir_str.startswith('eval/'):
        trimmed = Path(model_dir_str[len('eval/'):])
        candidates.extend([
            REPO_ROOT / trimmed,
            EVAL_ROOT / trimmed,
        ])
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (REPO_ROOT / model_dir).resolve()


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError('expected at least one numeric value')
    return values


def format_mb(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.2f}'


def format_ratio(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.3f}x'


def read_host_memory_mb() -> Dict[str, float | None]:
    result: Dict[str, float | None] = {
        'host_mem_total_mb': None,
        'host_mem_available_mb': None,
    }
    meminfo = Path('/proc/meminfo')
    if not meminfo.is_file():
        return result
    raw: Dict[str, float] = {}
    for line in meminfo.read_text(encoding='utf-8').splitlines():
        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            value_kb = float(fields[0])
        except ValueError:
            continue
        raw[key] = value_kb / 1024.0
    result['host_mem_total_mb'] = raw.get('MemTotal')
    result['host_mem_available_mb'] = raw.get('MemAvailable')
    return result


def read_gpu_memory_mb(device: torch.device) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {
        'gpu_mem_total_mb': None,
        'gpu_mem_free_mb': None,
    }
    if device.type != 'cuda' or not torch.cuda.is_available():
        return result
    index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    result['gpu_mem_total_mb'] = props.total_memory / (1024.0 * 1024.0)
    try:
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        result['gpu_mem_free_mb'] = free_bytes / (1024.0 * 1024.0)
        result['gpu_mem_total_mb'] = total_bytes / (1024.0 * 1024.0)
    except Exception:
        pass
    return result


def make_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output_dir = base_dir / f'model_size_limit_{stamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_sample_batch(actor: torch.nn.Module, family: FamilyConfig, device: torch.device) -> Dict[str, Any]:
    return {
        'rgbs': torch.randint(0, 256, family.rgb_shape, dtype=torch.uint8),
        'states': torch.randn(1, int(actor.state_dim), device=device, dtype=torch.float32),
        'input_ids': torch.randint(0, min(int(actor.full_vocab_size), 1000), (1, 16), device=device, dtype=torch.long),
    }


def build_training_batch(actor: torch.nn.Module, family: FamilyConfig, device: torch.device, batch_size: int) -> Dict[str, Any]:
    rgb_shape = (batch_size,) + tuple(family.rgb_shape[1:])
    rgbs = np.random.randint(0, 256, size=rgb_shape, dtype=np.uint8)
    states = np.random.randn(batch_size, int(actor.state_dim)).astype(np.float32)
    if hasattr(actor, 'policy_action_dim'):
        action_dim = int(actor.policy_action_dim)
    else:
        action_dim = int(actor.env_action_dim)
    action_bins = torch.randint(0, int(actor.num_action_bins), (batch_size, action_dim), device=device, dtype=torch.long)
    returns = torch.randn(batch_size, device=device, dtype=torch.float32)
    return {
        'rgbs': rgbs,
        'states': states,
        'action_bins': action_bins,
        'returns': returns,
    }


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=dtype)
    return contextlib.nullcontext()


def flatten_layer_names(items: Sequence[Any]) -> List[str]:
    flattened: List[str] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_layer_names(item))
        else:
            flattened.append(str(item))
    return flattened


def convert_actor_to_fbs(
    actor: torch.nn.Module,
    family: FamilyConfig,
    device: torch.device,
    *,
    max_sparsity: float,
    fbs_r: int,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    sample_batch = build_sample_batch(actor, family, device)
    vision_qkv, vision_proj, vision_ff1, vision_ff2 = _vision_transformer_layers(actor)
    lm_qkv, lm_proj, lm_ff1, lm_ff2 = _language_model_layers(actor)

    if vision_qkv:
        with autocast_context(device, dtype):
            actor = add_FBS_into_transformer(
                actor.to(device),
                vision_qkv,
                vision_proj,
                vision_ff1,
                vision_ff2,
                sample_batch,
                max_sparsity,
                fbs_r,
                lambda model, batch: model(
                    rgbs=batch['rgbs'],
                    states=batch['states'],
                    mode='policy',
                )[0].float().sum(),
                verify_outputs=False,
            ).cpu()

    if lm_qkv:
        with autocast_context(device, dtype):
            actor.vla.language_model = add_FBS_into_transformer(
                actor.vla.language_model.to(device),
                lm_qkv,
                lm_proj,
                lm_ff1,
                lm_ff2,
                sample_batch,
                max_sparsity,
                fbs_r,
                lambda model, batch: model(
                    input_ids=batch['input_ids'].to(device),
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1].float().mean(),
                verify_outputs=False,
            ).cpu()

    actor.vla.to(dtype=dtype)
    return actor, {
        'vision_qkv': vision_qkv,
        'vision_proj': vision_proj,
        'vision_ff1': vision_ff1,
        'vision_ff2': vision_ff2,
        'lm_qkv': lm_qkv,
        'lm_proj': lm_proj,
        'lm_ff1': lm_ff1,
        'lm_ff2': lm_ff2,
        'compatible_layer_count': len(flatten_layer_names(vision_qkv)) + len(flatten_layer_names(lm_qkv)),
    }


def build_small_actor(actor: torch.nn.Module, layer_info: Dict[str, Any]) -> torch.nn.Module:
    small_actor = actor
    if layer_info['vision_qkv']:
        small_actor = generate_small_model_v2(
            small_actor,
            layer_info['vision_qkv'],
            layer_info['vision_proj'],
            layer_info['vision_ff1'],
            layer_info['vision_ff2'],
        )
    if layer_info['lm_qkv'] and hasattr(small_actor, 'vla') and hasattr(small_actor.vla, 'language_model'):
        small_actor.vla.language_model = generate_small_model_v2(
            small_actor.vla.language_model,
            layer_info['lm_qkv'],
            layer_info['lm_proj'],
            layer_info['lm_ff1'],
            layer_info['lm_ff2'],
        )
    return small_actor


def estimate_supported_original_model_mb(original_model_mb: float, peak_train_memory_mb: float | None, budget_mb: float | None) -> float | None:
    if peak_train_memory_mb is None or peak_train_memory_mb <= 0 or budget_mb is None or budget_mb <= 0:
        return None
    return original_model_mb * budget_mb / peak_train_memory_mb


def summarize_budget_rows(results: Sequence[Dict[str, Any]], budgets_gb: Sequence[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        for budget_gb in budgets_gb:
            rows.append(
                {
                    'family': result['family'],
                    'max_sparsity': result['max_sparsity'],
                    'train_batch_size': result['train_batch_size'],
                    'budget_gb': budget_gb,
                    'estimated_supported_original_model_mb': estimate_supported_original_model_mb(
                        float(result['original_model_mb']),
                        result.get('peak_train_memory_mb'),
                        budget_gb * 1024.0,
                    ),
                    'status': result['status'],
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_resource_summary(resource_info: Dict[str, Any]) -> None:
    print('')
    print('Model size limit resource summary')
    print(f"host_mem_total_mb={format_mb(resource_info.get('host_mem_total_mb'))}")
    print(f"host_mem_available_mb={format_mb(resource_info.get('host_mem_available_mb'))}")
    print(f"gpu_mem_total_mb={format_mb(resource_info.get('gpu_mem_total_mb'))}")
    print(f"gpu_mem_free_mb={format_mb(resource_info.get('gpu_mem_free_mb'))}")
    print(f"train_batch_size={resource_info.get('train_batch_size')}")


def print_result_table(results: Sequence[Dict[str, Any]]) -> None:
    print('')
    print('Model size limit summary')
    print('family         sparsity  orig_mb   proxy_mb  peak_train_mb  supported_orig@gpu_total_mb  status')
    for row in results:
        print(
            f"{row['family']:<14}"
            f"{row['max_sparsity']:<10.2f}"
            f"{row['original_model_mb']:<10.2f}"
            f"{row['proxy_model_mb']:<10.2f}"
            f"{format_mb(row.get('peak_train_memory_mb')):<15}"
            f"{format_mb(row.get('estimated_supported_original_model_mb_at_gpu_total')):<29}"
            f"{row['status']}"
        )


def prepare_actor_for_training(actor: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    actor = actor.to(device)
    if hasattr(actor, 'vla'):
        actor.vla.to(device=device, dtype=dtype)
    if hasattr(actor, 'configure_trainable_modules'):
        actor.configure_trainable_modules(True)
    actor.train()
    return actor


def run_peak_training_step(actor: torch.nn.Module, family: FamilyConfig, device: torch.device, batch_size: int) -> Dict[str, Any]:
    training_batch = build_training_batch(actor, family, device, batch_size)
    params = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-4)
    action_dim = int(getattr(actor, 'policy_action_dim', actor.env_action_dim))

    metrics: Dict[str, Any] = {
        'model_resident_mb_before_train': None,
        'peak_train_memory_mb': None,
        'train_overhead_mb': None,
        'status': 'completed',
        'error': '',
    }

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize(device)
        metrics['model_resident_mb_before_train'] = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        torch.cuda.reset_peak_memory_stats(device)

    try:
        optimizer.zero_grad(set_to_none=True)
        _, log_prob, entropy, value, _ = actor.get_action_and_value(
            rgbs=training_batch['rgbs'],
            states=training_batch['states'],
            action_bins=training_batch['action_bins'],
            deterministic=False,
        )
        policy_loss = (-log_prob.mean()) / max(1, action_dim)
        value_loss = F.mse_loss(value, training_batch['returns'])
        entropy_bonus = entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            metrics['peak_train_memory_mb'] = peak_train_memory_mb
            if metrics['model_resident_mb_before_train'] is not None:
                metrics['train_overhead_mb'] = peak_train_memory_mb - metrics['model_resident_mb_before_train']
    except RuntimeError as exc:
        message = str(exc)
        metrics['status'] = 'oom' if 'out of memory' in message.lower() else 'failed'
        metrics['error'] = message
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            try:
                peak_train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                metrics['peak_train_memory_mb'] = peak_train_memory_mb
                if metrics['model_resident_mb_before_train'] is not None:
                    metrics['train_overhead_mb'] = peak_train_memory_mb - metrics['model_resident_mb_before_train']
            except Exception:
                pass
    finally:
        del optimizer
        del training_batch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return metrics


def run_single_sweep(
    family: FamilyConfig,
    model_dir: Path,
    device: torch.device,
    *,
    sparsity: float,
    fbs_r: int,
    dtype: torch.dtype,
    train_batch_size: int,
    resource_info: Dict[str, Any],
) -> Dict[str, Any]:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    actor = family.actor_builder(model_dir, device)
    original_model_mb = float(get_model_size(actor, True))
    actor, layer_info = convert_actor_to_fbs(
        actor,
        family,
        device,
        max_sparsity=sparsity,
        fbs_r=fbs_r,
        dtype=dtype,
    )
    small_actor = build_small_actor(actor, layer_info)
    proxy_model_mb = float(get_model_size(small_actor, True))
    compression_ratio = original_model_mb / proxy_model_mb if proxy_model_mb > 0 else 0.0

    del actor
    gc.collect()

    small_actor = prepare_actor_for_training(small_actor, device, dtype)
    train_metrics = run_peak_training_step(small_actor, family, device, train_batch_size)

    result = {
        'family': family.name,
        'model_dir': str(model_dir),
        'uses_random_init_fallback': not model_dir.is_dir(),
        'max_sparsity': sparsity,
        'train_batch_size': train_batch_size,
        'original_model_mb': original_model_mb,
        'proxy_model_mb': proxy_model_mb,
        'compression_ratio': compression_ratio,
        'compatible_layer_count': layer_info['compatible_layer_count'],
        'model_resident_mb_before_train': train_metrics['model_resident_mb_before_train'],
        'peak_train_memory_mb': train_metrics['peak_train_memory_mb'],
        'train_overhead_mb': train_metrics['train_overhead_mb'],
        'estimated_supported_original_model_mb_at_gpu_total': estimate_supported_original_model_mb(
            original_model_mb,
            train_metrics['peak_train_memory_mb'],
            resource_info.get('gpu_mem_total_mb'),
        ),
        'estimated_supported_original_model_mb_at_gpu_free': estimate_supported_original_model_mb(
            original_model_mb,
            train_metrics['peak_train_memory_mb'],
            resource_info.get('gpu_mem_free_mb'),
        ),
        'status': train_metrics['status'],
        'error': train_metrics['error'],
    }

    del small_actor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep compression ratios for the model-size-limit discussion.')
    parser.add_argument('--family', choices=sorted(FAMILY_CONFIGS.keys()), default='tinyvla')
    parser.add_argument('--model-dir', default='')
    parser.add_argument('--sparsities', default='0.00,0.25,0.50,0.75,0.90')
    parser.add_argument('--budget-gb', default='8,16,24,32,40,48,80')
    parser.add_argument('--train-batch-size', type=int, default=2)
    parser.add_argument('--fbs-r', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--output-dir', default='discussion/results')
    args = parser.parse_args()

    family = FAMILY_CONFIGS[args.family]
    sparsities = parse_float_list(args.sparsities)
    budgets_gb = parse_float_list(args.budget_gb)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    requested_model_dir = Path(args.model_dir) if args.model_dir else Path(family.default_model_dir)
    model_dir = resolve_model_dir_path(requested_model_dir)

    output_dir = make_output_dir(Path(args.output_dir))
    resource_info: Dict[str, Any] = {}
    resource_info.update(read_host_memory_mb())
    resource_info.update(read_gpu_memory_mb(device))
    resource_info['family'] = family.name
    resource_info['device'] = str(device)
    resource_info['dtype'] = args.dtype
    resource_info['model_dir'] = str(model_dir)
    resource_info['train_batch_size'] = args.train_batch_size

    results = [
        run_single_sweep(
            family,
            model_dir,
            device,
            sparsity=sparsity,
            fbs_r=args.fbs_r,
            dtype=dtype,
            train_batch_size=args.train_batch_size,
            resource_info=resource_info,
        )
        for sparsity in sparsities
    ]
    budget_rows = summarize_budget_rows(results, budgets_gb)

    write_csv(output_dir / 'summary.csv', results)
    write_csv(output_dir / 'budget_sweep.csv', budget_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(
            {
                'resource_info': resource_info,
                'results': results,
                'budget_rows': budget_rows,
            },
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    print_resource_summary(resource_info)
    print_result_table(results)
    print('')
    print(f'[summary] summary_csv={output_dir / "summary.csv"}')
    print(f'[summary] budget_csv={output_dir / "budget_sweep.csv"}')
    print(f'[summary] summary_json={output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
