"""Granularity-aware static small-model generation.

This module keeps the sampling, scheduling, feedback, and regeneration contract of
``SmallModelScalingInterface``.  It only changes the retained-neuron plan used by
the static FBS materializer.  The implementation intentionally lives under ``api`` so
the examples can be copied and run without importing the source tree directly.
"""

from __future__ import annotations

import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from api.small_model_scaling_interface import SmallModelScalingInterface

API_DIR = Path(__file__).resolve().parent
VENDOR_DIR = API_DIR / "vendor"
for path in (API_DIR, VENDOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_LAYER_RE = re.compile(r"(?P<prefix>.*?)(?:\.blocks|\.layers)\.(?P<index>\d+)(?:\.|$)")


def _aggregate_scores(scores: torch.Tensor) -> torch.Tensor:
    """Reduce a cached FBS score tensor to one score per output neuron."""
    scores = torch.as_tensor(scores)
    if scores.ndim == 1:
        return scores.detach().to(torch.float32)
    return scores.detach().to(torch.float32).mean(dim=0)


def _layer_key(name: str) -> Tuple[str, int, str]:
    """Return a stable family/index key for vision and language transformer layers."""
    match = _LAYER_RE.search(name)
    if match is None:
        return (name, -1, name)
    prefix = match.group("prefix")
    index = int(match.group("index"))
    return (prefix, index, f"{prefix}layer{index}")


class GranularitySmallModelScalingInterface(SmallModelScalingInterface):
    """Base class for layer- and block-grained generation strategies.

    ``max_sparsity`` is interpreted as the fraction of groups that are reduced to 2%
    of their neurons.  The complementary top-scoring groups are retained in full.
    """

    granularity_name = "layer"

    def __init__(self, *, low_group_retention: float = 0.02) -> None:
        if not 0.0 < low_group_retention <= 1.0:
            raise ValueError("low_group_retention must be in (0, 1]")
        self.low_group_retention = float(low_group_retention)

    def group_fbs_layers(self, fbs_layers: Sequence[str]) -> Mapping[str, List[str]]:
        """Group FBS module paths.  Subclasses override this for block grouping."""
        groups: Dict[str, List[str]] = {}
        for name in fbs_layers:
            family, index, key = _layer_key(name)
            if index < 0:
                key = name
            groups.setdefault(key, []).append(name)
        return groups

    def _ordered_groups(self, groups: Mapping[str, Sequence[str]]) -> List[str]:
        def sort_key(key: str) -> Tuple[str, int, str]:
            family, index, _ = _layer_key(key)
            return family, index, key

        return sorted(groups, key=sort_key)

    @staticmethod
    def _validate_max_sparsity(max_sparsity: float) -> float:
        value = float(max_sparsity)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"max_sparsity must be in [0, 1], got {max_sparsity}")
        return value

    def score_groups(
        self,
        actor: nn.Module,
        groups: Mapping[str, Sequence[str]],
    ) -> Dict[str, float]:
        """Compute the mean cached FBS score for every group."""
        result: Dict[str, float] = {}
        for group, names in groups.items():
            layer_scores: List[torch.Tensor] = []
            for name in names:
                module = self._get_fbs_module(actor, name)
                if module.cached_raw_w is None:
                    raise RuntimeError(f"FBS caches are empty for {name}; run a forward pass first")
                layer_scores.append(_aggregate_scores(module.cached_raw_w).reshape(-1))
            result[group] = float(torch.cat(layer_scores).mean().item()) if layer_scores else 0.0
        return result

    @staticmethod
    def _get_fbs_module(actor: nn.Module, name: str) -> nn.Module:
        from ours.utils.dl.common.model import get_module

        module = get_module(actor, name)
        language_model = getattr(getattr(actor, "vla", None), "language_model", None)
        if module is None and language_model is not None:
            module = get_module(language_model, name)
        if module is None:
            raise KeyError(f"FBS layer not found: {name}")
        return module

    def select_high_score_groups(
        self,
        group_scores: Mapping[str, float],
        max_sparsity: float,
    ) -> List[str]:
        """Select the highest-scoring retained groups, with deterministic tie breaks."""
        sparsity = self._validate_max_sparsity(max_sparsity)
        keys = list(group_scores)
        if not keys:
            return []
        retained_count = max(0, int(math.ceil((1.0 - sparsity) * len(keys))))
        retained_count = min(retained_count, len(keys))
        return sorted(keys, key=lambda key: (-float(group_scores[key]), key))[:retained_count]

    def build_retention_plan(
        self,
        actor: nn.Module,
        fbs_layers: Sequence[str],
        *,
        max_sparsity: float,
        previous_pruning_info: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, bool], Dict[str, float], Dict[str, List[str]]]:
        groups = {key: list(value) for key, value in self.group_fbs_layers(fbs_layers).items()}
        scores = self.score_groups(actor, groups)
        previous_groups = (previous_pruning_info or {}).get("retained_groups")
        if previous_groups:
            retained = [key for key in self._ordered_groups(groups) if key in set(previous_groups)]
            if not retained:
                retained = self.select_high_score_groups(scores, max_sparsity)
        else:
            retained = self.select_high_score_groups(scores, max_sparsity)
        retained_set = set(retained)
        plan = {name: group in retained_set for group, names in groups.items() for name in names}
        return plan, scores, groups

    @contextmanager
    def _patched_fbs_retention(
        self,
        actor: nn.Module,
        plan: Mapping[str, bool],
    ):
        """Temporarily make the reference generator materialize our retention plan."""
        originals: List[Tuple[Any, Any, Any]] = []
        try:
            for name, retain_full in plan.items():
                module = self._get_fbs_module(actor, name)
                raw_scores = _aggregate_scores(module.cached_raw_w)
                total = int(raw_scores.numel())
                keep = total if retain_full else max(1, int(math.ceil(total * self.low_group_retention)))
                selected = torch.argsort(raw_scores, descending=True, stable=True)[:keep]
                old_k = module.k_takes_all.k
                old_cached_w = module.cached_w
                raw_k = (total - keep) / max(total, 1)
                # ``_expected_kept_count`` uses ``int(total * k)``; nextafter keeps
                # that truncation from dropping one extra channel due to rounding.
                module.k_takes_all.k = (
                    math.nextafter(raw_k, math.inf) if 0.0 < raw_k < 1.0 else raw_k
                )
                # The reference generator uses non-zero cached_w entries as candidates.
                cached = torch.zeros_like(module.cached_raw_w)
                cached[..., selected] = module.cached_raw_w[..., selected]
                module.cached_w = cached
                originals.append((module, old_k, old_cached_w))
            yield
        finally:
            for module, old_k, old_cached_w in originals:
                module.k_takes_all.k = old_k
                module.cached_w = old_cached_w

    def _generate_grained_small_model(
        self,
        actor: nn.Module,
        *,
        sample_batch: Optional[Dict[str, Any]],
        device: torch.device,
        dtype: torch.dtype,
        max_sparsity: float,
        previous_pruning_info: Optional[dict],
        regeneration_increment_ratio: float,
        verify: bool,
    ) -> Tuple[nn.Module, dict]:
        from train.vla_adapter_new.ours import generate_static_small_model as reference_generator
        from ours.libs.gen_neuron_index import get_fbs_layers
        from ours.utils.common.data import flatten_2d_arr

        prepared_sample = None
        if sample_batch is not None:
            prepared_sample = reference_generator._materialize_fbs_caches(actor, sample_batch)
        groups_spec = reference_generator._collect_transformer_layer_groups(actor)
        all_fbs_layers: List[str] = []
        for layer_group in groups_spec:
            all_fbs_layers.extend(
                get_fbs_layers(
                    flatten_2d_arr(layer_group[0]),
                    flatten_2d_arr(layer_group[1]),
                    layer_group[2],
                    layer_group[3],
                )
            )
        plan, scores, groups = self.build_retention_plan(
            actor,
            all_fbs_layers,
            max_sparsity=max_sparsity,
            previous_pruning_info=previous_pruning_info,
        )
        generation_previous_pruning_info = previous_pruning_info
        generation_increment_ratio = regeneration_increment_ratio
        if getattr(self, "enforce_grouped_channels", False):
            # The reference merger operates on individual channels.  Grouped
            # strategies (attention heads) keep their group decision from above
            # and must not let that merger split a group during regeneration.
            generation_previous_pruning_info = None
            generation_increment_ratio = 1.0
        with self._patched_fbs_retention(actor, plan):
            small_model, pruning_info = reference_generator._generate_static_small_model_internal(
                actor,
                previous_pruning_info=generation_previous_pruning_info,
                regeneration_increment_ratio=generation_increment_ratio,
            )
        pruning_info["granularity"] = self.granularity_name
        pruning_info["group_scores"] = scores
        pruning_info["groups"] = {key: list(value) for key, value in groups.items()}
        # A specialized granularity may use a per-channel plan internally (for
        # example, attention heads).  Such implementations can expose the exact
        # retained group keys; the default remains the layer-plan projection.
        retained_group_names = getattr(self, "_retained_group_names", None)
        if retained_group_names is None:
            retained_group_names = {key for key, names in groups.items() if any(plan[name] for name in names)}
        pruning_info["retained_groups"] = [key for key in groups if key in retained_group_names]
        small_model.to(device=device, dtype=dtype)
        small_model.device = device
        if hasattr(small_model, "vla"):
            small_model.vla.to(device=device, dtype=dtype)
        for module_name in ("state_projector", "context_projector", "actor_head", "value_head"):
            if hasattr(small_model, module_name):
                getattr(small_model, module_name).to(device=device, dtype=torch.float32)
        if hasattr(small_model, "action_bin_centers") and "action_bin_centers" in small_model._buffers:
            small_model._buffers["action_bin_centers"] = small_model.action_bin_centers.to(
                device=device, dtype=torch.float32
            )
        if verify:
            reference_generator._verify_static_small_model(actor, small_model, prepared_sample)
        return small_model, pruning_info

    def generate_initial_small_model(self, *, large_agent, args, eval_envs, device, adapter, reference_api):
        sample = self.collect_sample_for_small_model_scaling(
            args, large_agent=large_agent, small_agent=None, eval_envs=eval_envs,
            device=device, adapter=adapter, reference_api=reference_api,
        )
        small_agent, pruning_info = self._generate_grained_small_model(
            large_agent, sample_batch=sample, device=device, dtype=torch.bfloat16,
            max_sparsity=args.max_sparsity, previous_pruning_info=None,
            regeneration_increment_ratio=1.0, verify=True,
        )
        self._prepare_generated_small_model(small_agent, args=args, device=device, adapter=adapter, initial_generation=True)
        self.after_small_model_scaling(
            large_agent=large_agent, small_agent=small_agent, sample_batch=sample,
            pruning_info=pruning_info, optimizer=None, args=args, device=device,
            adapter=adapter, reference_api=reference_api,
        )
        return small_agent, pruning_info

    def regenerate_small_model_in_place(self, *, large_agent, small_agent, current_pruning_info,
                                         optimizer, args, eval_envs, device, adapter, reference_api):
        from train.vla_adapter_new.ours.generate_static_small_model import inherit_static_small_model_retained_channels

        sample = self.collect_sample_for_small_model_scaling(
            args, large_agent=large_agent, small_agent=small_agent, eval_envs=eval_envs,
            device=device, adapter=adapter, reference_api=reference_api,
        )
        regenerated, pruning_info = self._generate_grained_small_model(
            large_agent, sample_batch=sample, device=device, dtype=torch.bfloat16,
            max_sparsity=args.max_sparsity, previous_pruning_info=current_pruning_info,
            regeneration_increment_ratio=args.small_model_regeneration_increment_ratio,
            verify=True,
        )
        self._prepare_generated_small_model(regenerated, args=args, device=device, adapter=adapter, initial_generation=False)
        if args.small_model_regeneration_increment_ratio < 1.0 and current_pruning_info is not None:
            inherit_static_small_model_retained_channels(regenerated, small_agent, pruning_info, current_pruning_info)
        small_agent.load_state_dict(regenerated.state_dict(), strict=True)
        if args.reset_optimizer_after_regeneration:
            optimizer.zero_grad(set_to_none=True)
            for parameter in small_agent.parameters():
                optimizer.state.pop(parameter, None)
        self.after_small_model_scaling(
            large_agent=large_agent, small_agent=small_agent, sample_batch=sample,
            pruning_info=pruning_info, optimizer=optimizer, args=args, device=device,
            adapter=adapter, reference_api=reference_api,
        )
        return pruning_info


KnowledgeExchangeGranularityInterface = GranularitySmallModelScalingInterface

__all__ = [
    "GranularitySmallModelScalingInterface",
    "KnowledgeExchangeGranularityInterface",
]
