"""Attention-head-grained static small-model generation.

Attention QKV outputs are selected in complete ``head_dim``-sized chunks.  All
other FBS layers use the same complete-layer grouping as ``layer_grained``.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.knowledge_exchange_granularity_interface import (
    GranularitySmallModelScalingInterface,
    _aggregate_scores,
)


def _without_fbs_suffix(name: str) -> str:
    return name[:-2] if name.endswith(".0") else name


def _qkv_kind(name: str) -> Optional[str]:
    base = _without_fbs_suffix(name)
    if base.endswith(".attn.qkv"):
        return "qkv"
    for kind in ("q", "k", "v"):
        if base.endswith(f".self_attn.{kind}_proj"):
            return kind
    return None


class AttentionHeadGrainedSmallModelScalingInterface(GranularitySmallModelScalingInterface):
    """Select complete attention heads while retaining layer granularity elsewhere."""

    granularity_name = "attention_head"
    enforce_grouped_channels = True

    def __init__(self, *, low_group_retention: float = 0.02) -> None:
        # Kept for API compatibility with the layer/block implementations.  QKV
        # groups are either fully selected or omitted; this value applies only to
        # non-attention layers.
        super().__init__(low_group_retention=low_group_retention)
        self._head_group_indices: Dict[str, Dict[str, torch.Tensor]] = {}
        self._head_layer_names: set[str] = set()
        self._retained_group_names: set[str] = set()

    @staticmethod
    def _module_out_features(module: torch.nn.Module) -> int:
        raw_linear = getattr(module, "raw_linear", None)
        if raw_linear is None:
            raise TypeError(f"FBS module {type(module).__name__} has no raw_linear")
        return int(raw_linear.out_features)

    @staticmethod
    def _module_metadata(actor: torch.nn.Module, name: str, out_features: int) -> Tuple[int, int]:
        """Infer ``(head_dim, number_of_heads)`` from the owning attention module.

        The VLA language model and vision backbone use slightly different module
        layouts, so inspect the longest matching ancestor first and then fall back
        to common HuggingFace config fields.
        """
        base = _without_fbs_suffix(name)
        candidates: List[Tuple[str, torch.nn.Module]] = []
        for root in (actor, getattr(getattr(actor, "vla", None), "language_model", None)):
            if root is None:
                continue
            for module_name, module in root.named_modules():
                if module_name and (base == module_name or base.startswith(module_name + ".")):
                    candidates.append((module_name, module))
        candidates.sort(key=lambda item: len(item[0]), reverse=True)

        head_dim = None
        num_heads = None
        for _, module in candidates:
            for attr in ("head_dim", "attention_head_dim"):
                value = getattr(module, attr, None)
                if isinstance(value, int) and value > 0:
                    head_dim = value
                    break
            for attr in ("num_heads", "num_attention_heads", "n_heads"):
                value = getattr(module, attr, None)
                if isinstance(value, int) and value > 0:
                    num_heads = value
                    break
            if head_dim is not None:
                break

        language_model = getattr(getattr(actor, "vla", None), "language_model", None)
        config = getattr(language_model, "config", None)
        if head_dim is None and config is not None:
            value = getattr(config, "head_dim", None)
            if isinstance(value, int) and value > 0:
                head_dim = value
        if num_heads is None and config is not None:
            value = getattr(config, "num_attention_heads", None)
            if isinstance(value, int) and value > 0:
                num_heads = value
        if head_dim is None and num_heads is not None and out_features % num_heads == 0:
            # A combined vision qkv has three projections; account for that before
            # using the standard hidden_size / num_heads relationship.
            projection_width = out_features // 3 if _qkv_kind(name) == "qkv" and out_features % 3 == 0 else out_features
            if projection_width % num_heads == 0:
                head_dim = projection_width // num_heads
        if head_dim is None:
            raise ValueError(
                f"unable to infer head_dim for attention QKV layer {name!r}; "
                "the owning attention module must expose head_dim"
            )
        if num_heads is None:
            projection_width = out_features // 3 if _qkv_kind(name) == "qkv" and out_features % 3 == 0 else out_features
            num_heads = max(1, projection_width // head_dim)
        return int(head_dim), int(num_heads)

    @staticmethod
    def _attention_family(name: str) -> str:
        base = _without_fbs_suffix(name)
        for suffix in (".attn.qkv", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    @staticmethod
    def _head_ranges(name: str, out_features: int, head_dim: int, head_index: int) -> List[Tuple[int, int]]:
        kind = _qkv_kind(name)
        if kind == "qkv":
            if out_features % 3 != 0:
                raise ValueError(f"combined QKV layer {name!r} has output width {out_features}, not divisible by 3")
            projection_width = out_features // 3
            start = head_index * head_dim
            if start + head_dim > projection_width:
                return []
            return [(offset + start, offset + start + head_dim) for offset in range(0, out_features, projection_width)]
        start = head_index * head_dim
        return [(start, start + head_dim)] if start + head_dim <= out_features else []

    def _qkv_groups(
        self,
        actor: torch.nn.Module,
        names: Sequence[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
        groups: Dict[str, List[str]] = {}
        indices: Dict[str, Dict[str, torch.Tensor]] = {}
        scores: Dict[str, float] = {}
        by_family: Dict[str, List[str]] = {}
        for name in names:
            if _qkv_kind(name) is not None:
                by_family.setdefault(self._attention_family(name), []).append(name)

        for family, family_names in by_family.items():
            per_head: Dict[int, List[Tuple[str, torch.Tensor, torch.Tensor]]] = {}
            for name in family_names:
                module = self._get_fbs_module(actor, name)
                raw_scores = _aggregate_scores(module.cached_raw_w).reshape(-1)
                head_dim, _ = self._module_metadata(actor, name, int(raw_scores.numel()))
                out_features = self._module_out_features(module)
                # The head count is derived from the actual output width.  This
                # naturally handles grouped-query attention where K/V have fewer
                # heads than Q.
                projection_width = out_features // 3 if _qkv_kind(name) == "qkv" else out_features
                head_count = projection_width // head_dim
                if head_count == 0 or projection_width % head_dim:
                    # SVD/FBS adapters can expose a rank smaller than one native
                    # head.  Keep that rank atomic rather than silently dropping
                    # its channels from the retention plan.
                    head_count = 1
                    head_dim = projection_width
                for head_index in range(head_count):
                    ranges = self._head_ranges(name, out_features, head_dim, head_index)
                    if not ranges:
                        continue
                    selected = torch.cat([torch.arange(start, end, device=raw_scores.device) for start, end in ranges])
                    per_head.setdefault(head_index, []).append((name, selected, raw_scores[selected]))
            for head_index, members in per_head.items():
                key = f"{family}head{head_index}"
                groups[key] = [name for name, _, _ in members]
                score_parts = []
                for name, selected, member_scores in members:
                    indices.setdefault(key, {})[name] = selected
                    score_parts.append(member_scores)
                scores[key] = float(torch.cat(score_parts).mean().item())
        return groups, indices, scores

    def build_retention_plan(
        self,
        actor: torch.nn.Module,
        fbs_layers: Sequence[str],
        *,
        max_sparsity: float,
        previous_pruning_info: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, bool], Dict[str, float], Dict[str, List[str]]]:
        regular_names = [name for name in fbs_layers if _qkv_kind(name) is None]
        qkv_names = [name for name in fbs_layers if _qkv_kind(name) is not None]
        groups: Dict[str, List[str]] = {}
        scores: Dict[str, float] = {}
        head_indices: Dict[str, Dict[str, torch.Tensor]] = {}
        head_groups, head_indices, head_scores = self._qkv_groups(actor, qkv_names)
        groups.update(head_groups)
        scores.update(head_scores)

        layer_groups = {key: list(value) for key, value in super().group_fbs_layers(regular_names).items()}
        groups.update(layer_groups)
        scores.update(self.score_groups(actor, layer_groups))

        previous_groups = (previous_pruning_info or {}).get("retained_groups")
        if previous_groups:
            retained = [key for key in self._ordered_groups(groups) if key in set(previous_groups)]
            if not retained:
                retained = self.select_high_score_groups(scores, max_sparsity)
        else:
            retained = self.select_high_score_groups(scores, max_sparsity)
        retained_set = set(retained)

        # At least one complete head per QKV module keeps the generated linear
        # layers valid even at very high sparsity.
        for name in qkv_names:
            if not any(name in groups[key] and key in retained_set for key in head_groups):
                candidates = [key for key in head_groups if name in head_groups[key]]
                if candidates:
                    retained_set.add(max(candidates, key=lambda key: (scores[key], key)))

        self._head_group_indices = {
            key: {name: value.detach() for name, value in members.items()}
            for key, members in head_indices.items()
            if key in retained_set
        }
        self._retained_group_names = retained_set
        self._head_layer_names = set(qkv_names)
        plan = {
            name: group in retained_set
            for group, names in groups.items()
            for name in set(names)
            if name not in self._head_layer_names
        }
        # QKV entries are handled by exact head indices in _patched_fbs_retention.
        plan.update({name: True for name in qkv_names})
        return plan, scores, groups

    @contextmanager
    def _patched_fbs_retention(self, actor: torch.nn.Module, plan: Mapping[str, bool]):
        originals: List[Tuple[Any, Any, Any]] = []
        try:
            retained_by_layer: Dict[str, List[torch.Tensor]] = {}
            for group_members in self._head_group_indices.values():
                for name, selected in group_members.items():
                    retained_by_layer.setdefault(name, []).append(selected)
            for name, retain_full in plan.items():
                module = self._get_fbs_module(actor, name)
                raw_scores = _aggregate_scores(module.cached_raw_w).reshape(-1)
                total = int(raw_scores.numel())
                if name in self._head_layer_names:
                    selected = torch.unique(torch.cat(retained_by_layer.get(name, [])), sorted=True)
                    keep = int(selected.numel())
                else:
                    keep = total if retain_full else max(1, int(math.ceil(total * self.low_group_retention)))
                    selected = torch.argsort(raw_scores, descending=True, stable=True)[:keep]
                if keep <= 0:
                    raise RuntimeError(f"attention-head selection removed every channel from {name}")
                old_k = module.k_takes_all.k
                old_cached_w = module.cached_w
                raw_k = (total - keep) / max(total, 1)
                module.k_takes_all.k = math.nextafter(raw_k, math.inf) if 0.0 < raw_k < 1.0 else raw_k
                cached = torch.zeros_like(module.cached_raw_w)
                cached[..., selected] = module.cached_raw_w[..., selected]
                module.cached_w = cached
                originals.append((module, old_k, old_cached_w))
            yield
        finally:
            for module, old_k, old_cached_w in originals:
                module.k_takes_all.k = old_k
                module.cached_w = old_cached_w


def make_attention_head_grained_interface() -> AttentionHeadGrainedSmallModelScalingInterface:
    return AttentionHeadGrainedSmallModelScalingInterface()


# Short aliases used by callers that name the granularity simply ``head``.
HeadGrainedSmallModelScalingInterface = AttentionHeadGrainedSmallModelScalingInterface


def make_head_grained_interface() -> AttentionHeadGrainedSmallModelScalingInterface:
    return make_attention_head_grained_interface()


def main() -> None:
    from api.unified_online_rl import parse_args, run_training
    from api.vla_model_interface_examples._reference_adapter import make_vla_adapter

    run_training(make_vla_adapter(), parse_args(), make_attention_head_grained_interface())


if __name__ == "__main__":
    main()
