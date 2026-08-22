"""Scaling-method implementations used by the VLA-Adapter example.

The runner intentionally owns rollout collection and PPO.  These strategies only
change how a static small policy is materialized, what data is used, and (for the
distillation variants) the objective applied after materialization.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from api.small_model_scaling_interface import SmallModelScalingInterface


def _randomize_student_parameters(student: nn.Module) -> None:
    for module in student.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()


def _as_device_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def reverse_kl_loss(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    """Compute ``KL(student || teacher)`` for batched logit/log-prob vectors."""
    student = F.log_softmax(student_log_probs.float(), dim=-1)
    teacher = F.log_softmax(teacher_log_probs.float(), dim=-1)
    return F.kl_div(teacher, student, log_target=True, reduction="batchmean")


def distillm_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    reverse_weight: float = 0.5,
) -> torch.Tensor:
    """DistiLLM-style adaptive mixture of forward and reverse KL terms."""
    if not 0.0 <= reverse_weight <= 1.0:
        raise ValueError("reverse_weight must be in [0, 1]")
    student = F.log_softmax(student_log_probs.float(), dim=-1)
    teacher = F.log_softmax(teacher_log_probs.float(), dim=-1)
    forward = F.kl_div(student, teacher, log_target=True, reduction="batchmean")
    reverse = F.kl_div(teacher, student, log_target=True, reduction="batchmean")
    return (1.0 - reverse_weight) * forward + reverse_weight * reverse


def _flatten_matching_features(student: torch.Tensor, teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    student = torch.as_tensor(student).float()
    teacher = torch.as_tensor(teacher).float()
    if student.ndim == 0 or teacher.ndim == 0:
        return student.reshape(1), teacher.reshape(1)
    batch = min(student.shape[0], teacher.shape[0])
    student = student[:batch].reshape(batch, -1)
    teacher = teacher[:batch].reshape(batch, -1)
    width = min(student.shape[-1], teacher.shape[-1])
    if width == 0:
        return student[:, :0], teacher[:, :0]
    return student[:, :width], teacher[:, :width]


def feature_distillation_loss(student_features: Iterable[torch.Tensor], teacher_features: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        student, teacher = _flatten_matching_features(student, teacher)
        if student.numel():
            losses.append(F.mse_loss(student, teacher.detach().to(student.device)))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def attention_distillation_loss(student_attentions: Iterable[torch.Tensor], teacher_attentions: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = []
    for student, teacher in zip(student_attentions, teacher_attentions):
        student, teacher = _flatten_matching_features(student, teacher)
        if student.numel():
            student = F.log_softmax(student, dim=-1)
            teacher = F.softmax(teacher.detach().to(student.device), dim=-1)
            losses.append(F.kl_div(student, teacher, reduction="batchmean"))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


class _EnvironmentAwareScaling(SmallModelScalingInterface):
    """Shared hooks for source-only and environment-triggered strategies."""

    regenerate_on_environment_switch = False
    source_env_only = False

    def on_environment_switch(self, **kwargs: Any) -> Tuple[bool, Optional[dict]]:
        if not self.regenerate_on_environment_switch:
            return False, kwargs.get("current_pruning_info")
        pruning_info = self.regenerate_small_model_in_place(**kwargs)
        return True, pruning_info

    def _source_env_id(self, args: Any) -> Optional[str]:
        raw = getattr(args, "envs_id", None)
        if raw is None:
            return getattr(args, "env_id", None)
        try:
            parsed = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in str(raw).split(",") if item.strip()]
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
        return str(parsed) if parsed else getattr(args, "env_id", None)

    def collect_sample_for_small_model_scaling(self, args: Any, **kwargs: Any) -> Dict[str, Any]:
        if not self.source_env_only:
            return super().collect_sample_for_small_model_scaling(args, **kwargs)
        adapter = kwargs["adapter"]
        eval_envs = kwargs["eval_envs"]
        source_env_id = self._source_env_id(args)
        current_env_id = getattr(args, "_current_env_id", getattr(args, "env_id", None))
        if not source_env_id or source_env_id == current_env_id:
            return super().collect_sample_for_small_model_scaling(args, **kwargs)
        source_envs = adapter.make_vector_env(
            args,
            device=kwargs["device"],
            env_id=source_env_id,
            num_envs=args.num_eval_envs,
            record_metrics=False,
        )
        try:
            kwargs["eval_envs"] = source_envs
            return super().collect_sample_for_small_model_scaling(args, **kwargs)
        finally:
            source_envs.close()


class _DistillationScaling(_EnvironmentAwareScaling):
    """Behavioral distillation with a strategy-specific loss hook."""

    objective = "logit"

    def __init__(self, *, distillation_steps: int = 4, learning_rate: float = 1e-5, temperature: float = 2.0, max_samples: int = 128) -> None:
        if distillation_steps < 0 or learning_rate <= 0 or temperature <= 0 or max_samples <= 0:
            raise ValueError("invalid distillation hyperparameter")
        self.distillation_steps = int(distillation_steps)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.max_samples = int(max_samples)

    def after_small_model_scaling(self, *, large_agent: nn.Module, small_agent: nn.Module, sample_batch: dict, **kwargs: Any) -> None:
        if self.distillation_steps == 0:
            return
        self._distill(large_agent, small_agent, sample_batch, kwargs["device"], kwargs["reference_api"])

    def _collect_outputs(self, policy: nn.Module, rgbs: Any, states: Any, action_bins: torch.Tensor, reference_api: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        _, log_prob, _, value, _ = reference_api.policy_get_action_and_value(
            policy, rgbs=rgbs, states=states, action_bins=action_bins, deterministic=True
        )
        return log_prob, value.view(-1)

    def _distill(self, teacher: nn.Module, student: nn.Module, sample_batch: dict, device: torch.device, reference_api: Any) -> None:
        # FBS materialization copies teacher channels.  Distillation methods start
        # from an independent initialization, as in the original KD formulation.
        _randomize_student_parameters(student)
        rgbs = sample_batch["rgbs"].detach().cpu().numpy() if isinstance(sample_batch["rgbs"], torch.Tensor) else np.asarray(sample_batch["rgbs"])
        states = np.asarray(sample_batch["states"], dtype=np.float32)
        action_bins = sample_batch.get("action_bins")
        if action_bins is None:
            _, _, _, _, action_bins = reference_api.batched_get_action_and_value_no_grad(teacher, rgbs, states, micro_batch_size=self.max_samples, deterministic=True)
        action_bins = _as_device_tensor(action_bins, device, torch.long)
        limit = min(len(rgbs), self.max_samples)
        rgbs, states, action_bins = rgbs[:limit], states[:limit], action_bins[:limit]
        teacher_training, student_training = teacher.training, student.training
        teacher.eval(); student.train()
        teacher_features, student_features = [], []
        hooks = []
        if self.objective in {"feature", "attention"}:
            def make_hook(bucket: List[torch.Tensor], *, attention: bool = False):
                def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                    if isinstance(output, (tuple, list)):
                        # HuggingFace attention modules return (hidden, weights) when
                        # output_attentions=True.  Fall back to hidden output for
                        # implementations that do not expose the distribution.
                        value = output[1] if attention and len(output) > 1 and torch.is_tensor(output[1]) else output[0]
                    else:
                        value = output
                    if torch.is_tensor(value):
                        # Keep the distillation signal while avoiding retention of
                        # every token/head activation from the multi-billion-
                        # parameter backbone.
                        if value.ndim > 2:
                            value = value.float().mean(dim=tuple(range(1, value.ndim - 1)))
                        if value.ndim > 1 and value.shape[-1] > 256:
                            value = value[..., :256]
                        bucket.append(value)
                return hook
            for name, module in teacher.named_modules():
                if (self.objective == "feature" and ("mlp" in name or "layernorm" in name)) or (self.objective == "attention" and ("attn" in name or "self_attn" in name)):
                    hooks.append(module.register_forward_hook(make_hook(teacher_features, attention=self.objective == "attention")))
            for name, module in student.named_modules():
                if (self.objective == "feature" and ("mlp" in name or "layernorm" in name)) or (self.objective == "attention" and ("attn" in name or "self_attn" in name)):
                    hooks.append(module.register_forward_hook(make_hook(student_features, attention=self.objective == "attention")))
        with torch.no_grad():
            teacher_log_prob, teacher_value = self._collect_outputs(teacher, rgbs, states, action_bins, reference_api)
        trainable = [p for p in student.parameters() if p.requires_grad]
        if not trainable:
            for hook in hooks:
                hook.remove()
            student.train(student_training); teacher.train(teacher_training)
            return
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        for _ in range(self.distillation_steps):
            optimizer.zero_grad(set_to_none=True)
            teacher_features.clear(); student_features.clear()
            student_log_prob, student_value = self._collect_outputs(student, rgbs, states, action_bins, reference_api)
            if self.objective == "reverse_kl":
                action_loss = reverse_kl_loss(student_log_prob / self.temperature, teacher_log_prob / self.temperature)
            elif self.objective == "distillm":
                action_loss = distillm_kl_loss(student_log_prob / self.temperature, teacher_log_prob / self.temperature)
            elif self.objective in {"feature", "attention"}:
                if self.objective == "feature":
                    action_loss = feature_distillation_loss(student_features, teacher_features).to(device)
                else:
                    action_loss = attention_distillation_loss(student_features, teacher_features).to(device)
            else:
                action_loss = F.mse_loss(student_log_prob, teacher_log_prob)
            loss = action_loss + 0.5 * F.mse_loss(student_value, teacher_value)
            loss.backward(); optimizer.step()
        student.train(student_training); teacher.train(teacher_training)
        for hook in hooks:
            hook.remove()


class LogitDistillationScaling(_DistillationScaling):
    """Logit/action-distribution distillation (renamed from knowledge_distillation)."""

    objective = "logit"


class FeatureDistillationScaling(_DistillationScaling):
    objective = "feature"


class AttentionDistillationScaling(_DistillationScaling):
    objective = "attention"


class MiniLLMScaling(_DistillationScaling):
    objective = "reverse_kl"


class DistiLLMScaling(_DistillationScaling):
    objective = "distillm"


class DataDistillationScaling(_DistillationScaling):
    """Always collect teacher-generated samples before training the student."""

    def collect_sample_for_small_model_scaling(self, args: Any, **kwargs: Any) -> Dict[str, Any]:
        original_policy = args.small_model_scaling_policy
        args.small_model_scaling_policy = "large"
        try:
            return super().collect_sample_for_small_model_scaling(args, **kwargs)
        finally:
            args.small_model_scaling_policy = original_policy


class LLMPrunerScaling(_EnvironmentAwareScaling):
    source_env_only = True


class _EnvironmentSwitchScaling(_EnvironmentAwareScaling):
    regenerate_on_environment_switch = True

    def should_regenerate_small_model_before_rollout(self, *args: Any, **kwargs: Any) -> bool:
        return False


class PowerInferScaling(_EnvironmentSwitchScaling):
    pass


class EdgeTAScaling(_EnvironmentSwitchScaling):
    pass


class LLMInAFlashScaling(_EnvironmentAwareScaling):
    """Keep attention dense and retain half as many FFN neurons as default."""

    @contextlib.contextmanager
    def _attention_dense_context(self, actor: nn.Module, max_sparsity: float):
        originals = []
        keep_ratio = max(0.0, min(1.0, (1.0 - float(max_sparsity)) * 0.5))
        for name, module in actor.named_modules():
            if not hasattr(module, "k_takes_all") or not hasattr(module, "cached_raw_w"):
                continue
            if module.cached_raw_w is None:
                continue
            is_attention = "attn" in name or "self_attn" in name
            old_k, old_cached = module.k_takes_all.k, module.cached_w
            raw = module.cached_raw_w
            total = int(raw.shape[-1])
            keep = total if is_attention else max(1, int(math.ceil(total * keep_ratio)))
            scores = raw.detach().float().mean(dim=0) if raw.ndim > 1 else raw.detach().float()
            selected = torch.argsort(scores, descending=True)[:keep]
            module.k_takes_all.k = (total - keep) / max(total, 1)
            cached = torch.zeros_like(raw); cached[..., selected] = raw[..., selected]
            module.cached_w = cached
            originals.append((module, old_k, old_cached))
        try:
            yield
        finally:
            for module, old_k, old_cached in originals:
                module.k_takes_all.k, module.cached_w = old_k, old_cached

    def generate_initial_small_model(self, *, large_agent: nn.Module, args: Any, eval_envs: Any, device: torch.device, adapter: Any, reference_api: Any):
        # The base generator performs cache materialization internally; patching the
        # dynamic FBS state around it gives a dense-attention, FFN-only materializer.
        with self._attention_dense_context(large_agent, args.max_sparsity):
            return super().generate_initial_small_model(large_agent=large_agent, args=args, eval_envs=eval_envs, device=device, adapter=adapter, reference_api=reference_api)


def make_scaling_method(name: str) -> SmallModelScalingInterface:
    factories = {
        "llm_pruner": LLMPrunerScaling,
        "logit_distillation": LogitDistillationScaling,
        "feature_distillation": FeatureDistillationScaling,
        "attn_distillation": AttentionDistillationScaling,
        "data_distillation": DataDistillationScaling,
        "minillm": MiniLLMScaling,
        "distillm": DistiLLMScaling,
        "llm_in_a_flash": LLMInAFlashScaling,
        "powerinfer": PowerInferScaling,
        "edgeta": EdgeTAScaling,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown scaling method {name!r}; supported: {', '.join(sorted(factories))}") from exc


SCALING_METHODS = {
    "llm_pruner": LLMPrunerScaling,
    "logit_distillation": LogitDistillationScaling,
    "feature_distillation": FeatureDistillationScaling,
    "attn_distillation": AttentionDistillationScaling,
    "data_distillation": DataDistillationScaling,
    "minillm": MiniLLMScaling,
    "distillm": DistiLLMScaling,
    "llm_in_a_flash": LLMInAFlashScaling,
    "powerinfer": PowerInferScaling,
    "edgeta": EdgeTAScaling,
}
