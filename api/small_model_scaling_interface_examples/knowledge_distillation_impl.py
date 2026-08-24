"""Legacy compatibility module for the former knowledge-distillation name.

The static FBS generator, trajectory selection, and regeneration schedule are inherited
unchanged.  This example only adds a short behavioral distillation phase after each
generated model is materialized.  All distillation hyperparameters live in the class
constructor, which is the intended extension pattern for user-defined generators.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.small_model_scaling_interface import SmallModelScalingInterface


class KnowledgeDistillationSmallModelScalingInterface(SmallModelScalingInterface):
    """Default static generation followed by teacher-student behavioral distillation."""

    def __init__(
        self,
        *,
        distillation_steps: int = 4,
        learning_rate: float = 1e-5,
        temperature: float = 2.0,
        action_loss_weight: float = 1.0,
        value_loss_weight: float = 0.5,
        max_samples: int = 128,
    ) -> None:
        if distillation_steps < 0:
            raise ValueError("distillation_steps must be non-negative")
        if learning_rate <= 0 or temperature <= 0:
            raise ValueError("learning_rate and temperature must be positive")
        if action_loss_weight < 0 or value_loss_weight < 0 or max_samples <= 0:
            raise ValueError("distillation weights must be non-negative and max_samples positive")
        self.distillation_steps = int(distillation_steps)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.action_loss_weight = float(action_loss_weight)
        self.value_loss_weight = float(value_loss_weight)
        self.max_samples = int(max_samples)

    def after_small_model_scaling(
        self,
        *,
        large_agent: nn.Module,
        small_agent: nn.Module,
        sample_batch: dict,
        pruning_info: dict,
        optimizer: Any,
        args: Any,
        device: torch.device,
        adapter: Any,
        reference_api: Any,
    ) -> None:
        del pruning_info, optimizer, args, adapter
        if self.distillation_steps == 0:
            return
        self._distill(
            large_agent=large_agent,
            small_agent=small_agent,
            sample_batch=sample_batch,
            device=device,
            reference_api=reference_api,
        )

    def _distill(
        self,
        *,
        large_agent: nn.Module,
        small_agent: nn.Module,
        sample_batch: dict,
        device: torch.device,
        reference_api: Any,
    ) -> None:
        self._randomize_student_parameters(small_agent)
        rgbs = sample_batch["rgbs"]
        if isinstance(rgbs, torch.Tensor):
            rgbs = rgbs.detach().cpu().numpy()
        else:
            rgbs = np.asarray(rgbs)
        states = np.asarray(sample_batch["states"], dtype=np.float32)
        action_bins = sample_batch.get("action_bins")
        if action_bins is None:
            _, _, _, _, action_bins = reference_api.batched_get_action_and_value_no_grad(
                large_agent,
                rgbs,
                states,
                micro_batch_size=getattr(small_agent, "eval_micro_batch_size", self.max_samples),
                deterministic=True,
            )
        action_bins = torch.as_tensor(action_bins, device=device, dtype=torch.long)
        if len(rgbs) > self.max_samples:
            rgbs = rgbs[: self.max_samples]
            states = states[: self.max_samples]
            action_bins = action_bins[: self.max_samples]

        student_was_training = small_agent.training
        teacher_was_training = large_agent.training
        small_agent.train()
        large_agent.eval()
        teacher_log_probs = []
        teacher_values = []
        with torch.no_grad():
            for start in range(0, len(rgbs), self.max_samples):
                end = min(start + self.max_samples, len(rgbs))
                _, teacher_log_prob, _, teacher_value, _ = reference_api.policy_get_action_and_value(
                    large_agent,
                    rgbs=rgbs[start:end],
                    states=states[start:end],
                    action_bins=action_bins[start:end],
                    deterministic=True,
                )
                teacher_log_probs.append(teacher_log_prob.detach())
                teacher_values.append(teacher_value.detach())
        teacher_log_prob = torch.cat(teacher_log_probs, dim=0)
        teacher_value = torch.cat(teacher_values, dim=0).view(-1)

        trainable_parameters = [parameter for parameter in small_agent.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            small_agent.train(student_was_training)
            large_agent.train(teacher_was_training)
            return
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate)
        for _ in range(self.distillation_steps):
            optimizer.zero_grad(set_to_none=True)
            _, student_log_prob, _, student_value, _ = reference_api.policy_get_action_and_value(
                small_agent,
                rgbs=rgbs,
                states=states,
                action_bins=action_bins,
                deterministic=True,
            )
            action_loss = F.mse_loss(
                student_log_prob / self.temperature,
                teacher_log_prob / self.temperature,
            )
            value_loss = F.mse_loss(student_value.view(-1), teacher_value)
            loss = self.action_loss_weight * action_loss + self.value_loss_weight * value_loss
            loss.backward()
            optimizer.step()
        print(
            f"[distill] steps={self.distillation_steps} samples={len(rgbs)} "
            f"action_loss={float(action_loss.detach().item()):.6f} "
            f"value_loss={float(value_loss.detach().item()):.6f}"
        )
        small_agent.train(student_was_training)
        large_agent.train(teacher_was_training)

    @staticmethod
    def _randomize_student_parameters(student: nn.Module) -> None:
        """Reset the generated student's learnable weights before distillation.

        Static FBS generation copies selected teacher channels into the student.  A
        distillation run should instead start from an independently initialized
        student while retaining its generated architecture and non-parameter buffers.
        ``reset_parameters`` is the standard initialization contract for PyTorch
        modules and is applied once per module by ``modules()``.
        """
        for module in student.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
                print('reset')


def make_knowledge_distillation_interface() -> KnowledgeDistillationSmallModelScalingInterface:
    return KnowledgeDistillationSmallModelScalingInterface()


def main() -> None:
    from api.unified_online_rl import parse_args, run_training
    from api.vla_model_interface_examples._reference_adapter import make_vla_adapter

    run_training(make_vla_adapter(), parse_args(), make_knowledge_distillation_interface())


if __name__ == "__main__":
    main()
