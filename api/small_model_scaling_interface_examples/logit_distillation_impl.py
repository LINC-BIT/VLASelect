"""Compatibility entry point for the renamed logit-distillation strategy."""

from api.small_model_scaling_interface_examples.scaling_methods import LogitDistillationScaling


LogitDistillationSmallModelScalingInterface = LogitDistillationScaling


def make_logit_distillation_interface(*, randomize_student_parameters: bool = True) -> LogitDistillationScaling:
    return LogitDistillationScaling(randomize_student_parameters=randomize_student_parameters)


__all__ = ["LogitDistillationScaling", "LogitDistillationSmallModelScalingInterface", "make_logit_distillation_interface"]
