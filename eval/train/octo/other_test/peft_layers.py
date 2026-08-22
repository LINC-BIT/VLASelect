from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.reset_parameters()

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if base.groups != 1:
            raise ValueError("LoRAConv2d currently supports groups=1 only")

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_a = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        self.lora_b = nn.Conv2d(
            in_channels=rank,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.reset_parameters()

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _replace_child(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(child_name)] = new_child
    else:
        setattr(parent, child_name, new_child)


def inject_lora_modules(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[int, int]:
    replaced_linear = 0
    replaced_conv = 0

    for child_name, child in list(module.named_children()):
        child_linear, child_conv = inject_lora_modules(child, rank, alpha, dropout)
        replaced_linear += child_linear
        replaced_conv += child_conv

        current_child = getattr(module, child_name) if not isinstance(module, nn.Sequential) else module[int(child_name)]
        if isinstance(current_child, nn.Linear):
            _replace_child(module, child_name, LoRALinear(current_child, rank, alpha, dropout))
            replaced_linear += 1
        elif isinstance(current_child, nn.Conv2d):
            _replace_child(module, child_name, LoRAConv2d(current_child, rank, alpha, dropout))
            replaced_conv += 1

    return replaced_linear, replaced_conv


def freeze_non_lora_parameters(module: nn.Module, train_actor_logstd: bool = True) -> None:
    for name, parameter in module.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            parameter.requires_grad = True
            continue
        if train_actor_logstd and name == "actor_logstd":
            parameter.requires_grad = True
            continue
        parameter.requires_grad = False

