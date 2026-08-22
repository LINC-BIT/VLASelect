"""Compatibility aliases for attention-head-grained generation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.knowledge_exchange_interface_examples.attention_head_grained import (
    AttentionHeadGrainedSmallModelScalingInterface,
    HeadGrainedSmallModelScalingInterface,
    make_attention_head_grained_interface,
    make_head_grained_interface,
)

__all__ = [
    "AttentionHeadGrainedSmallModelScalingInterface",
    "HeadGrainedSmallModelScalingInterface",
    "make_attention_head_grained_interface",
    "make_head_grained_interface",
]
