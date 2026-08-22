"""Block-grained static small-model generation example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.knowledge_exchange_granularity_interface import GranularitySmallModelScalingInterface, _layer_key


class BlockGrainedSmallModelScalingInterface(GranularitySmallModelScalingInterface):
    """Group consecutive transformer layers into fixed-size score-selection blocks."""

    granularity_name = "block"

    def __init__(self, *, block_size: int = 2, low_group_retention: float = 0.02) -> None:
        super().__init__(low_group_retention=low_group_retention)
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)

    def group_fbs_layers(self, fbs_layers: Sequence[str]) -> Mapping[str, List[str]]:
        families = {}
        for name in fbs_layers:
            family, index, _ = _layer_key(name)
            families.setdefault(family, {}).setdefault(index, []).append(name)
        groups = {}
        for family, indexed in families.items():
            ordered = sorted(indexed)
            for offset in range(0, len(ordered), self.block_size):
                chunk = ordered[offset : offset + self.block_size]
                key = f"{family}block{offset // self.block_size}"
                groups[key] = [name for index in chunk for name in indexed[index]]
        return groups


def make_block_grained_interface(*, block_size: int = 2) -> BlockGrainedSmallModelScalingInterface:
    return BlockGrainedSmallModelScalingInterface(block_size=block_size)


def main() -> None:
    from api.unified_online_rl import parse_args, run_training
    from api.vla_model_interface_examples._reference_adapter import make_tinyvla

    run_training(make_tinyvla(), parse_args(), make_block_grained_interface())


if __name__ == "__main__":
    main()
