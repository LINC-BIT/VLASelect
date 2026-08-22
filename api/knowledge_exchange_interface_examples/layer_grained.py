"""Layer-grained static small-model generation example."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.knowledge_exchange_granularity_interface import GranularitySmallModelScalingInterface


class LayerGrainedSmallModelScalingInterface(GranularitySmallModelScalingInterface):
    """Keep complete transformer layers with the highest mean FBS scores."""

    granularity_name = "layer"


def make_layer_grained_interface() -> LayerGrainedSmallModelScalingInterface:
    return LayerGrainedSmallModelScalingInterface()


def main() -> None:
    from api.unified_online_rl import parse_args, run_training
    from api.vla_model_interface_examples._reference_adapter import make_vla_adapter

    run_training(make_vla_adapter(), parse_args(), make_layer_grained_interface())


if __name__ == "__main__":
    main()
