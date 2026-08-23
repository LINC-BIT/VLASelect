"""EdgeVLA implementation of the shared VLA model interface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.knowledge_exchange_interface_examples.attention_head_grained import make_attention_head_grained_interface
from api.knowledge_exchange_interface_examples.block_grained import make_block_grained_interface
from api.knowledge_exchange_interface_examples.layer_grained import make_layer_grained_interface
from api.small_model_scaling_interface import SmallModelScalingInterface
from api.small_model_scaling_interface_examples.scaling_methods import SCALING_METHODS
from api.unified_online_rl import parse_args, run_training
from api.vla_model_interface_examples._reference_adapter import EdgeVLAImplementation, make_edgevla


KNOWLEDGE_EXCHANGE_GRANULARITIES = {
    "layer": make_layer_grained_interface,
    "block": make_block_grained_interface,
    "head": make_attention_head_grained_interface,
    "attention_head": make_attention_head_grained_interface,
}


def resolve_small_model_interface(args):
    if args.scaling_method and args.knowledge_exchange_granularity:
        raise ValueError("--scaling-method and --knowledge-exchange-granularity are mutually exclusive")
    if args.scaling_method:
        try:
            return SCALING_METHODS[args.scaling_method]()
        except KeyError as exc:
            supported = ", ".join(sorted(SCALING_METHODS))
            raise ValueError(f"unknown scaling method {args.scaling_method!r}; supported: {supported}") from exc
    if args.knowledge_exchange_granularity:
        try:
            return KNOWLEDGE_EXCHANGE_GRANULARITIES[args.knowledge_exchange_granularity]()
        except KeyError as exc:
            supported = ", ".join(sorted(KNOWLEDGE_EXCHANGE_GRANULARITIES))
            raise ValueError(
                f"unknown knowledge-exchange granularity {args.knowledge_exchange_granularity!r}; supported: {supported}"
            ) from exc
    return SmallModelScalingInterface()


def main() -> None:
    args = parse_args()
    run_training(make_edgevla(), args, resolve_small_model_interface(args))


if __name__ == "__main__":
    main()
