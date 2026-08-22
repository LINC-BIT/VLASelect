"""TinyVLA implementation of :class:`api.vla_model_interface.VLAModelInterface`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.unified_online_rl import parse_args, run_training
from api.vla_model_interface_examples._reference_adapter import TinyVLAImplementation, make_tinyvla


def main() -> None:
    run_training(make_tinyvla(), parse_args())


if __name__ == "__main__":
    main()
