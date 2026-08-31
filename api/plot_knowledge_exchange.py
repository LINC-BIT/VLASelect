"""Plot training accuracy for all completed knowledge-exchange granularities."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api._plot_result_curves import run_cli


_KNOWLEDGE_EXCHANGE_LABELS = {
    "default": "Channel/Neuron\n(VLASelect)",
    "attention_head": "Attention Head",
    "block": "Block",
    "layer": "Layer",
}


if __name__ == "__main__":
    results = Path(__file__).resolve().parent / "results" / "knowledge_exchange"
    run_cli(
        results,
        results / "training_accuracy_curve.png",
        "Training Accuracy by Knowledge-Exchange Granularity",
        label_overrides=_KNOWLEDGE_EXCHANGE_LABELS,
    )
