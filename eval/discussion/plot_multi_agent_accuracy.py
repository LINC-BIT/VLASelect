from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SMOOTHING_ALPHA = 0.6
METHOD_LABELS = {
    "mappo": "MAPPO",
    "ours": "VLASelect",
}


def smooth_scores(scores, alpha: float = SMOOTHING_ALPHA):
    """Apply exponential smoothing while preserving the original time points."""
    if not scores:
        return []
    smoothed = [scores[0]]
    for score in scores[1:]:
        smoothed.append(alpha * score + (1.0 - alpha) * smoothed[-1])
    return smoothed


def load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Invalid manifest: {args.manifest}")

    plotted = 0
    all_scores = []
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for run in manifest.get("runs", []):
        method = str(run.get("method", "unknown"))
        run_dir = Path(run.get("run_dir", ""))
        metrics_path = Path(run.get("result_json", ""))
        metrics = load_json(metrics_path) if metrics_path else None
        if not isinstance(metrics, list):
            metrics = load_json(run_dir / "metrics.json")
        if not isinstance(metrics, list):
            continue

        points = []
        for row in metrics:
            if not isinstance(row, dict):
                continue
            elapsed = row.get("elapsed_minutes", row.get("time_minutes"))
            score = row.get("score")
            try:
                points.append((float(elapsed), float(score)))
            except (TypeError, ValueError):
                continue
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        times = [item[0] for item in points]
        scores = smooth_scores([item[1] for item in points])
        all_scores.extend(scores)
        ax.plot(
            times,
            scores,
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=METHOD_LABELS.get(method.lower(), method),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise SystemExit("No time-indexed accuracy metrics found")

    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel("Accuracy (success rate)")
    ax.set_title("Multi-agent accuracy over training time")
    ax.set_ylim(min(all_scores) - 0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"[plot] accuracy_plot={args.output}")


if __name__ == "__main__":
    main()
