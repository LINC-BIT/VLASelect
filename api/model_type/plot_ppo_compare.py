#!/usr/bin/env python3
"""Plot VLASelect versus RLVLA accuracy for CNN and MLP experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(values, weight: float = 0.7):
    """Apply exponential smoothing; weight matches ``_plot_acc.py``."""
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(weight * smoothed[-1] + (1.0 - weight) * value)
    return smoothed


def load_series(run_dir: Path):
    # The comparison is based exclusively on the training curve emitted by
    # TensorBoard.  This also works for MWE runs that do not write JSON eval
    # snapshots because their short evaluation window has no completed episode.
    tb_dir = run_dir / "tb"
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if event_files:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # A stable run directory can contain event files from earlier reruns;
        # use only the newest file instead of concatenating separate runs.
        latest_event_file = max(event_files, key=lambda candidate: candidate.stat().st_mtime)
        accumulator = EventAccumulator(str(latest_event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        tag = "train/success_once" if "train/success_once" in tags else None
        if tag is not None:
            events = accumulator.Scalars(tag)
            if events:
                start_time = min(event.wall_time for event in events)
                points = [
                    (
                        max(0.0, (event.wall_time - start_time) / 60.0),
                        max(0.0, min(1.0, float(event.value))),
                    )
                    for event in events
                ]
                return zip(*points)

    raise FileNotFoundError(f"missing TensorBoard train/success_once metrics: {tb_dir}")


def latest(root: Path) -> Path:
    candidates = [
        p for p in root.iterdir()
        if p.is_dir()
        and list((p / "[agent]" / "tb").glob("events.out.tfevents.*"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no completed run with metrics under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime) / "[agent]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-vlaselect", type=Path, default=None)
    parser.add_argument("--cnn-conrft", type=Path, default=None)
    parser.add_argument("--mlp-vlaselect", type=Path, default=None)
    parser.add_argument("--mlp-conrft", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("MLP-CNN-ACC-COMPARE.png"))
    args = parser.parse_args()
    root = Path(__file__).with_name("ckpt") / "results"
    paths = {
        "CNN": (args.cnn_vlaselect or latest(root / "cnn"), args.cnn_conrft or latest(root / "cnn-conrft")),
        "MLP": (args.mlp_vlaselect or latest(root / "mlp"), args.mlp_conrft or latest(root / "mlp-conrft")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200, sharey=True)
    improvements = []
    for ax, (title, (vla_path, conrft_path)) in zip(axes, paths.items()):
        vx, vy = load_series(vla_path)
        cx, cy = load_series(conrft_path)
        vx, vy = list(vx), smooth(list(vy), 0.7)
        cx, cy = list(cx), smooth(list(cy), 0.7)
        ax.plot(vx, vy, linewidth=2.8, label="VLASelect")
        ax.plot(cx, cy, linewidth=2.8, label="RLVLA")
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Time (minutes)", fontsize=15)
        ax.set_ylim(-0.1, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
        vla_mean = sum(vy) / len(vy)
        conrft_mean = sum(cy) / len(cy)
        improvement = vla_mean - conrft_mean
        improvements.append(improvement)
        print(
            f"[compare] {title}: VLASelect mean={vla_mean:.4f}, "
            f"RLVLA mean={conrft_mean:.4f}, "
            f"absolute improvement={improvement:+.4f} ({improvement * 100:+.2f} pp)"
        )
    axes[0].set_ylabel("Accuracy", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[plot] output={args.output}")
    overall_improvement = sum(improvements) / len(improvements)
    print(
        f"[compare] overall mean absolute improvement="
        f"{overall_improvement:+.4f} ({overall_improvement * 100:+.2f} pp)"
    )


if __name__ == "__main__":
    main()
