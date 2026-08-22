from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="eval/success_at_end")
    parser.add_argument("--title", type=str, default="ConRFT Success Curve")
    args = parser.parse_args()

    tb_dir = args.run_dir / "tb"
    events = load_scalar_events(tb_dir, args.tag)
    if not events:
        raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")

    base_time = events[0].wall_time
    xs = [(event.wall_time - base_time) / 60.0 for event in events]
    ys = [event.value for event in events]

    plt.figure(figsize=(9, 6))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Wall-clock Time (minutes)")
    plt.ylabel(args.tag)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(args.title)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
