from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def candidate_scalar_tags(tag: str):
    candidates = [tag]
    if tag.startswith("eval/"):
        candidates.append(f"train/{tag.split('/', 1)[1]}")
    return candidates


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    for candidate in candidate_scalar_tags(tag):
        if candidate in scalar_tags:
            return accumulator.Scalars(candidate), candidate
    return [], tag


def draw_success_curve(run_dir: Path, output: Path, tag: str = "eval/success_at_end", title: str = "Self-Improvement Success Curve"):
    tb_dir = run_dir / "tb"
    events, resolved_tag = load_scalar_events(tb_dir, tag)
    if not events:
        tried = ", ".join(candidate_scalar_tags(tag))
        raise RuntimeError(f"Cannot find scalar tag(s) [{tried}] in {tb_dir}")

    base_time = events[0].wall_time
    xs = [(event.wall_time - base_time) / 60.0 for event in events]
    ys = [event.value for event in events]

    plt.figure(figsize=(9, 6))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Wall-clock Time (minutes)")
    plt.ylabel(resolved_tag)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="eval/success_at_end")
    parser.add_argument("--title", type=str, default="Self-Improvement Success Curve")
    args = parser.parse_args()

    output = draw_success_curve(
        run_dir=args.run_dir,
        output=args.output,
        tag=args.tag,
        title=args.title,
    )
    print(output)


if __name__ == "__main__":
    main()
