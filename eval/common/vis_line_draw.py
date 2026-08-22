from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_matplotlib_style(style: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.sans-serif": style["font_sans_serif"],
            "font.size": style["font_size"],
            "axes.labelsize": style["axes_labelsize"],
            "xtick.labelsize": style["xtick_labelsize"],
            "ytick.labelsize": style["ytick_labelsize"],
            "legend.fontsize": style["legend_fontsize"],
        }
    )


def draw_plot(plot_data: dict, render_config: dict, output_dir: Path) -> tuple[Path, Path]:
    figure = render_config["figure"]
    output_stem = plot_data["output_stem"]

    fig = plt.figure(figsize=(figure["width"], figure["height"]))
    ax = fig.add_subplot(111)

    for series in plot_data["series"]:
        style = series["style"]
        ax.plot(
            series["x"],
            series["y"],
            linewidth=style["linewidth"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=series["label"],
        )

    ax.set_xlabel(plot_data["xlabel"])
    ax.set_ylabel(plot_data["ylabel"])
    ax.set_xlim(plot_data["xlim"])
    ax.set_ylim(plot_data["ylim"])
    ax.grid(True, alpha=plot_data["grid_alpha"])

    if not plot_data["series"]:
        xmid = (plot_data["xlim"][0] + plot_data["xlim"][1]) / 2.0
        ymid = (plot_data["ylim"][0] + plot_data["ylim"][1]) / 2.0
        ax.text(xmid, ymid, "NaN", ha="center", va="center")

    fig.tight_layout()
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"
    fig.savefig(png_path, dpi=figure["dpi"])
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def draw_legend(plot_data: dict, render_config: dict, output_dir: Path) -> None:
    figure = render_config["legend_figure"]
    legend_entries = plot_data["legend_entries"]
    if not legend_entries:
        return

    handles = [
        Line2D(
            [0],
            [0],
            color=entry["style"]["color"],
            linestyle=entry["style"]["linestyle"],
            linewidth=entry["style"]["linewidth"],
        )
        for entry in legend_entries
    ]
    labels = [entry["label"] for entry in legend_entries]

    fig = plt.figure(figsize=(figure["width"], max(figure["min_height"], figure["height_per_item"] * len(labels))))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=1,
        handlelength=figure["handlelength"],
    )

    fig.tight_layout()
    output_stem = plot_data["output_stem"]
    png_path = output_dir / f"{output_stem}_legend.png"
    svg_path = output_dir / f"{output_stem}_legend.svg"
    fig.savefig(png_path, dpi=render_config["figure"]["dpi"], bbox_inches="tight", pad_inches=figure["pad_inches"])
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=figure["pad_inches"])
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=script_dir / "data" / "toy_cnn.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "success_once", "success_at_end"],
        default="all",
    )
    parser.add_argument(
        "--with-legend",
        action="store_true",
        help="Also render standalone legend images.",
    )
    args = parser.parse_args()

    payload = load_payload(args.json)
    apply_matplotlib_style(payload["render_config"]["matplotlib"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_names = payload["plot_order"] if args.plot == "all" else [args.plot]
    for plot_name in plot_names:
        plot_data = payload["plots"][plot_name]
        draw_plot(plot_data, payload["render_config"], args.output_dir)
        if args.with_legend:
            draw_legend(plot_data, payload["render_config"], args.output_dir)


if __name__ == "__main__":
    main()
