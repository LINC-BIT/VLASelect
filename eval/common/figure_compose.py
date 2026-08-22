from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def render_legend_image(
    legend_entries: Sequence[tuple[str, dict]],
    output_path: Path,
    *,
    ncol: int = 5,
    fontsize: int = 14,
    linewidth: float = 2.4,
    handlelength: float = 2.6,
    dpi: int = 200,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not legend_entries:
        return output_path

    rows = max(1, (len(legend_entries) + max(1, ncol) - 1) // max(1, ncol))
    fig = plt.figure(figsize=(max(6.0, ncol * 2.4), 0.72 + 0.38 * rows))
    ax = fig.add_subplot(111)
    ax.axis('off')
    handles = [
        Line2D(
            [0],
            [0],
            color=style.get('color', '#4C78A8'),
            linestyle=style.get('linestyle', '-'),
            linewidth=linewidth,
        )
        for _, style in legend_entries
    ]
    labels = [label for label, _ in legend_entries]
    ax.legend(
        handles,
        labels,
        loc='center',
        ncol=max(1, ncol),
        frameon=False,
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def compose_grid_figure(
    panel_paths: Sequence[Path],
    *,
    output_paths: Sequence[Path],
    rows: int,
    cols: int,
    figsize: tuple[float, float],
    legend_path: Path | None = None,
    legend_height_ratio: float = 0.12,
    wspace: float = 0.02,
    hspace: float = 0.02,
    dpi: int = 200,
) -> None:
    total_rows = rows + (1 if legend_path else 0)
    height_ratios = [1.0] * rows + ([legend_height_ratio * rows] if legend_path else [])
    fig = plt.figure(figsize=figsize, facecolor='white')
    grid = fig.add_gridspec(total_rows, cols, height_ratios=height_ratios, hspace=hspace, wspace=wspace)

    for index in range(rows * cols):
        r = index // cols
        c = index % cols
        ax = fig.add_subplot(grid[r, c])
        ax.axis('off')
        if index < len(panel_paths) and panel_paths[index].exists():
            image = mpimg.imread(panel_paths[index])
            ax.imshow(image)
            ax.set_aspect('auto')

    if legend_path and legend_path.exists():
        ax = fig.add_subplot(grid[rows, :])
        ax.axis('off')
        image = mpimg.imread(legend_path)
        ax.imshow(image)
        ax.set_aspect('auto')

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == '.png':
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
