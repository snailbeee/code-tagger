"""Figures for the EDA report.

Each function draws already-computed results (from ``stats``, ``signals``...)
and returns a matplotlib Figure; no statistics are computed here. Use
:func:`save_figure` to write and close a figure.
"""

import functools
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import ParamSpec

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- Palette (validated reference palette, light mode) -----------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"  # de-emphasised marks
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2: blue, orange
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)

STYLE: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 9,
    "legend.frameon": False,
}

P = ParamSpec("P")


def _styled[**P](plot: Callable[P, Figure]) -> Callable[P, Figure]:
    """Draw inside the report style without touching global matplotlib state."""

    @functools.wraps(plot)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        with plt.rc_context(STYLE):
            return plot(*args, **kwargs)

    return wrapper


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Recessive axes: no top/right spines, light grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: Figure, path: Path, dpi: int = 150) -> Path:
    """Write ``fig`` to ``path`` (parent folders created) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


@_styled
def plot_tag_frequencies(
    frequencies: pd.DataFrame, highlight: Collection[str] = ()
) -> Figure:
    """Horizontal bars of tag counts; ``highlight`` tags drawn in the accent colour.

    Args:
        frequencies: output of ``stats.tag_frequencies`` (``tag``, ``count``, ``pct``).
        highlight: tags to emphasise (typically the focus tags), which get a value label.
    """
    data = frequencies.sort_values("count")
    colors = [SERIES[0] if tag in highlight else NEUTRAL for tag in data["tag"]]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(data) + 1))
    bars = ax.barh(data["tag"], data["count"], color=colors, height=0.75)
    for bar, (tag, pct) in zip(
        bars, zip(data["tag"], data["pct"], strict=True), strict=True
    ):
        if tag in highlight:
            ax.annotate(
                f"{pct:.1f}%",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=8,
            )
    ax.set_xlabel("Number of problems")
    ax.set_title("Problems per tag")
    ax.margins(y=0.01)
    _style_axes(ax, grid_axis="x")
    if highlight:
        ax.legend(
            handles=[
                Patch(color=SERIES[0], label="Focus tag"),
                Patch(color=NEUTRAL, label="Other tag"),
            ],
            loc="lower right",
        )
    return fig


@_styled
def plot_cooccurrence(
    matrix: pd.DataFrame,
    title: str,
    value_format: str = "{:.0%}",
    hide_diagonal: bool = True,
) -> Figure:
    """Annotated heatmap of a square tag-by-tag matrix (e.g. P(row | column)).

    The diagonal (a tag with itself) is hidden by default so it doesn't stretch
    the colour scale and wash out the off-diagonal cells.
    """
    values = matrix.to_numpy(dtype=float).copy()
    if hide_diagonal:
        np.fill_diagonal(values, np.nan)
    fig, ax = plt.subplots(figsize=(0.75 * len(matrix) + 2, 0.65 * len(matrix) + 1.5))
    image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=np.nanmax(values))
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    threshold = 0.55 * np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            ax.text(
                j,
                i,
                value_format.format(values[i, j]),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if values[i, j] > threshold else TEXT_PRIMARY,
            )
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
    return fig


@_styled
def plot_length_distributions(
    word_counts: Mapping[str, pd.Series], bins: int = 40
) -> Figure:
    """Small multiples: histogram of token counts per field, log-scaled x axis."""
    n = len(word_counts)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 2.4 * n_rows), squeeze=False
    )
    for ax, (name, counts) in zip(axes.flat, word_counts.items(), strict=False):
        positive = counts[counts > 0]
        edges = np.logspace(0, np.log10(max(positive.max(), 10)), bins)
        ax.hist(positive, bins=edges, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(positive.median(), color=TEXT_PRIMARY, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{name}  (median {positive.median():.0f})", fontsize=9)
        _style_axes(ax)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.supxlabel("Tokens (log scale)", color=TEXT_SECONDARY, fontsize=9)
    fig.supylabel("Problems", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(
        "Length of each field", x=0.01, ha="left", fontweight="bold", color=TEXT_PRIMARY
    )
    fig.tight_layout()
    return fig


@_styled
def plot_problems_per_year(counts: pd.Series) -> Figure:
    """Bar chart of problems created per year."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=SERIES[0], width=0.75)
    ax.set_ylabel("Problems")
    ax.set_title("Problems per creation year")
    _style_axes(ax)
    return fig


@_styled
def plot_difficulty_by_tag(distributions: Mapping[str, pd.Series]) -> Figure:
    """Horizontal box plots of difficulty, one per tag (first entry is the reference)."""
    names = list(distributions)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1))
    ax.boxplot(
        [distributions[name].to_numpy() for name in names],
        orientation="horizontal",
        tick_labels=[f"{name} (n={distributions[name].size})" for name in names],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
        medianprops={"color": TEXT_PRIMARY, "linewidth": 1.5},
        whiskerprops={"color": TEXT_SECONDARY},
        capprops={"color": TEXT_SECONDARY},
    )
    ax.invert_yaxis()
    ax.set_xlabel("Codeforces difficulty rating")
    ax.set_title("Difficulty by tag (whiskers: 1.5 IQR)")
    _style_axes(ax, grid_axis="x")
    return fig


@_styled
def plot_signal_precision_recall(results: pd.DataFrame) -> Figure:
    """Scatter of each keyword signal's precision vs recall, by source.

    Args:
        results: output of ``signals.evaluate_signals``.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"text": (SERIES[0], "o"), "code": (SERIES[1], "s")}
    for source, (color, marker) in styles.items():
        subset = results[results["source"] == source]
        ax.scatter(
            subset["recall"],
            subset["precision"],
            s=45,
            color=color,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["tag"],
                (row["recall"], row["precision"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=TEXT_SECONDARY,
            )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (share of the tag's problems matched)")
    ax.set_ylabel("Precision (share of matches carrying the tag)")
    ax.set_title("Keyword signals used alone as classifiers")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="",
                label=f"{source} pattern",
            )
            for source, (color, marker) in styles.items()
        ],
        loc="upper right",
    )
    _style_axes(ax, grid_axis="both")
    return fig
