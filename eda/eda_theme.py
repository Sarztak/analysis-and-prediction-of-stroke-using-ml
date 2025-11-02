"""
Consistent visual theme for all EDA plots.
Applied across matplotlib and seaborn.

Usage:
    from src.eda.eda_theme import set_theme, create_fig, add_n_annotation
    set_theme()
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


# ======== COLOR CONSTANTS ========
STROKE_COLOR = "#d62728"       # red
NO_STROKE_COLOR = "#2ca02c"    # green
MISSING_COLOR = "#ffbf00"      # amber
NEUTRAL_COLOR = "#1f77b4"      # blue
RARE_COLOR = "#7f7f7f"         # grey

THEME_PALETTE = {
    0: NO_STROKE_COLOR,
    1: STROKE_COLOR,
    "Missing": MISSING_COLOR,
    "Neutral": NEUTRAL_COLOR,
    "Rare": RARE_COLOR,
}


# ======== STYLE SETUP ========
def set_theme():
    """
    Apply a unified style across matplotlib and seaborn.
    Call once at the start of any EDA script.
    """
    plt.style.use("default")

    rcParams.update({
        # Axes and grid
        "axes.grid": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        # Font
        "font.family": "DejaVu Sans",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Figure
        "figure.figsize": (7, 5),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # Legend
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,     # square corners (set True for rounded)
        "legend.borderpad": 0.6,      # spacing inside the box
        "legend.fontsize": 10,
        "legend.loc": "upper right"
    })

    sns.set_context("notebook")
    sns.set_style(
        "ticks",
        {
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "xtick.bottom": True,
            "ytick.left": True,
            "grid.color": "white",
        },
    )
    sns.set_palette([NO_STROKE_COLOR, STROKE_COLOR])


# ======== UTILITIES ========
def create_fig(ax=None, figsize=(7, 5)):
    """
    Create or reuse a matplotlib axis with theme applied.
    Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def add_n_annotation(ax, data, text=None, loc="upper right"):
    """
    Adds a small sample-size annotation (e.g., n = 5123) to a plot.
    """
    if text is None:
        text = f"n = {len(data):,}"

    loc_dict = {
        "upper right": (0.98, 0.95),
        "upper left": (0.02, 0.95),
        "lower right": (0.98, 0.05),
        "lower left": (0.02, 0.05),
    }
    if loc not in loc_dict:
        loc = "upper right"

    x, y = loc_dict[loc]
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right" if "right" in loc else "left",
        va="top" if "upper" in loc else "bottom",
        fontsize=9,
        color="black",
    )
    return ax


def save_fig(fig, name, output_dir="reports/eda"):
    """
    Save figure with theme defaults.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path
