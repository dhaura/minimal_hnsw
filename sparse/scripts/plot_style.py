"""Shared style constants for the alpha/beta sweep plotting scripts.

Categorical palette (validated colorblind-safe, fixed hue order) and chart
chrome from the dataviz skill's default palette (references/palette.md).
"""
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)


def alpha_color_map(alphas):
    """Fixed color per alpha, assigned by ascending value so a given alpha
    keeps its color across runs even if some combos are missing."""
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(sorted(alphas))}


def load_sweep_csv(csv_path):
    import sys
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"alpha", "beta", "pruning_time_sec", "indexing_time_sec", "searching_time_sec", "recall"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"{csv_path}: missing expected column(s): {sorted(missing)}")

    df = df.groupby(["alpha", "beta"], as_index=False).mean(numeric_only=True)
    df["total_indexing_time_sec"] = df["pruning_time_sec"] + df["indexing_time_sec"]
    return df
