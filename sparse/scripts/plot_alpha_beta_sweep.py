#!/usr/bin/env python3
"""Plots build time and search time vs. recall from a merged alpha/beta sweep CSV.

Input is the output of merge_alpha_beta_results.sh: one row per (alpha, beta)
combo with columns alpha, beta, dataset_size, threads, pruning_time_sec,
indexing_time_sec, searching_time_sec, recall.

Produces two figures, each with one line per alpha and one labeled point per
beta along that line (points ordered by beta, not by recall, so the line
shows the effect of increasing beta):
  - indexing_time_vs_recall.png: (pruning_time_sec + indexing_time_sec) vs recall
  - search_time_vs_recall.png:   searching_time_sec vs recall
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import INK_PRIMARY, INK_SECONDARY, SURFACE, alpha_color_map, load_sweep_csv, style_axes


def plot_metric(df, y_col, y_label, title, out_path):
    alphas = sorted(df["alpha"].unique())
    colors = alpha_color_map(alphas)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    style_axes(ax)

    for alpha in alphas:
        group = df[df["alpha"] == alpha].sort_values("beta")
        color = colors[alpha]
        ax.plot(
            group["recall"], group[y_col],
            color=color, linewidth=2, marker="o", markersize=8,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1,
            label=f"α = {alpha:g}", zorder=3,
        )
        # Points for the same alpha can sit very close together (beta has a
        # small effect once alpha is high), so cycle the label offset instead
        # of using a fixed corner -- otherwise adjacent labels overlap.
        offsets = [(7, 8), (7, -14), (-7, 8), (-7, -14)]
        for i, (_, row) in enumerate(group.iterrows()):
            ha = "left" if offsets[i % len(offsets)][0] >= 0 else "right"
            ax.annotate(
                f"β={int(row['beta'])}",
                (row["recall"], row[y_col]),
                textcoords="offset points", xytext=offsets[i % len(offsets)],
                fontsize=8, color=INK_SECONDARY, ha=ha,
            )

    ax.set_xlabel("Recall@k (%)", color=INK_SECONDARY)
    ax.set_ylabel(y_label, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(title="Alpha", frameon=False, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Merged results CSV (from merge_alpha_beta_results.sh)")
    parser.add_argument("--outdir", default=None,
                         help="Directory for the output PNGs (default: alongside csv_path)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"No such file: {csv_path}")

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_sweep_csv(csv_path)

    plot_metric(
        df, "total_indexing_time_sec", "Indexing time (pruning + build, s)",
        "Indexing time vs. recall", outdir / "indexing_time_vs_recall.png",
    )
    plot_metric(
        df, "searching_time_sec", "Searching time (s)",
        "Searching time vs. recall", outdir / "search_time_vs_recall.png",
    )


if __name__ == "__main__":
    main()
