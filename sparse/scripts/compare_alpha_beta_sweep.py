#!/usr/bin/env python3
"""Overlays two alpha/beta sweep runs (e.g. before/after a code change) on
the same recall-vs-time axes, so a given (alpha, beta) combo's movement
between runs is visible directly.

Input: two merged CSVs from merge_alpha_beta_results.sh (same schema as
plot_alpha_beta_sweep.py). Color still encodes alpha identity (same fixed
palette, same mapping as the single-run plot); which run a point came from
is a secondary encoding (linestyle + marker + opacity), never a repainted
color -- so alpha keeps its meaning across both series.

Usage:
  python3 compare_alpha_beta_sweep.py results/v1/msmarco_full_alpha_beta_sweep.csv \\
                                       results/v2/msmarco_full_alpha_beta_sweep.csv \\
                                       --label-a v1 --label-b v2 \\
                                       --competitors-csv results/competitors.csv

With --competitors-csv, each competitor method (grassRMA, sindi, ...) is
overlaid as a single labeled point on BOTH charts -- its indexing_time_sec on
the indexing chart, its searching_time_sec on the search chart -- since
competitors have no alpha/beta line, just one (recall, time) pair each.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from plot_style import INK_PRIMARY, INK_SECONDARY, PALETTE, SURFACE, alpha_color_map, load_sweep_csv, style_axes

# First run (a) is the faded/dashed reference; second run (b) is the
# foreground series and the one that gets beta labels.
STYLE = {
    "a": dict(linestyle="--", marker="s", markersize=7, alpha=0.5),
    "b": dict(linestyle="-", marker="o", markersize=8, alpha=1.0),
}


def plot_metric(df_a, df_b, label_a, label_b, y_col, y_label, title, out_path, competitors=None):
    alphas = sorted(set(df_a["alpha"]) | set(df_b["alpha"]))
    colors = alpha_color_map(alphas)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    style_axes(ax)

    for run_key, df in (("a", df_a), ("b", df_b)):
        st = STYLE[run_key]
        for alpha in sorted(df["alpha"].unique()):
            group = df[df["alpha"] == alpha].sort_values("beta")
            color = colors[alpha]
            ax.plot(
                group["recall"], group[y_col],
                color=color, linewidth=2, linestyle=st["linestyle"],
                marker=st["marker"], markersize=st["markersize"],
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1,
                alpha=st["alpha"], zorder=3 if run_key == "b" else 2,
            )
            if run_key == "b":
                offsets = [(7, 8), (7, -14), (-7, 8), (-7, -14)]
                for i, (_, row) in enumerate(group.iterrows()):
                    ha = "left" if offsets[i % len(offsets)][0] >= 0 else "right"
                    ax.annotate(
                        f"β={int(row['beta'])}",
                        (row["recall"], row[y_col]),
                        textcoords="offset points", xytext=offsets[i % len(offsets)],
                        fontsize=8, color=INK_SECONDARY, ha=ha,
                    )

    # Competitor methods (grassRMA, sindi, ...) have no alpha/beta -- just a
    # single point each. Continue the palette past the alpha slots (never
    # reuse an alpha's color for a different identity) and label directly
    # rather than adding a third legend block for one or two points.
    if competitors is not None and not competitors.empty and y_col in competitors.columns:
        for i, (_, row) in enumerate(competitors.iterrows()):
            color = PALETTE[(len(alphas) + i) % len(PALETTE)]
            ax.plot(
                row["recall"], row[y_col],
                color=color, marker="D", markersize=9,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1,
                linestyle="none", zorder=4,
            )
            # A beta label from a nearby alpha line can land underneath this
            # (e.g. grassRMA sitting right next to alpha=0.9's tightly
            # clustered points) -- an opaque backing keeps the bold
            # competitor label legible regardless of what's behind it.
            ax.annotate(
                str(row["method"]),
                (row["recall"], row[y_col]),
                textcoords="offset points", xytext=(10, 8),
                fontsize=9, color=INK_PRIMARY, fontweight="bold",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85),
            )

    alpha_handles = [
        mlines.Line2D([], [], color=colors[a], linewidth=2, marker="o", markersize=7, label=f"α = {a:g}")
        for a in alphas
    ]
    run_handles = [
        mlines.Line2D([], [], color=INK_SECONDARY, linewidth=2, **{k: v for k, v in STYLE["a"].items() if k != "alpha"}, label=label_a),
        mlines.Line2D([], [], color=INK_SECONDARY, linewidth=2, **{k: v for k, v in STYLE["b"].items() if k != "alpha"}, label=label_b),
    ]

    alpha_legend = ax.legend(handles=alpha_handles, title="Alpha", loc="upper left",
                              frameon=False, labelcolor=INK_SECONDARY)
    ax.add_artist(alpha_legend)
    # Every corner inside the axes is data-dependent (which corner is empty
    # differs between the indexing and search charts, and shifts again once
    # a competitor lands somewhere new) -- so this legend goes outside the
    # axes, below the x-axis label, where it can never collide with a point.
    ax.legend(handles=run_handles, title="Run", loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, labelcolor=INK_SECONDARY)

    ax.set_xlabel("Recall@k (%)", color=INK_SECONDARY)
    ax.set_ylabel(y_label, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_a", help="First (reference) run's merged CSV")
    parser.add_argument("csv_b", help="Second (foreground) run's merged CSV")
    parser.add_argument("--label-a", default=None, help="Legend label for csv_a (default: its parent directory name)")
    parser.add_argument("--label-b", default=None, help="Legend label for csv_b (default: its parent directory name)")
    parser.add_argument("--outdir", default=None, help="Directory for the output PNGs (default: alongside csv_b)")
    parser.add_argument("--competitors-csv", default=None,
                         help="Optional competitors CSV (method,params,dataset_size,threads,"
                              "indexing_time_sec,searching_time_sec,recall) from "
                              "append_competitor_result.sh -- overlaid as single points on the "
                              "search-time chart only (competitors have no alpha/beta line)")
    args = parser.parse_args()

    path_a, path_b = Path(args.csv_a), Path(args.csv_b)
    for p in (path_a, path_b):
        if not p.exists():
            sys.exit(f"No such file: {p}")

    label_a = args.label_a or path_a.parent.name
    label_b = args.label_b or path_b.parent.name
    outdir = Path(args.outdir) if args.outdir else path_b.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df_a = load_sweep_csv(path_a)
    df_b = load_sweep_csv(path_b)

    competitors = None
    if args.competitors_csv:
        comp_path = Path(args.competitors_csv)
        if not comp_path.exists():
            sys.exit(f"No such file: {comp_path}")
        competitors = pd.read_csv(comp_path)
        # Competitors have no separate pruning step, so their indexing_time_sec
        # IS the total -- alias it so plot_metric's y_col lookup (which uses
        # "total_indexing_time_sec" for the sweep data) finds a match here too.
        competitors["total_indexing_time_sec"] = competitors["indexing_time_sec"]

    plot_metric(
        df_a, df_b, label_a, label_b,
        "total_indexing_time_sec", "Indexing time (pruning + build, s)",
        f"Indexing time vs. recall — {label_a} vs {label_b}",
        outdir / f"compare_indexing_time_{label_a}_vs_{label_b}.png",
        competitors=competitors,
    )
    plot_metric(
        df_a, df_b, label_a, label_b,
        "searching_time_sec", "Searching time (s)",
        f"Searching time vs. recall — {label_a} vs {label_b}",
        outdir / f"compare_search_time_{label_a}_vs_{label_b}.png",
        competitors=competitors,
    )


if __name__ == "__main__":
    main()
