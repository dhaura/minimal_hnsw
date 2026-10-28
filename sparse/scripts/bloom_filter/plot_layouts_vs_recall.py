#!/usr/bin/env python3
"""Filter layouts against no filter, by recall level (one panel per dataset).

Reads <layouts_dir>/inline_<dataset>/ and listsketch_<dataset>/ (PRESET=inline /
listsketch) through analyze_gate_layouts.matched(): every gated point's QPS / the sep
gate-off QPS of its own config and job at the same recall. Plots the median per 0.01
recall bin for the separate Bloom filter, the in-row Bloom filter and the top-8 / top-16
term summaries, all at T=10. Writes, to <out_dir>:
  layouts_vs_recall.png/.pdf  the chart
  layouts_vs_recall.md        the plotted medians (the chart's table view)

  python3 plot_layouts_vs_recall.py <layouts_dir> <out_dir> <dataset> [<dataset> ...]
  (layouts_dir = results/bloom_filter/gate_layouts, out_dir = its figures/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gate_layouts import matched  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# (job folder, series in that job, label, reference categorical slot 1-4, validated light)
SERIES = [("listsketch", "list16 T=10", "top-16 terms per neighbor (32 B)", "#2a78d6"),
          ("listsketch", "list8 T=10", "top-8 terms per neighbor (16 B)", "#eb6834"),
          ("inline", "inline T=10", "Bloom, filter in the row", "#1baf7a"),
          ("listsketch", "sep T=10", "Bloom, separate filter", "#eda100")]


def medians(root, dataset):
    out = {}
    for job, s, label, _ in SERIES:
        d = Path(root) / f"{job}_{dataset}"
        if not d.exists():
            continue
        mr = matched(d, dataset)[2]
        g = mr[mr.series == s]
        b = np.floor(g.Recall * 100) / 100
        m = g.groupby(b).ratio.agg(["median", "size"])
        out[label] = m[m["size"] >= 3]["median"]
    return pd.DataFrame(out)


def main(root, out_dir, *datasets):
    tabs = {ds: medians(root, ds) for ds in datasets}
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(9.0, 6.0 * len(datasets)), 4.6), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, ds in zip(axes[0], datasets):
        t = tabs[ds]
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color=INK, lw=1, ls=(0, (4, 3)))
        ends = []
        for _, _, label, color in SERIES:
            if label in t:
                y = t[label].dropna()
                ax.plot(y.index + 0.005, y.values, "-o", color=color, lw=2, ms=5, label=label, zorder=3)
                ends.append([y.values[-1], y.index[-1] + 0.005, label])
        ends.sort()
        for i in range(1, len(ends)):  # keep end labels at least 0.045 apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045)
        for yv, xv, label in ends:
            ax.annotate(label.split(" (")[0], (xv, yv), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.text(ax.get_xlim()[0], 1.0, " no filter", va="bottom", fontsize=8, color=INK)
        ax.set_title(ds, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("recall@10 (0.01 bins)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.012)  # room for end labels
    axes[0][0].set_ylabel("QPS ÷ no-filter QPS at the same recall", color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper center", ncol=len(labels),
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"layouts_vs_recall.{ext}", dpi=150, facecolor=SURFACE)
    lines = ["# Filters vs no filter by recall (median QPS ratio per 0.01 recall bin, T=10)", ""]
    for ds, t in tabs.items():
        lines += [f"## {ds}", "", "| recall bin | " + " | ".join(t.columns) + " |", "|---" * (len(t.columns) + 1) + "|"]
        lines += [f"| {r:.2f} | " + " | ".join("-" if np.isnan(v) else f"{v:.2f}" for v in row) + " |"
                  for r, row in t.iterrows()]
        lines.append("")
    (out / "layouts_vs_recall.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])
