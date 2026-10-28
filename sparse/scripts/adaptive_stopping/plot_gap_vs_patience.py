#!/usr/bin/env python3
"""Score gap vs patience in the real search, by recall (run_stop_online_perlmutter.sh output).

Every rule point's QPS / fixed-ef QPS at the same recall (analyze_stop_online.load), as dots
per family, and per family the best setting in each 0.0025 recall window as a line.
Writes, next to the CSVs:
  gap_vs_patience_<dataset>.png/.pdf  the chart
  gap_vs_patience_<dataset>.md        the plotted window maxima (the chart's table view)

  python3 plot_gap_vs_patience.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stop_online import load  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FAMILIES = [("score gap", "#2a78d6"), ("patience", "#eb6834"), ("combo", "#1baf7a")]  # reference slots 1-3
LABEL = {"score gap": "score gap", "patience": "patience (current rule)", "combo": "gap or top-m patience"}
STEP = 0.0025
BELOW = {"patience"}  # end labels set below-right of their endpoint (the combo line ends beside patience)


def main(sweep, dataset):
    rules = load(sweep, dataset)[2]
    rules = rules[rules.Recall >= 0.975]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(1.0, color=INK, lw=1, ls=(0, (4, 3)))
    lines = [f"# Best speedup over fixed ef per {STEP} recall window, {dataset}", "",
             "| window start | " + " | ".join(LABEL[f] for f, _ in FAMILIES) + " |", "|---" * (len(FAMILIES) + 1) + "|"]
    env, ends = {}, []
    for fam, color in FAMILIES:
        g = rules[rules.family == fam]
        ax.scatter(g.Recall, g.ratio, s=22, color=color, alpha=0.35, linewidths=0, zorder=2)
        w = g.groupby(np.floor(g.Recall / STEP) * STEP).ratio.max()
        env[fam] = w
        ax.plot(w.index + STEP / 2, w.values, "-o", color=color, lw=2, ms=5, label=LABEL[fam], zorder=3)
        ends.append([w.values[-1], w.index[-1] + STEP / 2, fam])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least 0.06 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for yv, xv, fam in ends:
        below = fam in BELOW
        ax.annotate(LABEL[fam], (xv, yv), xytext=(8, -14) if below else (8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK2)
    starts = sorted(set().union(*[set(e.index) for e in env.values()]))
    for s in starts:
        lines.append(f"| {s:.4f} | " + " | ".join(f"{env[f][s]:.2f}x" if s in env[f].index else "-"
                                                  for f, _ in FAMILIES) + " |")
    ax.text(0.975, 1.0, " fixed ef", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("recall@10", color=INK2)
    ax.set_ylabel("QPS ÷ fixed-ef QPS at the same recall", color=INK2)
    ax.set_title(f"{dataset}: stopping rules in the search (M=32, efC=800)", fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_xlim(0.9745, rules.Recall.max() + 0.0045)  # room for end labels
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"gap_vs_patience_{dataset}.{ext}", dpi=150, facecolor=SURFACE)
    (Path(sweep) / f"gap_vs_patience_{dataset}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
