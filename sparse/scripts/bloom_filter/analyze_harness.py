#!/usr/bin/env python3
"""Summarize and plot one Bloom-gate harness run (sparse_bloom_sweep, any step).

Reads <run_dir>/results.csv and writes, into the same directory:
  summary.md  - baseline drift, shadow overhead and gate QPS/recall per (h, T)
  harness.png / harness.pdf - QPS vs T, recall vs T, QPS vs recall

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_harness.py <run_dir> [--compare old_results.csv]
--compare pairs gate rows with an earlier run (e.g. the previous step) on the same grid.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import BASELINE, INK_MUTED, PALETTE, SURFACE, style_axes

# Fixed hue per hash count. PALETTE[3] is replaced by #4a3aa7: it is too close
# to PALETTE[1] for overlapping curves (see plot_deadweight_hist.py).
HASH_COLOR = {2: PALETTE[0], 3: PALETTE[1], 4: PALETTE[2], 5: "#4a3aa7", 1: PALETTE[4]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--compare", help="earlier results.csv (e.g. the GCC sweep)")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.run_dir, "results.csv"))
    off = df[df["mode"] == "off"]
    gate = df[df["mode"] == "gate"].copy()
    nq = gate["queries"].iloc[0]
    base_qps = off["qps"].mean()
    base_recall = off["recall_pct"].iloc[0]
    gate["shadow_qps"] = nq / gate["shadow_search_time_sec"]
    gate["gate_x"] = gate["qps"] / base_qps
    gate["shadow_x"] = gate["shadow_qps"] / base_qps
    gate["reject_pct"] = 100 * gate["bitmap_rejections"] / gate["eligible_neighbors"]

    lines = [f"# Bloom gate harness: {os.path.basename(os.path.normpath(args.run_dir))}", "",
             f"Filter-off baseline: QPS {off['qps'].iloc[0]:,.0f} (start), "
             f"{off['qps'].iloc[-1]:,.0f} (end), drift "
             f"{100 * (off['qps'].iloc[-1] / off['qps'].iloc[0] - 1):+.1f}%; "
             f"recall {base_recall:.2f}%. Ratios use the mean of the two.", "",
             "Shadow = every check runs, nothing is rejected (pure gate overhead).", "",
             "| h | T (%) | rejected (%) | shadow QPS | shadow x | gate QPS | gate x | recall (%) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in gate.itertuples():
        lines.append(f"| {r.bloom_hashes} | {r.threshold_pct:g} | {r.reject_pct:.1f} | "
                     f"{r.shadow_qps:,.0f} | {r.shadow_x:.2f} | {r.qps:,.0f} | "
                     f"{r.gate_x:.2f} | {r.recall_pct:.2f} |")
    if args.compare:
        old = pd.read_csv(args.compare)
        m = gate.merge(old, on=["bloom_hashes", "threshold_pct"], suffixes=("", "_old"))
        lines += ["", f"Gate QPS vs `{args.compare}` (same hashes, T):", "",
                  "| h | T (%) | old QPS | new QPS | new/old | old recall | new recall |",
                  "|---|---|---|---|---|---|---|"]
        for r in m.itertuples():
            lines.append(f"| {r.bloom_hashes} | {r.threshold_pct:g} | {r.qps_old:,.0f} | "
                         f"{r.qps:,.0f} | {r.qps / r.qps_old:.2f} | "
                         f"{r.recall_pct_old:.2f} | {r.recall_pct:.2f} |")
    with open(os.path.join(args.run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for h, g in gate.groupby("bloom_hashes"):
        c = HASH_COLOR.get(h, INK_MUTED)
        axes[0].plot(g["threshold_pct"], g["qps"], "-o", color=c, lw=2, ms=8, label=f"h={h} gate")
        axes[0].plot(g["threshold_pct"], g["shadow_qps"], "--", color=c, lw=2, label=f"h={h} shadow")
        axes[1].plot(g["threshold_pct"], g["recall_pct"], "-o", color=c, lw=2, ms=8, label=f"h={h}")
        axes[2].plot(g["recall_pct"], g["qps"], "-o", color=c, lw=2, ms=8, label=f"h={h} gate")
    axes[0].axhline(base_qps, color=BASELINE, lw=2, label="filter off")
    axes[1].axhline(base_recall, color=BASELINE, lw=2, label="filter off")
    axes[2].plot([base_recall], [base_qps], "s", color=INK_MUTED, ms=9, label="filter off")
    for ax, xl, yl, t in [(axes[0], "threshold T (%)", "QPS", "QPS (gate solid, shadow dashed)"),
                          (axes[1], "threshold T (%)", "recall@10 (%)", "Recall"),
                          (axes[2], "recall@10 (%)", "QPS", "QPS vs recall")]:
        style_axes(ax)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.run_dir, f"harness.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
