#!/usr/bin/env python3
"""Summarize and plot a lossless-gate trace run (sparse_bloom_sweep --trace, step 4).

Reads <run_dir>/results.csv (headroom per ef) and, when present,
<run_dir>/results.csv.bounds.csv (head+tail bound designs). Writes into the
same directory:
  trace_summary.md - headroom table and bound-design table
  trace.png / trace.pdf - skip share per ef: perfect gate, hypothetical bounds
                          within 2x / 3x, and every measured one-line design

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_trace.py <run_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import INK_MUTED, PALETTE, SURFACE, style_axes

EF_COLOR = {100: PALETTE[0], 400: PALETTE[2], 1600: "#4a3aa7"}


def bars(ax, labels, values_by_ef, title, ylabel):
    efs = list(values_by_ef)
    x = np.arange(len(labels))
    w = 0.8 / len(efs)
    for i, ef in enumerate(efs):
        ax.bar(x + (i - (len(efs) - 1) / 2) * w, values_by_ef[ef], w * 0.92,
               color=EF_COLOR.get(ef, INK_MUTED), label=f"ef={ef}")
    style_axes(ax)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    hd = pd.read_csv(os.path.join(run_dir, "results.csv")).sort_values("ef")
    bounds_path = os.path.join(run_dir, "results.csv.bounds.csv")
    bd = pd.read_csv(bounds_path) if os.path.exists(bounds_path) else None

    lines = [f"# Lossless-gate trace: {os.path.basename(os.path.normpath(run_dir))}", "",
             "Calls traced after top_candidates filled, every 10th query. 'Within c x' = a "
             "hypothetical bound never more than c times the true score.", "",
             "| ef | calls | perfect gate (%) | within 1.25x | 1.5x | 2x | 3x | 5x | 10x "
             "| 62-bucket sketch (%) | sketch violations |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in hd.itertuples(index=False):
        v = list(r)
        # columns: ef, traced_queries, calls, oracle_start, oracle_now, 6 x ub_c*, sketch62, violations
        lines.append(f"| {v[0]} | {v[2]:,} | " + " | ".join(f"{x:.1f}" for x in [v[3]] + v[5:11])
                     + f" | {v[11]:.1f} | {v[12]} |")
    if bd is not None:
        lines += ["", "One-cache-line head+tail designs (K exact heaviest terms, tail bounded by B "
                  "bucket max codes; B=0 = one max code). Skip share of traced calls:", "",
                  "| ef | K | B | bytes | skipped (%) | violations |", "|---|---|---|---|---|---|"]
        for r in bd.sort_values(["ef", "head_terms"], ascending=[True, False]).itertuples():
            lines.append(f"| {r.ef} | {r.head_terms} | {r.tail_buckets} | {r.bytes} | "
                         f"{r.skip_pct:.1f} | {r.violations} |")
    with open(os.path.join(run_dir, "trace_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ncols = 2 if bd is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.4), facecolor=SURFACE, squeeze=False)
    labels = ["perfect gate", "within 2x", "within 3x", "62-bucket\nsketch"]
    cols = ["oracle_start_pct", "ub_c2_pct", "ub_c3_pct", "sketch62_skip_pct"]
    bars(axes[0][0], labels, {r.ef: [getattr(r, c) for c in cols] for r in hd.itertuples()},
         "Late distance calls a lossless gate skips", "skipped (%)")
    if bd is not None:
        designs = bd.drop_duplicates(["head_terms", "tail_buckets"]).sort_values("head_terms", ascending=False)
        dl = [f"K={k}\n{'tail max' if b == 0 else f'B={b}'}" for k, b in
              zip(designs["head_terms"], designs["tail_buckets"])] + ["62 bkts\nno head"]
        vals = {}
        for r in hd.itertuples():
            sub = bd[bd["ef"] == r.ef].set_index(["head_terms", "tail_buckets"])
            vals[r.ef] = [sub.loc[(k, b), "skip_pct"] for k, b in
                          zip(designs["head_terms"], designs["tail_buckets"])] + [r.sketch62_skip_pct]
        bars(axes[0][1], dl, vals, "One-cache-line bound designs", "skipped (%)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(run_dir, f"trace.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
