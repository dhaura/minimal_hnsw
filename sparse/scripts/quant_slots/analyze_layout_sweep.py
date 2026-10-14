#!/usr/bin/env python3
"""Compare original vs slotted row layouts over a full SparseHNSW sweep.

Reads every layout_sweep_*.csv in <dir> (written by sparse_hnsw_sweep with
slot_size>0: each point appears twice, Model "..._slots" for the slotted
layout, measured alternately on the same graph). Writes into <dir>:
  layout_sweep_summary.md - per recall threshold r: best QPS with recall >= r
                            for each layout, their ratio, and the configs;
                            plus the paired per-config ratio distribution
  layout_sweep.png / .pdf - Pareto fronts (QPS vs recall) and ratio per threshold

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_layout_sweep.py <dir> [--thresholds 0.80,0.85,...]
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import BASELINE, INK_SECONDARY, PALETTE, SURFACE, style_axes

DEFAULT_T = "0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.995,0.997,0.998"


def load(d):
    files = sorted(glob.glob(os.path.join(d, "layout_sweep_*.csv")))
    if not files:
        sys.exit(f"no layout_sweep_*.csv in {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["layout"] = np.where(df["Model"].str.endswith("_slots"), "slotted", "original")
    df["config"] = df["Params"].str.replace(r" slots=\d+", "", regex=True)
    return df, files


def pareto(g):
    g = g.sort_values("Recall", ascending=False)
    keep, best = [], -1.0
    for r in g.itertuples():
        if r.QPS > best:
            keep.append(r.Index)
            best = r.QPS
    return g.loc[keep].sort_values("Recall")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--thresholds", default=DEFAULT_T)
    args = ap.parse_args()
    df, files = load(args.dir)
    o = df[df["layout"] == "original"]
    s = df[df["layout"] == "slotted"]

    paired = o.set_index("config")[["QPS", "Recall"]].join(
        s.set_index("config")[["QPS", "Recall"]], lsuffix="_o", rsuffix="_s", how="inner")
    if not np.allclose(paired["Recall_o"], paired["Recall_s"]):
        sys.exit("recall differs between layouts for some config")
    ratio = paired["QPS_s"] / paired["QPS_o"]

    rows = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        bo, bs = o[o["Recall"] >= t], s[s["Recall"] >= t]
        if bo.empty or bs.empty:
            continue
        ro, rs = bo.loc[bo["QPS"].idxmax()], bs.loc[bs["QPS"].idxmax()]
        rows.append((t, ro.QPS, rs.QPS, rs.QPS / ro.QPS, ro.config, rs.config))

    lines = [f"# Row layout full sweep: {os.path.basename(os.path.normpath(args.dir))}", "",
             f"{len(files)} CSV files, {len(paired)} configs measured in both layouts (same graph, alternated runs, "
             "identical predictions).", "",
             f"Paired per-config ratio (slotted / original QPS): min {ratio.min():.3f}x, "
             f"p5 {ratio.quantile(0.05):.3f}x, median {ratio.median():.3f}x, max {ratio.max():.3f}x; "
             f"{(ratio < 1).sum()} of {len(ratio)} configs slower.", "",
             "Best QPS among configs with recall >= r:", "",
             "| recall >= | original QPS | slotted QPS | slotted / original | original config | slotted config |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {t:g} | {qo:,.0f} | {qs:,.0f} | {x:.2f}x | {co} | {cs} |" for t, qo, qs, x, co, cs in rows]
    with open(os.path.join(args.dir, "layout_sweep_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    ax = axes[0]
    for name, g, c in (("original layout", o, INK_SECONDARY), ("fixed-size slots", s, PALETTE[0])):
        p = pareto(g)
        ax.plot(p["Recall"] * 100, p["QPS"], "-o", color=c, lw=2, ms=5, label=f"{name} (Pareto front)")
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (%)")
    ax.set_ylabel("QPS (log)")
    ax.set_title("SparseHNSW on msmarco_full: Pareto fronts", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    t = [r[0] for r in rows]
    ax.bar(range(len(t)), [r[3] for r in rows], color=PALETTE[0], width=0.7)
    ax.axhline(1.0, color=BASELINE, lw=2)
    style_axes(ax)
    ax.set_xticks(range(len(t)), [f"{x:g}" for x in t], rotation=45, fontsize=8)
    ax.set_xlabel("recall threshold r")
    ax.set_ylabel("best slotted QPS / best original QPS")
    ax.set_title("Speedup of the best config at each recall", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.dir, f"layout_sweep.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
