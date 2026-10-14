#!/usr/bin/env python3
"""Compare patience rules that watch the best 10 / 20 / 40 results against fixed ef
(sparse_bloom_sweep --patience-k, step 7).

Reads <run_dir>/results.csv (mode,ef,patience,track_k,recall_pct,searching_time_sec,qps)
and writes into the same directory:
  patience_k_summary.md - every patience point with the fixed-ef QPS at the same
                          recall (linear interpolation on the fixed-ef curve)
  patience_k.png / .pdf - QPS vs recall: fixed ef and one curve per track_k

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_patience_k.py <run_dir> [--oracle <step6 run_dir>]
--oracle adds the step 6 oracle-stop estimate to the plot (different job/graph).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import INK_SECONDARY, PALETTE, SURFACE, style_axes

K_COLOR = {10: PALETTE[0], 20: PALETTE[2], 40: "#4a3aa7"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--oracle", help="step 6 run directory (results.csv with mean_oracle_k_call)")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.run_dir, "results.csv"))
    fixed = df[df["mode"] == "fixed"].sort_values("recall_pct")
    pat = df[df["mode"] == "patience"].copy()
    pat["fixed_qps"] = np.interp(pat["recall_pct"], fixed["recall_pct"], fixed["qps"], left=np.nan, right=np.nan)
    pat["vs_fixed"] = pat["qps"] / pat["fixed_qps"]

    lines = [f"# Patience watching fewer results: {os.path.basename(os.path.normpath(args.run_dir))}", "",
             "track_k = how many best results patience watches for improvement (40 = k_hat, the default). "
             "All patience runs at ef=1600. 'vs fixed ef' = QPS over the fixed-ef QPS interpolated at the "
             "same recall (blank = outside its range).", "",
             "| fixed ef | recall (%) | QPS |", "|---|---|---|"]
    lines += [f"| {r.ef} | {r.recall_pct:.2f} | {r.qps:,.0f} |" for r in fixed.sort_values("ef").itertuples()]
    lines += ["", "| track_k | patience | recall (%) | QPS | fixed-ef QPS at same recall | vs fixed ef |",
              "|---|---|---|---|---|---|"]
    for r in pat.sort_values(["track_k", "patience"]).itertuples():
        m = "" if np.isnan(r.fixed_qps) else f"{r.fixed_qps:,.0f}"
        v = "" if np.isnan(r.vs_fixed) else f"{r.vs_fixed:.2f}x"
        lines.append(f"| {r.track_k} | {r.patience} | {r.recall_pct:.2f} | {r.qps:,.0f} | {m} | {v} |")
    with open(os.path.join(args.run_dir, "patience_k_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.plot(fixed["recall_pct"], fixed["qps"], "-s", color=INK_SECONDARY, lw=2, ms=7, label="fixed ef")
    for tk, g in pat.groupby("track_k"):
        g = g.sort_values("recall_pct")
        ax.plot(g["recall_pct"], g["qps"], "-o", color=K_COLOR.get(tk, PALETTE[4]), lw=2, ms=6,
                label=f"patience watching best {tk}")
    if args.oracle:
        o = pd.read_csv(os.path.join(args.oracle, "results.csv"))
        o = o[o["mode"] == "fixed"].sort_values("recall_pct")
        ax.plot(o["recall_pct"], o["qps"] * o["mean_calls"] / o["mean_oracle_k_call"], "--o", color=PALETTE[1],
                lw=2, ms=6, label="oracle stop, step 6 (estimate)")
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (%)")
    ax.set_ylabel("QPS (log)")
    ax.set_title("Patience by number of results watched (ef=1600)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.run_dir, f"patience_k.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
