#!/usr/bin/env python3
"""Summarize and plot a per-query early-stopping run (sparse_bloom_sweep --work, step 6).

Reads <run_dir>/results.csv (fixed-ef and patience timings, mean work per ef)
and <run_dir>/results.csv.queries.csv (one row per query and ef). Writes into
the same directory:
  work_summary.md - fixed ef vs the per-query oracle stop vs patience
  work.png / work.pdf - QPS vs recall (fixed ef, patience, oracle estimate) and
                        the per-query distribution of oracle work at the largest ef

The oracle stops each query right after the last member of its final top-k_hat
was scored, so its predictions (and recall) equal fixed ef's. Its QPS is an
ESTIMATE: fixed-ef QPS x (mean calls / mean oracle calls), i.e. it assumes
search time is proportional to distance calls.

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_work.py <run_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import INK_SECONDARY, PALETTE, SURFACE, style_axes

PAT_COLOR = {400: PALETTE[0], 800: PALETTE[2], 1600: "#4a3aa7"}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    pq = pd.read_csv(os.path.join(run_dir, "results.csv.queries.csv"))
    fixed = df[df["mode"] == "fixed"].sort_values("ef").copy()
    pat = df[df["mode"] == "patience"].copy()

    fixed["after_fill_pct"] = 100 * (fixed["mean_calls"] - fixed["mean_fill_call"]) / fixed["mean_calls"]
    fixed["oracle_x"] = fixed["mean_calls"] / fixed["mean_oracle_k_call"]
    fixed["oracle_qps_est"] = fixed["qps"] * fixed["oracle_x"]
    by_recall = fixed.sort_values("recall_pct")
    pat["fixed_qps_at_recall"] = np.interp(pat["recall_pct"], by_recall["recall_pct"], by_recall["qps"],
                                           left=np.nan, right=np.nan)
    pat["vs_fixed"] = pat["qps"] / pat["fixed_qps_at_recall"]

    lines = [f"# Early-stopping headroom: {os.path.basename(os.path.normpath(run_dir))}", "",
             "Calls = layer-0 distance calls per query (mean). Oracle = stop each query right after "
             "the last member of its final top-k_hat was scored: identical predictions. "
             "Oracle QPS is an estimate (time assumed proportional to calls).", "",
             "| ef | recall (%) | QPS | calls | after list filled (%) | oracle calls | top-10 oracle calls "
             "| oracle saving | oracle QPS (est.) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in fixed.itertuples():
        lines.append(f"| {r.ef} | {r.recall_pct:.2f} | {r.qps:,.0f} | {r.mean_calls:,.0f} | {r.after_fill_pct:.1f} "
                     f"| {r.mean_oracle_k_call:,.0f} | {r.mean_oracle_10_call:,.0f} | {r.oracle_x:.2f}x "
                     f"| {r.oracle_qps_est:,.0f} |")
    lines += ["", "Patience (stop after N expansions without improving the top-k_hat), measured QPS; "
              "'vs fixed ef' interpolates the fixed-ef curve at the same recall (blank = outside its range).", "",
              "| ef | patience | recall (%) | QPS | fixed-ef QPS at same recall | vs fixed ef |",
              "|---|---|---|---|---|---|"]
    for r in pat.sort_values(["ef", "patience"]).itertuples():
        m = "" if np.isnan(r.fixed_qps_at_recall) else f"{r.fixed_qps_at_recall:,.0f}"
        v = "" if np.isnan(r.vs_fixed) else f"{r.vs_fixed:.2f}x"
        lines.append(f"| {r.ef} | {r.patience} | {r.recall_pct:.2f} | {r.qps:,.0f} | {m} | {v} |")
    top = pq[pq["ef"] == pq["ef"].max()]
    frac = (top["oracle_k_call"] / top["total_calls"]).to_numpy()
    lines += ["", f"Per query at ef={top['ef'].iloc[0]} ({len(top)} queries): oracle work / total work "
              f"p10 {np.percentile(frac, 10):.2f}, median {np.median(frac):.2f}, p90 {np.percentile(frac, 90):.2f}, "
              f"max {frac.max():.2f}."]
    with open(os.path.join(run_dir, "work_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    ax = axes[0]
    ax.plot(fixed["recall_pct"], fixed["qps"], "-s", color=INK_SECONDARY, lw=2, ms=7, label="fixed ef (measured)")
    ax.plot(fixed["recall_pct"], fixed["oracle_qps_est"], "--o", color=PALETTE[1], lw=2, ms=7,
            label="per-query oracle stop (estimate)")
    for e, g in pat.groupby("ef"):
        g = g.sort_values("recall_pct")
        ax.plot(g["recall_pct"], g["qps"], "-o", color=PAT_COLOR.get(e, PALETTE[4]), lw=2, ms=6,
                label=f"patience 16-256, ef={e}")
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (%)")
    ax.set_ylabel("QPS (log)")
    ax.set_title("Fixed ef vs early stopping", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.hist(frac, bins=np.linspace(0, 1, 41), color=PALETTE[0])
    style_axes(ax)
    ax.set_xlabel(f"oracle calls / total calls, per query (ef={top['ef'].iloc[0]})")
    ax.set_ylabel("queries")
    ax.set_title("How early each query's final top-k is complete", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(run_dir, f"work.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
