#!/usr/bin/env python3
"""Matched-recall comparison of the Bloom gate against simply lowering ef.

Reads <run_dir>/results.csv written by `sparse_bloom_sweep ... --ef-sweep`
(columns mode,bloom_hashes,threshold_pct,ef,recall_pct,searching_time_sec,qps)
and writes into the same directory:
  ef_sweep_summary.md - every gated point next to the filter-off QPS at the same
                        recall (linear interpolation on the filter-off ef curve)
  ef_sweep.png / .pdf - QPS vs recall: filter-off curve plus one curve per T

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_ef_sweep.py <run_dir>
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

T_COLOR = {20.0: PALETTE[0], 25.0: PALETTE[1], 30.0: PALETTE[2], 35.0: "#4a3aa7"}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    off = df[df["mode"] == "off"].sort_values("recall_pct")
    gate = df[df["mode"] == "gate"].copy()

    # Filter-off QPS at the gated point's recall; NaN outside the off curve's range.
    gate["off_qps_at_recall"] = np.interp(gate["recall_pct"], off["recall_pct"], off["qps"],
                                          left=np.nan, right=np.nan)
    gate["speedup_matched"] = gate["qps"] / gate["off_qps_at_recall"]

    lines = [f"# Gate vs lowering ef: {os.path.basename(os.path.normpath(run_dir))}", "",
             "Filter-off curve (one point per ef):", "",
             "| ef | recall (%) | QPS |", "|---|---|---|"]
    lines += [f"| {r.ef} | {r.recall_pct:.2f} | {r.qps:,.0f} |" for r in off.sort_values("ef").itertuples()]
    lines += ["", "h=1 gate; 'off QPS at same recall' interpolates the curve above "
              "(blank = recall outside its range).", "",
              "| T (%) | ef | recall (%) | gate QPS | off QPS at same recall | speedup |",
              "|---|---|---|---|---|---|"]
    for r in gate.sort_values(["threshold_pct", "ef"]).itertuples():
        matched = "" if np.isnan(r.off_qps_at_recall) else f"{r.off_qps_at_recall:,.0f}"
        speed = "" if np.isnan(r.speedup_matched) else f"{r.speedup_matched:.2f}x"
        lines.append(f"| {r.threshold_pct:g} | {r.ef} | {r.recall_pct:.2f} | {r.qps:,.0f} | {matched} | {speed} |")
    with open(os.path.join(run_dir, "ef_sweep_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.plot(off["recall_pct"], off["qps"], "-s", color=INK_SECONDARY, lw=2, ms=8, label="filter off (ef sweep)")
    for t, g in gate.groupby("threshold_pct"):
        g = g.sort_values("recall_pct")
        ax.plot(g["recall_pct"], g["qps"], "-o", color=T_COLOR.get(t, PALETTE[4]), lw=2, ms=8,
                label=f"h=1 gate, T={t:g}%")
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (%)")
    ax.set_ylabel("QPS (log)")
    ax.set_title("Gate vs lowering ef (one point per ef)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(run_dir, f"ef_sweep.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
