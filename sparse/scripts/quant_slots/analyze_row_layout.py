#!/usr/bin/env python3
"""Summarize and plot a row layout A/B run (the former sparse_row_layout driver, job 59713310).

Reads <run_dir>/results.csv (slot_size, ef, layout, median_sec, qps, ...) and
writes into the same directory:
  layout_summary.md - slotted vs original QPS per (slot_size, ef), with overflow
  layout.png / layout.pdf - slotted / original QPS ratio vs ef, one line per slot_size

Both layouts run on the same graph and return identical predictions (the
driver aborts otherwise), so the ratio is a pure speed comparison.

Usage (needs pandas + matplotlib: `module load python`):
  python3 analyze_row_layout.py <run_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import BASELINE, PALETTE, SURFACE, style_axes


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    # The driver that wrote these results called the slot size "stride".
    df = pd.read_csv(os.path.join(run_dir, "results.csv")).rename(columns={"stride": "slot_size"})
    key = ["slot_size", "ef"]
    o = df[df["layout"] == "original"].set_index(key)
    s = df[df["layout"] == "slotted"].set_index(key)
    m = o.join(s, lsuffix="_o", rsuffix="_s").reset_index()
    m["ratio"] = m["qps_s"] / m["qps_o"]

    lines = [f"# Row layout A/B: {os.path.basename(os.path.normpath(run_dir))}", "",
             "Original = offset-indexed quantized rows (row_off lookup per call). Slotted = fixed-size, "
             "64-byte-aligned slots; rows longer than a slot overflow to the original blob. "
             "Medians of 5 alternated runs; predictions identical.", "",
             "| slot_size (B) | overflow rows (%) | slot memory (GB) | ef | recall (%) | original QPS | slotted QPS | slotted / original |",
             "|---|---|---|---|---|---|---|---|"]
    for r in m.sort_values(key).itertuples():
        lines.append(f"| {r.slot_size} | {r.overflow_pct_o:.1f} | {r.slot_bytes_o / 1e9:.2f} | {r.ef} | {r.recall_pct_o:.2f} "
                     f"| {r.qps_o:,.0f} | {r.qps_s:,.0f} | {r.ratio:.3f}x |")
    with open(os.path.join(run_dir, "layout_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    for i, (slot_size, g) in enumerate(m.groupby("slot_size")):
        g = g.sort_values("ef")
        ax.plot(g["ef"], g["ratio"], "-o", color=PALETTE[i % len(PALETTE)], lw=2, ms=7,
                label=f"slot_size {slot_size} B ({g['overflow_pct_o'].iloc[0]:.1f}% overflow)")
    ax.axhline(1.0, color=BASELINE, lw=2, label="original layout")
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xticks(sorted(m["ef"].unique()), [str(e) for e in sorted(m["ef"].unique())])
    ax.set_xlabel("ef")
    ax.set_ylabel("slotted QPS / original QPS")
    ax.set_title("Fixed-size row slots vs offset-indexed rows (same recall)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(run_dir, f"layout.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
