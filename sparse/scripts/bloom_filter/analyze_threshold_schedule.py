#!/usr/bin/env python3
"""Advisor question: lower the Bloom threshold T towards 0 as recall rises.

Reads bloom_<dataset>_*.csv from a GATE_SET=schedule sweep (gate off and
T = 0,1,2,3,4,5,8,10) and writes, next to them:
  schedule_summary_<dataset>.md  best QPS at recall >= r per series; the best
                                 Bloom-on choice per recall bin vs gate off;
                                 matched-recall ratio of each series vs gate off
  schedule_<dataset>.png/.pdf    best-QPS fronts for off, T=0, 2, 5, 10

  python3 analyze_threshold_schedule.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
INK, T_COLORS = "#0b0b0b", {0: "#86b6ef", 2: "#3987e5", 5: "#1c5cab", 10: "#0d366b"}  # ordinal blue ramp


def load(sweep, dataset):
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(sweep).glob(f"bloom_{dataset}_*.csv"))], ignore_index=True)
    p = df.Params
    df["ef"] = p.str.extract(r" ef=(\d+)")[0].astype(int)
    df["T"] = p.str.extract(r" T=([\d.]+)")[0].astype(float)  # NaN = gate off
    df["curve"] = p.str.replace(r" (ef=\d+|bloom_h=\d+|T=[\d.]+)", "", regex=True)
    df["series"] = ["off" if np.isnan(t) else f"T={t:g}" for t in df["T"]]
    return df


def best(d, r):
    d = d[d.Recall >= r]
    return d.QPS.max() if len(d) else np.nan


def main(sweep, dataset):
    df = load(sweep, dataset)
    on = df[df["T"].notna()]
    lines = [f"# Threshold schedule, {dataset} ({len(df):,} points)", ""]
    cols = ["off"] + sorted(on.series.unique(), key=lambda s: float(s[2:]))
    lines += ["## Best QPS (k) at recall >= r", "", "| recall | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
    for r in BINS:
        lines.append(f"| {r:.2f} | " + " | ".join(f"{best(df[df.series == c], r) / 1e3:.0f}" for c in cols) + " |")
    lines.append("")

    lines += ["## Best Bloom-on choice per recall bin vs gate off", "",
              "| recall | best Bloom-on series | QPS | gate off QPS | ratio |", "|---|---|---|---|---|"]
    for r in BINS:
        d = on[on.Recall >= r]
        if not len(d):
            continue
        w = d.loc[d.QPS.idxmax()]
        lines.append(f"| {r:.2f} | {w.series} | {w.QPS:,.0f} | {best(df[df['T'].isna()], r):,.0f} | "
                     f"{w.QPS / best(df[df['T'].isna()], r):.3f} |")

    # Matched recall: each gated point vs the gate-off ef curve of the same config.
    rows = []
    for curve, o in df[df["T"].isna()].groupby("curve"):
        o = o.groupby("Recall", as_index=False).QPS.max().sort_values("Recall")
        g = on[(on.curve == curve) & on.Recall.between(o.Recall.min(), o.Recall.max())]
        if len(o) > 1 and len(g):
            rows.append(g.assign(ratio=g.QPS / np.exp(np.interp(g.Recall, o.Recall, np.log(o.QPS)))))
    same = on.merge(df[df["T"].isna()][["curve", "ef", "Recall"]], on=["curve", "ef"], suffixes=("", "_off"))
    lines += ["", "## Matched recall vs gate off (same config, interpolated ef curve)", "",
              "| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp, median) |",
              "|---|---|---|---|"]
    mr = pd.concat(rows)
    for s, g in mr.groupby("series", sort=False):
        dr = 100 * (same[same.series == s].Recall - same[same.series == s].Recall_off).median()
        lines.append(f"| {s} | {len(g)} | {g.ratio.median():.3f} [{g.ratio.quantile(.1):.2f}, "
                     f"{g.ratio.quantile(.9):.2f}] | {dr:+.2f} |")
    out = Path(sweep) / f"schedule_summary_{dataset}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    grid = np.linspace(0.80, df.Recall.max(), 300)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(grid, [best(df[df["T"].isna()], r) for r in grid], color=INK, lw=2, label="gate off (SparseHNSW)")
    for t, c in T_COLORS.items():
        ax.plot(grid, [best(on[on["T"] == t], r) for r in grid], color=c, lw=2, label=f"T = {t}")
    ax.set_yscale("log")
    ax.set_title(f"{dataset}: Bloom gate")
    ax.set_xlabel("recall@10 (best QPS at recall >= x)")
    ax.set_ylabel("QPS (64 threads)")
    ax.grid(alpha=0.25, color="#52514e", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"schedule_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
