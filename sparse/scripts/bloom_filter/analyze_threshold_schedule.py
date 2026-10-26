#!/usr/bin/env python3
"""Advisor question: lower the Bloom threshold T towards 0 as recall rises.

Reads bloom_<dataset>_*.csv from GATE_SET=schedule / schedule_high sweeps
(gate off and any thresholds T; each job is compared with its own gate-off
pass) and writes, next to them:
  schedule_summary_<dataset>.md  best QPS at recall >= r per series; the best
                                 Bloom-on choice per recall bin vs gate off;
                                 matched-recall ratio of each series vs gate off
  schedule_<dataset>.png/.pdf    left: best-QPS fronts for gate off and every T;
                                 right: gate off vs the best gate-on QPS at each
                                 recall over all T (a per-recall threshold choice),
                                 with the winning T labelled at each recall bin

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
INK, T_LIGHT, T_DARK = "#0b0b0b", "#86b6ef", "#0d366b"  # T <= 10: ordinal blue ramp, light = low T
HI_LIGHT, HI_DARK = "#f2a65a", "#8a2f0a"  # T > 10: ordinal orange ramp, a separate hue


def load(sweep, dataset):
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(sweep).glob(f"bloom_{dataset}_*.csv"))], ignore_index=True)
    p = df.Params
    df["ef"] = p.str.extract(r" ef=(\d+)")[0].astype(int)
    df["T"] = p.str.extract(r" T=([\d.]+)")[0].astype(float)  # NaN = gate off
    df["curve"] = p.str.replace(r" (ef=\d+|bloom_h=\d+|T=[\d.]+)", "", regex=True) + " job=" + df.JobID.astype(str)
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
    ts = sorted(on["T"].unique())
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("t", [T_LIGHT, T_DARK])
    hi_ramp = matplotlib.colors.LinearSegmentedColormap.from_list("hi", [HI_LIGHT, HI_DARK])
    lo_ts, hi_ts = [t for t in ts if t <= 10], [t for t in ts if t > 10]
    off = np.array([best(df[df["T"].isna()], r) for r in grid])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax.plot(grid, off, color=INK, lw=2, label="gate off (SparseHNSW)")
    for i, t in enumerate(ts):
        front = np.array([best(on[on["T"] == t], r) for r in grid])
        c = (ramp(lo_ts.index(t) / max(1, len(lo_ts) - 1)) if t <= 10
             else hi_ramp(hi_ts.index(t) / max(1, len(hi_ts) - 1)))
        ax.plot(grid, front, color=c, lw=1.5, label=f"T = {t:g}")
    ax.set_yscale("log")
    ax.set_title(f"{dataset}: best QPS at recall >= x")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (64 threads)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    # Right: a per-recall threshold choice = best gate-on QPS over all T at each recall.
    ax2.plot(grid, off, color=INK, lw=2, label="gate off (SparseHNSW)")
    ax2.plot(grid, [best(on, r) for r in grid], color=T_DARK, lw=2, label="gate on, best T at each recall")
    for r in BINS:
        d = on[on.Recall >= r]
        if not len(d) or r > grid[-1]:
            continue
        w = d.loc[d.QPS.idxmax()]
        ax2.plot(r, w.QPS, "o", color=T_DARK, ms=5)
        if r in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99):  # label row along the bottom, clear of the curves
            ax2.text(r, 0.03, f"T={w['T']:g}\n{w.QPS / best(df[df['T'].isna()], r):.2f}x",
                     transform=ax2.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color="#48515b")
    ax2.set_yscale("log")
    ax2.set_title("Different threshold per recall target vs gate off")
    ax2.set_xlabel("recall@10 (labels: winning T, QPS vs gate off)")
    ax2.set_ylabel("QPS (64 threads)")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo / 2.2, hi)  # room for the label row
    for a in (ax, ax2):
        a.grid(alpha=0.25, color="#52514e", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"schedule_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
