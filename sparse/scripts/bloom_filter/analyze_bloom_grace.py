#!/usr/bin/env python3
"""Grace (Intel Cascade Lake) vs Perlmutter (AMD Zen3) Bloom-gate verification.

Reuses load()/best() from analyze_threshold_schedule.py, then reports, per
dataset, the Grace ratio next to the published Perlmutter ratio:

  1. matched-recall QPS ratio (filter ON vs the filter-OFF ef curve of the SAME
     config, log-QPS interpolated in recall) per T, vs the Perlmutter reference;
  2. best filter-on vs best filter-off QPS per recall bin, with the winning T;
  3. schedule_<dataset>.png with every T actually present (the original plot
     hard-codes T=0,2,5,10) and the real thread count.

  python3 analyze_bloom_grace.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_threshold_schedule import BINS, best, load  # noqa: E402

# Published Perlmutter (AMD EPYC 7763, 64 threads) matched-recall ratios.
# None = not reported for that T.  "~" values are the flat T=0-10 band.
PM_RATIO = {
    "msmarco_full": {0: 0.746, 5: 0.75, 10: 0.759, 15: 0.72, 20: 0.61, 25: 0.45,
                     30: None, 35: None},
    "nq_splade": {0: 0.755, 5: 0.755, 10: 0.755, 15: 0.76, 20: 0.69, 25: 0.49},
}
PM_BIN_RANGE = {"msmarco_full": (0.67, 0.85), "nq_splade": (0.64, 0.86)}
PM_APPROX = {"msmarco_full": {5}, "nq_splade": {0, 5, 10}}  # reported as a band, not a point


def matched_recall(df, on):
    """Each gated point vs the gate-off ef curve of its own config."""
    rows = []
    for curve, o in df[df["T"].isna()].groupby("curve"):
        o = o.groupby("Recall", as_index=False).QPS.max().sort_values("Recall")
        g = on[(on.curve == curve) & on.Recall.between(o.Recall.min(), o.Recall.max())]
        if len(o) > 1 and len(g):
            rows.append(g.assign(ratio=g.QPS / np.exp(np.interp(g.Recall, o.Recall, np.log(o.QPS)))))
    return pd.concat(rows) if rows else pd.DataFrame()


def main(sweep, dataset):
    df = load(sweep, dataset)
    on, off = df[df["T"].notna()], df[df["T"].isna()]
    threads = int(df.Threads.mode()[0])
    mr = matched_recall(df, on)
    same = on.merge(off[["curve", "ef", "Recall"]], on=["curve", "ef"], suffixes=("", "_off"))
    pm, approx = PM_RATIO.get(dataset, {}), PM_APPROX.get(dataset, set())

    L = [f"# Bloom gate on Grace vs Perlmutter - {dataset}", "",
         f"Grace: Intel Xeon Gold 6248R (Cascade Lake), {threads} threads, "
         f"{len(df):,} points, {df.curve.nunique()} configs.",
         "Perlmutter reference: AMD EPYC 7763 (Zen3), 64 threads.", "",
         "## 1. Matched-recall QPS ratio (filter on vs filter off, same config)", "",
         "| T | points | Grace ratio median [p10, p90] | Perlmutter ratio | Grace - PM | recall change (pp) |",
         "|---|---|---|---|---|---|"]
    for t, g in sorted(mr.groupby("T"), key=lambda kv: kv[0]) if len(mr) else []:
        s = same[same["T"] == t]
        dr = 100 * (s.Recall - s.Recall_off).median() if len(s) else float("nan")
        p = pm.get(int(t))
        ptxt = "n/a" if p is None else (f"~{p:.3f}" if int(t) in approx else f"{p:.3f}")
        dtxt = "n/a" if p is None else f"{g.ratio.median() - p:+.3f}"
        L.append(f"| {t:g} | {len(g)} | {g.ratio.median():.3f} "
                 f"[{g.ratio.quantile(.1):.2f}, {g.ratio.quantile(.9):.2f}] | {ptxt} | {dtxt} | {dr:+.2f} |")

    lo, hi = PM_BIN_RANGE.get(dataset, (float("nan"), float("nan")))
    L += ["", "## 2. Best filter-on vs best filter-off QPS per recall bin", "",
          f"Perlmutter reference range for this ratio: {lo:.2f}-{hi:.2f}x.", "",
          "| recall | winning T | filter-on QPS | filter-off QPS | Grace ratio |", "|---|---|---|---|---|"]
    for r in BINS:
        d, o = on[on.Recall >= r], best(off, r)
        if not len(d) or not np.isfinite(o):
            continue
        w = d.loc[d.QPS.idxmax()]
        L.append(f"| {r:.2f} | {w['T']:g} | {w.QPS:,.0f} | {o:,.0f} | {w.QPS / o:.3f} |")

    out = Path(sweep) / f"grace_vs_perlmutter_{dataset}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")

    ts = sorted(on["T"].unique())
    cmap = plt.get_cmap("Blues")
    grid = np.linspace(0.80, df.Recall.max(), 300)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(grid, [best(off, r) for r in grid], color="#0b0b0b", lw=2, label="filter off (SparseHNSW)")
    for i, t in enumerate(ts):
        ax.plot(grid, [best(on[on["T"] == t], r) for r in grid],
                color=cmap(0.35 + 0.6 * i / max(len(ts) - 1, 1)), lw=1.8, label=f"T = {t:g}")
    ax.set_yscale("log")
    ax.set_title(f"{dataset}: Bloom gate (Grace, Cascade Lake)")
    ax.set_xlabel("recall@10 (best QPS at recall >= x)")
    ax.set_ylabel(f"QPS ({threads} threads)")
    ax.grid(alpha=0.25, color="#52514e", lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"schedule_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
