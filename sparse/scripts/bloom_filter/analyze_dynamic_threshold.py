#!/usr/bin/env python3
"""Per-query Bloom threshold schedules vs fixed thresholds vs gate off.

Reads bloom_<dataset>_*.csv from a PRESET=dynamic sweep. A schedule
"sched=Te>Tl@F" uses threshold Te for the first F*ef expansions after the
result list fills, then Tl (-1 = gate off). Writes, next to the CSVs:
  dynamic_summary_<dataset>.md  per series: matched-recall QPS ratio vs gate off
                                (same config, gate-off ef curve interpolated in
                                log QPS) and recall change at the same ef; best
                                QPS at recall >= r for the best series per group
  dynamic_<dataset>.png/.pdf    matched-recall ratio per series, by group

  python3 analyze_dynamic_threshold.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
GROUPS = {"fixed T": "#3987e5", "off early, on late": "#c4632f", "on early, off late": "#7a7a74"}
INK = "#0b0b0b"


def readable(series):
    """'sched -1>25@0.5' -> 'no filter for first 50%, then T=25'; 'T=10' -> 'fixed T=10'."""
    if not series.startswith("sched "):
        return "fixed " + series if series.startswith("T=") else series
    early, rest = series[6:].split(">")
    late, frac = rest.split("@")
    phase = lambda t: "no filter" if float(t) < 0 else f"T={float(t):g}"
    return f"{phase(early)} for first {100 * float(frac):g}%, then {phase(late)}"


def load(sweep, dataset):
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(sweep).glob(f"bloom_{dataset}_*.csv"))], ignore_index=True)
    p = df.Params
    df["ef"] = p.str.extract(r" ef=(\d+)")[0].astype(int)
    t = p.str.extract(r" T=([\d.]+)")[0]
    s = p.str.extract(r" sched=(\S+)")[0]
    df["series"] = np.where(s.notna(), "sched " + s.fillna(""), np.where(t.notna(), "T=" + t.fillna(""), "off"))
    df["group"] = np.where(s.notna(), np.where(s.fillna("").str.startswith("-1>"), "off early, on late",
                                               "on early, off late"), np.where(t.notna(), "fixed T", "off"))
    df["label"] = df.series.map(readable)
    df["curve"] = p.str.replace(r" (ef=\d+|bloom_h=\d+|T=[\d.]+|sched=\S+)", "", regex=True)
    return df


def main(sweep, dataset):
    df = load(sweep, dataset)
    off, on = df[df.group == "off"], df[df.group != "off"]
    rows = []
    for curve, o in off.groupby("curve"):
        o = o.groupby("Recall", as_index=False).QPS.max().sort_values("Recall")
        g = on[(on.curve == curve) & on.Recall.between(o.Recall.min(), o.Recall.max())]
        if len(o) > 1 and len(g):
            rows.append(g.assign(ratio=g.QPS / np.exp(np.interp(g.Recall, o.Recall, np.log(o.QPS)))))
    mr = pd.concat(rows)
    same = on.merge(off[["curve", "ef", "Recall"]], on=["curve", "ef"], suffixes=("", "_off"))
    stats = (mr.groupby(["group", "label"]).ratio.quantile([0.1, 0.5, 0.9]).unstack()
             .join(same.assign(dr=100 * (same.Recall - same.Recall_off)).groupby(["group", "label"]).dr.median())
             .join(mr.groupby(["group", "label"]).size().rename("n")).reset_index()
             .sort_values(["group", 0.5], ascending=[True, False]))
    lines = [f"# Per-query threshold schedules, {dataset} ({len(df):,} points)", "",
             "Percentages are of the search after the result list fills (a query runs about ef "
             "expansions there). Ratio = QPS / gate-off QPS at the same recall (same config).", "",
             "| group | schedule | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |",
             "|---|---|---|---|---|"]
    for r in stats.itertuples(index=False):
        lines.append(f"| {r[0]} | {r[1]} | {r.n} | {r[3]:.3f} [{r[2]:.2f}, {r[4]:.2f}] | {r.dr:+.2f} |")
    best = stats.groupby("group").head(1)
    lines += ["", "## Best QPS at recall >= r (best series per group by median ratio)", "",
              "| recall | gate off | " + " | ".join(f"{g}: {s}" for g, s in zip(best.group, best.label)) + " |",
              "|---" * (len(best) + 2) + "|"]
    bq = lambda d, r: d[d.Recall >= r].QPS.max() if (d.Recall >= r).any() else np.nan
    for r in BINS:
        lines.append(f"| {r:.2f} | {bq(off, r):,.0f} | " + " | ".join(
            f"{bq(df[df.label == s], r):,.0f}" for s in best.label) + " |")
    (Path(sweep) / f"dynamic_summary_{dataset}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(stats) + 1.6))
    y = np.arange(len(stats))[::-1]
    for yi, r in zip(y, stats.itertuples(index=False)):
        ax.errorbar(r[3], yi, xerr=[[r[3] - r[2]], [r[4] - r[3]]], fmt="o", color=GROUPS[r[0]], capsize=2, ms=5)
    ax.axvline(1.0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  ({dr:+.2f} pp recall)" for s, dr in zip(stats.label, stats.dr)], fontsize=8.5)
    ax.set_xlabel("QPS / gate-off QPS at the same recall (median, p10-p90)")
    ax.set_title(f"{dataset}: per-query Bloom threshold schedules vs no filter (= 1.0)")
    for g, c in GROUPS.items():
        ax.plot([], [], "o", color=c, label=g)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, color="#52514e", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"dynamic_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
