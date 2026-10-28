#!/usr/bin/env python3
"""Gate layout tests (PRESET=inline / listsketch) with gate-stats counts.

Reads bloom_<dataset>_*.csv and their .gate.csv sidecars. Series = layout x threshold:
sep (separate filter array), inline (filter row as each slot's first line), list<n>
(top-n terms beside each neighbor id). Writes, next to them:
  layouts_summary_<dataset>.md  QPS / gate-off QPS at the same recall (the sep gate-off ef
                                curve of the same config and job, interpolated in log QPS)
                                and recall change at the same ef; gate decisions vs an
                                ideal gate (exact query-term overlap > T% of query terms)
  layouts_<dataset>.png/.pdf    matched-recall ratio per series; reject shares per series

  python3 analyze_gate_layouts.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED = "#0b0b0b", "#7a7a74"
COLORS = {"sep": "#2a78d6", "inline": "#c4632f", "list8": "#8fbf6a", "list16": "#2e8b57", "list4": "#b9d99f"}


def series(p):
    lay = p.str.extract(r" layout=(\S+)")[0].fillna("sep")
    t = p.str.extract(r" T=([\d.]+)")[0]
    return lay, np.where(t.isna(), "off", "T=" + t.fillna("")), lay + " " + np.where(t.isna(), "off", "T=" + t.fillna(""))


def matched(sweep, dataset):
    """Result files, all points, gated points with their matched-recall ratio, same-ef pairs."""
    files = sorted(f for f in Path(sweep).glob(f"bloom_{dataset}_*.csv") if not f.name.endswith(".gate.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ef"] = df.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    df["layout"], df["thr"], df["series"] = series(df.Params)
    df["curve"] = df.Params.str.replace(r" (ef=\d+|bloom_h=\d+|T=[\d.]+|layout=\S+)", "", regex=True) \
        + " job=" + df.JobID.astype(str)
    base = df[df.series == "sep off"]
    rows = []
    for curve, o in base.groupby("curve"):
        o = o.groupby("Recall", as_index=False).QPS.max().sort_values("Recall")
        g = df[(df.curve == curve) & (df.series != "sep off") & df.Recall.between(o.Recall.min(), o.Recall.max())]
        if len(o) > 1 and len(g):
            rows.append(g.assign(ratio=g.QPS / np.exp(np.interp(g.Recall, o.Recall, np.log(o.QPS)))))
    same = df.merge(base[["curve", "ef", "Recall"]], on=["curve", "ef"], suffixes=("", "_off"))
    return files, df, pd.concat(rows), same


def main(sweep, dataset):
    files, df, mr, same = matched(sweep, dataset)
    lines = [f"# Gate layouts, {dataset} ({len(df):,} points)", "",
             "Ratio = QPS / sep gate-off QPS at the same recall (same config and job).", "",
             "| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |", "|---|---|---|---|"]
    for s, g in mr.groupby("series"):
        dr = 100 * (same[same.series == s].Recall - same[same.series == s].Recall_off).median()
        lines.append(f"| {s} | {len(g)} | {g.ratio.median():.3f} [{g.ratio.quantile(.1):.2f}, "
                     f"{g.ratio.quantile(.9):.2f}] | {dr:+.2f} |")
    edges = [0.0, 0.90, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0]
    mr["bin"] = pd.cut(mr.Recall, edges, right=False)
    by = mr.groupby(["series", "bin"], observed=True).ratio.median().unstack()
    lines += ["", "## Median matched-recall ratio by recall bin", "",
              "| series | " + " | ".join(f"[{b.left:.2f}, {b.right:.2f})" for b in by.columns) + " |",
              "|---" * (len(by.columns) + 1) + "|"]
    for s, r in by.iterrows():
        lines.append(f"| {s} | " + " | ".join("-" if np.isnan(v) else f"{v:.2f}" for v in r) + " |")

    side = [f.with_name(f.name + ".gate.csv") for f in files if f.with_name(f.name + ".gate.csv").exists()]
    tot = None  # gate-stats sidecars: only runs of the pre-cleanup engine wrote them
    if side:
        gs = pd.concat([pd.read_csv(s) for s in side], ignore_index=True)
        gs["layout"], gs["thr"], gs["series"] = series(gs.Params)
        tot = gs.groupby("series")[["checked", "gate_rej", "ideal_rej", "false_pass", "false_rej", "useful",
                                    "useful_gate_rej", "useful_ideal_rej"]].sum()
        pct = lambda a, b: 100 * tot[a] / tot[b]
        lines += ["", "## Gate decisions vs the ideal exact-overlap gate (all gated neighbors, all points)", "",
                  "Ideal gate: pass iff the neighbor's exact query-term overlap > T% of the query's terms. "
                  "Useful: the neighbor would enter the result list (distance < current worst).", "",
                  "| series | checked | gate rejects % | ideal rejects % | false passes % of checked "
                  "(% of ideal rejects) | false rejects % | useful % | useful rejected: gate % / ideal % |",
                  "|---|---|---|---|---|---|---|---|"]
        for s, r in tot.iterrows():
            lines.append(f"| {s} | {r.checked:,.0f} | {pct('gate_rej', 'checked')[s]:.1f} | {pct('ideal_rej', 'checked')[s]:.1f} "
                         f"| {pct('false_pass', 'checked')[s]:.1f} ({pct('false_pass', 'ideal_rej')[s]:.1f}) "
                         f"| {pct('false_rej', 'checked')[s]:.1f} | {pct('useful', 'checked')[s]:.1f} "
                         f"| {pct('useful_gate_rej', 'useful')[s]:.1f} / {pct('useful_ideal_rej', 'useful')[s]:.1f} |")
    (Path(sweep) / f"layouts_summary_{dataset}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    st = mr.groupby("series").ratio.quantile([0.1, 0.5, 0.9]).unstack().sort_values(0.5)
    fig, axs = plt.subplots(1, 1 if tot is None else 2, figsize=(6 if tot is None else 12, 0.42 * len(st) + 1.8),
                            squeeze=False)
    ax = axs[0][0]
    for y, (s, r) in enumerate(st.iterrows()):
        c = COLORS.get(s.split()[0], MUTED)
        ax.errorbar(r[0.5], y, xerr=[[r[0.5] - r[0.1]], [r[0.9] - r[0.5]]], fmt="o", color=c, capsize=2, ms=5)
    ax.axvline(1.0, color=INK, lw=1)
    ax.set_yticks(range(len(st)))
    ax.set_yticklabels(st.index, fontsize=8.5)
    ax.set_xlabel("QPS / gate-off QPS at the same recall (median, p10-p90)")
    ax.set_title(f"{dataset}: gate layouts vs no filter (= 1.0)")
    if tot is not None:
        ax2 = axs[0][1]
        on = tot.sort_values("checked")
        y2 = np.arange(len(on))
        ax2.barh(y2 + 0.2, pct("gate_rej", "checked")[on.index], 0.4, color="#2a78d6", label="gate rejects")
        ax2.barh(y2 - 0.2, pct("ideal_rej", "checked")[on.index], 0.4, color=MUTED, label="ideal gate rejects")
        ax2.set_yticks(y2)
        ax2.set_yticklabels(on.index, fontsize=8.5)
        ax2.set_xlabel("% of gated neighbors")
        ax2.set_title("rejected by the gate vs by an ideal exact-overlap gate")
        ax2.legend(frameon=False, fontsize=8)
    for a in axs[0]:
        a.grid(axis="x", alpha=0.25, color="#52514e", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"layouts_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
