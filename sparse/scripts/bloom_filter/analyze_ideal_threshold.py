#!/usr/bin/env python3
"""How good could an overlap-threshold filter be? (PRESET=idealT)

Reads bloom_<dataset>_*.csv and the .gate.csv sidecar of the T=0 Bloom gate (gate-off
recall), whose gate-stats pass scored an ideal gate -- pass iff the neighbor's exact
query-term overlap > T% of the query's terms -- at several T on every neighbor checked
after the result list filled. Writes, next to them:
  ideal_summary_<dataset>.md  per ef: % of checked neighbors the ideal gate rejects and %
                              of useful ones (would enter the list) it loses, per T
  ideal_<dataset>.png/.pdf    rejected vs useful lost, one line per ef, points labelled T
Optional compare dirs (layout jobs with .gate.csv): their gates at the same config
(beta=3, no patience, same ef) are tabulated and overlaid as markers.

  python3 analyze_ideal_threshold.py <sweep_dir> <dataset> [compare_dir ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TS = [0, 2, 5, 10, 15, 20, 25, 30]
INK = "#0b0b0b"
EF_LIGHT, EF_DARK = "#86b6ef", "#0d366b"  # ordinal ramp: light = small ef
ROW_LINES = 4.0  # lines a row fetch costs; a separate 1-line check breaks even at 1/ROW_LINES rejected
MARKERS = {"list16": "*", "list8": "D", "list4": "v", "sep": "s", "inline": "^"}


def compare_gates(dirs, efs):
    """Other jobs' gates at the idealT config: per series and ef, rejected % and useful lost %."""
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.csv.gate.csv")):
            o = pd.read_csv(f)
            o = o[o.Params.str.contains(r" beta=3(?:\s|$)") & ~o.Params.str.contains(" pat=")].copy()
            o["ef"] = o.Params.str.extract(r" ef=(\d+)")[0].astype(int)
            o["series"] = (o.Params.str.extract(r" layout=(\S+)")[0].fillna("sep") + " T="
                           + o.Params.str.extract(r" T=([\d.]+)")[0])
            rows.append(o[o.ef.isin(efs)])
    if not rows:
        return None
    o = pd.concat(rows).drop_duplicates(["series", "ef"])
    o = o[~o.series.str.startswith("inline")]  # same filter rows as sep: identical decisions
    return o.assign(rej=100 * o.gate_rej / o.checked, lost=100 * o.useful_gate_rej / o.useful)


def main(sweep, dataset, *compare_dirs):
    row_lines = ROW_LINES
    files = sorted(f for f in Path(sweep).glob(f"bloom_{dataset}_*.csv") if not f.name.endswith(".gate.csv"))
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    gs = pd.concat([pd.read_csv(f.with_name(f.name + ".gate.csv")) for f in files], ignore_index=True)
    gs = gs[gs.Params.str.contains(r" T=0(?:\s|$)")].copy()
    gs["ef"] = gs.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    res["ef"] = res.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    off = res[~res.Params.str.contains(" T=")].groupby("ef").Recall.first()
    t0 = res[res.Params.str.contains(r" T=0(?:\s|$)")].groupby("ef").Recall.first()
    g = gs.groupby("ef").sum(numeric_only=True)
    lines = [f"# Ideal overlap-threshold gate, {dataset}", "",
             "Neighbors checked after the result list filled, on the T=0 Bloom path (recall = gate off). "
             "Cells: % of checked rejected / % of useful neighbors lost (useful = would enter the list). "
             f"A separate 1-line check pays for itself only above ~{100 / row_lines:.0f}% rejected "
             f"({row_lines:g}-line rows).", "",
             "| ef | recall off / T=0 | checked | useful % | Bloom T=0 rejects % | "
             + " | ".join(f"T={t}" for t in TS) + " |", "|---" * (5 + len(TS)) + "|"]
    for ef, r in g.iterrows():
        cells = [f"{100 * r[f'ideal_rej_T{t}'] / r.checked:.1f} / {100 * r[f'useful_ideal_rej_T{t}'] / r.useful:.1f}"
                 for t in TS]
        lines.append(f"| {ef} | {off.get(ef, float('nan')):.4f} / {t0.get(ef, float('nan')):.4f} | {r.checked:,.0f} | "
                     f"{100 * r.useful / r.checked:.1f} | {100 * r.gate_rej / r.checked:.2f} | " + " | ".join(cells) + " |")
    cmp = compare_gates(compare_dirs, set(g.index)) if compare_dirs else None
    if cmp is not None:
        lines += ["", "## Other gates at the same config (rejected % / useful lost %)", "",
                  "| series | " + " | ".join(f"ef={e}" for e in g.index) + " |", "|---" * (len(g) + 1) + "|"]
        for s, c in cmp.groupby("series"):
            c = c.set_index("ef")
            lines.append(f"| {s} | " + " | ".join(f"{c.rej[e]:.1f} / {c.lost[e]:.1f}" if e in c.index else "-"
                                                  for e in g.index) + " |")
    (Path(sweep) / f"ideal_summary_{dataset}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ef", [EF_LIGHT, EF_DARK])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (ef, r) in enumerate(g.iterrows()):
        x = [100 * r[f"useful_ideal_rej_T{t}"] / r.useful for t in TS]
        y = [100 * r[f"ideal_rej_T{t}"] / r.checked for t in TS]
        ax.plot(x, y, "-o", color=ramp(i / max(1, len(g) - 1)), ms=4, lw=1.6, label=f"ef = {ef}")
        if i == len(g) - 1:
            for t, xi, yi in zip(TS, x, y):
                ax.annotate(f"T={t}", (xi, yi), textcoords="offset points", xytext=(6, -10), fontsize=7.5,
                            color="#48515b")
    if cmp is not None:
        efs = list(g.index)
        for s, c in cmp.groupby("series"):
            m = MARKERS.get(s.split()[0], "o")
            for _, r in c.iterrows():
                ax.plot(r.lost, r.rej, m, color=ramp(efs.index(r.ef) / max(1, len(efs) - 1)), ms=7, mec=INK, mew=0.6)
            ax.plot([], [], m, color="#c3c2be", mec=INK, mew=0.6, ms=7, label=s)
    ax.axhline(100 / row_lines, color="#52514e", lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 100 / row_lines, " 1-line check break-even", va="bottom", ha="right",
            fontsize=7.5, color="#52514e")
    ax.set_xlabel("useful neighbors lost (%)")
    ax.set_ylabel("checked neighbors rejected (%)")
    ax.set_title(f"{dataset}: ideal exact-overlap gate, rejected vs useful lost")
    ax.grid(alpha=0.25, color="#52514e", lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"ideal_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:])
