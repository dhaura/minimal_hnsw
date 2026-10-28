#!/usr/bin/env python3
"""Adaptive stopping step 1: online stopping rules timed against fixed ef.

Reads stop_<dataset>_*.csv from run_stop_online_perlmutter.sh. Model suffix _p<spec>:
"0" = fixed ef, "P" = patience, "gG" = score gap, "gG:mM:pP" = combo. Writes, next to them:
  online_summary_<dataset>.md  per rule: QPS / fixed-ef QPS at the same recall (fixed-ef
                               curve interpolated in log QPS), best QPS at recall >= r
  online_<dataset>.png/.pdf    best-QPS fronts: fixed ef vs each rule family

  python3 analyze_stop_online.py <sweep_dir> <dataset>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.996, 0.997]
FAMILY = {"fixed ef": "#0b0b0b", "patience": "#7a7a74", "score gap": "#2a78d6", "combo": "#c4632f"}


def family(spec):
    return ("fixed ef" if spec == "0" else "combo" if ":" in spec
            else "score gap" if spec.startswith("g") else "patience")


def best(d, r):
    d = d[d.Recall >= r]
    return d.QPS.max() if len(d) else np.nan


def load(sweep, dataset):
    """All points, the fixed-ef rows, and rule rows with QPS / fixed-ef QPS at the same recall."""
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(sweep).glob(f"stop_{dataset}_*.csv"))], ignore_index=True)
    df["ef"] = df.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    df["rule"] = df.Model.str.extract(r"_p([^_]+)$")[0]
    df["family"] = df.rule.map(family)
    fixed = df[df.rule == "0"].sort_values("Recall")
    rules = df[df.rule != "0"]
    ok = rules.Recall.between(fixed.Recall.min(), fixed.Recall.max())
    rules = rules[ok].assign(ratio=lambda d: d.QPS / np.exp(np.interp(d.Recall, fixed.Recall, np.log(fixed.QPS))))
    return df, fixed, rules


def main(sweep, dataset):
    df, fixed, rules = load(sweep, dataset)

    lines = [f"# Online stopping rules, {dataset} ({len(df)} points)", "",
             "Ratio = QPS / fixed-ef QPS at the same recall (fixed-ef curve interpolated in log QPS).",
             f"Fixed ef spans recall {fixed.Recall.min():.4f}-{fixed.Recall.max():.4f}.", "",
             "## Per rule: recall reached (largest ef cap) and best matched-recall ratio", "",
             "| rule | family | recall at cap | best ratio (at recall, ef cap) | median ratio over caps |",
             "|---|---|---|---|---|"]
    for spec, g in rules.groupby("rule"):
        top = df[df.rule == spec].Recall.max()
        w = g.loc[g.ratio.idxmax()]
        lines.append(f"| {spec} | {w.family} | {top:.4f} | {w.ratio:.3f} ({w.Recall:.4f}, ef={w.ef}) | "
                     f"{g.ratio.median():.3f} |")
    win = [0.975, 0.985, 0.99, 0.993, 0.995, 0.997, 1.0]
    lines += ["", "## Best matched-recall ratio per family by recall window (setting, ef cap)", "",
              "| recall window | " + " | ".join(f for f in FAMILY if f != "fixed ef") + " |", "|---" * 4 + "|"]
    for lo, hi in zip(win[:-1], win[1:]):
        cells = []
        for fam in [f for f in FAMILY if f != "fixed ef"]:
            g = rules[(rules.family == fam) & (rules.Recall >= lo) & (rules.Recall < hi)]
            w = g.loc[g.ratio.idxmax()] if len(g) else None
            cells.append(f"{w.ratio:.2f}x ({w.rule}, ef={w.ef})" if w is not None else "-")
        lines.append(f"| [{lo}, {hi}) | " + " | ".join(cells) + " |")
    lines += ["", "## Best QPS at recall >= r", "",
              "| recall | fixed ef | best rule | rule QPS | speedup |", "|---|---|---|---|---|"]
    for r in BINS:
        d = df[(df.rule != "0") & (df.Recall >= r)]
        if not len(d):
            continue
        w = d.loc[d.QPS.idxmax()]
        lines.append(f"| {r} | {best(fixed, r):,.0f} | {w.rule} (ef={w.ef}) | {w.QPS:,.0f} | "
                     f"{w.QPS / best(fixed, r):.2f}x |")
    (Path(sweep) / f"online_summary_{dataset}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    grid = np.linspace(BINS[0], df.Recall.max(), 300)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for fam, c in FAMILY.items():
        d = df[df.family == fam]
        ax.plot(grid, [best(d, r) for r in grid], color=c, lw=2 if fam == "fixed ef" else 1.6,
                label=fam + (" (best over its settings)" if fam != "fixed ef" else ""))
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (64 threads)")
    ax.set_title(f"{dataset}: online stopping rules vs fixed ef (best QPS at recall >= x)")
    ax.grid(alpha=0.25, color="#52514e", lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sweep) / f"online_{dataset}.{ext}", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
