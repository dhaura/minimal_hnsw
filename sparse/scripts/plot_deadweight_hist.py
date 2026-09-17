#!/usr/bin/env python3
"""Plot dead-weight histograms produced by sparse_hnsw_demo_distlog.

Consumes the `*_hist.csv` files written by dist_log.h
(bin_lo,bin_hi,count,fraction) and, when present, the sibling `*_result.csv`
for recall. Emits two figures per dataset:

  <dataset>_deadweight_overlay.png    density + CDF, all runs overlaid
  <dataset>_deadweight_panels.png     one histogram per run, 2-up grid

Usage
-----
  python3 plot_deadweight_hist.py                          # all datasets found
  python3 plot_deadweight_hist.py --dataset msmarco_full
  python3 plot_deadweight_hist.py --indir results/deadweight_efc --outdir figs

A NOTE ON WHICH AVERAGE THIS IS
-------------------------------
Bins hold one sample per DISTANCE CALL, so every statistic derived here is
call-weighted: a 12-entry row counts as much as a 900-entry row. The headline
figure quoted elsewhere ("91.10% dead weight") is ENTRY-weighted
(sum(unused)/sum(total)) and is not recoverable from these bins. The two differ
by a couple of points and must not be mixed in one sentence; annotations below
say "call-wtd" to keep that straight.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE, style_axes

# Re-stepped from plot_style.PALETTE: the shipped 4th hue (#eda100) pairs with
# #eb6834 at normal-vision dE 13.7, under the floor of 15, and these curves
# overlap so any pair gets compared. Swapping in #4a3aa7 clears all-pairs at
# CVD 9.2 / normal 16.3. Same shipped steps, re-ordered -- not new hues.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

# Directory names are slugs; figures get the published spelling.
DATASET_LABELS = {
    "msmarco_full": "MS MARCO (Full)",
    "msmarco_1M": "MS MARCO (1M)",
    "msmarco_small": "MS MARCO (Small)",
    "msmarco_v2": "MS MARCO v2",
    "msmarco_v2_splade": "MS MARCO v2 (SPLADE)",
    "nq_splade": "Natural Questions (SPLADE)",
}


def pretty_dataset(ds):
    if ds in DATASET_LABELS:
        return DATASET_LABELS[ds]
    return ds.replace("_", " ").title()


FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_M(?P<M>\d+)_efc(?P<efc>\d+)_ef(?P<ef>\d+)_a(?P<alpha>[\d.]+)_hist\.csv$"
)


def discover(indir):
    """Group hist CSVs by dataset, parsing params out of the filename."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(indir, "*_hist.csv"))):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            print(f"  skip (unrecognized name): {os.path.basename(path)}")
            continue
        g = m.groupdict()
        rec = None
        res = path.replace("_hist.csv", "_result.csv")
        if os.path.exists(res):
            try:
                rec = float(pd.read_csv(res)["recall"].iloc[0])
            except Exception:
                pass
        runs.setdefault(g["dataset"], []).append(
            dict(path=path, M=int(g["M"]), efc=int(g["efc"]), ef=int(g["ef"]),
                 alpha=float(g["alpha"]), recall=rec)
        )
    # Stable order: efC ascending, then ef ascending. Colour follows this
    # identity, so a run keeps its hue whether or not its siblings are present.
    for ds in runs:
        runs[ds].sort(key=lambda r: (r["efc"], r["ef"]))
    return runs


def load(run):
    df = pd.read_csv(run["path"])
    total = df["count"].sum()
    if total == 0:
        raise ValueError(f"{run['path']}: histogram is empty")
    df["mid"] = (df["bin_lo"] + df["bin_hi"]) / 2.0
    # Recompute share rather than trusting the stored `fraction`, so a
    # concatenated or hand-edited file cannot silently misreport.
    df["share"] = df["count"] / total
    df["cum"] = df["share"].cumsum()
    run["df"] = df
    run["total"] = int(total)
    run["mean"] = float((df["mid"] * df["share"]).sum())
    idx = int(np.searchsorted(df["cum"].values, 0.5))
    run["median"] = float(df["bin_hi"].iloc[min(idx, len(df) - 1)])
    return run


def label_for(run, multi_ef):
    lab = f"efC={run['efc']}"
    if multi_ef:
        lab += f", ef={run['ef']}"
    if run["recall"] is not None:
        lab += f"  ({run['recall']:.2f}% recall)"
    return lab


def xlim_for(runs, keep=0.99):
    """Clip the x-axis to the smallest round window holding `keep` of every
    run's mass, so 80 empty bins do not squash the part with the story in it.
    The CDF panel always shows 0-100, so nothing is hidden outright."""
    lo = 100.0
    for r in runs:
        df = r["df"]
        below = df["cum"].values
        i = int(np.searchsorted(below, 1.0 - keep))
        lo = min(lo, float(df["bin_lo"].iloc[max(i - 1, 0)]))
    return max(0.0, np.floor(lo / 10.0) * 10.0), 100.0


def draw_overlay(runs, dataset, outdir):
    multi_ef = len({r["ef"] for r in runs}) > 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    lo, hi = xlim_for(runs)

    for i, r in enumerate(runs):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        df = r["df"]
        ax1.step(df["mid"], df["share"] * 100.0, where="mid",
                 color=c, linewidth=2.0, label=label_for(r, multi_ef), zorder=3)
        ax2.step(df["bin_hi"], df["cum"] * 100.0, where="post",
                 color=c, linewidth=2.0, label=label_for(r, multi_ef), zorder=3)

    for ax, title, ylab in (
        (ax1, "Distribution", "Share of distance calls (%)"),
        (ax2, "Cumulative Distribution", "Calls at or below x (%)"),
    ):
        style_axes(ax)
        ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=9)
        ax.set_xlabel("Dead weight of row scored (%)", color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel(ylab, color=INK_SECONDARY, fontsize=10)
    ax1.set_xlim(lo, hi)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    # Direct labels: identity is never colour-alone, and the green step carries
    # a contrast WARN against this surface. But runs at the same ef sit almost
    # exactly on top of each other -- that coincidence IS the finding -- so
    # labelling every peak just stacks four labels in the same few pixels.
    # Label each ef-group once, anchored on the rising edge where there is
    # open space rather than at the peak against the right spine.
    groups = {}
    for i, r in enumerate(runs):
        groups.setdefault(r["ef"], []).append((i, r))
    for k, (ef, members) in enumerate(sorted(groups.items())):
        i, r = members[len(members) // 2]
        df = r["df"]
        peak = df["share"].max()
        # Stagger the anchor down the rising edge per group. Curves that sit
        # close together would otherwise anchor at the same x and their leader
        # lines would converge through each other's text.
        frac = max(0.12, 0.45 - 0.16 * k)
        rising = df.index[df["share"] >= frac * peak]
        j = int(rising[0]) if len(rising) else int(df["share"].values.argmax())
        efcs = "/".join(str(m[1]["efc"]) for m in members)
        txt = f"ef={ef}" + (f"  (efC {efcs})" if len(members) > 1 else f", efC={efcs}")
        if len(members) > 1:
            txt += "\nnearly identical"
        ax1.annotate(txt,
                     xy=(df["mid"].iloc[j], df["share"].iloc[j] * 100.0),
                     xytext=(-8, 30 + 22 * k), textcoords="offset points",
                     color=SERIES_COLORS[i % len(SERIES_COLORS)],
                     fontsize=8.5, fontweight="bold", ha="right", va="bottom",
                     zorder=5, annotation_clip=True,
                     arrowprops=dict(arrowstyle="-", color=SERIES_COLORS[i % len(SERIES_COLORS)],
                                     linewidth=0.8, shrinkA=1, shrinkB=2))

    leg = ax1.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.suptitle(f"Dead Weight per Distance Call \u2014 {pretty_dataset(dataset)}",
                 color=INK_PRIMARY, fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.007, 0.015,
             f"One sample per distance call (call-weighted, not entry-weighted). "
             f"Left panel clipped to {lo:.0f}-100%; the cumulative panel shows the full range.",
             color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])
    out = os.path.join(outdir, f"{dataset}_deadweight_overlay.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def draw_panels(runs, dataset, outdir):
    multi_ef = len({r["ef"] for r in runs}) > 1
    n = len(runs)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 3.5 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    lo, hi = xlim_for(runs)
    ymax = max(r["df"]["share"].max() for r in runs) * 100.0

    for k, r in enumerate(runs):
        ax = axes[k // ncol][k % ncol]
        c = SERIES_COLORS[k % len(SERIES_COLORS)]
        df = r["df"]
        style_axes(ax)
        ax.fill_between(df["mid"], df["share"] * 100.0, step="mid",
                        color=c, alpha=0.30, zorder=2, linewidth=0)
        ax.step(df["mid"], df["share"] * 100.0, where="mid",
                color=c, linewidth=1.8, zorder=3)
        ax.axvline(r["median"], color=c, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ymax * 1.18)
        head = f"efC={r['efc']}" + (f", ef={r['ef']}" if multi_ef else "")
        ax.set_title(head, color=INK_PRIMARY, fontsize=12, pad=7)
        sub = f"median {r['median']:.0f}%  ·  mean {r['mean']:.1f}% (call-wtd)  ·  {r['total']/1e6:.1f}M calls"
        if r["recall"] is not None:
            sub += f"  ·  recall {r['recall']:.2f}%"
        ax.annotate(sub, xy=(0.015, 0.90), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for k in range(n):
        ax = axes[k // ncol][k % ncol]
        if k // ncol == nrow - 1 or k + ncol >= n:
            ax.set_xlabel("Dead weight of row scored (%)", color=INK_SECONDARY, fontsize=9.5)
        if k % ncol == 0:
            ax.set_ylabel("Share of calls (%)", color=INK_SECONDARY, fontsize=9.5)

    fig.suptitle(f"Dead-Weight Distribution by Run \u2014 {pretty_dataset(dataset)}",
                 color=INK_PRIMARY, fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.007, 0.012, "Dashed line marks the median. Call-weighted; x clipped to "
                           f"{lo:.0f}-100%.", color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = os.path.join(outdir, f"{dataset}_deadweight_panels.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=os.path.join(here, "results", "deadweight_efc"))
    ap.add_argument("--outdir", default=None, help="default: --indir")
    ap.add_argument("--dataset", default=None, help="only this dataset")
    args = ap.parse_args()

    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)

    runs_by_ds = discover(args.indir)
    if args.dataset:
        runs_by_ds = {k: v for k, v in runs_by_ds.items() if k == args.dataset}
    if not runs_by_ds:
        sys.exit(f"no *_hist.csv found under {args.indir}"
                 + (f" for dataset {args.dataset}" if args.dataset else ""))

    for ds, runs in sorted(runs_by_ds.items()):
        for r in runs:
            load(r)
        print(f"\n{ds}: {len(runs)} run(s)")
        for r in runs:
            print("  efC=%-5d ef=%-5d  %8.1fM calls  median %3.0f%%  mean %5.1f%% (call-wtd)%s"
                  % (r["efc"], r["ef"], r["total"] / 1e6, r["median"], r["mean"],
                     f"  recall {r['recall']:.2f}%" if r["recall"] is not None else ""))
        print("  ->", draw_overlay(runs, ds, outdir))
        print("  ->", draw_panels(runs, ds, outdir))


if __name__ == "__main__":
    main()
