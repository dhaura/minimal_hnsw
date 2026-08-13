#!/usr/bin/env python3
"""Visualise an alpha/beta sweep.
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3, all-pairs validated
MARKERS = ["o", "s", "^"]
# sequential blue 100 -> 700, light = low magnitude
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQ)
NODATA = "#eeede8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def frontier(points):
    out, best = [], -1.0
    for rec, qps in sorted(points, reverse=True):
        if qps > best:
            out.append((rec, qps))
            best = qps
    return sorted(out)


def qps_at(curve, target):
    if not curve or target < curve[0][0] or target > curve[-1][0]:
        return None
    for (r0, q0), (r1, q1) in zip(curve, curve[1:]):
        if r0 <= target <= r1:
            if r1 == r0:
                return max(q0, q1)
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return curve[-1][1]


def load(paths):
    by_cfg = defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            prm = r["Params"]
            a = float(prm.split("alpha=")[1].split()[0])
            b = int(prm.split("beta=")[1].split()[0])
            by_cfg[(a, b)].append((float(r["Recall"]), float(r["QPS"])))
    return {k: frontier(v) for k, v in by_cfg.items()}


def heat(ax, curves, alphas, betas, valfn, title, fmt, log_scale):
    grid = [[valfn(curves.get((a, b))) for b in betas] for a in alphas]
    flat = [v for row in grid for v in row if v is not None]
    if not flat:
        return
    lo, hi = min(flat), max(flat)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            v = grid[i][j]
            if v is None:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=NODATA, edgecolor=SURFACE,
                                           linewidth=2))
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            if log_scale:
                t = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo) or 1)
            else:
                t = (v - lo) / ((hi - lo) or 1)
            # 2px surface gap between cells, per mark spec
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor=SEQ_CMAP(t), edgecolor=SURFACE,
                                       linewidth=2))
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if t > 0.55 else INK)
    best = max(((v, i, j) for i, row in enumerate(grid)
                for j, v in enumerate(row) if v is not None), default=None)
    if best:
        _, bi, bj = best
        ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=2.2, zorder=5))
    ax.set_xticks(range(len(betas)), [str(b) for b in betas])
    ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlim(-.5, len(betas) - .5)
    ax.set_ylim(len(alphas) - .5, -.5)
    ax.set_xlabel("beta (refine over-fetch)")
    ax.set_ylabel("alpha (mass kept)")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--candidates", default="auto",
                    help="'a/b,a/b,a/b', or 'auto' to pick the winner at each "
                         "recall target (deduped, capped at 3 series -- the "
                         "documented all-pairs-safe categorical limit)")
    ap.add_argument("--auto-targets", default="0.95,0.98,0.99")
    ap.add_argument("--baseline", default="0.8/3")
    args = ap.parse_args()

    curves = load(args.csvs)
    alphas = sorted({a for a, _ in curves})
    betas = sorted({b for _, b in curves})

    if args.candidates == "auto":
        picked = []
        for t in [float(x) for x in args.auto_targets.split(",") if x]:
            ranked = sorted(((qps_at(c, t), k) for k, c in curves.items()
                             if qps_at(c, t) is not None), reverse=True)
            if ranked and ranked[0][1] not in picked:
                picked.append(ranked[0][1])
        args.candidates = ",".join(f"{a:g}/{b}" for a, b in picked[:3])
        print("auto-selected candidates:", args.candidates)
    os.makedirs(args.outdir, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.28)

    # --- A: candidate frontiers -------------------------------------------
    axA = fig.add_subplot(gs[0, :])
    def parse(c):
        a, b = c.split("/"); return (float(a), int(b))

    base = parse(args.baseline)
    if base in curves:
        xs, ys = zip(*curves[base])
        axA.plot(xs, ys, color=MUTED, lw=2, ls="--", marker="", zorder=2)
        axA.annotate(f"baseline a{base[0]:g}/b{base[1]}", (xs[0], ys[0]),
                     textcoords="offset points", xytext=(4, 7),
                     color=MUTED, fontsize=8.5, va="bottom", ha="left")

    for i, c in enumerate(args.candidates.split(",")):
        cfg = parse(c)
        if cfg not in curves:
            continue
        xs, ys = zip(*curves[cfg])
        axA.plot(xs, ys, color=SERIES[i], lw=2, marker=MARKERS[i], ms=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                 label=f"alpha={cfg[0]:g} beta={cfg[1]}")
        axA.annotate(f"a{cfg[0]:g}/b{cfg[1]}", (xs[0], ys[0]),
                     textcoords="offset points", xytext=(-8, 0),
                     color=SERIES[i], fontsize=9, fontweight="bold",
                     va="center", ha="right")

    axA.set_yscale("log")
    axA.set_xlabel("Recall@10")
    axA.set_ylabel("QPS (64 threads, log scale)")
    axA.set_title(f"Pruning configurations on the recall/throughput frontier"
                  + (f"  —  {args.label}" if args.label else ""),
                  loc="left", fontsize=11.5, color=INK, pad=10)
    axA.legend(frameon=False, loc="upper right", fontsize=9)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # --- B/C/D: grids ------------------------------------------------------
    heat(fig.add_subplot(gs[1, 0]), curves, alphas, betas,
         lambda c: qps_at(c, 0.95) if c else None,
         "QPS at recall 0.95", lambda v: f"{v/1000:.0f}k", True)
    heat(fig.add_subplot(gs[1, 1]), curves, alphas, betas,
         lambda c: qps_at(c, 0.98) if c else None,
         "QPS at recall 0.98", lambda v: f"{v/1000:.1f}k", True)
    heat(fig.add_subplot(gs[1, 2]), curves, alphas, betas,
         lambda c: c[-1][0] if c else None,
         "Recall ceiling (max reachable)", lambda v: f"{v:.3f}", False)

    fig.text(0.008, 0.012,
             "Heatmaps: darker = higher; black outline = best cell; “–” = the ef "
             "ladder never reaches that recall. Panel A: markers are measured ef "
             "points (10…3200), lines interpolate.",
             fontsize=8, color=MUTED)

    for ext in ("png", "pdf"):
        path = os.path.join(args.outdir, f"alpha_beta_sweep.{ext}")
        fig.savefig(path, dpi=170 if ext == "png" else None, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()
