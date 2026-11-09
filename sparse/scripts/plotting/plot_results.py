"""Plot figures for the sparse-kNN comparison.

    python3 common/plot_results.py -i merged.csv -o figures/

Produces, as PDF (vector) and PNG (for quick viewing):

    fig1_qps_vs_recall.*        the headline figure
    fig2_searchtime_vs_recall.* wall time for the whole query batch
    fig3_indexing_vs_recall.*   build cost against the recall it buys
    fig4_memory_vs_recall.*     peak RSS against the recall it buys
    fig5_rr_vs_recall.*         ranking quality (only ordered methods)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------- tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Reference categorical palette, fixed order. Validated for the adjacent
# pairlist in light mode: all hard checks PASS.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#7b52d1", "#00868b", "#b3442e"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

SUBJECT = "SparseHNSW"


def is_subject_model(m):
    """Any SparseHNSW variant, e.g. SparseHNSW_a0.8b5, counts as the
    subject -- alpha/beta variants are still the method under study."""
    return m == SUBJECT or m.startswith(SUBJECT)

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
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 140,
})


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def num(row, col):
    v = row.get(col, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def frontier(rows):
    """Keep only points not dominated on (recall, QPS): higher is better on both.
    """
    pts = [(num(r, "Recall"), num(r, "QPS"), r) for r in rows]
    keep, seen = [], set()
    for rec, qps, r in pts:
        if any(orec >= rec and oqps >= qps and (orec > rec or oqps > qps)
               for orec, oqps, _ in pts):
            continue
        key = (round(rec, 12), round(qps, 6))
        if key in seen:                 # ties on both axes: keep one
            continue
        seen.add(key)
        keep.append(r)
    return keep


def order_models(rows):
    """Stable, deterministic series order: subject first, then alphabetical.
    """
    models = sorted({r["Model"] for r in rows})
    subj = [m for m in models if is_subject_model(m)]
    rest = [m for m in models if not is_subject_model(m)]
    return subj + rest


def style(models):
    return {m: (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)])
            for i, m in enumerate(models)}


def curve(rows, model, xcol, ycol):
    pts = [(num(r, xcol), num(r, ycol)) for r in rows if r["Model"] == model]
    pts = [(x, y) for x, y in pts if x == x and y == y and y > 0]
    return zip(*sorted(pts)) if pts else ((), ())


def finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def line_figure(rows, models, st, xcol, ycol, xlabel, ylabel, title,
                out, logy=True, xlim=None):
    """Multi-series line chart. Identity comes from the legend below the axes.

    No direct end labels here: with six curves converging at high recall, the
    label sat a few pixels from the same name in the legend and cluttered the
    one region the figure exists to show. Identity is still not color-alone --
    every series carries a distinct marker shape, and figure_data.tsv is the
    table view. The scatter figures keep their labels because they have no
    legend.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in models:
        xs, ys = curve(rows, m, xcol, ycol)
        if not xs:
            continue
        color, marker = st[m]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=m, zorder=3,
                clip_on=True)
    if logy:
        ax.set_yscale("log")
    if xlim:
        # Only a hair of right headroom now; the 14% was room for end labels.
        lo, hi = xlim
        ax.set_xlim(lo, hi + (hi - lo) * 0.02)
    ax.margins(y=0.08)
    finish(ax, xlabel, ylabel, title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=min(len(models), 6), fontsize=8, labelcolor=INK_2,
              handlelength=1.6, columnspacing=1.4, borderpad=0,
              handletextpad=0.5)
    fig.tight_layout()
    save(fig, out)


def emphasis_scatter(rows, models, st, ycol, ylabel, title, out,
                     logy=True):
    """One marker per distinct (method, index configuration).

    These two figures carry no legend, so the direct labels are the only thing
    naming a series and an overlap between two of them is a real defect rather
    than clutter. Labels are therefore de-collided vertically after layout.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = []
    for m in models:
        rs = [r for r in rows if r["Model"] == m]
        best = {}
        for r in rs:
            key = round(num(r, ycol), 6)
            if key != key:
                continue
            if key not in best or num(r, "Recall") > num(best[key], "Recall"):
                best[key] = r
        if not best:
            continue
        is_subject = is_subject_model(m)
        color = st[m][0] if is_subject else "#b8b7b0"
        marker = st[m][1]
        xs = [num(r, "Recall") for r in best.values()]
        ys = [num(r, ycol) for r in best.values()]
        ax.scatter(xs, ys, s=64 if is_subject else 46, c=color, marker=marker,
                   edgecolors=SURFACE, linewidths=1.2,
                   zorder=4 if is_subject else 3, label=m)
        bx, by = max(zip(xs, ys), key=lambda p: p[0])
        labels.append((m, bx, by, is_subject))
    if logy:
        ax.set_yscale("log")
    ax.margins(x=0.14, y=0.18)
    finish(ax, "Recall@10", ylabel, title)
    fig.tight_layout()          # must precede placement; it resizes the axes
    place_scatter_labels(ax, labels)
    save(fig, out)


def place_scatter_labels(ax, labels, min_gap_pt=9.5):
    """Annotate each anchor, nudging apart only those that would overlap.

    Offsets are emitted in POINTS while transData speaks PIXELS, so convert --
    at the default dpi they differ by dpi/72 and the nudge lands in the wrong
    place.
    """
    if not labels:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    items = []
    for m, x, y, is_subject in labels:
        _, py = ax.transData.transform((x, y))
        items.append([py / px_per_pt, m, x, y, is_subject])
    items.sort(key=lambda it: it[0])
    for i in range(1, len(items)):
        items[i][0] = max(items[i][0], items[i - 1][0] + min_gap_pt)
    for py, m, x, y, is_subject in items:
        _, orig_py = ax.transData.transform((x, y))
        ax.annotate(m, xy=(x, y), xytext=(5, 4 + py - orig_py / px_per_pt),
                    textcoords="offset points",
                    fontsize=7.5, color=INK if is_subject else INK_2,
                    fontweight="bold" if is_subject else "normal")


def save(fig, out):
    for ext in ("pdf", "png"):
        p = f"{out}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--xmin", type=float, default=None,
                    help="left edge of the recall axis; default auto-zooms")
    ap.add_argument("--pareto", action="store_true",
                    help="plot only the (recall, QPS) frontier of each method, "
                         "dropping dominated points; applied to EVERY method")
    ap.add_argument("--bin", type=float, default=None, metavar="PCT",
                    help="thin each series to one marker per PCT recall bin "
                         "(e.g. 0.2). Bins start at multiples of PCT and keep "
                         "their highest-QPS point, so the best QPS at recall >= "
                         "each bin edge is preserved; each series' "
                         "highest-recall point is always kept.")
    args = ap.parse_args()

    rows = read(args.input)
    if args.pareto:
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["Model"]].append(r)
        kept = []
        print("pareto filter (recall, QPS), all methods:")
        for m in sorted(by_model):
            f = frontier(by_model[m])
            print(f"  {m:<12} {len(by_model[m]):3d} points -> {len(f):3d} on the frontier")
            kept += f
        rows = kept

    if args.bin:
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["Model"]].append(r)
        kept = []
        print(f"binning to one point per {args.bin}% recall (highest QPS wins):")
        for m in sorted(by_model):
            best = {}
            for r in by_model[m]:
                rec, q = num(r, "Recall"), num(r, "QPS")
                if rec is None or q is None:
                    continue
                b = math.floor(rec * 100.0 / args.bin + 1e-9)
                if b not in best or q > num(best[b], "QPS"):
                    best[b] = r
            top = max((r for r in by_model[m] if num(r, "Recall") is not None and num(r, "QPS") is not None),
                      key=lambda r: num(r, "Recall"), default=None)
            out = sorted({id(r): r for r in list(best.values()) + ([top] if top else [])}.values(),
                         key=lambda r: num(r, "Recall"))
            print(f"  {m:<12} {len(by_model[m]):3d} points -> {len(out):3d} after binning")
            kept += out
        rows = kept

    os.makedirs(args.outdir, exist_ok=True)
    models = order_models(rows)
    st = style(models)

    # Figures carry no subtitle, so the run conditions are echoed here and are
    # recorded in figure_data.tsv. They are not lost, just not on the canvas.
    threads = sorted({r["Threads"] for r in rows})
    nq = sorted({r["NumQueries"] for r in rows})
    k = sorted({r["K"] for r in rows})
    print(f"conditions: k={'/'.join(k)} | {'/'.join(nq)} queries | "
          f"{'/'.join(threads)} threads | median of repeated runs")

    recalls = [num(r, "Recall") for r in rows]
    lo = args.xmin if args.xmin is not None else max(0.0, min(recalls) - 0.02)
    xlim = (lo, 1.005)

    print(f"plotting {len(rows)} rows, {len(models)} methods -> {args.outdir}")

    line_figure(rows, models, st, "Recall", "QPS",
                "Recall@10", f"QPS ({threads[0] if len(threads)==1 else 'n'} threads, log scale)",
                "Throughput vs Recall",
                os.path.join(args.outdir, "fig1_qps_vs_recall"), xlim=xlim)

    line_figure(rows, models, st, "Recall", "SearchSecMedian",
                "Recall@10", f"Batch Search Time, seconds (log scale)",
                "Search time vs Recall",
                os.path.join(args.outdir, "fig2_searchtime_vs_recall"), xlim=xlim)

    emphasis_scatter(rows, models, st, "IndexSec",
                     "Index Build Time, seconds (log scale)",
                     "Index Build Cost vs Recall",
                     os.path.join(args.outdir, "fig3_indexing_vs_recall"))

    emphasis_scatter(rows, models, st, "PeakRSSGB",
                     "Peak Resident Memory, GB (log scale)",
                     "Memory Cost vs Recall",
                     os.path.join(args.outdir, "fig4_memory_vs_recall"))

    ordered = [r for r in rows if str(r["ResultsOrdered"]).lower().startswith("t")]
    dropped = {r["Model"] for r in rows} - {r["Model"] for r in ordered}
    if ordered:
        # A method missing from this figure is a material caveat, and the
        # subtitle that used to carry it is gone -- put it in the title.
        title = "Ranking Quality vs Recall"
        if dropped:
            title += f" (excludes unranked: {', '.join(sorted(dropped))})"
            print(f"  fig5 excludes unranked methods: {', '.join(sorted(dropped))}")
        line_figure(ordered, [m for m in models if m in {r['Model'] for r in ordered}],
                    st, "Recall", "RR@10", "Recall@10", "RR@10 vs Exact NN",
                    title,
                    os.path.join(args.outdir, "fig5_rr_vs_recall"),
                    logy=False, xlim=xlim)

    tsv = os.path.join(args.outdir, "figure_data.tsv")
    cols = ["Model", "Params", "Recall", "RR@10", "QPS", "SearchSecMedian",
            "IndexSec", "PeakRSSGB", "Threads"]
    with open(tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["Model"], num(r, "Recall"))):
            w.writerow([r.get(c, "") for c in cols])
    print(f"  wrote {tsv}  (table view)")


if __name__ == "__main__":
    main()
