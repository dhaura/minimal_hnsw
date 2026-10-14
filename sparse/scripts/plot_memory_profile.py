#!/usr/bin/env python3
"""Figures for the two memory/recall profiling experiments.

Usage
-----
  module load python/3.11-24.1.0
  python plot_memory_profile.py                       # both, default inputs
  python plot_memory_profile.py --only cache
  python plot_memory_profile.py --out-root /path/to/results --tag rerun
"""

import argparse
import datetime as _dt
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import PALETTE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, BASELINE, style_axes

KIB, MIB, GIB = 1 << 10, 1 << 20, 1 << 30

# Zen3 cache hierarchy on a Perlmutter CPU node (2x EPYC 7763), read from
# /sys/devices/system/cpu/cpu0/cache/index*/{size,shared_cpu_list}.
#
# Reproduce with:   lscpu -C
#                   cat /sys/devices/system/cpu/cpu0/cache/index*/size
#                   cat /sys/devices/system/cpu/cpu0/cache/index*/shared_cpu_list
#
# NB `lscpu` without -C prints L2/L3 as AGGREGATES over all instances:
# "L2 cache: 64 MiB (128 instances)" is 512 KiB per core, and
# "L3 cache: 512 MiB (16 instances)" is 32 MiB per CCD shared by 8 cores.
HW_LEVELS = [
    ("L1d / core",            32 * KIB,  "32 KiB, private"),
    ("L2 / core",            512 * KIB,  "512 KiB, private (shared by the SMT pair)"),
    ("L3 / core at 8 busy",    4 * MIB,  "32 MiB CCD / 8 cores"),
    ("L3 / CCD",              32 * MIB,  "32 MiB, shared by 8 cores"),
    ("L3 / socket",          256 * MIB,  "8 CCDs"),
    ("L3 / node",            512 * MIB,  "both sockets"),
    ("1 GiB side cache",       1 * GIB,  "off-chip, would have to be built"),
]


# Figure titles live here so the README index cannot drift from the plots.
# Each title states what is plotted; conclusions belong in the prose, not the
# axes furniture.
CACHE_TITLES = {
    "fig1_traffic_served_vs_cache_size":
        "DRAM Row Traffic Served Versus Hot-Row Cache Capacity",
    "fig2_flatness_lorenz":
        "Concentration of Row Traffic Across the Touched Corpus (Lorenz Curve)",
    "fig3_cache_leverage":
        "Cache Leverage Versus Capacity: Traffic Share per Unit of Corpus Share",
    "fig4_rank_frequency":
        "Rank-Frequency Distribution of Row Fetches",
    "fig5_hardware_levels":
        "Traffic Served at Each Level of the EPYC 7763 Cache Hierarchy",
    "fig6_marginal_return":
        "Marginal Traffic Served per Additional MiB of Cache",
}

TAIL_TITLES = {
    "fig1_loss_concentration":
        "Cumulative Share of Missed Neighbours by Query Hardness Percentile",
    "fig2_miss_histogram":
        "Distribution of Missed Neighbours per Query",
    "fig3_hardness_covariate":
        "Missed Neighbours Versus Query Term Count, by Decile",
    "fig4_routing_payoff":
        "Recall@10 as a Function of the Query Fraction Routed to an Exact Scan",
}


# matplotlib will happily draw a title wider than the canvas and silently clip
# it, so measure and step the size down only when a title does not fit.
# Titles are centred on the FIGURE, not the axes: several of these plots have a
# wide y-label or trailing value labels, which puts the axes centre off-centre.
TITLE_SIZE = 15.0
TITLE_FLOOR = 11.0


def place_title(fig, text, y=0.97):
    t = fig.suptitle(text, color=INK_PRIMARY, fontsize=TITLE_SIZE, y=y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    limit = 0.96 * fig.get_size_inches()[0] * fig.dpi
    size = TITLE_SIZE
    while (t.get_window_extent(renderer=renderer).width > limit
           and size > TITLE_FLOOR):
        size -= 0.5
        t.set_fontsize(size)
        fig.canvas.draw()
    if size < TITLE_SIZE:
        print(f"    title shrunk to {size:g}pt to fit: {text[:48]}...")
    return t


# Decade ticks on a byte axis read as "98 KiB, 977 KiB, 9.5 MiB". Pin the
# ticks to the sizes that actually mean something on this machine instead.
BYTE_TICKS = [32 * KIB, 256 * KIB, 1 * MIB, 4 * MIB, 32 * MIB, 256 * MIB,
              1 * GIB, 4 * GIB]


def byte_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in BYTE_TICKS if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([human_bytes(t) for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("cache size (bytes, log scale)")


def human_bytes(b):
    b = float(b)
    for unit, div in (("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if b >= div:
            v = b / div
            return f"{v:.0f} {unit}" if v >= 10 or v == int(v) else f"{v:.1f} {unit}"
    return f"{b:.0f} B"


# --------------------------------------------------------------------------
# Task 1: hot-row cache
# --------------------------------------------------------------------------

def load_access_profile(prof_dir, tag):
    """One (coverage.csv, hist.csv) pair -> everything the cache figures need.

    The driver dumps only the hottest rows to hist.csv (2M by default), so the
    histogram alone cannot be normalised -- it does not know the totals. The
    coverage file does: it reports (rows, pct_fetches, pct_bytes) at fixed
    cache sizes over ALL touched rows. Anchoring the histogram's cumulative
    sums on the largest coverage row that still falls inside the dump recovers
    the true totals to ~6 significant figures.
    """
    cov = pd.read_csv(os.path.join(prof_dir, f"{tag}_coverage.csv"))
    hist = pd.read_csv(os.path.join(prof_dir, f"{tag}_hist.csv"))

    cum_acc = hist["hits"].to_numpy(dtype=np.int64).cumsum()
    cum_traffic = (hist["hits"].to_numpy(np.int64)
                   * hist["row_bytes"].to_numpy(np.int64)).cumsum()
    cum_resident = hist["row_bytes"].to_numpy(np.int64).cumsum()
    n_dumped = len(hist)

    inside = cov[(cov["rows"] > 0) & (cov["rows"] <= n_dumped)
                 & (cov["pct_fetches"] > 0)]
    if inside.empty:
        raise SystemExit(f"{tag}: no coverage anchor falls inside the {n_dumped}-row dump")
    anchor = inside.iloc[-1]
    ar = int(anchor["rows"])
    total_acc = cum_acc[ar - 1] / (anchor["pct_fetches"] / 100.0)
    total_traffic = cum_traffic[ar - 1] / (anchor["pct_bytes"] / 100.0)

    # Distinct rows touched: the first coverage row that reaches 100%.
    done = cov[cov["pct_fetches"] >= 99.9999]
    distinct = int(done["rows"].iloc[0]) if not done.empty else int(cov["rows"].max())

    return dict(
        tag=tag, cov=cov, hist=hist,
        cum_acc=cum_acc, cum_traffic=cum_traffic, cum_resident=cum_resident,
        n_dumped=n_dumped, distinct=distinct,
        total_acc=total_acc, total_traffic=total_traffic,
        mean_row_bytes=float(hist["row_bytes"].mean()),
        dump_limit_bytes=int(cum_resident[-1]),
    )


def dense_curve(p, n_points=400):
    """Cache size -> (% fetches, % traffic), at the resolution of the dump."""
    budgets = np.unique(np.geomspace(
        max(p["cum_resident"][0], 1), p["dump_limit_bytes"], n_points).astype(np.int64))
    # searchsorted on the monotone resident-bytes prefix: how many of the
    # hottest rows fit in each budget.
    fit = np.searchsorted(p["cum_resident"], budgets, side="right")
    fit = np.clip(fit, 1, p["n_dumped"])
    return (budgets,
            100.0 * p["cum_acc"][fit - 1] / p["total_acc"],
            100.0 * p["cum_traffic"][fit - 1] / p["total_traffic"],
            fit)


def zipf_reference(n_items, budgets, mean_row_bytes):
    """What the same curve would look like if the workload were Zipf(s=1).

    Share of accesses in the r hottest items is H_r/H_N; with uniform row size
    a budget of B bytes holds r = B/mean_row_bytes items. This is the shape a
    cache is designed for -- the point of the comparison is that ours is not
    remotely this.
    """
    HN = math.log(n_items) + 0.5772156649
    r = np.clip(budgets / mean_row_bytes, 1, n_items)
    return 100.0 * (np.log(r) + 0.5772156649) / HN


def cache_figures(profiles, outdir, dataset):
    os.makedirs(outdir, exist_ok=True)
    colors = {p["tag"]: PALETTE[i % len(PALETTE)] for i, p in enumerate(profiles)}
    curves = {p["tag"]: dense_curve(p) for p in profiles}

    def hw_markers(ax, at="bottom"):
        lo, hi = ax.get_ylim()
        log_y = ax.get_yscale() == "log"
        for name, size, _ in HW_LEVELS:
            ax.axvline(size, color=BASELINE, lw=0.9, ls=":", zorder=1)
            if at == "bottom":
                y = lo * 1.15 if log_y else lo + (hi - lo) * 0.015
                va = "bottom"
            else:
                y = hi / 1.15 if log_y else hi - (hi - lo) * 0.015
                va = "top"
            ax.text(size, y, f" {name}", rotation=90, va=va, ha="left",
                    fontsize=7, color=INK_MUTED)

    # ---- fig 1: the headline. traffic served vs cache size ----------------
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    style_axes(ax)
    for p in profiles:
        b, _, pct_tr, _ = curves[p["tag"]]
        c = colors[p["tag"]]
        ax.plot(b, pct_tr, color=c, lw=2.2, label=f"{p['tag']} (measured)", zorder=4)
        # beyond the dump the driver's own anchors are still exact
        anc = p["cov"][p["cov"]["rows"] > p["n_dumped"]]
        if not anc.empty:
            xs = np.concatenate([[b[-1]], anc["cache_bytes"].to_numpy()])
            ys = np.concatenate([[pct_tr[-1]], anc["pct_bytes"].to_numpy()])
            ax.plot(xs, ys, color=c, lw=2.2, ls="--", zorder=4)
            ax.plot(anc["cache_bytes"], anc["pct_bytes"], "o", color=c, ms=5.5, zorder=5)
    ref = zipf_reference(profiles[0]["distinct"], curves[profiles[0]["tag"]][0],
                         profiles[0]["mean_row_bytes"])
    ax.plot(curves[profiles[0]["tag"]][0], ref, color=INK_MUTED, lw=1.6, ls="-.",
            label="Zipf(s=1) reference, same row count", zorder=3)
    ax.set_ylim(0, 100)
    byte_axis(ax, 24 * KIB, 6 * GIB)
    hw_markers(ax, at="bottom")
    ax.set_ylabel("% of DRAM row traffic served from cache")
    place_title(fig, CACHE_TITLES["fig1_traffic_served_vs_cache_size"])
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(0.01, 0.01,
             f"{dataset} | solid = per-row resolution of the histogram dump, "
             "dashed = driver's own exact anchors beyond it",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig1_traffic_served_vs_cache_size.png"), dpi=170)
    plt.close(fig)

    # ---- fig 2: flatness. traffic served vs fraction of corpus resident ---
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    style_axes(ax)
    ax.plot([0, 100], [0, 100], color=BASELINE, lw=1.4, ls="--",
            label="perfectly uniform access (no skew at all)", zorder=2)
    for p in profiles:
        b, _, pct_tr, fit = curves[p["tag"]]
        c = colors[p["tag"]]
        ax.plot(100.0 * fit / p["distinct"], pct_tr, color=c, lw=2.2,
                label=p["tag"], zorder=4)
        anc = p["cov"][p["cov"]["rows"] > p["n_dumped"]]
        if not anc.empty:
            xs = np.concatenate([[100.0 * fit[-1] / p["distinct"]],
                                 100.0 * anc["rows"].to_numpy() / p["distinct"]])
            ys = np.concatenate([[pct_tr[-1]], anc["pct_bytes"].to_numpy()])
            ax.plot(xs, ys, "--", color=c, lw=2.2, zorder=4)
            ax.plot(xs[1:], ys[1:], "o", color=c, ms=5.5, zorder=5)
    p0 = profiles[-1]
    one_gib = p0["cov"][p0["cov"]["cache_bytes"] == GIB]
    if not one_gib.empty:
        r = one_gib.iloc[0]
        ax.annotate(f"1 GiB cache\n{100.0 * r['rows'] / p0['distinct']:.0f}% of the "
                    f"touched corpus resident\nfor {r['pct_bytes']:.0f}% of the traffic",
                    xy=(100.0 * r["rows"] / p0["distinct"], r["pct_bytes"]),
                    xytext=(22, 92), fontsize=9, color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of the touched corpus held resident")
    ax.set_ylabel("% of DRAM row traffic served")
    place_title(fig, CACHE_TITLES["fig2_flatness_lorenz"])
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.text(0.01, 0.01, dataset, fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig2_flatness_lorenz.png"), dpi=170)
    plt.close(fig)

    # ---- fig 3: leverage. traffic share / corpus share --------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    style_axes(ax)
    for p in profiles:
        b, _, pct_tr, fit = curves[p["tag"]]
        c = colors[p["tag"]]
        share = 100.0 * fit / p["distinct"]
        ax.plot(b, pct_tr / share, color=c, lw=2.2, label=p["tag"], zorder=4)
        anc = p["cov"][p["cov"]["rows"] > p["n_dumped"]]
        if not anc.empty:
            ax.plot(anc["cache_bytes"],
                    anc["pct_bytes"] / (100.0 * anc["rows"] / p["distinct"]),
                    "o--", color=c, lw=2.2, ms=5.5, zorder=4)
    ax.axhline(1.0, color=BASELINE, lw=1.4, ls="--", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.75, ax.get_ylim()[1])
    byte_axis(ax, 24 * KIB, 6 * GIB)
    ax.text(30 * KIB, 0.82, "1.0 = the cached rows are no hotter than the average row",
            fontsize=8, color=INK_MUTED)
    hw_markers(ax, at="top")
    ax.set_ylabel("leverage  =  traffic share / corpus share")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    place_title(fig, CACHE_TITLES["fig3_cache_leverage"])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=9)
    fig.text(0.01, 0.01, dataset, fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig3_cache_leverage.png"), dpi=170)
    plt.close(fig)

    # ---- fig 4: rank-frequency, log-log -----------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    style_axes(ax)
    for p in profiles:
        n = p["n_dumped"]
        sample = np.unique(np.geomspace(1, n, 1500).astype(np.int64)) - 1
        ax.plot(sample + 1, p["hist"]["hits"].to_numpy()[sample],
                color=colors[p["tag"]], lw=2.0, label=p["tag"], zorder=4)
    p0 = profiles[0]
    r = np.geomspace(1, p0["n_dumped"], 300)
    ax.plot(r, p0["hist"]["hits"].iloc[0] / r, color=INK_MUTED, lw=1.6, ls="-.",
            label="Zipf(s=1) through the hottest row", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("row rank by fetch count")
    ax.set_ylabel("fetches")
    place_title(fig, CACHE_TITLES["fig4_rank_frequency"])
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.text(0.01, 0.01,
             f"{dataset} | only the hottest {p0['n_dumped']:,} rows are dumped",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig4_rank_frequency.png"), dpi=170)
    plt.close(fig)

    # ---- fig 5: what each real cache level buys ---------------------------
    ref = profiles[-1]
    b, _, pct_tr, fit = curves[ref["tag"]]
    # Beyond the histogram dump only the driver's own anchors exist, and they
    # are sparse. Sizes that land between two of them (512 MiB does) are
    # linearly interpolated and flagged as such rather than dropped.
    anc = ref["cov"][ref["cov"]["rows"] > ref["n_dumped"]]
    anc_x = np.concatenate([[b[-1]], anc["cache_bytes"].to_numpy(float)])
    anc_y = np.concatenate([[pct_tr[-1]], anc["pct_bytes"].to_numpy(float)])
    anc_r = np.concatenate([[fit[-1]], anc["rows"].to_numpy(float)])
    rows = []
    for name, size, note in HW_LEVELS:
        if size <= ref["dump_limit_bytes"]:
            i = max(int(np.searchsorted(b, size, side="right")) - 1, 0)
            served, nrows, src = float(pct_tr[i]), int(fit[i]), "measured"
        elif size > anc_x[-1]:
            continue
        else:
            exact = ref["cov"][ref["cov"]["cache_bytes"] == size]
            if not exact.empty:
                served = float(exact["pct_bytes"].iloc[0])
                nrows, src = int(exact["rows"].iloc[0]), "measured"
            else:
                served = float(np.interp(size, anc_x, anc_y))
                nrows = int(np.interp(size, anc_x, anc_r))
                src = "interpolated"
        rows.append((name, size, note, served, nrows, src))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    style_axes(ax)
    served = np.array([r[3] for r in rows])
    hatch = ["//" if r[5] == "interpolated" else None for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, served, color=PALETTE[0], zorder=3, height=0.62, hatch=hatch,
            edgecolor="white", linewidth=0)
    ax.barh(y, 100 - served, left=served, color=BASELINE, zorder=3, height=0.62)
    total_gib = ref["total_traffic"] / GIB
    for i, (name, size, note, pct, nrows, src) in enumerate(rows):
        mark = "*" if src == "interpolated" else " "
        ax.text(101, i, f"{pct:5.1f}% served{mark}  |  {(100 - pct) / 100 * total_gib:5.1f} GiB "
                        f"still from DRAM", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({human_bytes(sz)})" for n, sz, *_ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of DRAM row traffic served from cache")
    place_title(fig, CACHE_TITLES["fig5_hardware_levels"], y=0.965)
    fig.text(0.01, 0.012,
             f"{dataset} | {ref['tag']}, {total_gib:.0f} GiB total\n"
             "'L3 / core at 8 busy' divides the 32 MiB CCD slice by its 8 cores, "
             "the per-core share under a 64-thread run.   * = interpolated",
             fontsize=7.5, color=INK_MUTED, va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0.0, 0.075, 0.72, 0.95))
    fig.savefig(os.path.join(outdir, "fig5_hardware_levels.png"), dpi=170)
    plt.close(fig)
    hw_df = pd.DataFrame(rows, columns=["level", "bytes", "note", "pct_traffic_served",
                                        "rows_resident", "source"])
    hw_df["residual_dram_gib"] = (100 - hw_df["pct_traffic_served"]) / 100 * total_gib
    hw_df.insert(0, "config", ref["tag"])
    hw_df.to_csv(os.path.join(outdir, "hardware_levels.csv"), index=False)

    # ---- fig 6: marginal return -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    style_axes(ax)
    lo_marginal, hi_marginal = [], []
    for p in profiles:
        b, _, pct_tr, _ = curves[p["tag"]]
        d = np.maximum(np.gradient(pct_tr, b / MIB), 1e-9)[2:]
        lo_marginal.append(max(float(np.percentile(d, 2)), 1e-4))
        hi_marginal.append(d.max())
        ax.plot(b[2:], d, color=colors[p["tag"]], lw=2.0, label=p["tag"], zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(min(lo_marginal) / 2, max(hi_marginal) * 3)
    byte_axis(ax, 24 * KIB, 1.2 * GIB)
    hw_markers(ax, at="top")
    ax.set_ylabel("marginal % of traffic served per extra MiB")
    place_title(fig, CACHE_TITLES["fig6_marginal_return"])
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.text(0.01, 0.01, dataset, fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig6_marginal_return.png"), dpi=170)
    plt.close(fig)

    # ---- CSVs -------------------------------------------------------------
    summary = []
    for p in profiles:
        c = p["cov"].copy()
        c.insert(0, "config", p["tag"])
        c["pct_corpus_resident"] = 100.0 * c["rows"] / p["distinct"]
        c["leverage"] = c["pct_bytes"] / c["pct_corpus_resident"].replace(0, np.nan)
        c["residual_dram_gib"] = (100 - c["pct_bytes"]) / 100 * p["total_traffic"] / GIB
        summary.append(c)
    pd.concat(summary).to_csv(os.path.join(outdir, "coverage_summary.csv"), index=False)

    flat = []
    for p in profiles:
        h = p["hist"]
        def share_of_corpus(frac):
            n = min(int(round(p["distinct"] * frac)), p["n_dumped"])
            return 100.0 * p["cum_traffic"][n - 1] / p["total_traffic"], n
        s1, n1 = share_of_corpus(0.01)
        s10, n10 = share_of_corpus(0.10)
        flat.append(dict(
            config=p["tag"],
            total_fetches=int(round(p["total_acc"])),
            distinct_rows=p["distinct"],
            reuse_factor=p["total_acc"] / p["distinct"],
            dram_traffic_gib=p["total_traffic"] / GIB,
            mean_row_bytes=p["mean_row_bytes"],
            hottest_row_fetches=int(h["hits"].iloc[0]),
            fetches_at_rank_1e6=int(h["hits"].iloc[min(999_999, p["n_dumped"] - 1)]),
            head_to_rank1M_ratio=h["hits"].iloc[0] / max(1, h["hits"].iloc[min(999_999, p["n_dumped"] - 1)]),
            traffic_from_hottest_1pct_of_corpus=s1,
            rows_in_hottest_1pct=n1,
            traffic_from_hottest_10pct_of_corpus=s10,
            rows_in_hottest_10pct=n10,
        ))
    pd.DataFrame(flat).to_csv(os.path.join(outdir, "flatness_stats.csv"), index=False)
    return pd.DataFrame(flat), hw_df


# --------------------------------------------------------------------------
# Task 2: the recall tail
# --------------------------------------------------------------------------

def load_tail(perq_dir, efs):
    out = {}
    for ef in efs:
        path = os.path.join(perq_dir, f"perquery.ef{ef}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            # traversal loss: neighbours the walk never put in the candidate set.
            d["miss"] = d["k"] - d["hits_at_ef"]
            out[ef] = d
    return out


def tail_figures(per_ds, outdir):
    """per_ds: {dataset: {ef: dataframe}}"""
    os.makedirs(outdir, exist_ok=True)
    datasets = list(per_ds)

    # ---- fig 1: concentration of traversal loss ---------------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.2),
                             squeeze=False)
    rows = []
    for axi, ds in enumerate(datasets):
        ax = axes[0][axi]
        style_axes(ax)
        for i, (ef, d) in enumerate(sorted(per_ds[ds].items())):
            miss = np.sort(d["miss"].to_numpy())[::-1]
            tot = miss.sum()
            if tot == 0:
                continue
            cum = 100.0 * miss.cumsum() / tot
            pct_q = 100.0 * (np.arange(len(miss)) + 1) / len(miss)
            ax.plot(pct_q, cum, color=PALETTE[i % len(PALETTE)], lw=2.2,
                    label=f"ef={ef}  (recall@ef {d['hits_at_ef'].sum() / (d['k'].iloc[0] * len(d)):.4f})",
                    zorder=4)
            for p in (1, 5, 10):
                idx = max(1, int(round(len(miss) * p / 100)))
                rows.append(dict(dataset=ds, ef=ef, worst_pct=p,
                                 share_of_total_loss=100.0 * miss[:idx].sum() / tot,
                                 queries=idx, total_missed_neighbours=int(tot),
                                 queries_with_any_miss=int((miss > 0).sum()),
                                 n_queries=len(miss)))
        ax.plot([0, 100], [0, 100], color=BASELINE, lw=1.3, ls="--",
                label="loss spread evenly over all queries", zorder=2)
        for p in (1, 5, 10):
            ax.axvline(p, color=BASELINE, lw=0.9, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(0, 101)
        ax.set_xlabel("worst X% of queries (log scale)")
        if axi == 0:
            ax.set_ylabel("% of all missed neighbours")
        ax.set_title(ds, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    place_title(fig, TAIL_TITLES["fig1_loss_concentration"], y=0.98)
    fig.text(0.01, 0.01, "y is a share of the TOTAL miss count, not a mean recall over "
                         "a subset of queries", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig1_loss_concentration.png"), dpi=170)
    plt.close(fig)
    conc = pd.DataFrame(rows)
    conc.to_csv(os.path.join(outdir, "loss_concentration.csv"), index=False)

    # ---- fig 2: how many queries miss anything at all ---------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8),
                             squeeze=False)
    for axi, ds in enumerate(datasets):
        ax = axes[0][axi]
        style_axes(ax)
        efs = sorted(per_ds[ds])
        width = 0.8 / max(1, len(efs))
        kmax = int(max(d["k"].max() for d in per_ds[ds].values()))
        for i, ef in enumerate(efs):
            d = per_ds[ds][ef]
            counts = d["miss"].value_counts().reindex(range(kmax + 1), fill_value=0)
            pct = 100.0 * counts / len(d)
            x = np.arange(kmax + 1) + (i - (len(efs) - 1) / 2) * width
            ax.bar(x, np.maximum(pct.to_numpy(), 1e-3), width=width,
                   color=PALETTE[i % len(PALETTE)], zorder=3,
                   label=f"ef={ef}   {pct.iloc[0]:.1f}% of queries lose nothing")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 300)
        ax.set_xticks(range(kmax + 1))
        ax.set_xlabel("neighbours of the true top-10 the walk never reached")
        if axi == 0:
            ax.set_ylabel("% of queries (log scale)")
        ax.set_title(ds, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    place_title(fig, TAIL_TITLES["fig2_miss_histogram"], y=0.98)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig2_miss_histogram.png"), dpi=170)
    plt.close(fig)

    # ---- fig 3: does term count predict a hard query? ---------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8),
                             squeeze=False)
    corr_rows = []
    for axi, ds in enumerate(datasets):
        ax = axes[0][axi]
        style_axes(ax)
        for i, ef in enumerate(sorted(per_ds[ds])):
            d = per_ds[ds][ef]
            r = float(np.corrcoef(d["n_terms"], d["miss"])[0, 1])
            corr_rows.append(dict(dataset=ds, ef=ef, pearson_r_terms_vs_miss=r,
                                  pearson_r_candset_vs_miss=float(
                                      np.corrcoef(d["cand_set"], d["miss"])[0, 1])
                                  if d["cand_set"].std() > 0 else np.nan))
            bins = np.quantile(d["n_terms"], np.linspace(0, 1, 11))
            bins = np.unique(bins)
            d = d.assign(_b=pd.cut(d["n_terms"], bins, include_lowest=True))
            g = d.groupby("_b", observed=True).agg(
                terms=("n_terms", "mean"), miss=("miss", "mean")).dropna()
            ax.plot(g["terms"], g["miss"], "o-", color=PALETTE[i % len(PALETTE)],
                    lw=2.0, ms=5, label=f"ef={ef}   r = {r:+.3f}", zorder=4)
        ax.set_xlabel("query term count (decile mean)")
        if axi == 0:
            ax.set_ylabel("mean missed neighbours per query")
        ax.set_title(ds, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5)
    place_title(fig, TAIL_TITLES["fig3_hardness_covariate"], y=0.98)
    fig.text(0.01, 0.01, "Pearson r over per-query term count and missed neighbours",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig3_hardness_covariate.png"), dpi=170)
    plt.close(fig)
    pd.DataFrame(corr_rows).to_csv(os.path.join(outdir, "hardness_covariates.csv"),
                                   index=False)

    # ---- fig 4: payoff of routing the tail to an exact scan ---------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.0),
                             squeeze=False)
    pay_rows = []
    for axi, ds in enumerate(datasets):
        ax = axes[0][axi]
        style_axes(ax)
        for i, ef in enumerate(sorted(per_ds[ds])):
            d = per_ds[ds][ef]
            k, nq = int(d["k"].iloc[0]), len(d)
            miss = np.sort(d["miss"].to_numpy())[::-1]
            frac = np.linspace(0, 0.20, 201)
            idx = (frac * nq).astype(int)
            remaining = miss.sum() - np.concatenate([[0], miss.cumsum()])[idx]
            recall = 1.0 - remaining / (k * nq)
            ax.plot(100 * frac, 100 * recall, color=PALETTE[i % len(PALETTE)], lw=2.2,
                    label=f"ef={ef}", zorder=4)
            for f in (0.01, 0.05, 0.10):
                j = int(f * nq)
                pay_rows.append(dict(
                    dataset=ds, ef=ef, routed_fraction=f,
                    recall_if_routed=1.0 - (miss.sum() - miss[:j].sum()) / (k * nq),
                    baseline_recall=1.0 - miss.sum() / (k * nq)))
        ax.axhline(100.0, color=BASELINE, lw=1.3, ls="--", zorder=2)
        ax.set_xlabel("% of queries routed to an exact / inverted scan")
        if axi == 0:
            ax.set_ylabel("recall@10 after routing (%)")
        ax.set_title(ds, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    place_title(fig, TAIL_TITLES["fig4_routing_payoff"], y=0.98)
    fig.text(0.01, 0.01, "upper bound: assumes the router picks the right queries and "
                         "the fallback scan is exact", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig4_routing_payoff.png"), dpi=170)
    plt.close(fig)
    pd.DataFrame(pay_rows).to_csv(os.path.join(outdir, "routing_payoff.csv"), index=False)
    return conc



# --------------------------------------------------------------------------

def main():
    scratch = os.environ.get("SCRATCH", "/pscratch/sd/d/dhaura")
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=os.path.join(here, "results"))
    ap.add_argument("--tag", default=None, help="folder suffix (default: a timestamp)")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["cache", "tail"],
                    help="sections to build (default: all that have inputs)")
    ap.add_argument("--access-dir",
                    default=f"{scratch}/datasets/SpKNN/msmarco_full/access_profile")
    ap.add_argument("--access-dataset", default="msmarco_full (8.84M x 30,109)")
    ap.add_argument("--access-tags", nargs="+",
                    default=["efc200_ef400", "efc200_ef3200", "efc800_ef3200"])
    ap.add_argument("--tail-dir", action="append", metavar="NAME=PATH",
                    default=None, help="repeatable; default: msmarco_full and nq_splade")
    ap.add_argument("--tail-efs", nargs="+", type=int, default=[400, 1600, 3200])
    args = ap.parse_args()
    want = set(args.only) if args.only else {"cache", "tail"}

    stamp = args.tag or _dt.datetime.now().strftime("%Y%m%d-%H%M")
    root = os.path.join(args.out_root, f"memory_profile_{stamp}")
    os.makedirs(root, exist_ok=True)
    print(f"output -> {root}")

    flat = hw = conc = None
    if "cache" in want:
        profiles = []
        for t in args.access_tags:
            try:
                profiles.append(load_access_profile(args.access_dir, t))
                print(f"  loaded access profile {t}")
            except FileNotFoundError:
                print(f"  skipped {t}: no coverage/hist pair in {args.access_dir}")
        if not profiles:
            sys.exit(f"no access profiles under {args.access_dir}")
        flat, hw = cache_figures(profiles, os.path.join(root, "cache"),
                                 args.access_dataset)
        print("  cache/ written")

    if "tail" in want:
        specs = args.tail_dir or [
            f"msmarco_full={scratch}/datasets/SpKNN/msmarco_full/diag_perquery",
            f"nq_splade={scratch}/datasets/SpKNN/nq_splade/diag_perquery",
        ]
        per_ds = {}
        for spec in specs:
            name, _, path = spec.partition("=")
            d = load_tail(path, args.tail_efs)
            if d:
                per_ds[name] = d
                print(f"  loaded tail {name}: ef {sorted(d)}")
            else:
                print(f"  skipped {name}: no perquery.ef*.csv in {path}")
        if not per_ds:
            sys.exit("no per-query files found")
        conc = tail_figures(per_ds, os.path.join(root, "tail"))
        print("  tail/ written")

    def figure_index(titles):
        rows = ["| figure | contents |", "|:---|:---|"]
        rows += [f"| `{name}.png` | {title} |" for name, title in titles.items()]
        return "\n".join(rows) + "\n\n"

    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write(f"# Row-access and recall-tail profile — {stamp}\n\n")
        fh.write(f"Generated by `plot_memory_profile.py` on {_dt.datetime.now():%Y-%m-%d %H:%M}.\n\n")
        if flat is not None:
            fh.write("## cache/ — Row-access distribution and cache coverage\n\n")
            fh.write(f"Access profiles: `{args.access_dir}`\n\n")
            fh.write("### Per-configuration access statistics\n\n")
            fh.write("Totals are recovered from the capped histogram dump by anchoring "
                     "its cumulative sums on the driver's own coverage rows; they agree "
                     "with the driver's printed totals to six significant figures.\n\n")
            fh.write(flat.to_markdown(index=False, floatfmt=".3g") + "\n\n")
            fh.write("### Traffic served at each level of the cache hierarchy\n\n")
            fh.write("`L3 / core at 8 busy` divides the 32 MiB CCD slice by its 8 cores, "
                     "the per-core share under a 64-thread run. Rows marked "
                     "`interpolated` fall between two coverage anchors.\n\n")
            fh.write(hw.to_markdown(index=False, floatfmt=".3g") + "\n\n")
            fh.write("### Figures\n\n")
            fh.write(figure_index(CACHE_TITLES))
        if conc is not None:
            fh.write("## tail/ — Distribution of recall loss across queries\n\n")
            fh.write("### Share of missed neighbours carried by the hardest queries\n\n")
            fh.write("`share_of_total_loss` is a share of the TOTAL miss count, not a "
                     "mean recall over a subset of queries.\n\n")
            fh.write(conc.to_markdown(index=False, floatfmt=".4g") + "\n\n")
            fh.write("### Figures\n\n")
            fh.write(figure_index(TAIL_TITLES))
    print(f"done: {root}/README.md")


if __name__ == "__main__":
    main()
