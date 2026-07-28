#!/usr/bin/env python3
"""Parse profiling-ladder output and plot it.

    module load python
    python3 plot_profile.py logs/profile_12345.out -o plots/

Outputs (only for sections found in the input):
    fig1_ablation.png
    fig2_working_set.png
    fig3_scaling.png
    fig4_counters.png
    fig5_cold_warm.png
    profile_data.csv
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")           # headless: no DISPLAY on compute nodes
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# ---------------------------------------------------------------- palette ----
# Reference categorical palette, fixed slot order (blue, aqua, yellow).
# Aqua/yellow fall below 3:1 contrast on this surface, so every mark carries a
# direct value label (the relief rule) and identity is never colour-alone.
S1, S2, S3 = "#2a78d6", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "figure.dpi": 130,
})


def style(ax, xgrid=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", alpha=0.7)
    ax.grid(axis="y" if xgrid else "x", visible=False)


# ----------------------------------------------------------------- parse -----
VARIANT_RE = re.compile(
    r"^([1-6])\s+(\S.*?)\s{2,}([\d.]+)\s*ns/call\s+([\d.]+)\s*GB/s\s+\(([\d.]+)\s*s\)")
WORKSET_RE = re.compile(r"^working set:\s+(\d+)\s+rows\s+=\s+([\d.]+)\s+MB")
SCALE_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
PROF_RE = re.compile(r"PROF\s+(.*)")
COUNTER_RE = re.compile(r"^\s*(\d+)\s+(\S+:u)\s*(?:([\d.]+)\s+/dist-call)?\s*$")
GROUP_RE = re.compile(r"^=+\s*(G\d\S*)\s*=+\s*$")
VARIANT_HDR_RE = re.compile(r"^=+\s*variant\s+(\d)\s*=+\s*$")
BANNER_RE = re.compile(r"^#+\s*(.+?)\s*#+\s*$")

VARIANT_MEANING = {
    1: "compute floor (rows in L1)",
    2: "memory only (no ALU)",
    3: "production kernel (dense)",
    4: "+ batched prefetch",
    5: "+ gather then dense",
    6: "old merge kernel",
}
DENSE_VARIANTS = (1, 2, 3, 4, 5)   # share one kernel; V6 does not


def parse(text):
    """Return a dict of the sections found. Missing sections are absent keys."""
    # best-of: (workset_mb, variant) -> min ns/call
    ablation = {}
    ablation_meta = {}
    scaling = []
    prof_runs = []           # list of dicts, one per `PROF mode=` run
    counters = defaultdict(dict)   # context -> {event: (count, per_dist)}

    cur_ws = None
    cur_ctx = "unknown"
    cur_prof = None

    for line in text.splitlines():
        m = BANNER_RE.match(line)
        if m and "#" in line:
            cur_ctx = m.group(1).strip()
            continue

        m = VARIANT_HDR_RE.match(line)
        if m:
            v = int(m.group(1))
            cur_ctx = "variant %d (%s)" % (v, VARIANT_MEANING[v])
            continue

        m = GROUP_RE.match(line)
        if m:
            continue  # group name is not needed; events are self-identifying

        m = WORKSET_RE.match(line)
        if m:
            cur_ws = float(m.group(2))
            continue

        m = VARIANT_RE.match(line)
        if m and cur_ws is not None:
            v = int(m.group(1))
            name = m.group(2).strip()
            ns = float(m.group(3))
            gbps = float(m.group(4))
            key = (cur_ws, v)
            # best-of-N across reps and repeated invocations
            if key not in ablation or ns < ablation[key]:
                ablation[key] = ns
                ablation_meta[key] = (name, gbps)
            continue

        m = SCALE_RE.match(line)
        if m:
            scaling.append({
                "threads": int(m.group(1)),
                "stream_gbps": float(m.group(2)),
                "dense_ns": float(m.group(3)),
                "dense_mcalls": float(m.group(4)),
                "dense_gbps": float(m.group(5)),
                "dense_gbps_core": float(m.group(6)),
            })
            continue

        m = PROF_RE.search(line)
        if m:
            kv = dict(
                (k, v) for k, v in
                (tok.split("=", 1) for tok in m.group(1).split() if "=" in tok))
            if "mode" in kv:                 # a new run starts
                cur_prof = dict(kv)
                prof_runs.append(cur_prof)
            elif cur_prof is not None:
                cur_prof.update(kv)
            elif kv:
                cur_prof = dict(kv, mode="setup")
                prof_runs.append(cur_prof)
                cur_prof = None
            continue

        m = COUNTER_RE.match(line)
        if m:
            evt = m.group(2)
            cnt = int(m.group(1))
            per = float(m.group(3)) if m.group(3) else None
            counters[cur_ctx][evt] = (cnt, per)
            continue

    out = {}
    if ablation:
        out["ablation"] = ablation
        out["ablation_meta"] = ablation_meta
    if scaling:
        # de-dup by thread count, keep the fastest (best-of across runs)
        by_t = {}
        for r in scaling:
            t = r["threads"]
            if t not in by_t or r["dense_ns"] < by_t[t]["dense_ns"]:
                by_t[t] = r
        out["scaling"] = [by_t[t] for t in sorted(by_t)]
    if prof_runs:
        out["prof"] = prof_runs
    if counters:
        out["counters"] = dict(counters)
    return out


def fnum(d, key):
    try:
        return float(d[key])
    except (KeyError, TypeError, ValueError):
        return None


# ----------------------------------------------------------------- fig 1 -----
def fig_ablation(data, ax=None):
    """E3: stacked compute-floor vs memory-exposed, with the two bounds drawn.

    The bounds carry the argument:
      max(V1,V2)  what perfect overlap of load and compute would cost
      V1 + V2     what running them back-to-back, serially, would cost
    A production bar (V3) that exceeds V1+V2 means fusion is *destroying*
    memory-level parallelism -- restructuring is free money.
    """
    ab = data["ablation"]
    full_ws = max(ws for ws, _ in ab)
    vals = {v: ab[(full_ws, v)] for (ws, v) in ab if ws == full_ws}
    if not {1, 2, 3} <= set(vals):
        return None

    v1, v2, v3 = vals[1], vals[2], vals[3]
    bound = max(v1, v2)
    serial = v1 + v2

    order = [v for v in (1, 2, 3, 4, 5, 6) if v in vals]
    labels = ["V%d  %s" % (v, VARIANT_MEANING[v]) for v in order]
    totals = [vals[v] for v in order]
    comp = [0.0 if v in (2, 6) else min(v1, vals[v]) for v in order]
    mem = [t - c for t, c in zip(totals, comp)]

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = range(len(order))

    ax.barh(y, mem, color=[S3 if v == 6 else S1 for v in order], height=0.62,
            edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.barh(y, comp, left=mem, color=S2, height=0.62,
            edgecolor=SURFACE, linewidth=1.5, zorder=3)
    handles = [Patch(facecolor=S1, label="memory-exposed"),
               Patch(facecolor=S2, label="compute (ALU)")]
    if 6 in vals:
        handles.append(Patch(facecolor=S3, label="old merge kernel (total)"))

    for i, t in enumerate(totals):
        ax.text(t + max(totals) * 0.012, i, "%.0f ns" % t, va="center",
                ha="left", fontsize=9.5, color=INK, fontweight="bold")

    # Where the real traversal actually lands (production, 1 thread).
    prod = None
    for r in data.get("prof", []):
        if r.get("mode") == "batch" and r.get("search_threads") == "1":
            prod = fnum(r, "ns_per_dist")

    n = len(order)
    xmax = max(totals + [serial]) * 1.24
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, xmax)
    # Reserve a strip above the bars for the reference-line labels, so they
    # never collide with the title, the bars, or each other.
    ax.set_ylim(n - 0.4, -2.0)
    ax.set_xlabel("nanoseconds per distance call", labelpad=8)
    ax.set_title("Distance Calculation Profile", pad=12)

    refs = [(bound, "--", INK2, "max(V1,V2) = %.0f" % bound, -0.5),
            (serial, ":", INK2, "V1+V2 = %.0f" % serial, -1.05)]
    if prod:
        refs.append((prod, "-", S3, "real search = %.0f ns/dist" % prod, -1.6))
    for xv, ls, col, lab, ylab in refs:
        ax.axvline(xv, color=col, ls=ls, lw=1.6 if ls == "-" else 1.4, zorder=4)
        # Flip the label to the left of its line if it would run off the right.
        left = xv > xmax * 0.62
        ax.text(xv + (-1 if left else 1) * xmax * 0.012, ylab, lab,
                fontsize=8.5, color=col if col != INK2 else INK2,
                va="center", ha="right" if left else "left",
                fontweight="bold" if ls == "-" else "normal",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.13),
              ncol=len(handles), fontsize=9)
    style(ax)

    mem_share = (v3 - v1) / v3 * 100.0
    penalty = v3 - serial
    note = ("memory-attributable  (V3-V1)/V3 = %.0f%%          "
            "pipelining penalty  V3-(V1+V2) = %+.0f ns" % (mem_share, penalty))
    if 6 in vals:
        note += ("          dense rewrite  V6/V3 = %.2fx"
                 % (vals[6] / v3 if v3 else 0.0))
    ax.annotate(note, xy=(0.0, -0.28), xycoords="axes fraction", fontsize=9,
                color=INK2, va="top")
    if own:
        fig.tight_layout()
    return {"v1": v1, "v2": v2, "v3": v3, "v6": vals.get(6),
            "bound": bound, "serial": serial,
            "mem_share": mem_share, "penalty": penalty, "prod": prod}


# ----------------------------------------------------------------- fig 2 -----
def fig_working_set(data, ax=None):
    """E4: ns/call vs working-set size, with the hardware knees marked."""
    ab = data["ablation"]
    pts = sorted((ws, ns) for (ws, v), ns in ab.items() if v == 3)
    if len(pts) < 3:
        return None

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    for x, lab in ((1.0, "L2\n1 MB"), (6.0, "4K-page STLB\nreach 6 MB"),
                   (36.6, "L3/socket\n36 MB")):
        if min(xs) <= x <= max(xs):
            ax.axvline(x, color=GRID, lw=1.2, ls="-", zorder=1)
            ax.text(x, max(ys) * 1.04, lab, fontsize=8, color=INK2,
                    ha="center", va="bottom")

    ax.plot(xs, ys, color=S1, marker="o", markersize=7, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, yv in pts:
        ax.annotate("%.0f" % yv, (x, yv), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=INK)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g MB" % v) if v >= 1 else ("%g KB" % (v * 1024))))
    ax.set_xlabel("Working Set (Dataset Size in MB)")
    ax.set_ylabel("ns per Distance Call")
    ax.set_ylim(0, max(ys) * 1.25)
    ax.set_title("Cost vs Working Set Size")
    style(ax, xgrid=False)
    if own:
        fig.tight_layout()
    return True


# ----------------------------------------------------------------- fig 3 -----
def fig_scaling(data, axes=None):
    """E5: the verdict. Latency-bound scales; bandwidth-bound saturates."""
    sc = data["scaling"]
    if len(sc) < 3:
        return None
    t = [r["threads"] for r in sc]
    dense = [r["dense_gbps"] for r in sc]
    stream = [r["stream_gbps"] for r in sc]
    per_core = [r["dense_gbps_core"] for r in sc]

    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1, a2 = axes

    # No dual axis: two measures of different scale get two panels.
    a1.plot(t, stream, color=S1, marker="o", markersize=6, label="stream ceiling",
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    a1.plot(t, dense, color=S2, marker="s", markersize=6, label="distanceDense() achieved",
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    a1.annotate("%.0f" % stream[-1], (t[-1], stream[-1]), textcoords="offset points",
                xytext=(-4, 8), fontsize=8.5, color=INK, ha="right")
    a1.annotate("%.0f" % dense[-1], (t[-1], dense[-1]), textcoords="offset points",
                xytext=(-4, -14), fontsize=8.5, color=INK, ha="right")
    a1.set_xscale("log", base=2)
    a1.set_xticks(t)
    a1.set_xticklabels([str(x) for x in t])
    a1.set_xlabel("threads")
    a1.set_ylabel("aggregate GB/s")
    a1.set_ylim(0, max(stream) * 1.18)          # headroom for the end labels
    a1.set_title("Achieved vs Achievable Bandwidth", pad=10)
    a1.legend(loc="upper left", fontsize=9)
    style(a1, xgrid=False)

    a2.plot(t, per_core, color=S2, marker="s", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    a2.axhline(per_core[0], color=INK2, ls="--", lw=1.3)
    a2.text(t[0], per_core[0], " single-thread rate", fontsize=8.5,
            color=INK2, va="bottom")
    for x, yv in ((t[0], per_core[0]), (t[-1], per_core[-1])):
        a2.annotate("%.3f" % yv, (x, yv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    a2.set_xscale("log", base=2)
    a2.set_xticks(t)
    a2.set_xticklabels([str(x) for x in t])
    a2.set_xlabel("threads")
    a2.set_ylabel("GB/s per core (dense kernel)")
    a2.set_ylim(0, max(per_core) * 1.3)
    a2.set_title("Per-core Throughput")
    style(a2, xgrid=False)

    # Verdict, stated with the numbers that produce it.
    eff = dense[-1] / (dense[0] * t[-1] / t[0]) if dense[0] else 0.0
    ratio = dense[-1] / stream[-1] if stream[-1] else 0.0
    if ratio > 0.8 and eff < 0.5:
        verdict = ("BANDWIDTH-BOUND: the dense kernel reaches %.0f%% of the stream ceiling "
                   "and scaling efficiency is %.0f%%." % (ratio * 100, eff * 100))
    elif eff > 0.7:
        verdict = ("LATENCY-BOUND: still scaling (%.0f%% efficiency at %d threads) at "
                   "only %.0f%% of the stream ceiling."
                   % (eff * 100, t[-1], ratio * 100))
    else:
        verdict = ("MIXED: %.0f%% scaling efficiency at %d threads, %.0f%% of the "
                   "stream ceiling."
                   % (eff * 100, t[-1], ratio * 100))
    a1.annotate(verdict, xy=(0.0, -0.30), xycoords="axes fraction",
                fontsize=9, color=INK2, va="top")
    if own:
        fig.tight_layout()
    return True


# ----------------------------------------------------------------- fig 4 -----
INTERESTING = [
    ("mem_load_l3_miss_retired.local_dram", "DRAM loads (local)"),
    ("mem_load_l3_miss_retired.remote_dram", "DRAM loads (remote)"),
    ("mem_load_retired.l2_miss", "L2 misses"),
    ("dtlb_load_misses.miss_causes_a_walk", "page walks"),
]


def fig_counters(data, axes=None):
    """E2: the counters, per distance call -- the only unit that compares."""
    ctrs = data["counters"]
    ctxs = [c for c in ctrs if any(
        ev + ":u" in ctrs[c] and ctrs[c][ev + ":u"][1] is not None
        for ev, _ in INTERESTING)]
    if not ctxs:
        return None
    ctxs = ctxs[:5]

    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                 gridspec_kw={"width_ratios": [2, 1]})
    a1, a2 = axes

    n = len(ctxs)
    width = 0.8 / n
    colors = [S1, S2, S3, "#008300", "#4a3aa7"]
    for i, ctx in enumerate(ctxs):
        vals, xs = [], []
        for j, (ev, _) in enumerate(INTERESTING):
            per = ctrs[ctx].get(ev + ":u", (None, None))[1]
            vals.append(per if per is not None else 0.0)
            xs.append(j + (i - (n - 1) / 2) * width)
        b = a1.bar(xs, vals, width=width * 0.9, color=colors[i % len(colors)],
                   label=short_ctx(ctx), edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for rect, v in zip(b, vals):
            if v > 0:
                a1.text(rect.get_x() + rect.get_width() / 2, v,
                        "%.2f" % v, ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    a1.set_xticks(range(len(INTERESTING)))
    a1.set_xticklabels([lab for _, lab in INTERESTING], fontsize=9)
    a1.set_ylabel("Events per Distance Call")
    allv = [ctrs[c].get(e + ":u", (None, 0))[1] or 0 for c in ctxs for e, _ in INTERESTING]
    a1.set_ylim(0, max(allv + [1.0]) * 1.30)     # headroom for bar labels
    a1.set_title("Memory Events per Distance Call", pad=10)
    a1.legend(fontsize=8.5, loc="upper left")
    style(a1, xgrid=False)

    ipcs, names = [], []
    for ctx in ctxs:
        c = ctrs[ctx].get("cycles:u", (None, None))[0]
        ins = ctrs[ctx].get("instructions:u", (None, None))[0]
        if c and ins:
            ipcs.append(ins / c)
            names.append(short_ctx(ctx))
    if ipcs:
        bars = a2.bar(range(len(ipcs)), ipcs, color=[colors[i % len(colors)]
                                                     for i in range(len(ipcs))],
                      width=0.6, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for rect, v in zip(bars, ipcs):
            a2.text(rect.get_x() + rect.get_width() / 2, v, "%.2f" % v,
                    ha="center", va="bottom", fontsize=9, color=INK,
                    fontweight="bold")
        a2.set_xticks(range(len(ipcs)))
        a2.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        a2.set_ylabel("Instructions per Cycle")
        a2.set_ylim(0, max(ipcs) * 1.25)
        a2.set_title("IPC")
    else:
        a2.axis("off")
    style(a2, xgrid=False)
    if own:
        fig.tight_layout()
    return True


def fig_time_split(data, ax=None):
    """E6: does distance() actually own the search time?
    """
    rep = None
    for r in data.get("prof", []):
        if r.get("mode") == "replay":
            rep = r
    if not rep:
        return None
    share = fnum(rep, "distance_share")
    if share is None:
        return None
    if rep.get("ndist_match") == "0":
        share = None                       # replay diverged; refuse to plot it
    if share is None:
        return None

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9.5, 2.9))

    dist_pct = share * 100.0
    ovh_pct = 100.0 - dist_pct
    ax.barh([0], [dist_pct], color=S1, height=0.5, label="distanceDense()",
            edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.barh([0], [ovh_pct], left=[dist_pct], color=S2, height=0.5,
            label="traversal overhead (heaps, visited bits, neighbor walks, query scatter)",
            edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(dist_pct / 2, 0, "%.1f%%" % dist_pct, ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")
    ax.text(min(dist_pct + ovh_pct / 2, 97), 0.30, "%.1f%%" % ovh_pct,
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.5)          # headroom so the legend clears the bar
    ax.set_yticks([])
    ax.set_xlabel("Percentage of Search-phase Time")
    ax.set_title("Timing Profile", pad=10)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    style(ax)

    true_ns = fnum(rep, "ns_per_dist_true")
    naive_ns = fnum(rep, "ns_per_dist_naive")
    gfrac = fnum(rep, "graph_frac_of_bytes")
    bits = []
    if true_ns and naive_ns:
        bits.append("True %.0f ns/dist vs Naive %.0f ns/dist (search_time/ndist "
                    "overstates by %.0f%%)" % (true_ns, naive_ns,
                                               (naive_ns / true_ns - 1) * 100))
    if gfrac is not None:
        bits.append("Graph Traffic = %.1f%% of bytes" % (gfrac * 100))
    if bits:
        ax.annotate("   |   ".join(bits), xy=(0.0, -0.42),
                    xycoords="axes fraction", fontsize=9, color=INK2, va="top")
    if own:
        fig.tight_layout()
    return True


# Explicit legend/x-axis overrides for fig4 contexts (checked before the
# generic prefix-stripping below).
CTX_LABEL = {
    "E2: counters on the real search (gated)": "Real Search (Batch)",
    "E6 + E3b + E1: the real driver (one index build)": "Real Search",
    "E2/E3 counters: per-variant groups (1 core)": "Micro-benchmark",
}


def short_ctx(c):
    if c in CTX_LABEL:
        return CTX_LABEL[c]
    m = re.match(r"variant (\d)", c)
    if m:
        return "V%s %s" % (m.group(1), VARIANT_MEANING[int(m.group(1))].split(" (")[0])
    c = re.sub(r"^E\d[+E\d]*:\s*", "", c)
    return (c[:26] + "...") if len(c) > 29 else c


# ----------------------------------------------------------------- fig 5 -----
def fig_cold_warm(data, ax=None):
    """E3b: the same measurement, but on the real traversal instead of a proxy."""
    rep = None
    for r in data.get("prof", []):
        if r.get("mode") == "repeat":
            rep = r
    if not rep:
        return None
    cold = fnum(rep, "cold_us_per_query")
    warm = fnum(rep, "warm_us_per_query")
    share = fnum(rep, "mem_share")
    if cold is None or warm is None:
        return None

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.bar([0], [warm], color=S2, width=0.55, label="compute + cache-resident",
           edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.bar([0], [cold - warm], bottom=[warm], color=S1, width=0.55,
           label="memory-exposed (DRAM + page walks)",
           edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.bar([1], [warm], color=S2, width=0.55, edgecolor=SURFACE,
           linewidth=1.5, zorder=3)

    ax.text(0, cold, "%.1f us\ncold" % cold, ha="center", va="bottom",
            fontsize=10, color=INK, fontweight="bold")
    ax.text(1, warm, "%.1f us\nwarm" % warm, ha="center", va="bottom",
            fontsize=10, color=INK, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1st pass (cold)", "2nd pass (warm)"])
    ax.set_ylabel("Microseconds per Query")
    ax.set_ylim(0, cold * 1.62)          # room for bar labels, legend and share
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("Same Query Twice: Cold vs Warm (Real Traversal)", pad=10)
    ax.legend(loc="upper right", fontsize=8.5)
    if share is not None:
        ax.text(0.02, 0.97, "Memory Percentage of Search = %.0f%%" % (share * 100),
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                color=INK, fontweight="bold")
    style(ax, xgrid=False)

    warn = []
    if rep.get("path_mismatches", "0") != "0":
        warn.append("path_mismatches=%s (passes diverged -- result invalid)"
                    % rep["path_mismatches"])
    bmax = fnum(rep, "bytes_per_query_max")
    if bmax and bmax > 24e6:
        warn.append("footprint %.0f MB > L3: warm pass polluted, share is a floor"
                    % (bmax / 1e6))
    if warn:
        ax.annotate("WARNING: " + "; ".join(warn), xy=(0.0, -0.16),
                    xycoords="axes fraction", fontsize=8.5, color="#b03030")
    if own:
        fig.tight_layout()
    return True


# ------------------------------------------------------------------ csv ------
def write_csv(data, path):
    rows = []
    for (ws, v), ns in sorted(data.get("ablation", {}).items()):
        rows.append(["ablation", "variant_%d" % v, "%.1f" % ws, "ns_per_call", ns])
    for r in data.get("scaling", []):
        for k in ("stream_gbps", "dense_gbps", "dense_gbps_core", "dense_ns"):
            rows.append(["scaling", "threads_%d" % r["threads"], "", k, r[k]])
    for r in data.get("prof", []):
        tag = "%s_%sthr" % (r.get("mode", "?"), r.get("search_threads", "?"))
        for k, v in r.items():
            try:
                rows.append(["production", tag, "", k, float(v)])
            except ValueError:
                pass
    for ctx, evs in data.get("counters", {}).items():
        for ev, (cnt, per) in evs.items():
            rows.append(["counters", ctx, "", ev, cnt])
            if per is not None:
                rows.append(["counters", ctx, "", ev + "_per_dist", per])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "series", "working_set_mb", "metric", "value"])
        w.writerows(rows)
    return len(rows)


# ----------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="profiling output (default: stdin)")
    ap.add_argument("-o", "--outdir", default="plots")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    args = ap.parse_args()

    text = open(args.log).read() if args.log else sys.stdin.read()
    data = parse(text)
    if not data:
        sys.exit("no profiling output recognized -- is this a ladder log?")

    os.makedirs(args.outdir, exist_ok=True)

    def save(fig, name):
        for ext in (["png"] + (["pdf"] if args.pdf else [])):
            p = os.path.join(args.outdir, "%s.%s" % (name, ext))
            fig.savefig(p, bbox_inches="tight")
        print("  %-22s %s.png" % (name, os.path.join(args.outdir, name)))
        plt.close(fig)

    print("wrote:")
    stats = None
    have = {}

    if "ablation" in data:
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        stats = fig_ablation(data, ax)
        if stats:
            fig.tight_layout(); save(fig, "fig1_ablation"); have["ablation"] = True
        else:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        if fig_working_set(data, ax):
            fig.tight_layout(); save(fig, "fig2_working_set"); have["ws"] = True
        else:
            plt.close(fig)

    if "scaling" in data:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        if fig_scaling(data, axes):
            fig.tight_layout(); save(fig, "fig3_scaling"); have["scaling"] = True
        else:
            plt.close(fig)

    if "counters" in data:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                 gridspec_kw={"width_ratios": [2, 1]})
        if fig_counters(data, axes):
            fig.tight_layout(); save(fig, "fig4_counters"); have["counters"] = True
        else:
            plt.close(fig)

    if "prof" in data:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        if fig_cold_warm(data, ax):
            fig.tight_layout(); save(fig, "fig5_cold_warm"); have["coldwarm"] = True
        else:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 2.9))
        if fig_time_split(data, ax):
            fig.tight_layout(); save(fig, "fig6_time_split"); have["split"] = True
        else:
            plt.close(fig)

    n = write_csv(data, os.path.join(args.outdir, "profile_data.csv"))
    print("  %-22s %s (%d rows)" % ("tidy data",
                                    os.path.join(args.outdir, "profile_data.csv"), n))

    for r in data.get("prof", []):
        if r.get("mode") == "replay" and r.get("ndist_match") == "1":
            s = fnum(r, "distance_share")
            print("\nis distanceDense() the bottleneck?  (E6 replay ablation)")
            print("  distanceDense()   = %6.1f%% of search time" % (s * 100))
            print("  traversal overhead= %6.1f%%   (heaps, visited bits, "
                  "neighbor walks)" % ((1 - s) * 100))
            print("  ns/dist true      = %7.0f   (naive search_time/ndist = %.0f)"
                  % (fnum(r, "ns_per_dist_true"), fnum(r, "ns_per_dist_naive")))

    for r in data.get("prof", []):
        if r.get("mode") != "batch":
            continue
        rf = fnum(r, "refine_frac_of_ndist")
        print("\nreal search, %s thread(s)  (E1 batch)" % r.get("search_threads", "?"))
        for key, label, fmt in (
                ("qps", "throughput", "%10.0f QPS"),
                ("recall", "recall@k", "%10.2f %%"),
                ("ns_per_dist", "ns per distance call", "%10.1f ns"),
                ("achieved_GBps", "achieved bandwidth (doc rows)", "%10.2f GB/s"),
                ("total_GBps", "achieved bandwidth (all traffic)", "%10.2f GB/s")):
            v = fnum(r, key)
            if v is not None:
                print(("  %-32s" + fmt) % (label, v))
        if rf is not None:
            print("  %-32s%10.1f %% of distance calls (beta-refine, merge kernel"
                  " on unpruned rows)" % ("exact re-scoring", rf * 100))

    if stats:
        print("\nheadline numbers")
        print("  compute floor  V1 = %7.0f ns/call" % stats["v1"])
        print("  memory only    V2 = %7.0f ns/call" % stats["v2"])
        print("  production     V3 = %7.0f ns/call" % stats["v3"])
        if stats.get("v6"):
            print("  old merge      V6 = %7.0f ns/call  -> dense rewrite is %.2fx"
                  % (stats["v6"], stats["v6"] / stats["v3"]))
        print("  memory share      = %6.0f%%   (V3-V1)/V3" % stats["mem_share"])
        print("  pipelining penalty= %+7.0f ns  V3-(V1+V2)%s" % (
            stats["penalty"],
            "   <- loads and ALU are not overlapping" if stats["penalty"] > 0 else ""))
        print("  pipelining bound  = %7.0f ns/call  -> %.1fx headroom"
              % (stats["bound"], stats["v3"] / stats["bound"]))
        if stats["prod"]:
            print("  real search       = %7.0f ns/dist  (vs %.0f proxy)"
                  % (stats["prod"], stats["v3"]))


if __name__ == "__main__":
    main()
