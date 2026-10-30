#!/usr/bin/env python3
"""Figures of the "Inline Top-K Gate" page, drawn from the saved analysis JSONs.

  gt_hits_<ds>    true top-10 docs by query hits among their first K terms (topk_gt_overlap.py)
  lines_qps_<ds>  lines read (inline_outcome.py) and QPS (inline_pilot.py, preset confirm) of each
                  gated point over the same-job gate-off curve at its recall
  sweep_<ds>      full sweep, best QPS at recall >= r: gate over gate off, gate + gap stop over the
                  gap stop alone (compare_inline_sweep.py)
  slot_layout_<ds> the mean pruned row in its slot, written top-K first, for K = 16 and 24
Writes results/sketch_theory/figures/<name>.png/.pdf; the table views are those scripts' .md files.

  module load python/3.11-24.1.0
  python plot_inline_figures.py msmarco_full
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results" / "sketch_theory"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a93a3"
K_COLOR = {16: "#2a78d6", 24: "#eb6834"}  # reference slots 1-2, as on the page
NAME = {"msmarco_full": "MS MARCO", "nq_splade": "NQ"}
# Slot bytes and the mean pruned row at alpha 0.85 (overlap_theory/ideal_alpha.py: 62.1 and 75.8 terms)
SLOT = {"msmarco_full": (320, 62), "nq_splade": (448, 76)}


def load(name):
    return json.loads((RES / name).read_text())


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def baseline(ax, label):
    ax.axhline(1.0, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (0, 1.0), xycoords=("axes fraction", "data"), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color=INK)


def save(fig, name):
    (RES / "figures").mkdir(exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"{name}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def gt_hits(ds):
    d = load(f"topk_gt_overlap_{ds}.json")
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for k, color, ls in ((16, K_COLOR[16], "-"), (24, K_COLOR[24], "-"), (32, REF, (0, (5, 4)))):
        s = d[f"top{k}"]
        ax.plot(range(len(s["hist"])), [100 * h / d["pairs"] for h in s["hist"]], color=color, ls=ls, lw=2,
                label=f"K = {k}{', past line 0' if k > 30 else ''}: 0 hits {100 * s['zero']:.3f}%, "
                      f"recall@10 ceiling {s['recall_ceiling']:.4f}")
    style(ax, f"{NAME[ds]}: true top-10 docs by query hits among their first K terms",
          "query terms among the doc's first K terms", "% of query / GT pairs")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    save(fig, f"gt_hits_{ds}")


def lines_qps(ds):
    lines, qps = load(f"inline_outcome_{ds}.json")["series"], load(f"inline_confirm_{ds}.json")["points"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for k, color in K_COLOR.items():
        for ax, pts, key in ((axes[0], lines[f"inline{k}"], "traffic_vs_off_matched"),
                             (axes[1], qps[f"SparseHNSW_inline{k}"], "speedup")):
            pts = [p for p in pts if p[key]]  # ef 20 and 30 fall below the gate-off curve's range
            ax.plot([p["recall"] for p in pts], [p[key] for p in pts], "-o", color=color, lw=2, ms=5,
                    label=f"inline, K = {k}")
    style(axes[0], f"{NAME[ds]}: lines read ÷ gate off", "recall@10", "ratio at the same recall")
    style(axes[1], f"{NAME[ds]}: QPS ÷ gate off", "recall@10", "ratio at the same recall")
    for ax in axes:
        baseline(ax, "gate off")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    save(fig, f"lines_qps_{ds}")


def sweep(ds):
    d = load(f"inline_sweep_{ds}.json")
    q, x = d["best_qps"], range(len(d["recall"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for k, color in K_COLOR.items():
        for ax, num, den in ((axes[0], f"inline{k}", "off"), (axes[1], f"inline{k}+Gap", "Gap")):
            ax.plot(x, [a / b for a, b in zip(q[num], q[den])], "-o", color=color, lw=2, ms=5, label=f"inline, K = {k}")
    style(axes[0], f"{NAME[ds]} full sweep: gate ÷ gate off", "recall ≥", "best QPS ratio")
    style(axes[1], f"{NAME[ds]} full sweep: gate + gap stop ÷ gap stop alone", "recall ≥", "")
    for ax, label in zip(axes, ("gate off", "gap stop alone")):
        baseline(ax, label)
        ax.set_xticks(list(x), [f"{r:g}" for r in d["recall"]])
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    save(fig, f"sweep_{ds}")


def slot_layout(ds):
    size, n = SLOT[ds]
    used = 4 + 3 * n
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    for y, k in ((1, 16), (0, 24)):
        x = 0
        for label, w, color, dark in (("H", 4, INK2, True), (f"top-{k} IDs", 2 * k, K_COLOR[k], True),
                                      ("other IDs", 2 * (n - k), "#b3bac6", False), (f"top-{k} w", k, "#8fd6bb", False),
                                      ("other weights", n - k, "#1baf7a", True), ("unused", size - used, "#ecebe7", False)):
            ax.barh(y, w, left=x, height=0.6, color=color, edgecolor=SURFACE, lw=1.2)
            ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
            x += w
    for i, b in enumerate(range(0, size, 64)):
        ax.axvline(b, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(b + 32, 1.45, f"line {i}", ha="center", va="bottom", fontsize=8, color=INK2)
    style(ax, f"{NAME[ds]}: mean pruned row ({n} terms, {used} B) in its {size} B slot, written top-K first",
          f"byte offset (dashed: 64-byte lines). The test reads line 0 only (header + 30 IDs); a row that passes "
          f"is scored over lines 0-{(used - 1) // 64}.", "")
    ax.grid(False)
    ax.set_xlim(0, size)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xticks(range(0, size + 1, 64))
    ax.set_yticks([0, 1], ["K = 24", "K = 16"])
    save(fig, f"slot_layout_{ds}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["msmarco_full", "nq_splade"]:
        gt_hits(ds)
        lines_qps(ds)
        sweep(ds)
        slot_layout(ds)
        print(f"{ds}: figures in {RES / 'figures'}")
