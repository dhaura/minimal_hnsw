#!/usr/bin/env python3
"""Query / ground-truth dimension overlap of the top-K terms, as in
dimension_overlap/analyze_dimension_overlap.py, for K = 16, 24, 32 and the full pruned row.

Per query and each of its ten exact neighbors (topk_gt_overlap.py, per-pair arrays in
topk_gt_overlap_<ds>_pairs.npz), hits = query terms among the doc's first K pruned terms:
  document-normalized = 100 * hits / min(K, pruned nnz)   share of the tested terms in the query
  query-normalized    = 100 * hits / query nnz            share of the query the test sees
Histograms are densities (% of pairs per overlap point) so series with different bin widths
compare by area: document-normalized top-K values sit on the grid j / K and get bins one step
wide; the full row and the query-normalized view use 2-point bins.
Writes results/sketch_theory/figures/topk_overlap_{document,query}.png/.pdf, both datasets.

  module load python/3.11-24.1.0
  python plot_topk_overlap.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_inline_figures import INK, K_COLOR, REF, RES, save, style  # noqa: E402

SERIES = [(16, K_COLOR[16], "-"), (24, K_COLOR[24], "-"), (32, REF, (0, (5, 4))), (0, "#1baf7a", "-")]  # 0 = full row
NAME = {"msmarco_full": "MS MARCO (full)", "nq_splade": "NQ (SPLADE)"}
WHAT = {"document": "|query ∩ doc's first K terms| / min(K, doc terms)",
        "query": "|query ∩ doc's first K terms| / |query|"}


def overlap(p, k, norm):
    hits = p[f"hits{k}"] if k else p["full"]
    den = p["qnnz"] if norm == "query" else (np.minimum(k, p["nnz"]) if k else p["nnz"])
    return 100.0 * hits / den


def edges(k, norm):
    if k and norm == "document":
        return (np.arange(k + 2) - 0.5) * 100.0 / k  # one bin per value j / K
    return np.arange(0, 102, 2.0)


def figure(norm):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    for ax, ds in zip(axes, NAME):
        p = np.load(RES / f"topk_gt_overlap_{ds}_pairs.npz")
        for k, color, ls in SERIES:
            v, e = overlap(p, k, norm), edges(k, norm)
            h = np.histogram(v, bins=e)[0]
            ax.stairs(100 * h / v.size / np.diff(e), e, color=color, ls=ls, lw=1.8,
                      label=f"{f'top-{k}' if k else 'full pruned row'}: mean {v.mean():.1f}%, "
                            f"0 overlap {100 * (v == 0).mean():.3f}%")
        style(ax, NAME[ds], f"{norm}-normalized overlap (%) = 100 × {WHAT[norm]}", "% of query / GT pairs per overlap point")
        ax.set_xlim(-4, 104)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.suptitle(f"Query / ground-truth overlap of the doc's first K pruned terms (α = 0.85, one sample per "
                 f"query / GT pair), {norm} normalized", color=INK, fontsize=12, x=0.01, ha="left")
    save(fig, f"topk_overlap_{norm}")


if __name__ == "__main__":
    for norm in ("document", "query"):
        figure(norm)
    print(f"figures in {RES / 'figures'}")
