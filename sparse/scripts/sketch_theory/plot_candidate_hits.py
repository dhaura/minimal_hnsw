#!/usr/bin/env python3
"""Candidate-list insertion by top-K query hits, the top-K counterpart of
candidate_overlap/plot_candidate_overlap_hist.py, from the instrumented inline runs
(run_sketch_outcome_perlmutter.sh, results/sketch_theory/outcome_inline_<ds>/*.outcome.csv).

Every layer-0 neighbor scored once the result list holds ef candidates (the ones the gate can
act on) is counted per search by class (0 not added to the candidate list, 1 added and never
expanded, 2 expanded, 3 returned in the top k*beta) and by hits_K = query terms among its top K
pruned terms, K = 16 and 32 (the build records no other K). Added = classes 1-3. Per layout
(off = gate off; inline16 = gate on, so no scored neighbor has 0 top-16 hits) at one ef:
  candidate_hits_*       share within each class by hits_K; the shaded 0 bin is what the gate skips
  candidate_retention_*  100 * added / scored for each hits_K
Writes results/sketch_theory/figures/candidates/<ds>/<layout>/<name>_ef<ef>_k<K>.png/.pdf.

  module load python/3.11-24.1.0
  python plot_candidate_hits.py msmarco_full [off,inline16] [1600]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_inline_figures import GRID, INK, INK2, RES, SURFACE, style  # noqa: E402

ADD, NOT = "#1baf7a", "#eb6834"
NAME = {"msmarco_full": "MS MARCO (full)", "nq_splade": "NQ (SPLADE)"}
GATE = {"off": "gate off", "inline16": "inline gate K = 16", "inline24": "inline gate K = 24"}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def shade(ax):
    ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.7, lw=0)
    ax.axvline(0.5, color=INK2, ls=(0, (4, 3)), lw=1)


def plot(ds, layout, ef, row, k, out):
    h = {c: np.array([row[f"{c}_h{k}_{j}"] for j in range(k + 1)], float)
         for c in ("notadded", "unexpanded", "expanded", "returned")}
    na, ad = h["notadded"], h["unexpanded"] + h["expanded"] + h["returned"]
    xmax = int(np.searchsorted(np.cumsum(na + ad) / (na + ad).sum(), 0.999)) + 1  # >= 99.9% shown
    sub = (f"M=32 · efC=200 · ef={ef} · α=0.85 · β=3 · {GATE[layout]} · scored once the result list "
           f"is full · recall {100 * row['recall']:.2f}%")
    note = (f"0 hits: {ad[0]:,.0f} of {ad[0] + na[0]:,.0f} added ({100 * ad[0] / max(ad[0] + na[0], 1):.2f}%); "
            f"returned with 0 hits: {h['returned'][0]:,.0f} of {h['returned'].sum():,.0f}; overall add rate "
            f"{100 * ad.sum() / (ad + na).sum():.2f}%. x-axis 0-{xmax} (>= 99.9% of candidates).")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, (title, v, color) in zip(axes, (("Added for future expansion", ad, ADD), ("Not added", na, NOT))):
        shade(ax)
        e, s = np.arange(k + 2) - 0.5, 100 * v / v.sum()
        ax.stairs(s, e, color=color, fill=True, alpha=0.35, lw=0)
        ax.stairs(s, e, color=color, lw=2)
        ax.text(0.98, 0.96, f"{v.sum():,.0f} candidates\n{100 * v[0] / v.sum():.2f}% with 0 hits", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK2)
        style(ax, title, f"query terms among the doc's top {k} terms", "share within candidate class (%)" if color == ADD else "")
        ax.set_xlim(-0.5, xmax + 0.5)
    fig.suptitle(f"Candidate list insertion vs top-{k} query hits, {NAME[ds]}\n{sub}", color=INK, fontsize=11.5)
    fig.text(0.01, 0.012, note, color=INK2, fontsize=8.5)
    save(fig, out / f"candidate_hits_ef{ef}_k{k}")

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    shade(ax)
    rate = 100 * ad / np.maximum(ad + na, 1)
    ax.bar(np.arange(xmax + 1), rate[:xmax + 1], width=1.0, color=ADD, alpha=0.55, edgecolor=ADD, lw=0.6)
    ax.text(0.98, 0.96, f"{(ad + na).sum():,.0f} candidates\n{100 * ad.sum() / (ad + na).sum():.2f}% retained overall",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    style(ax, "", f"query terms among the doc's top {k} terms", "candidates added for future expansion (%)")
    ax.set_xlim(-0.5, xmax + 0.5)
    ax.set_ylim(0, 100)
    fig.suptitle(f"Candidate retention rate by top-{k} query hits, {NAME[ds]}\n{sub}", color=INK, fontsize=11.5)
    fig.text(0.01, 0.012, f"Each bar is 100 × added / scored for that hit count. x-axis 0-{xmax} (>= 99.9% of candidates).",
             color=INK2, fontsize=8.5)
    save(fig, out / f"candidate_retention_ef{ef}_k{k}")


def main(ds, layouts="off,inline16", ef="1600"):
    df = pd.read_csv(next((RES / f"outcome_inline_{ds}").glob("*.outcome.csv")))
    df["ef"] = df.Params.str.extract(r"ef=(\d+)")[0].astype(int)
    for layout in layouts.split(","):
        model = "SparseHNSW" if layout == "off" else f"SparseHNSW_{layout}"
        row = df[(df.Model == model) & (df.ef == int(ef))].iloc[0]
        for k in (16, 32):
            plot(ds, layout, int(ef), row, k, RES / "figures" / "candidates" / ds / layout)
    print(f"{ds}: figures in {RES / 'figures' / 'candidates' / ds}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
