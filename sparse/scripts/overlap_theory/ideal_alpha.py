#!/usr/bin/env python3
"""Ideal overlap filter against the Bloom break-even alpha >= 1 / C (overlap-check theory, step 3).

alpha(T, ef): share of checked neighbors (checked = after the result list fills) that an
ideal filter skips, i.e. whose exact query-term overlap is <= T% of the query's terms, from
the ideal-gate counts (results/bloom_filter/gate_layouts/ideal_threshold_<ds>, beta=3,
M=32 efC=200 alpha=0.85), plus the Bloom T=10 share from the listsketch run of the same
config. C_i = ceil((3 n_i + 4) / 64) over a row sample of the base pruned like the engine
(fp16 weights, keep the largest until alpha of the row mass). Writes
results/overlap_theory/ideal_alpha_<ds>.md and .json.

  python3 ideal_alpha.py <dataset> <base.csr> [slot=320] [sample=200000]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
TS = [0, 2, 5, 10, 15, 20, 25, 30]


def lines_sample(path, slot, sample, prune=0.85):
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
        indptr = np.fromfile(f, dtype=np.int64, count=nrow + 1)
    data = np.memmap(path, dtype=np.float32, mode="r", offset=24 + 8 * (nrow + 1) + 4 * nnz, shape=(nnz,))
    rows = np.sort(np.random.default_rng(0).choice(nrow, size=min(sample, nrow), replace=False))
    n = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        v = np.sort(data[indptr[r]:indptr[r + 1]].astype(np.float16).astype(np.float32))[::-1]
        c = np.cumsum(v)
        n[i] = min(len(v), int(np.searchsorted(c, prune * c[-1])) + 1) if len(v) else 0
    read = 3 * n + 4
    C = -(-read // 64)
    over = (read + 3) // 4 * 4 > slot  # overflow: slot line holding the pointer, then the row
    C_model = np.where(over, 1 + C, C)
    C_ref = np.where(over, 4 + C, np.maximum(C, 4))  # reference: 4 prefetched slot lines
    return n, C_model, C_ref, over


def main(ds, base, slot=320, sample=200000):
    n, Cm, Cr, over = lines_sample(base, slot, sample)
    hist = {int(c): float((Cm == c).mean()) for c in np.unique(Cm)}
    a_star, a_ref = 1 / Cm.mean(), 1 / Cr.mean()
    ideal = pd.read_csv(next((RES / f"bloom_filter/gate_layouts/ideal_threshold_{ds}").glob("*.gate.csv")))
    ideal["ef"] = ideal.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    ls = pd.read_csv(next((RES / f"bloom_filter/gate_layouts/listsketch_{ds}").glob("*.gate.csv")))
    ls = ls[ls.Params.str.contains(r" T=10\b") & ~ls.Params.str.contains("layout=|pat=") & ls.Params.str.contains("beta=3")]
    ls["ef"] = ls.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    out = {"dataset": ds, "slot": slot, "sample": len(n), "nnz_mean": float(n.mean()), "overflow": float(over.mean()),
           "C_hist": hist, "C_mean": float(Cm.mean()), "C_ref_mean": float(Cr.mean()),
           "alpha_star": a_star, "alpha_star_ref": a_ref, "T": TS, "rows": []}
    for _, r in ideal.sort_values("ef").iterrows():
        b = ls[ls.ef == r.ef]
        out["rows"].append({"ef": int(r.ef), "checked": int(r.checked), "useful": int(r.useful),
            "alpha": [r[f"ideal_rej_T{t}"] / r.checked for t in TS],
            "useful_lost": [r[f"useful_ideal_rej_T{t}"] / r.useful for t in TS],
            "bloom10": float(b.gate_rej.sum() / b.checked.sum()) if len(b) else None,
            "bloom10_useful_lost": float(b.useful_gate_rej.sum() / b.useful.sum()) if len(b) else None})
    (RES / "overlap_theory").mkdir(parents=True, exist_ok=True)
    (RES / f"overlap_theory/ideal_alpha_{ds}.json").write_text(json.dumps(out, indent=1))
    pc = lambda x: f"{100 * x:.1f}%"
    L = [f"# Ideal overlap filter vs break-even alpha, {ds}", "",
         f"Row sample: {len(n):,} rows, pruned nnz mean {n.mean():.1f}, overflow {pc(over.mean())} of a {slot} B slot.",
         "C_i histogram: " + ", ".join(f"C={c}: {pc(v)}" for c, v in hist.items()),
         f"Mean C = {Cm.mean():.2f} -> break-even alpha* = 1/C = {pc(a_star)} "
         f"(reference, current code: mean max(C,4) = {Cr.mean():.2f} -> {pc(a_ref)}).", "",
         "Cells: alpha skipped / useful neighbors lost / filter traffic vs no filter (1 - alpha + 1/C).", "",
         "| ef | Bloom T=10 | " + " | ".join(f"ideal T={t}" for t in TS) + " |", "|---" * (len(TS) + 2) + "|"]
    for r in out["rows"]:
        cell = lambda a, u: f"{pc(a)} / {pc(u)} / {1 - a + a_star:.2f}x"
        bl = cell(r["bloom10"], r["bloom10_useful_lost"]) if r["bloom10"] is not None else "-"
        L.append(f"| {r['ef']} | {bl} | " + " | ".join(cell(a, u) for a, u in zip(r["alpha"], r["useful_lost"])) + " |")
    (RES / f"overlap_theory/ideal_alpha_{ds}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:5]))
