#!/usr/bin/env python3
"""Lossless-oracle ceiling for skipping checked neighbors, against the ideal and Bloom filters
(overlap-check theory, step 4).

A checked neighbor (after the result list fills) that would not enter the result list can be
skipped without changing the search: it never reaches the candidate queue, the results or the
rerank. The oracle skips exactly those, a share o = 1 - useful / checked. Per (T, ef): the share
of those unused neighbors each filter catches (ideal from the ideal-gate run, Bloom from the
bloom_skip run at that T), and the share of true top-10 documents with overlap <= T% of the query
terms (results/dimension_overlap pair file, pruned rows). Also the September in-engine exact-
overlap gate runs at ef=1600 (results/candidate_overlap_*), the only measured recall for an ideal
gate. Run ideal_alpha.py first. Writes results/overlap_theory/oracle_ceiling_<ds>.md and .json.

  python3 oracle_ceiling.py <dataset>
"""
import json
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
TS = [0, 2, 5, 10, 15, 20, 25, 30]


def gate_rows(pattern):
    g = pd.concat([pd.read_csv(f) for f in sorted(RES.glob(pattern))], ignore_index=True)
    g["ef"] = g.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    g["T"] = g.Params.str.extract(r" T=([\d.]+)")[0].astype(float).astype(int)
    return g.groupby(["T", "ef"], as_index=False).sum(numeric_only=True)


def closed_loop(ds):
    """ef=1600 runs: scored distance calls and recall, no gate (gate 0) and gate at 5/10/15%."""
    runs = [(0, RES / f"candidate_overlap_query_normalized/{ds}/pruned")]
    runs += [(g, RES / f"candidate_overlap_query_gate_true_distance/{ds}/pruned/gate_{g:02d}pct") for g in (5, 10, 15)]
    out = []
    for g, d in runs:
        h = pd.read_csv(next(d.glob("*_query_hist.csv")))
        r = pd.read_csv(next(d.glob("*_result.csv"))).iloc[0]
        out.append({"gate": g, "calls": int(h["count"].sum()), "recall": float(r.recall),
                    "beta": int(r.beta), "build_s": float(r.indexing_time_sec)})
    for r in out:
        r["calls_ratio"] = r["calls"] / out[0]["calls"]
    return out


def main(ds):
    meta = json.loads((RES / f"overlap_theory/ideal_alpha_{ds}.json").read_text())
    a_star = meta["alpha_star"]
    ideal = gate_rows(f"bloom_filter/gate_layouts/ideal_threshold_{ds}/*.gate.csv")
    bloom = gate_rows(f"overlap_theory/bloom_skip_{ds}/*.gate.csv")
    pairs = pd.read_csv(next(RES.glob(f"dimension_overlap/*/{ds}_pair_overlaps.csv.gz")))
    gt = [float((100 * pairs.pruned_intersection_nnz <= t * pairs.query_nnz).mean()) for t in TS]
    rows = []
    for r in ideal.sort_values("ef").itertuples():
        unused, b = r.checked - r.useful, bloom[bloom.ef == r.ef].set_index("T")
        rows.append({"ef": int(r.ef), "o": unused / r.checked,
            "ic": [(getattr(r, f"ideal_rej_T{t}") - getattr(r, f"useful_ideal_rej_T{t}")) / unused for t in TS],
            "bc": [float((b.gate_rej[t] - b.useful_gate_rej[t]) / (b.checked[t] - b.useful[t])) for t in TS]})
    cl = closed_loop(ds)
    (RES / f"overlap_theory/oracle_ceiling_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "C_mean": meta["C_mean"], "alpha_star": a_star, "T": TS,
         "gt_at_risk": gt, "rows": rows, "closed_loop": cl}, indent=1))
    pc = lambda x: f"{100 * x:.1f}%"
    L = [f"# Lossless oracle vs overlap filters, {ds}", "",
         f"Break-even alpha* = 1/C = {pc(a_star)}. Oracle = skip every checked neighbor that would not enter "
         "the result list (no effect on the search); bytes vs no filter = 1 - o + 1/C.", "",
         "| ef | " + " | ".join(str(r["ef"]) for r in rows) + " |", "|---" * (len(rows) + 1) + "|",
         "| oracle skip share | " + " | ".join(pc(r["o"]) for r in rows) + " |",
         "| oracle bytes | " + " | ".join(f"{1 - r['o'] + a_star:.2f}x" for r in rows) + " |", "",
         "## Share of unused neighbors caught (ideal / Bloom), and true top-10 at risk (overlap <= T)", "",
         "| T | " + " | ".join(f"ef={r['ef']}" for r in rows) + " | top-10 at risk |", "|---" * (len(rows) + 2) + "|"]
    for i, t in enumerate(TS):
        L.append(f"| {t} | " + " | ".join(f"{pc(r['ic'][i])} / {pc(r['bc'][i])}" for r in rows) + f" | {pc(gt[i])} |")
    L += ["", "## In-engine exact-overlap gate, ef=1600 (September runs; each run its own build)", "",
          "Bytes estimate = scored-call ratio + 1/C (assumes the gate checks no more neighbors than the "
          "ungated run scores).", "",
          "| gate | scored calls | vs no gate | recall | change (pp) | beta | build (s) | bytes est |",
          "|---|---|---|---|---|---|---|---|"]
    for r in cl:
        est = "1.00x" if r["gate"] == 0 else f"{r['calls_ratio'] + a_star:.2f}x"
        L.append(f"| {r['gate'] or 'off'} | {r['calls']:,} | {r['calls_ratio']:.2f}x | {r['recall']:.2f}% | "
                 f"{r['recall'] - cl[0]['recall']:+.2f} | {r['beta']} | {r['build_s']:.0f} | {est} |")
    (RES / f"overlap_theory/oracle_ceiling_{ds}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
