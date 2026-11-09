#!/usr/bin/env python3
"""Work counts for the final method sweep (run_final_profile_perlmutter.sh).

Headline graph (M32 efC200 alpha 0.85 beta 3, seeds 8:4, u8 slots), series off / gap / inline16 /
inline16gap (gap 0.1), ef 50-400, sparse_profile batch mode at 64 threads (SPARSE_HNSW_PROFILE build;
its QPS is not used). Per point: recall, distance calls (traversal and beta refine), row bytes and 64 B
row lines per query, inline gate checks / rejects per query, and the per-query recall distribution.

  module load python/3.11-24.1.0
  python analyze_final_profile.py msmarco_full    # -> results/final_sweep/final_profile_<ds>.{md,json}
"""
import csv
import json
import re
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "final_sweep"
SERIES = ["off", "gap", "inline16", "inline16gap"]


def parse(log):
    kv = {}
    for line in open(log):
        if line.startswith("PROF "):
            kv.update(dict(re.findall(r"(\w+)=(\S+)", line)))
    return kv


def main(ds):
    folders = sorted(RES.glob(f"profile_{ds}_*"))
    if not folders:
        sys.exit(f"no profile_{ds}_* folder in {RES}")
    d, rows = folders[-1], {}
    for log in sorted(d.glob("*_ef*.log")):
        series, ef = log.stem.rsplit("_ef", 1)
        kv = parse(log)
        rq = [float(r["recall"]) for r in csv.DictReader(open(log.with_suffix(".perq.csv")))]
        nq = len(rq)
        rows[(series, int(ef))] = dict(
            recall=float(kv["recall"]) / 100, ndist=float(kv["ndist_per_query"]),
            refine=float(kv["refine_ndist"]) / nq, bytes=float(kv["bytes_per_query"]),
            lines=float(kv["lines_per_query"]), checks=float(kv["gate_checks_per_query"]),
            rejects=float(kv["gate_rejects_per_query"]), rmin=float(kv["perq_recall_min"]),
            p5=float(kv["p5"]), p10=float(kv["p10"]), eq1=float(kv["share_eq1"]),
            ge09=sum(r >= 0.9 for r in rq) / nq)
    efs = sorted({ef for _, ef in rows})
    md = [f"# Final method sweep, work counts, {ds} ({d.name})", "",
          "sparse_profile batch mode, 64 threads, M32 efC200 alpha 0.85 beta 3, seeds 8:4, u8 slots, gap 0.1.",
          "Per query: distance calls (traversal + beta refine), 64 B row lines (scored rows + gate reads of",
          "rejected neighbors), inline gate checks / rejects; x = ratio to off at the same ef.", "",
          "| ef | series | recall | dist calls | refine calls | lines | lines x | KiB rows | gate checks | rejects |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ef in efs:
        o = rows.get(("off", ef))
        for s in SERIES:
            r = rows.get((s, ef))
            if not r:
                continue
            md.append(f"| {ef} | {s} | {r['recall']:.4f} | {r['ndist']:,.0f} | {r['refine']:.0f} | {r['lines']:,.0f} | "
                      f"{r['lines'] / o['lines']:.2f} | {r['bytes'] / 1024:,.0f} | {r['checks']:,.0f} | {r['rejects']:,.0f} |")
    md += ["", "Per-query recall@10 distribution (share of queries with recall >= 0.9 and = 1, p5, p10, min).", "",
           "| ef | series | mean | >= 0.9 | = 1 | p5 | p10 | min |", "|---|---|---|---|---|---|---|---|"]
    for ef in efs:
        for s in SERIES:
            r = rows.get((s, ef))
            if r:
                md.append(f"| {ef} | {s} | {r['recall']:.4f} | {r['ge09']:.3f} | {r['eq1']:.3f} | {r['p5']:.1f} | "
                          f"{r['p10']:.1f} | {r['rmin']:.1f} |")
    (RES / f"final_profile_{ds}.md").write_text("\n".join(md) + "\n")
    (RES / f"final_profile_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "folder": d.name, "points": {f"{s}_ef{ef}": v for (s, ef), v in rows.items()}}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
