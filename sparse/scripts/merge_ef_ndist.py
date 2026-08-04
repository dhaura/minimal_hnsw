#!/usr/bin/env python3
"""Merge the two passes of run_ef_ndist.sh into one per-ef table.
"""
import csv
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    __file__).resolve().parent / "results" / "ef_ndist"

def ef_of(params: str) -> int:
    return int(re.search(r"\bef=(\d+)", params).group(1))

bench = {}
with open(out_dir / "bench_timing.csv") as f:
    for r in csv.DictReader(f):
        bench[ef_of(r["Params"])] = r

prof = {}
with open(out_dir / "prof_ndist.csv") as f:
    for r in csv.DictReader(f):
        prof[ef_of(r["Params"])] = r

print(f"{'ef':>6} {'ndist/query':>12} {'recall@10':>10} {'search s':>10} "
      f"{'us/query':>9} {'QPS':>10} {'prof QPS':>10} {'overhead':>9}")
rows = []
for ef in sorted(bench):
    b, p = bench[ef], prof.get(ef)
    med = float(b["SearchSecMedian"])
    qps = float(b["QPS"])
    nd = float(p["NDistPerQuery"]) if p else float("nan")
    pq = float(p["QPS"]) if p else float("nan")
    ovh = (qps / pq - 1) * 100 if p else float("nan")
    print(f"{ef:>6} {nd:>12.1f} {float(b['Recall'])*100:>9.2f}% {med:>10.4f} "
          f"{med/int(b['NumQueries'])*1e6:>9.2f} {qps:>10.0f} {pq:>10.0f} {ovh:>8.1f}%")
    rows.append((ef, nd, float(b["Recall"]), med, qps))

with open(out_dir / "merged.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ef", "ndist_per_query", "recall_at_10", "search_sec_median",
                "qps_48threads"])
    for row in rows:
        w.writerow(row)
print(f"\nwrote {out_dir / 'merged.csv'}")
