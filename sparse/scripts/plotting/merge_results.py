"""Merge per-method result CSVs into one table, and refuse to merge bad data.

    python3 common/merge_results.py -o merged.csv a.csv b.csv ...
    python3 common/merge_results.py -o merged.csv --pareto results/*.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

SCHEMA = [
    "Model", "Params", "Threads", "Recall", "RR@10", "IndexSec", "LoadSec",
    "ConvertSec", "SearchSecMedian", "SearchSecRuns", "AmortizedUsPerQuery",
    "QPS", "LatencyUsSingleThread", "PeakRSSGB", "IndexBytes", "NumQueries",
    "K", "ResultsOrdered", "Host", "JobID",
]


def read(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in SCHEMA if c not in (r.fieldnames or [])]
        if missing:
            sys.exit(
                f"{path}: missing column(s) {missing}\n"
                "This file predates the current schema. Old results cannot be\n"
                "mixed with new ones -- their columns do not mean the same thing."
            )
        return list(r)


def f(row, col):
    v = row.get(col, "")
    return float(v) if str(v).strip() else float("nan")


def gate_rr(rows):
    bad = [r for r in rows
           if str(r["ResultsOrdered"]).lower().startswith("t")
           and f(r, "RR@10") < f(r, "Recall") - 1e-9]
    for r in bad:
        print(f"  FAIL RR<recall  {r['Model']:<12} {r['Params'][:52]}  "
              f"recall={f(r,'Recall'):.4f} RR={f(r,'RR@10'):.4f}", file=sys.stderr)
    return len(bad)


def gate_ground(rows):
    ks = {(r["K"], r["NumQueries"]) for r in rows}
    if len(ks) > 1:
        print(f"  FAIL mixed ground truth: (K, NumQueries) = {sorted(ks)}", file=sys.stderr)
        return 1
    return 0


def gate_monotonic(rows):
    """Within a method, sorting by recall must not show time going backwards.
    """
    bad = 0
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["Model"]].append(r)
    for model, rs in by_model.items():
        rs = sorted(rs, key=lambda r: f(r, "Recall"))
        for a, b in zip(rs, rs[1:]):
            ta, tb = f(a, "SearchSecMedian"), f(b, "SearchSecMedian")
            if tb < ta * 0.85:          # >15% faster at strictly higher recall
                print(f"  WARN non-monotonic  {model:<12} "
                      f"recall {f(a,'Recall'):.4f}->{f(b,'Recall'):.4f} but "
                      f"time {ta:.4f}s->{tb:.4f}s  [{b['Params'][:40]}]",
                      file=sys.stderr)
                bad += 1
    return bad


def pareto(rows):
    """Keep points not dominated on (recall, QPS): higher is better on both."""
    keep, seen = [], set()
    pts = [(f(r, "Recall"), f(r, "QPS"), r) for r in rows]
    for rec, qps, r in pts:
        if any(orec >= rec and oqps >= qps and (orec > rec or oqps > qps)
               for orec, oqps, _ in pts):
            continue
        key = (round(rec, 12), round(qps, 6))
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--pareto", action="store_true",
                    help="keep only non-dominated (recall, QPS) points, ALL methods")
    ap.add_argument("--force", action="store_true",
                    help="write the merge even if a gate fails")
    args = ap.parse_args()

    rows = []
    for p in args.csvs:
        rs = read(p)
        models = sorted({r["Model"] for r in rs})
        print(f"{p}: {len(rs)} rows ({', '.join(models)})")
        rows.extend(rs)

    if not rows:
        sys.exit("no rows")

    print("\n--- sanity gates ---")
    failures = gate_rr(rows) + gate_ground(rows)
    warns = gate_monotonic(rows)
    print(f"  {failures} hard failure(s), {warns} warning(s)")

    unordered = {r["Model"] for r in rows
                 if not str(r["ResultsOrdered"]).lower().startswith("t")}
    if unordered:
        print(f"  NOTE unranked results (RR not comparable): {sorted(unordered)}")

    if failures and not args.force:
        sys.exit("\nrefusing to merge; fix the failures above or pass --force")

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["Model"]].append(r)

    out = []
    print("\n--- per method ---")
    for model in sorted(by_model):
        rs = by_model[model]
        if args.pareto:
            kept = pareto(rs)
            print(f"  {model:<12} {len(rs):3d} points -> {len(kept):3d} on the frontier")
            rs = kept
        else:
            print(f"  {model:<12} {len(rs):3d} points")
        out.extend(sorted(rs, key=lambda r: f(r, "Recall")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
