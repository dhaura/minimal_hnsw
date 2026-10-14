#!/usr/bin/env python3
"""Rebuild speedup_vs_baselines.csv from a playground pareto.csv.

For each recall threshold r, every method's best QPS among its points with
recall >= r, and SparseHNSW's ratio to each baseline. Same columns as the
original table: recall,ours_qps,<B>_qps,vs_<B>,...

Usage:
  python3 make_speedup_table.py <pareto.csv> <out.csv> [--thresholds 0.90,...,0.99]
"""
import argparse
import csv

OURS = "SparseHNSW"
BASELINES = ["SEISMIC", "kANNolo", "PyANNS", "GrassRMA", "SINDI"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pareto")
    ap.add_argument("out")
    ap.add_argument("--thresholds", default="0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.pareto)))

    def best(model, r):
        q = [float(x["QPS"]) for x in rows if x["Model"] == model and float(x["Recall"]) >= r]
        return max(q) if q else None

    header = ["recall", "ours_qps"]
    for b in BASELINES:
        header += [f"{b}_qps", f"vs_{b}"]
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for t in [float(x) for x in args.thresholds.split(",")]:
            ours = best(OURS, t)
            line = [f"{t:g}", round(ours) if ours else ""]
            for b in BASELINES:
                q = best(b, t)
                line += [round(q) if q else "", round(ours / q, 2) if ours and q else ""]
            w.writerow(line)
    print(open(args.out).read())


if __name__ == "__main__":
    main()
