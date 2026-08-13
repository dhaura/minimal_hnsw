#!/usr/bin/env python3
"""Collapse several SparseHNSW alpha/beta ef-sweeps into ONE envelope curve.

Reads   <outdir>/sparse_hnsw_results_*.csv   (one per configuration)
Writes  <outdir>/sparse_hnsw_results.csv     (the envelope, single Model)
Moves   the per-configuration files to <outdir>/configs/ so make_figures'
        sparse_hnsw_results*.csv glob cannot pick them up as extra series.
"""
import argparse
import csv
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--model", default="SparseHNSW")
    args = ap.parse_args()

    srcs = sorted(glob.glob(os.path.join(args.outdir, "sparse_hnsw_results_*.csv")))
    if not srcs:
        raise SystemExit(f"no sparse_hnsw_results_*.csv in {args.outdir}")

    rows, fields = [], None
    for s in srcs:
        with open(s) as fh:
            rd = csv.DictReader(fh)
            fields = fields or rd.fieldnames
            n = 0
            for r in rd:
                rows.append(r); n += 1
        print(f"  read {os.path.basename(s)}  ({n} points)")

    pts = sorted(rows, key=lambda r: (-float(r["Recall"]), -float(r["QPS"])))
    keep, best_qps = [], -1.0
    for r in pts:
        q = float(r["QPS"])
        if q > best_qps:
            keep.append(r)
            best_qps = q
    keep.sort(key=lambda r: float(r["Recall"]))

    for r in keep:
        r["Model"] = args.model

    out = os.path.join(args.outdir, "sparse_hnsw_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    cfgdir = os.path.join(args.outdir, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    for s in srcs:
        shutil.move(s, os.path.join(cfgdir, os.path.basename(s)))

    print(f"\nenvelope: {len(keep)} of {len(rows)} points survive as '{args.model}'")
    print(f"  recall {float(keep[0]['Recall']):.4f} -> {float(keep[-1]['Recall']):.4f}")
    print("  winning configuration per point:")
    for r in keep:
        p = r["Params"]
        a = p.split("alpha=")[1].split()[0]
        b = p.split("beta=")[1].split()[0]
        ef = p.split("ef=")[1].split()[0]
        print(f"    recall {float(r['Recall']):.4f}  QPS {float(r['QPS']):>10,.0f}"
              f"   alpha={a} beta={b} ef={ef}")
    print(f"\nwrote {out}")
    print(f"per-configuration CSVs moved to {cfgdir}/ (kept, not globbed)")


if __name__ == "__main__":
    main()
