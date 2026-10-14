#!/usr/bin/env python3
"""Collapse several SparseHNSW alpha/beta ef-sweeps into ONE envelope curve.

Reads   <outdir>/sparse_hnsw_results_*.csv   (one per configuration), plus
        any paths selected with --include
Writes  <outdir>/sparse_hnsw_results.csv     (the envelope, single Model)
Moves   the per-configuration files to <outdir>/configs/ so make_figures'
        sparse_hnsw_results*.csv glob cannot pick them up as extra series.
        Explicitly included files are left in place.
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
    ap.add_argument("--include", action="append", default=[], metavar="GLOB",
                    help="additional input glob, relative to outdir (repeatable)")
    ap.add_argument("--bin", type=float, default=None, metavar="PCT",
                    help="keep the highest-QPS point from all inputs per "
                         "PCT-wide recall bucket (e.g. 0.2); the downstream "
                         "merge can compute a Pareto frontier separately")
    ap.add_argument("--targets", default=None,
                    help="comma-separated recall targets, e.g. "
                         "0.80,0.85,0.90,0.94,0.96,0.97,0.98,0.99. For each, keep "
                         "the fastest point that MEETS it; omit to keep the full front")
    args = ap.parse_args()

    legacy_srcs = glob.glob(os.path.join(args.outdir, "sparse_hnsw_results_*.csv"))
    included_srcs = []
    for pattern in args.include:
        included_srcs.extend(glob.glob(os.path.join(args.outdir, pattern)))
    output = os.path.abspath(os.path.join(args.outdir, "sparse_hnsw_results.csv"))
    srcs = sorted({os.path.abspath(s) for s in legacy_srcs + included_srcs
                   if os.path.abspath(s) != output})
    if not srcs:
        raise SystemExit(f"no input CSVs found in {args.outdir}")

    rows, fields = [], None
    for s in srcs:
        with open(s) as fh:
            rd = csv.DictReader(fh)
            fields = fields or rd.fieldnames
            n = 0
            for r in rd:
                rows.append(r); n += 1
        print(f"  read {os.path.basename(s)}  ({n} points)")

    if args.bin:
        if args.bin <= 0:
            raise SystemExit("--bin must be greater than zero")
        best = {}
        for r in rows:
            bucket = round(float(r["Recall"]) * 100.0 / args.bin)
            if bucket not in best or float(r["QPS"]) > float(best[bucket]["QPS"]):
                best[bucket] = r
        print(f"  binned {len(rows)} input points -> {len(best)} "
              f"(highest QPS per {args.bin:g}% recall bucket)")
        keep = sorted(best.values(), key=lambda r: float(r["Recall"]))
    else:
        pts = sorted(rows, key=lambda r: (-float(r["Recall"]), -float(r["QPS"])))
        keep, best_qps = [], -1.0
        for r in pts:
            q = float(r["QPS"])
            if q > best_qps:
                keep.append(r)
                best_qps = q
        keep.sort(key=lambda r: float(r["Recall"]))

    if args.targets:
        targets = [float(t) for t in args.targets.split(",") if t.strip()]
        picked, seen = [], set()
        for t in sorted(targets):
            # fastest point at or above the target; the front is sorted by
            # recall so the earliest qualifying entry is also the fastest.
            hit = next((r for r in keep if float(r["Recall"]) >= t), None)
            if hit is not None and id(hit) not in seen:
                seen.add(id(hit))
                picked.append(hit)
        if picked:
            print(f"  thinned {len(keep)} front points -> {len(picked)} "
                  f"at targets {args.targets}")
            keep = picked

    for r in keep:
        r["Model"] = args.model

    out = os.path.join(args.outdir, "sparse_hnsw_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)

    cfgdir = os.path.join(args.outdir, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    legacy_srcs = {os.path.abspath(s) for s in legacy_srcs}
    for s in srcs:
        if s not in legacy_srcs:
            continue
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
    if legacy_srcs:
        print(f"legacy per-configuration CSVs moved to {cfgdir}/ "
              "(kept, not globbed)")
    if included_srcs:
        print("explicitly included CSVs left in place")


if __name__ == "__main__":
    main()
