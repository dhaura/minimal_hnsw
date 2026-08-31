#!/usr/bin/env python3
"""Iso-recall speedup table: our QPS vs every baseline, at fixed recall steps.
"""
import argparse, csv, math, os, re, sys

def front(pts):
    best = {}
    for r, q in pts:
        if r not in best or q > best[r]:
            best[r] = q
    out, b = [], -1.0
    for r in sorted(best, reverse=True):
        if best[r] > b:
            out.append((r, best[r])); b = best[r]
    return sorted(out)

def qat(c, t):
    if not c or t < c[0][0] or t > c[-1][0]:
        return None
    for a, b in zip(c, c[1:]):
        if a[0] <= t <= b[0]:
            if b[0] == a[0]:
                return max(a[1], b[1])
            f = (t - a[0]) / (b[0] - a[0])
            return math.exp(math.log(a[1]) + f * (math.log(b[1]) - math.log(a[1])))
    return c[-1][1]

def load_ours(paths):
    pts = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            pts.append((float(r["Recall"]), float(r["QPS"])))
    return front(pts)

def load_baselines(pareto_csv, skip):
    by = {}
    for r in csv.DictReader(open(pareto_csv)):
        m = r["Model"].strip()
        if m.upper().startswith(skip.upper()):
            continue
        by.setdefault(m.split("_")[0], []).append((float(r["Recall"]), float(r["QPS"])))
    return {k: front(v) for k, v in by.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lo", type=float, default=0.80)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args()

    ours = load_ours(args.ours)
    base = load_baselines(args.pareto, "SparseHNSW")
    names = sorted(base, key=lambda n: -(qat(base[n], 0.95) or 0))

    rows, t = [], args.lo
    while t <= args.hi + 1e-9:
        o = qat(ours, t)
        if o:
            row = {"recall": round(t, 4), "ours_qps": round(o)}
            for n in names:
                v = qat(base[n], t)
                row[f"{n}_qps"] = round(v) if v else ""
                row[f"vs_{n}"] = round(o / v, 2) if v else ""
            rows.append(row)
        t += args.step

    hdr = f"{'recall':>7} {'ours QPS':>10} " + " ".join(f"{n[:9]:>10}" for n in names)
    print(f"\n===== {args.label} =====  ours covers "
          f"{ours[0][0]*100:.1f}-{ours[-1][0]*100:.1f}%")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = " ".join(f"{r['vs_'+n]:>9}x" if r["vs_" + n] != "" else f"{'-':>10}"
                         for n in names)
        print(f"{r['recall']*100:6.1f}% {r['ours_qps']:10,} {cells}")

    if args.out and rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
