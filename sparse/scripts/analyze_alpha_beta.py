#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def frontier(points):
    """(recall, qps) points that are not dominated by another point."""
    pts = sorted(points)
    out, best = [], -1.0
    for rec, qps in reversed(pts):          # high recall -> low
        if qps > best:
            out.append((rec, qps))
            best = qps
    return sorted(out)


def qps_at(curve, target):
    """Interpolate QPS at `target` recall. None if the curve does not reach it.
    """
    if not curve or target < curve[0][0] or target > curve[-1][0]:
        return None
    for (r0, q0), (r1, q1) in zip(curve, curve[1:]):
        if r0 <= target <= r1:
            if r1 == r0:
                return max(q0, q1)
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return curve[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="+",
                    help="one or more sweep CSVs (main + any extension)")
    ap.add_argument("--targets", default="0.90,0.95,0.98",
                    help="recall targets to compare at")
    ap.add_argument("--baseline", default="0.8/3",
                    help="alpha/beta currently in use, for the verdict")
    args = ap.parse_args()

    targets = [float(t) for t in args.targets.split(",") if t]

    rows = []
    for path in args.csv_path:
        rows += list(csv.DictReader(open(path)))
    by_cfg = defaultdict(list)
    index_sec, peak_gb = {}, {}
    for r in rows:
        p = r["Params"]
        alpha = p.split("alpha=")[1].split()[0]
        beta = p.split("beta=")[1].split()[0]
        cfg = f"{alpha}/{beta}"
        by_cfg[cfg].append((float(r["Recall"]), float(r["QPS"])))
        index_sec[cfg] = float(r["IndexSec"])
        peak_gb[cfg] = float(r["PeakRSSGB"])

    curves = {c: frontier(pts) for c, pts in by_cfg.items()}

    def sort_key(c):
        a, b = c.split("/")
        return (float(a), int(b))

    cfgs = sorted(curves, key=sort_key)

    hdr = f"{'alpha/beta':<12}{'maxRec':>8}{'idx_s':>8}{'RSS_GB':>8}"
    for t in targets:
        hdr += f"{'QPS@'+format(t,'.2f'):>12}"
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for c in cfgs:
        cur = curves[c]
        line = f"{c:<12}{cur[-1][0]:8.4f}{index_sec[c]:8.1f}{peak_gb[c]:8.1f}"
        table[c] = {}
        for t in targets:
            q = qps_at(cur, t)
            table[c][t] = q
            line += f"{q:12.0f}" if q else f"{'-':>12}"
        print(line)

    print()
    best_at = {}
    for t in targets:
        ranked = sorted(((d[t], c) for c, d in table.items()
                         if d[t] is not None), reverse=True)
        if not ranked:
            print(f"recall {t:.2f}: no configuration reaches this recall")
            continue
        best_at[t] = ranked[0][1]
        podium = "  ".join(f"{c}={q:,.0f}" for q, c in ranked[:3])
        print(f"recall {t:.2f}: best {ranked[0][1]:<8} | top3: {podium}")

    base = args.baseline
    print()
    if base not in table:
        print(f"baseline {base} not present in the sweep")
        return
    print(f"=== verdict vs baseline alpha/beta = {base} ===")
    verdict_rows = []
    for t in targets:
        b = table[base][t]
        w = best_at.get(t)
        if b is None or w is None:
            print(f"  recall {t:.2f}: baseline does not reach this recall")
            continue
        wq = table[w][t]
        gain = (wq / b - 1) * 100
        verdict_rows.append((t, w, gain))
        flag = "" if w == base else f"   <-- {w} is {gain:+.1f}% faster"
        print(f"  recall {t:.2f}: baseline {b:,.0f} QPS, best {w} {wq:,.0f} QPS{flag}")

    winners = {w for _, w, g in verdict_rows if g > 3.0}
    if not winners:
        print(f"\nCONCLUSION: keep alpha/beta = {base} "
              "(nothing beats it by >3% at any target).")
    else:
        print(f"\nCONCLUSION: {base} is NOT optimal. "
              f"Better at one or more targets: {', '.join(sorted(winners))}.")
        print("Rerun the ef sweep with the winner before using it in the "
              "cross-method benchmark.")


if __name__ == "__main__":
    main()
