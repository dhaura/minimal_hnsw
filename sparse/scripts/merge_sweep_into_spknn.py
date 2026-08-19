#!/usr/bin/env python3
"""Append sweep rows into a spknn-playground results CSV.

Usage:
  merge_sweep_into_spknn.py -target <results.csv> <sweep.csv> [<sweep.csv> ...]
                            [--pareto] [--dry-run]
"""

import argparse
import csv
import os
import sys

CANONICAL = [
    "Model",
    "Recall",
    "Indexing Time",
    "Single Query Time (microseconds)",
    "Searching Time (Seconds)",
    "QPS",
    "RR@10",
]


def read_sweep(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        # The sweep header names the RR column "RR@10"; tolerate either.
        rr = r.get("RR@10") or r.get("RR@10 (vs exact-NN gt)")
        out.append(
            {
                "Model": r["Model"],
                "Recall": r["Recall"],
                "Indexing Time": r["Indexing Time"],
                "Single Query Time (microseconds)": r["Single Query Time (microseconds)"],
                "Searching Time (Seconds)": r["Searching Time (Seconds)"],
                "QPS": r["QPS"],
                "RR@10": rr,
                # Diagnostics only; stripped before writing the canonical row.
                "_params": r.get("params", ""),
            }
        )
    return out


def pareto(rows):
    """Keep points not dominated in both recall and QPS.

    A point is dominated when some other point is at least as good on both
    axes and strictly better on one. Exact ties (same recall and same QPS,
    which happens when two knob settings turn out to describe the same
    configuration) collapse to the first one seen.
    """
    keep, seen = [], set()
    for r in rows:
        rec, qps = float(r["Recall"]), float(r["QPS"])
        if any(
            float(o["Recall"]) >= rec
            and float(o["QPS"]) >= qps
            and (float(o["Recall"]) > rec or float(o["QPS"]) > qps)
            for o in rows
        ):
            continue
        key = (round(rec, 12), round(qps, 6))
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-target", required=True, help="spknn results CSV to append to")
    ap.add_argument("sweeps", nargs="+", help="sweep CSVs written by the sweep drivers")
    ap.add_argument(
        "--pareto",
        action="store_true",
        help="keep only non-dominated (recall, QPS) points per model",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_rows = []
    for path in args.sweeps:
        rows = read_sweep(path)
        print(f"{path}: {len(rows)} rows ({', '.join(sorted({r['Model'] for r in rows}))})")
        new_rows.extend(rows)

    # Group by model, then order each model's curve by ascending recall.
    models, seen = [], set()
    for r in new_rows:
        if r["Model"] not in seen:
            seen.add(r["Model"])
            models.append(r["Model"])
    ordered = []
    for m in models:
        rows = [r for r in new_rows if r["Model"] == m]
        if args.pareto:
            kept = pareto(rows)
            print(f"  {m}: {len(rows)} points -> {len(kept)} on the Pareto frontier")
            rows = kept
        ordered.extend(sorted(rows, key=lambda r: float(r["Recall"])))

    for r in ordered:
        print(
            f"    recall={float(r['Recall']) * 100:6.2f}%  "
            f"QPS={float(r['QPS']):10.2f}  [{r['_params']}]"
        )

    existing_models = set()
    if os.path.exists(args.target):
        with open(args.target, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CANONICAL:
                sys.exit(
                    f"{args.target} has unexpected header:\n  {reader.fieldnames}\n"
                    f"expected:\n  {CANONICAL}"
                )
            existing_models = {r["Model"] for r in reader}

    clash = existing_models & set(models)
    if clash:
        print(
            f"WARNING: {args.target} already has rows for {sorted(clash)}; "
            "appending will duplicate that curve.",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"\n--dry-run: would append {len(ordered)} rows to {args.target}\n")
        w = csv.DictWriter(sys.stdout, fieldnames=CANONICAL, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
        return

    write_header = not os.path.exists(args.target) or os.path.getsize(args.target) == 0
    with open(args.target, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(ordered)
    print(f"\nAppended {len(ordered)} rows to {args.target}")


if __name__ == "__main__":
    main()
