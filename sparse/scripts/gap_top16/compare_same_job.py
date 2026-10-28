#!/usr/bin/env python3
"""Same-job check for the gap / top-16 full sweep.

Best QPS at recall >= r for each series of the sweep (no filter = the same-job
SparseHNSW rerun, +Gap, +Top16, +Top16+Gap) and for the existing playground
SparseHNSW front, so the variants can be compared with a baseline from the same
jobs and binary, and the rerun with the existing front (drift). Writes
results/gap_top16/same_job_<dataset>.md.

  python3 compare_same_job.py <dataset>
"""
import os
import sys
from pathlib import Path

import pandas as pd

R = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
HERE = Path(__file__).resolve().parent


def series(p):
    return ("Top16" if " layout=list16" in p else "") + ("Gap" if " gap=" in p else "") or "rerun"


def best(d, r):
    d = d[d.Recall >= r]
    return d.QPS.max() if len(d) else float("nan")


def main(ds):
    sweep = HERE.parent / "results" / "gap_top16" / f"full_sweep_{ds}"
    df = pd.concat([pd.read_csv(f) for f in sorted(sweep.glob(f"gaptop16_{ds}_*.csv"))], ignore_index=True)
    df["series"] = df.Params.map(series)
    old = pd.read_csv(Path(os.environ["SCRATCH"]) / "datasets/SpKNN" / ds / "sparse_hnsw/sparse_hnsw_results.csv")
    cols = ["rerun", "Gap", "Top16", "Top16Gap"]
    lines = [f"# Same-job comparison, {ds} ({len(df):,} points)", "",
             "Best QPS at recall >= r. rerun = no filter, patience list, same jobs as the variants;",
             "existing = the playground SparseHNSW front (older build, other jobs).", "",
             "| recall | existing | rerun | rerun / existing | Gap / rerun | Top16 / rerun | Top16+Gap / rerun |",
             "|---|---|---|---|---|---|---|"]
    for r in R:
        q = {c: best(df[df.series == c], r) for c in cols}
        e = best(old, r)
        lines.append(f"| {r} | {e:,.0f} | {q['rerun']:,.0f} | {q['rerun'] / e:.2f} | " +
                     " | ".join(f"{q[c] / q['rerun']:.2f}" for c in cols[1:]) + " |")
    out = HERE.parent / "results" / "gap_top16" / f"same_job_{ds}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
