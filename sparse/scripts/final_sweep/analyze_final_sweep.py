#!/usr/bin/env python3
"""Tables for the final method sweep (run_final_sweep_perlmutter.sh): how much faster each feature makes
the search than plain SparseHNSW, measured inside the same jobs so that node-to-node speed differences
cancel out.

Series: off (no gate, no early stop: the baseline), Gap (score-gap stop), inline16 (inline top-16 gate),
inline16+Gap. Rows come from seeded (8:4) and, in the lowrec runs, unseeded searches. Seeding is a
tuning knob like beta, so seeded and unseeded rows are kept as separate ef curves. Rows with a patience
stop or other gate layouts are skipped, so CSVs from other sweeps can be read too.
  1. QPS at matched recall: along each ef curve of a fixed (graph, beta, gap) configuration, QPS is
     interpolated linearly in log QPS at the target recall. A series' QPS is its best curve, and the
     tables give each series' QPS divided by off's.
  2. Recall and QPS ratio at the same ef on the headline graph (M32 efC200 alpha 0.85, beta 3, gap 0.1).

  module load python/3.11-24.1.0
  python analyze_final_sweep.py msmarco_full    # -> results/final_sweep/final_sweep_<ds>.{md,json}
  python analyze_final_sweep.py msmarco_full --dir <csv dir> --glob '<pattern>.csv' --out <prefix>
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

R = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
RES = Path(__file__).resolve().parent.parent / "results"
SERIES = ["off", "Gap", "inline16", "inline16+Gap"]
HEAD = dict(M=32, efC=200, alpha=0.85, beta=3, gap=0.1)


def parse(p):
    kv = dict(t.split("=", 1) for t in p.split())
    lay, gap = kv.get("layout", ""), kv.get("gap")
    ok = "pat" not in kv and lay in ("", "inline16")
    series = (lay + "+Gap" if lay else "Gap") if gap else (lay or "off")
    return pd.Series(dict(ok=ok, series=series, seeded="seedT" in kv,
                          M=int(kv["M"]), efC=int(kv["efC"]), alpha=float(kv["alpha"]), beta=int(kv["beta"]),
                          ef=int(kv["ef"]), gap=float(gap) if gap else -1.0))


def at_recall(c, r):
    """log-QPS interpolation along one ef curve (sorted by ef) at recall r; nan if never reached."""
    rec, q = c.Recall.to_list(), c.QPS.to_list()
    if rec[0] >= r:
        return q[0]
    for i in range(1, len(rec)):
        if rec[i - 1] < r <= rec[i]:
            t = (r - rec[i - 1]) / (rec[i] - rec[i - 1])
            return math.exp(math.log(q[i - 1]) + t * (math.log(q[i]) - math.log(q[i - 1])))
    return float("nan")


def series_qps(df, s, r):
    best = float("nan")
    for _, c in df[df.series == s].groupby(["JobID", "seeded", "M", "efC", "alpha", "beta", "gap"]):
        v = at_recall(c.sort_values("ef"), r)
        best = v if best != best or v > best else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--dir"), ap.add_argument("--glob"), ap.add_argument("--out")
    a = ap.parse_args()
    ds = a.dataset
    folder = Path(a.dir) if a.dir else RES / "final_sweep" / f"full_sweep_{ds}"
    df = pd.concat([pd.read_csv(f) for f in sorted(folder.glob(a.glob or f"final_{ds}_*.csv"))], ignore_index=True)
    df = pd.concat([df, df.Params.apply(parse)], axis=1)
    df = df[df.ok]
    q = {s: [series_qps(df, s, r) for r in R] for s in SERIES}
    f = lambda x, y: "—" if x != x or y != y else f"{x / y:.2f}"
    jobs = sorted(df.JobID.unique().tolist())
    md = [f"# Final method sweep, same-job comparison, {ds} ({len(df):,} points, jobs {', '.join(map(str, jobs))})", "",
          "QPS at matched recall: each fixed (graph, beta, gap) ef curve interpolated in log QPS; best curve per "
          "series. u8 slots; seeds 8:4 or off (lowrec jobs). Ratios against off (no gate, no stop) from the same jobs.", "",
          "| recall | off QPS | Gap | inline16 | inline16+Gap | i16+Gap / Gap |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(R):
        o = q["off"][i]
        md.append(f"| {r} | {'—' if o != o else f'{o:,.0f}'} | " + " | ".join(f(q[s][i], o) for s in SERIES[1:])
                  + f" | {f(q['inline16+Gap'][i], q['Gap'][i])} |")
    h = df[(df.M == HEAD["M"]) & (df.efC == HEAD["efC"]) & (df.alpha == HEAD["alpha"]) & (df.beta == HEAD["beta"])]
    h = h[h.gap.isin([-1.0, HEAD["gap"]]) & h.seeded]
    per_ef = {}
    if len(h):
        h = h[h.JobID == h.groupby("JobID").ef.max().idxmax()]  # the job with the widest ef list
        md += ["", f"Same ef, M{HEAD['M']} efC{HEAD['efC']} alpha {HEAD['alpha']} beta {HEAD['beta']}, gap "
               f"{HEAD['gap']} (job {int(h.JobID.iloc[0])}): recall, and QPS / off QPS at that ef.", "",
               "| ef | off recall | off QPS | " + " | ".join(SERIES[1:]) + " |", "|---|---|---|---|---|---|"]
        for ef, g in h.groupby("ef"):
            row = {s: g[g.series == s].iloc[0] for s in SERIES if (g.series == s).any()}
            if "off" not in row:
                continue
            o = row["off"]
            per_ef[int(ef)] = {s: [float(v.Recall), float(v.QPS)] for s, v in row.items()}
            md.append(f"| {ef} | {o.Recall:.4f} | {o.QPS:,.0f} | " + " | ".join(
                f"{row[s].Recall:.4f}, {row[s].QPS / o.QPS:.2f}x" if s in row else "—" for s in SERIES[1:]) + " |")
    out = Path(a.out) if a.out else RES / "final_sweep" / f"final_sweep_{ds}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"dataset": ds, "points": len(df), "jobs": jobs, "recall": R, "qps": q, "head": HEAD, "per_ef": per_ef}, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
