#!/usr/bin/env python3
"""Same-job comparison for the inline top-K full sweep (run_inline_sweep_perlmutter.sh).

Best QPS at recall >= r for each series of the sweep: off (no gate, patience list, the
same-job baseline), Gap, inline16, inline16+Gap, inline24, inline24+Gap, each over every
graph, beta, seed and stop setting of its jobs. Ratios are against same-job off, plus
inline+Gap against Gap (does the gate add to the stop rule). For reference, the per-edge
Top16 / Top16+Gap ratios against their own same-job rerun (results/gap_top16, older jobs
and binary), and this sweep's off against that rerun (drift between the two sets of jobs).

  module load python/3.11-24.1.0
  python compare_inline_sweep.py msmarco_full   # -> results/sketch_theory/inline_sweep_<ds>.{md,json}
"""
import json
import sys
from pathlib import Path

import pandas as pd

R = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
RES = Path(__file__).resolve().parent.parent / "results"
SERIES = ["off", "Gap", "inline16", "inline16+Gap", "inline24", "inline24+Gap"]


def label(p):
    lay = p.split(" layout=")[1].split()[0] if " layout=" in p else ""
    gap = " gap=" in p
    if lay.startswith("list"):
        return "Top16" + ("+Gap" if gap else "")
    return (lay + ("+Gap" if gap else "") if lay else ("Gap" if gap else "off"))


def best(d, r):
    d = d[d.Recall >= r]
    return float(d.QPS.max()) if len(d) else float("nan")


def load(folder, pattern):
    df = pd.concat([pd.read_csv(f) for f in sorted(folder.glob(pattern))], ignore_index=True)
    df["series"] = df.Params.map(label)
    return df


def main(ds):
    df = load(RES / "sketch_theory" / f"full_sweep_{ds}", f"inline_{ds}_*.csv")
    ref = load(RES / "gap_top16" / f"full_sweep_{ds}", f"gaptop16_{ds}_*.csv")
    q = {s: [best(df[df.series == s], r) for r in R] for s in SERIES}
    t = {s: [best(ref[ref.series == s], r) for r in R] for s in ("off", "Top16", "Top16+Gap")}
    out = {"dataset": ds, "points": len(df), "recall": R, "best_qps": q, "gap_top16_best_qps": t,
           "jobs": sorted(df.JobID.unique().tolist())}
    (RES / "sketch_theory" / f"inline_sweep_{ds}.json").write_text(json.dumps(out, indent=1))
    f = lambda a, b: "—" if a != a or b != b else f"{a / b:.2f}"
    md = [f"# Inline top-K full sweep, same-job comparison, {ds} ({len(df):,} points, jobs "
          + ", ".join(map(str, out["jobs"])) + ")", "",
          "Best QPS at recall >= r. off = no gate, patience list; ratios are against off from the same jobs.",
          "Top16 = per-edge sketch (removed), against its own same-job rerun in results/gap_top16.", "",
          "| recall | off QPS | Gap | inline16 | inline16+Gap | inline24 | inline24+Gap | i16+Gap / Gap | i24+Gap / Gap "
          "| Top16 (ref) | Top16+Gap (ref) | off / ref rerun |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(R):
        o = q["off"][i]
        md.append(f"| {r} | {'—' if o != o else f'{o:,.0f}'} | " + " | ".join(f(q[s][i], o) for s in SERIES[1:]) + " | "
                  + f"{f(q['inline16+Gap'][i], q['Gap'][i])} | {f(q['inline24+Gap'][i], q['Gap'][i])} | "
                  + f"{f(t['Top16'][i], t['off'][i])} | {f(t['Top16+Gap'][i], t['off'][i])} | {f(o, t['off'][i])} |")
    (RES / "sketch_theory" / f"inline_sweep_{ds}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
