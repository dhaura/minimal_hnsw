#!/usr/bin/env python3
"""Top-K sketch layout pilot: QPS of each gated series against the same-job gate-off
curve at matched recall.

Reads results/sketch_theory/inline_<preset>_<ds>/*.csv: SparseHNSW (no gate),
SparseHNSW_list16 (per-edge sketch, removed from the engine 2026-10-29) and
SparseHNSW_inline<K>. Presets: pilot (rows heaviest first), diag (row order alone, extra
prefetch; variants since removed) and confirm (rows top-K first, the kept layout). The
one-off runner is deleted; its sweep was M=32 efC=200 alpha 0.85 beta 3 seeds 8:4, stop
rules off, ef 20-1600 (build-inline/bin/sparse_hnsw_sweep):
  32 200 EF 1 0 0 0.85 3 BASE Q GT CSV SparseHNSW 3 1 1 8 8:4 0 -SLOT off 1 LAYOUTS
QPS is interpolated in log space over recall along each series' ef curve. Two views:
  - at fixed recall targets: QPS(series) / QPS(gate off), where both curves reach it;
  - per gated point: its QPS over the gate-off QPS at the same recall.
Decision rule (user, 2026-10-29): inline is kept if it beats gate off at matched recall.

  module load python/3.11-24.1.0
  python inline_pilot.py msmarco_full [diag]   # -> results/sketch_theory/inline_<preset>_<ds>.{md,json}
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results" / "sketch_theory"
TARGETS = (0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99)


def load(ds, preset):
    files = sorted((RES / f"inline_{preset}_{ds}").glob("*.csv"))
    if not files:
        sys.exit(f"no pilot CSV for {ds}")
    series = {}
    for row in csv.DictReader(open(files[-1])):
        ef = int(re.search(r"ef=(\d+)", row["Params"]).group(1))
        series.setdefault(row["Model"], []).append((ef, float(row["Recall"]), float(row["QPS"])))
    return files[-1].name, {m: sorted(p) for m, p in series.items()}


def at(points, target):
    """log-QPS at a recall target along the ef curve; None outside its range."""
    rec = np.maximum.accumulate([p[1] for p in points])
    if not rec[0] <= target <= rec[-1]:
        return None
    return float(np.exp(np.interp(target, rec, np.log([p[2] for p in points]))))


def main(ds, preset="pilot"):
    name, series = load(ds, preset)
    base = series.pop("SparseHNSW")
    out = {"dataset": ds, "csv": name, "targets": {}, "points": {}}
    for m, pts in series.items():
        out["targets"][m] = {t: (q / b if (q := at(pts, t)) and (b := at(base, t)) else None) for t in TARGETS}
        out["points"][m] = [{"ef": ef, "recall": r, "qps": q, "base_recall": base[i][1], "base_qps": base[i][2],
                             "speedup": (q / b) if (b := at(base, r)) else None} for i, (ef, r, q) in enumerate(pts)]
    out["base"] = [{"ef": ef, "recall": r, "qps": q} for ef, r, q in base]
    (RES / f"inline_{preset}_{ds}.json").write_text(json.dumps(out, indent=1))
    names = list(series)
    cell = lambda x: "—" if x is None else f"{x:.2f}x"
    md = [f"# Sketch layout pilot, {ds} ({name})", "", "QPS / gate-off QPS at matched recall.", "",
          "| recall | gate-off QPS | " + " | ".join(n.replace("SparseHNSW_", "") for n in names) + " |",
          "|---|---|" + "---|" * len(names)]
    for t in TARGETS:
        b = at(base, t)
        md.append(f"| {t:.3f} | {'—' if b is None else f'{b:,.0f}'} | " + " | ".join(cell(out["targets"][n][t]) for n in names) + " |")
    md += ["", "Per ef: recall (gate off -> gated) and QPS / gate-off QPS at the gated recall.", "",
           "| ef | gate-off recall | " + " | ".join(n.replace("SparseHNSW_", "") for n in names) + " |",
           "|---|---|" + "---|" * len(names)]
    for i, (ef, r, _) in enumerate(base):
        md.append(f"| {ef} | {r:.4f} | " + " | ".join(
            f"{p['recall']:.4f}, {cell(p['speedup'])}" for p in (out["points"][n][i] for n in names)) + " |")
    (RES / f"inline_{preset}_{ds}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:3])
