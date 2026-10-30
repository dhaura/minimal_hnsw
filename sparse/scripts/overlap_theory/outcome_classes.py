#!/usr/bin/env python3
"""Ideal gate inside the search: what becomes of the checked neighbors, and row bytes at
matched recall (overlap-check theory, step 4c).

Reads results/overlap_theory/outcome_<ds>/*.outcome.csv (run_outcome_perlmutter.sh). Gate off:
each checked layer-0 neighbor's class (not added, added but never expanded, expanded, returned in
the top k*beta) and the share of each class with overlap <= T% of the query terms. Bytes per
query = 64 x (row lines of every traversal distance call + one filter line per gate check);
"free" drops the filter lines. Lossless oracle at the gate-off points: skips every not-added
checked neighbor (rows at the mean lines per call) and pays one line per checked neighbor; its
recall is the gate-off recall. Matched recall: interpolate log bytes over recall on each curve.
Writes results/overlap_theory/outcome_<ds>.md and .json.

  python3 outcome_classes.py <dataset>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "overlap_theory"
TS = [0, 2, 5, 10, 15, 20, 25, 30]
CLS = ["notadded", "unexpanded", "expanded", "returned"]
TARGETS = [0.93, 0.95, 0.97, 0.98]


def at(curve, target, col):
    """Bytes of a curve at a recall target, or None outside its recall range."""
    c = curve.sort_values("recall")
    if not c.recall.min() <= target <= c.recall.max():
        return None
    return float(np.exp(np.interp(target, c.recall, np.log(c[col]))))


def main(ds):
    d = pd.concat([pd.read_csv(f) for f in sorted((RES / f"outcome_{ds}").glob("*.outcome.csv"))], ignore_index=True)
    d["ef"] = d.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    d["T"] = d.Params.str.extract(r"idealT=(\d+)")[0].fillna(-1).astype(int)
    d["kib"] = 64 * (d.lines + d.checks) / d.queries / 1024
    d["kib_free"] = 64 * d.lines / d.queries / 1024
    off = d[d["T"] == -1].sort_values("ef").copy()
    checked = off[CLS].sum(axis=1)
    off["oracle_kib"] = 64 * (off.lines - off.notadded * off.lines / off.calls + checked) / off.queries / 1024
    out = {"dataset": ds, "T": TS, "targets": TARGETS, "off": [], "gated": {}, "matched": {}}
    for r, n in zip(off.itertuples(), checked):
        out["off"].append({"ef": int(r.ef), "recall": float(r.recall), "kib": float(r.kib), "oracle_kib": float(r.oracle_kib),
                           "cls": [getattr(r, c) / n for c in CLS],
                           "ov": [[getattr(r, f"{c}_T{t}") / max(1, getattr(r, c)) for t in TS] for c in CLS]})
    for t in TS:
        g = d[d["T"] == t].sort_values("ef")
        out["gated"][t] = [{"ef": int(r.ef), "recall": float(r.recall), "kib": float(r.kib), "kib_free": float(r.kib_free),
                            "skip": r.skips / max(1, r.checks)} for r in g.itertuples()]
        out["matched"][t] = [[None if at(g, x, "kib") is None or at(off, x, "kib") is None else at(g, x, "kib") / at(off, x, "kib"),
                              None if at(g, x, "kib_free") is None or at(off, x, "kib") is None else at(g, x, "kib_free") / at(off, x, "kib")]
                             for x in TARGETS]
    out["oracle_ratio"] = [float(v) for v in off.oracle_kib / off.kib]
    (RES / f"outcome_{ds}.json").write_text(json.dumps(out, indent=1))
    pc, fx = (lambda x: f"{100 * x:.1f}%"), (lambda x: "-" if x is None else f"{x:.2f}x")
    L = [f"# Ideal gate in the search, {ds}", "",
         f"Lossless oracle at the same recall: {min(out['oracle_ratio']):.2f}-{max(out['oracle_ratio']):.2f}x the bytes of gate off.", "",
         "## Checked neighbors by class, gate off (share of checked; share of the class with overlap <= 10%)", "",
         "| ef | recall | KiB/query | " + " | ".join(CLS) + " |", "|---" * (len(CLS) + 3) + "|"]
    for o in out["off"]:
        L.append(f"| {o['ef']} | {pc(o['recall'])} | {o['kib']:.0f} | " +
                 " | ".join(f"{pc(s)} ({pc(v[TS.index(10)])})" for s, v in zip(o["cls"], o["ov"])) + " |")
    L += ["", "## Bytes vs gate off at matched recall: with one filter line per check / filter free", "",
          "| T | " + " | ".join(f"recall {x}" for x in TARGETS) + " |", "|---" * (len(TARGETS) + 1) + "|"]
    for t in TS:
        L.append(f"| {t} | " + " | ".join(f"{fx(a)} / {fx(b)}" for a, b in out["matched"][t]) + " |")
    (RES / f"outcome_{ds}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
