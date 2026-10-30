#!/usr/bin/env python3
"""Inline top-K gate, data movement from the instrumented runs (run_sketch_outcome_perlmutter.sh).

Reads results/sketch_theory/outcome_inline_<ds>/*.outcome.csv by column name: per layout and
ef, recall, distance calls, row lines, gate checks and skips, layer-0 expansions (run totals).
Traffic = row lines + one line per skip: the test reads line 0 of every checked row, and a row
that passes counts line 0 in its own row lines when it is scored. Gate off = SparseHNSW. At
matched recall, log-traffic is interpolated over recall along each ef curve (as inline_pilot.py
does for QPS).

  module load python/3.11-24.1.0
  python inline_outcome.py msmarco_full   # -> results/sketch_theory/inline_outcome_<ds>.{md,json}
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results" / "sketch_theory"
TARGETS = (0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99)
COLS = ("queries", "recall", "calls", "lines", "checks", "skips", "expansions")


def load(ds):
    files = sorted((RES / f"outcome_inline_{ds}").glob("*.outcome.csv"))
    if not files:
        sys.exit(f"no outcome CSV for {ds}")
    series = {}
    for row in csv.DictReader(open(files[-1])):
        v = {c: float(row[c]) for c in COLS}
        q, e = v["queries"], v["expansions"]
        series.setdefault(row["Model"].replace("SparseHNSW_", "").replace("SparseHNSW", "off"), []).append({
            "ef": int(re.search(r"ef=(\d+)", row["Params"]).group(1)), "recall": v["recall"],
            "calls": v["calls"] / q, "row_lines": v["lines"] / q, "traffic": (v["lines"] + v["skips"]) / q,
            "skip_share": v["skips"] / v["checks"] if v["checks"] else 0.0,
            "checks_per_exp": v["checks"] / e, "lines_per_exp": v["lines"] / e})
    return files[-1].name, {m: sorted(p, key=lambda p: p["ef"]) for m, p in series.items()}


def at(points, target, key="traffic"):
    """log-<key> at a recall target along the ef curve; None outside its range."""
    rec = np.maximum.accumulate([p["recall"] for p in points])
    if not rec[0] <= target <= rec[-1]:
        return None
    return float(np.exp(np.interp(target, rec, np.log([p[key] for p in points]))))


def main(ds):
    name, series = load(ds)
    off = series["off"]
    gated = [m for m in series if m != "off"]
    for m in gated:
        for p, b in zip(series[m], off):
            p["traffic_vs_off_same_ef"] = p["traffic"] / b["traffic"]
            p["traffic_vs_off_matched"] = (p["traffic"] / o) if (o := at(off, p["recall"])) else None
    matched = {m: {t: (g / b if (g := at(series[m], t)) and (b := at(off, t)) else None) for t in TARGETS}
               for m in gated}
    out = {"dataset": ds, "csv": name, "series": series, "matched_traffic_vs_off": matched}
    (RES / f"inline_outcome_{ds}.json").write_text(json.dumps(out, indent=1))
    cell = lambda x: "—" if x is None else f"{x:.2f}x"
    md = [f"# Inline top-K gate, data movement, {ds} ({name})", "",
          "Traffic = row lines + one line per skip, per query. Lines vs gate off at matched recall:", "",
          "| recall | gate-off lines/q | " + " | ".join(gated) + " |", "|---|---|" + "---|" * len(gated)]
    for t in TARGETS:
        b = at(off, t)
        md.append(f"| {t:.3f} | {'—' if b is None else f'{b:,.0f}'} | " + " | ".join(cell(matched[m][t]) for m in gated) + " |")
    md += ["", "Per ef: gate-off recall, row lines per expansion; per layout recall, skip share, checks per expansion,"
           " traffic vs gate off at the same ef.", "",
           "| ef | off recall | off lines/exp | " + " | ".join(gated) + " |", "|---|---|---|" + "---|" * len(gated)]
    for i, b in enumerate(off):
        md.append(f"| {b['ef']} | {b['recall']:.4f} | {b['lines_per_exp']:.0f} | " + " | ".join(
            f"{p['recall']:.4f}, {p['skip_share']:.1%}, {p['checks_per_exp']:.1f}, {p['traffic_vs_off_same_ef']:.2f}x"
            for p in (series[m][i] for m in gated)) + " |")
    (RES / f"inline_outcome_{ds}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
