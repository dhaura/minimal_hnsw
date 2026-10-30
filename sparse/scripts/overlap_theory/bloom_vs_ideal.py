#!/usr/bin/env python3
"""Real Bloom filter vs the ideal overlap filter at every T, against the break-even 1 / C.

Reads results/overlap_theory/bloom_skip_<ds>/*.gate.csv (run_bloom_skip_perlmutter.sh):
one row per (T, ef) with the Bloom gate on at T. Within a row, the Bloom and the ideal
filter judge the same checked neighbors at the same T: alpha_bloom = gate_rej / checked,
alpha_ideal = ideal_rej / checked, useful lost = useful_*_rej / useful, false passes =
Bloom passes that the ideal filter skips. C_mean comes from ideal_alpha_<ds>.json (run
ideal_alpha.py first). Writes results/overlap_theory/bloom_vs_ideal_<ds>.md and .json.

  python3 bloom_vs_ideal.py <dataset>
"""
import json
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "overlap_theory"


def main(ds):
    meta = json.loads((RES / f"ideal_alpha_{ds}.json").read_text())
    a_star = meta["alpha_star"]
    g = pd.concat([pd.read_csv(f) for f in sorted((RES / f"bloom_skip_{ds}").glob("*.gate.csv"))], ignore_index=True)
    g["ef"] = g.Params.str.extract(r" ef=(\d+)")[0].astype(int)
    g["T"] = g.Params.str.extract(r" T=([\d.]+)")[0].astype(float).astype(int)
    g = g.groupby(["T", "ef"], as_index=False).sum(numeric_only=True)
    rows = [{"T": int(r.T), "ef": int(r.ef), "checked": int(r.checked),
             "a_bloom": r.gate_rej / r.checked, "a_ideal": r.ideal_rej / r.checked,
             "u_bloom": r.useful_gate_rej / r.useful, "u_ideal": r.useful_ideal_rej / r.useful,
             "false_pass": r.false_pass / r.checked, "false_rej": r.false_rej / r.checked}
            for r in g.sort_values(["T", "ef"]).itertuples()]
    (RES / f"bloom_vs_ideal_{ds}.json").write_text(json.dumps(
        {"dataset": ds, "alpha_star": a_star, "C_mean": meta["C_mean"], "rows": rows}, indent=1))
    pc = lambda x: f"{100 * x:.1f}%"
    efs = sorted({r["ef"] for r in rows})
    L = [f"# Bloom vs ideal filter at each T, {ds}", "",
         f"Break-even alpha* = 1/C = {pc(a_star)} (C mean {meta['C_mean']:.2f}). Bloom gate on at T; both filters "
         "judge the same checked neighbors. Cells: alpha Bloom / alpha ideal (bytes vs no filter for Bloom, 1 - alpha + 1/C).", "",
         "| T | " + " | ".join(f"ef={e}" for e in efs) + " |", "|---" * (len(efs) + 1) + "|"]
    for t in sorted({r["T"] for r in rows}):
        cells = []
        for e in efs:
            r = next(x for x in rows if x["T"] == t and x["ef"] == e)
            cells.append(f"{pc(r['a_bloom'])} / {pc(r['a_ideal'])} ({1 - r['a_bloom'] + a_star:.2f}x)")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## Useful neighbors lost (Bloom / ideal) and Bloom false passes (share of checked)", "",
          "| T | " + " | ".join(f"ef={e}" for e in efs) + " |", "|---" * (len(efs) + 1) + "|"]
    for t in sorted({r["T"] for r in rows}):
        cells = []
        for e in efs:
            r = next(x for x in rows if x["T"] == t and x["ef"] == e)
            cells.append(f"{pc(r['u_bloom'])} / {pc(r['u_ideal'])}, fp {pc(r['false_pass'])}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    bad = [r for r in rows if r["false_rej"] > 0]
    L += ["", f"False rejects (Bloom skips a neighbor the ideal filter keeps): {'none' if not bad else f'{len(bad)} rows'}."]
    (RES / f"bloom_vs_ideal_{ds}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
