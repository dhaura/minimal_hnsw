#!/usr/bin/env python3
"""Export the slotted-layout sweep as SparseHNSW inputs for the spknn-playground figures.

Reads the full sweep (results/quant_slots/step2_full_sweep/layout_sweep_*.csv),
keeps only the slotted rows ("..._slots" models), renames them "SparseHNSW",
and replaces PeakRSSGB with the slots-only peak measured for the same M
(results/quant_slots/step3_peak_memory/*.csv). In the sweep both layouts lived
in one process, so its own PeakRSSGB overstates a slotted deployment.

Writes one sparse_hnsw_results_slots<slot_size>_M<M>_efc<E>.csv per graph into
<out_dir> (the playground's SparseHNSW data folder), where
combine_hnsw_envelope.py picks them up and folds them into the envelope.

Usage:
  python3 prepare_playground_slots.py <out_dir>
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "quant_slots")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = sys.argv[1]

    peak = {}  # M -> slots-only peak RSS (GB)
    for f in glob.glob(os.path.join(RES, "step3_peak_memory", "*.csv")):
        for r in csv.DictReader(open(f)):
            m = re.match(r"SparseHNSW_M(\d+)_slot-(\d+)", r["Model"])
            if m:
                peak[int(m.group(1))] = r["PeakRSSGB"]
    if not peak:
        sys.exit("no slots-only peak memory rows found")

    files = sorted(glob.glob(os.path.join(RES, "step2_full_sweep", "layout_sweep_*.csv")))
    for f in files:
        rows = list(csv.DictReader(open(f)))
        fields = list(rows[0].keys())
        slotted = [r for r in rows if r["Model"].endswith("_slots")]
        M = int(re.search(r"M=(\d+)", slotted[0]["Params"]).group(1))
        efc = int(re.search(r"efC=(\d+)", slotted[0]["Params"]).group(1))
        slot_size = re.search(r"slots=(\d+)", slotted[0]["Params"]).group(1)
        if M not in peak:
            sys.exit(f"no slots-only peak memory for M={M}")
        for r in slotted:
            r["Model"] = "SparseHNSW"
            r["PeakRSSGB"] = peak[M]
        out = os.path.join(out_dir, f"sparse_hnsw_results_slots{slot_size}_M{M}_efc{efc}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(slotted)
        print(f"  {os.path.basename(f)} -> {os.path.basename(out)}  ({len(slotted)} slotted points, PeakRSSGB={peak[M]})")


if __name__ == "__main__":
    main()
