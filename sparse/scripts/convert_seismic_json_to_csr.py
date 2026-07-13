#!/usr/bin/env python3
"""Convert Seismic-style sparse-vector JSON (nq_splade) to big-ann CSR binary.

Input JSON layout (one object per file):
    {"vectors": [{"id": "...", "coordinates": [...], "values": [...]}, ...]}

Output layout (what csr_matrix.h's CSRMatrix(path, ori=true) reads, same as
the MSMARCO big-ann files):
    int64 nrow, int64 ncol, int64 nnz,
    int64 indptr[nrow+1], int32 indices[nnz], float32 data[nnz]

Rows are emitted in file order (files processed in sorted name order:
xaa.json, xab.json, ...), so row id == insertion order in the demo drivers.
Compute the ground truth from the converted .csr files (compute_gt.py) so ids
are consistent with that ordering.

Usage (needs `module load python` on Perlmutter):
    python3 convert_seismic_json_to_csr.py --json queries.test.json --out queries.test.csr
    python3 convert_seismic_json_to_csr.py --json 'documents/*.json' --out base_nq.csr
"""
import argparse
import glob
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

# SPLADE uses the BERT WordPiece vocabulary; keep ncol identical across the
# base and query files regardless of the max coordinate actually present.
BERT_VOCAB_SIZE = 30522


def convert(json_files, out_path, ncol):
    tmp_idx = out_path + ".indices.tmp"
    tmp_val = out_path + ".values.tmp"
    nnz_counts = []
    nrow = 0
    max_coord = -1
    with open(tmp_idx, "wb") as fi, open(tmp_val, "wb") as fv:
        for path in json_files:
            t0 = time.time()
            with open(path, "rb") as f:
                vectors = json.load(f)["vectors"]
            idx_arrs = []
            val_arrs = []
            for v in vectors:
                c = np.asarray(v["coordinates"], dtype=np.int32)
                d = np.asarray(v["values"], dtype=np.float32)
                if c.size != d.size:
                    sys.exit(f"{path}: id={v['id']} coordinates/values length mismatch")
                # The merge-based distance kernel needs indices sorted per row.
                if c.size > 1 and np.any(np.diff(c) < 0):
                    order = np.argsort(c, kind="stable")
                    c, d = c[order], d[order]
                nnz_counts.append(c.size)
                idx_arrs.append(c)
                val_arrs.append(d)
            all_idx = np.concatenate(idx_arrs) if idx_arrs else np.empty(0, np.int32)
            if all_idx.size:
                max_coord = max(max_coord, int(all_idx.max()))
            all_idx.tofile(fi)
            (np.concatenate(val_arrs) if val_arrs else np.empty(0, np.float32)).tofile(fv)
            nrow += len(vectors)
            print(f"  {os.path.basename(path)}: {len(vectors)} vectors "
                  f"({time.time() - t0:.1f}s, running total {nrow})", flush=True)

    if max_coord >= ncol:
        sys.exit(f"max coordinate {max_coord} >= ncol {ncol}; pass a larger --ncol")

    indptr = np.zeros(nrow + 1, dtype=np.int64)
    np.cumsum(np.asarray(nnz_counts, dtype=np.int64), out=indptr[1:])
    nnz = int(indptr[-1])

    with open(out_path, "wb") as out:
        out.write(struct.pack("<3q", nrow, ncol, nnz))
        indptr.tofile(out)
        with open(tmp_idx, "rb") as fi:
            shutil.copyfileobj(fi, out, 1 << 24)
        with open(tmp_val, "rb") as fv:
            shutil.copyfileobj(fv, out, 1 << 24)
    os.remove(tmp_idx)
    os.remove(tmp_val)
    print(f"wrote {out_path}: nrow={nrow} ncol={ncol} nnz={nnz} "
          f"(avg nnz/row {nnz / max(nrow, 1):.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True,
                    help="input JSON file or glob (quote globs in the shell)")
    ap.add_argument("--out", required=True, help="output .csr path")
    ap.add_argument("--ncol", type=int, default=BERT_VOCAB_SIZE)
    args = ap.parse_args()

    files = sorted(glob.glob(args.json))
    if not files:
        sys.exit(f"no files match {args.json}")
    print(f"converting {len(files)} file(s) -> {args.out}")
    convert(files, args.out, args.ncol)


if __name__ == "__main__":
    main()
