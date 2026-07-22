#!/usr/bin/env python3
"""Compute exact max-inner-product ground truth for big-ann CSR datasets.

Reads the base and query .csr files (fp32 big-ann layout, same files the demo
drivers load) and writes the .gt layout main.cpp's get_gt() expects, matching
the MSMARCO dev GT files:
    uint32 nq, uint32 k, uint32 ids[nq*k], float32 dists[nq*k]

Row ids in the output refer to row order in the base .csr, which is also the
label order the demos assign at insertion time.

Usage (needs `module load python` on Perlmutter):
    python3 compute_gt.py --base base_nq.csr --queries queries.test.csr \
        --out base_nq.test.gt --k 10
"""
import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp


def load_csr(path):
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
        indptr = np.fromfile(f, dtype=np.int64, count=nrow + 1)
        indices = np.fromfile(f, dtype=np.int32, count=nnz)
        data = np.fromfile(f, dtype=np.float32, count=nnz)
    if indices.size != nnz or data.size != nnz:
        sys.exit(f"{path}: truncated file (is it the fp32 ori format?)")
    return sp.csr_matrix((data, indices, indptr), shape=(int(nrow), int(ncol)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=64, help="queries per matmul chunk")
    args = ap.parse_args()

    t0 = time.time()
    X = load_csr(args.base)
    Q = load_csr(args.queries)
    if Q.shape[1] != X.shape[1]:
        sys.exit(f"dim mismatch: base ncol={X.shape[1]} queries ncol={Q.shape[1]}")
    print(f"base {X.shape} nnz={X.nnz}, queries {Q.shape} nnz={Q.nnz} "
          f"(loaded in {time.time() - t0:.1f}s)", flush=True)

    XT = X.T.tocsr()
    print(f"transposed base in {time.time() - t0:.1f}s", flush=True)

    nq, k = Q.shape[0], args.k
    ids = np.zeros((nq, k), dtype=np.uint32)
    dists = np.full((nq, k), -np.inf, dtype=np.float32)
    for lo in range(0, nq, args.chunk):
        hi = min(lo + args.chunk, nq)
        S = (Q[lo:hi] @ XT).tocsr()
        for i in range(hi - lo):
            row_vals = S.data[S.indptr[i]:S.indptr[i + 1]]
            row_cols = S.indices[S.indptr[i]:S.indptr[i + 1]]
            if row_vals.size < k:
                sys.exit(f"query {lo + i} matched only {row_vals.size} docs (< k={k})")
            top = np.argpartition(row_vals, -k)[-k:]
            order = top[np.argsort(-row_vals[top], kind="stable")]
            ids[lo + i] = row_cols[order]
            dists[lo + i] = row_vals[order]
        print(f"  queries {hi}/{nq} ({time.time() - t0:.1f}s elapsed)", flush=True)

    with open(args.out, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        ids.tofile(f)
        dists.tofile(f)
    print(f"wrote {args.out}: nq={nq} k={k} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
