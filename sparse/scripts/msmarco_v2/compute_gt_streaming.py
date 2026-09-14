#!/usr/bin/env python3
"""Exact max-inner-product ground truth for big-ann CSR datasets, streamed.

Same inputs, same output layout as compute_gt.py:
    uint32 nq, uint32 k, uint32 ids[nq*k], float32 dists[nq*k]
Row ids refer to row order in the base .csr, which is the label order the demo
drivers assign at insertion time.

Usage (needs `module load python` on Perlmutter):
    python3 compute_gt_streaming.py --base base_v2_splade.csr \
        --queries queries.dev.csr --out base_v2_splade.dev.gt \
        --k 10 --workers 32 --block-rows 250000
"""
import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp

HEADER = 24  # int64 nrow, ncol, nnz


def read_header(path):
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
        indptr = np.fromfile(f, dtype=np.int64, count=int(nrow) + 1)
    if indptr.size != nrow + 1:
        sys.exit(f"{path}: truncated indptr")
    return int(nrow), int(ncol), int(nnz), indptr


def load_csr(path):
    nrow, ncol, nnz, indptr = read_header(path)
    off = HEADER + 8 * (nrow + 1)
    indices = np.fromfile(path, dtype=np.int32, count=nnz, offset=off)
    data = np.fromfile(path, dtype=np.float32, count=nnz, offset=off + 4 * nnz)
    if indices.size != nnz or data.size != nnz:
        sys.exit(f"{path}: truncated (is it the fp32 ori format?)")
    return sp.csr_matrix((data, indices, indptr), shape=(nrow, ncol))


def read_block(path, indptr, r0, r1, ncol, nnz):
    """Rows [r0, r1) of the .csr as a standalone csr_matrix."""
    s, e = int(indptr[r0]), int(indptr[r1])
    base = HEADER + 8 * (indptr.size)
    indices = np.fromfile(path, dtype=np.int32, count=e - s, offset=base + 4 * s)
    data = np.fromfile(path, dtype=np.float32, count=e - s,
                       offset=base + 4 * nnz + 4 * s)
    local = (indptr[r0:r1 + 1] - s).astype(np.int64)
    return sp.csr_matrix((data, indices, local), shape=(r1 - r0, ncol))


_G = {}


def _init(base, queries, k, chunk):
    nrow, ncol, nnz, indptr = read_header(base)
    _G.update(base=base, indptr=indptr, ncol=ncol, nnz=nnz, k=k, chunk=chunk,
              Q=load_csr(queries))


def score_block(rng):
    """Top-k of one document block for every query. Ids are global row ids."""
    r0, r1 = rng
    B = read_block(_G["base"], _G["indptr"], r0, r1, _G["ncol"], _G["nnz"])
    BT = B.T.tocsr()
    del B

    Q, k, chunk = _G["Q"], _G["k"], _G["chunk"]
    nq = Q.shape[0]
    ids = np.zeros((nq, k), dtype=np.uint32)
    dists = np.full((nq, k), -np.inf, dtype=np.float32)

    for lo in range(0, nq, chunk):
        hi = min(lo + chunk, nq)
        S = (Q[lo:hi] @ BT).tocsr()
        # scipy's sparse matmul does not promise sorted column indices, and the
        # tie-break below reads S.indices as ascending doc id.
        S.sort_indices()
        for i in range(hi - lo):
            vals = S.data[S.indptr[i]:S.indptr[i + 1]]
            cols = S.indices[S.indptr[i]:S.indptr[i + 1]]
            if vals.size == 0:
                continue
            if vals.size > k:
                # Cut to the k-th largest value, keeping every tie with it, then
                # order that small superset. `cols` is ascending (csr row order)
                # and flatnonzero preserves it, so a stable sort on -vals breaks
                # ties by ascending doc id -- the same rule compute_gt.py gets
                # from stable-sorting a whole corpus row.
                thr = np.partition(vals, -k)[-k]
                sel = np.flatnonzero(vals >= thr)
                vals, cols = vals[sel], cols[sel]
            order = np.argsort(-vals, kind="stable")[:k]
            n = order.size
            ids[lo + i, :n] = cols[order] + r0
            dists[lo + i, :n] = vals[order]
    return ids, dists


def merge(acc_ids, acc_dists, ids, dists, k):
    """Keep the best k of the running top-k and one block's top-k.

    Ordered by descending score, ties by ascending doc id, so the result does
    not depend on how the corpus was split into blocks.
    """
    cat_d = np.concatenate([acc_dists, dists], axis=1)
    cat_i = np.concatenate([acc_ids, ids], axis=1)
    order = np.lexsort((cat_i, -cat_d), axis=1)[:, :k]
    rows = np.arange(cat_d.shape[0])[:, None]
    return cat_i[rows, order], cat_d[rows, order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=64, help="queries per matmul")
    ap.add_argument("--block-rows", type=int, default=250000,
                    help="documents per block; drives peak memory")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    nrow, ncol, nnz, _ = read_header(args.base)
    Q = load_csr(args.queries)
    if Q.shape[1] != ncol:
        sys.exit(f"dim mismatch: base ncol={ncol} queries ncol={Q.shape[1]}")
    nq, k = Q.shape[0], args.k
    blocks = [(r, min(r + args.block_rows, nrow))
              for r in range(0, nrow, args.block_rows)]
    print(f"base {nrow:,}x{ncol} nnz={nnz:,} | queries {nq}x{ncol} nnz={Q.nnz:,}\n"
          f"{len(blocks)} blocks of {args.block_rows:,} rows, {args.workers} workers",
          flush=True)

    acc_ids = np.zeros((nq, k), dtype=np.uint32)
    acc_dists = np.full((nq, k), -np.inf, dtype=np.float32)

    done = 0
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.base, args.queries, k, args.chunk)) as pool:
            for ids, dists in pool.imap_unordered(score_block, blocks, chunksize=1):
                acc_ids, acc_dists = merge(acc_ids, acc_dists, ids, dists, k)
                done += 1
                if done % 20 == 0 or done == len(blocks):
                    el = time.time() - t0
                    print(f"  [{done}/{len(blocks)}] {el:.0f}s "
                          f"eta={el / done * (len(blocks) - done):.0f}s", flush=True)
    else:
        _init(args.base, args.queries, k, args.chunk)
        for b in blocks:
            ids, dists = score_block(b)
            acc_ids, acc_dists = merge(acc_ids, acc_dists, ids, dists, k)
            done += 1

    short = int((acc_dists == -np.inf).any(axis=1).sum())
    if short:
        sys.exit(f"{short} queries matched fewer than k={k} documents")

    with open(args.out, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        acc_ids.astype(np.uint32).tofile(f)
        acc_dists.astype(np.float32).tofile(f)
    print(f"wrote {args.out}: nq={nq} k={k} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
