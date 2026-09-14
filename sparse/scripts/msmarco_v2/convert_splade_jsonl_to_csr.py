#!/usr/bin/env python3
"""Convert an Anserini/Pyserini SPLADE-encoded corpus to a big-ann CSR binary.

Input: one or more `*.jsonl.gz` shards, one JSON object per line, as shipped in
    https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco_v2_passage_splade_pp_ed.tar

    {"id": "msmarco_passage_06_797714185", "content": "",
     "vector": {"receptor": 20, "##s": 48, ...}}

    Keys of `vector` are BERT WordPiece tokens, values are integer impacts.

Output: the layout csr_matrix.h's CSRMatrix(path, ori=true) reads, same as the
    MSMARCO big-ann files:
        int64 nrow, int64 ncol, int64 nnz,
        int64 indptr[nrow+1], int32 indices[nnz], float32 data[nnz]
    plus a `<out>.pids` sidecar: one passage id per line, in row order.

Usage (needs `module load python` on Perlmutter):
    python3 convert_splade_jsonl_to_csr.py \
        --shards 'msmarco_v2_passage_splade_pp_ed/*.jsonl.gz' \
        --vocab bert_vocab.txt --out base_v2_splade.csr --workers 64
"""
import argparse
import glob
import gzip
import os
import re
import shutil
import sys
import time

import numpy as np

try:
    import orjson as _json

    def loads(b):
        return _json.loads(b)
except ImportError:
    import json as _json

    def loads(b):
        return _json.loads(b)

BERT_VOCAB_SIZE = 30522


def load_vocab(path):
    """token -> id, by line order (line 1 is id 0, as in HF vocab.txt)."""
    with open(path, encoding="utf-8") as f:
        vocab = {tok.rstrip("\n"): i for i, tok in enumerate(f)}
    if len(vocab) != BERT_VOCAB_SIZE:
        print(f"warning: {path} has {len(vocab)} tokens, expected {BERT_VOCAB_SIZE}",
              file=sys.stderr)
    return vocab


def shard_key(path):
    """Numeric sort key so 2.jsonl.gz precedes 10.jsonl.gz."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return (int(m.group(1)) if m else 0, os.path.basename(path))


_VOCAB = None
_TMP = None


def _init(vocab_path, tmp):
    global _VOCAB, _TMP
    _VOCAB = load_vocab(vocab_path)
    _TMP = tmp


def convert_shard(args):
    """Write <tmp>/<n>.{idx,val,cnt,pid} for one shard; return its stats."""
    idx_in_order, path = args
    vocab = _VOCAB
    stem = os.path.join(_TMP, f"{idx_in_order:05d}")

    indices, values, counts, pids = [], [], [], []
    unknown = 0
    empty_rows = 0

    with gzip.open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            rec = loads(line)
            vec = rec["vector"]

            row = []
            for tok, w in vec.items():
                tid = vocab.get(tok)
                if tid is None:
                    unknown += 1
                    continue
                if w <= 0:
                    continue
                row.append((tid, w))
            # CSR rows must be index-sorted: the distance kernel merges two
            # sorted lists, and pruneMatrixWithAlpha re-sorts on that assumption.
            row.sort()

            if not row:
                empty_rows += 1
            counts.append(len(row))
            pids.append(rec["id"])
            for tid, w in row:
                indices.append(tid)
                values.append(w)

    np.asarray(indices, dtype=np.int32).tofile(stem + ".idx")
    np.asarray(values, dtype=np.float32).tofile(stem + ".val")
    np.asarray(counts, dtype=np.int64).tofile(stem + ".cnt")
    with open(stem + ".pid", "w", encoding="utf-8") as f:
        f.write("\n".join(pids))
        f.write("\n")

    return (idx_in_order, os.path.basename(path), len(counts), len(indices),
            unknown, empty_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True,
                    help="glob for the *.jsonl.gz shards (quote it)")
    ap.add_argument("--vocab", required=True, help="bert-base-uncased vocab.txt")
    ap.add_argument("--out", required=True, help="output .csr path")
    ap.add_argument("--tmp", default=None,
                    help="scratch dir for the per-shard parts (default <out>.parts)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ncol", type=int, default=BERT_VOCAB_SIZE)
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="convert only the first N shards (for a smaller subset)")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards), key=shard_key)
    if not shards:
        sys.exit(f"no shards matched {args.shards!r}")
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"{len(shards)} shards, {args.workers} workers -> {args.out}", flush=True)

    tmp = args.tmp or (args.out + ".parts")
    os.makedirs(tmp, exist_ok=True)

    t0 = time.time()
    jobs = list(enumerate(shards))
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.vocab, tmp)) as pool:
            for r in pool.imap_unordered(convert_shard, jobs, chunksize=1):
                results.append(r)
                done = len(results)
                if done % 10 == 0 or done == len(jobs):
                    rows = sum(x[2] for x in results)
                    nnz = sum(x[3] for x in results)
                    el = time.time() - t0
                    print(f"  [{done}/{len(jobs)}] rows={rows:,} nnz={nnz:,} "
                          f"{el:.0f}s eta={el / done * (len(jobs) - done):.0f}s",
                          flush=True)
    else:
        _init(args.vocab, tmp)
        for j in jobs:
            results.append(convert_shard(j))

    results.sort()
    nrow = sum(r[2] for r in results)
    nnz = sum(r[3] for r in results)
    unknown = sum(r[4] for r in results)
    empty = sum(r[5] for r in results)
    print(f"phase 1 done in {time.time() - t0:.0f}s: nrow={nrow:,} nnz={nnz:,} "
          f"(mean {nnz / max(nrow, 1):.1f}/row) unknown_tokens={unknown:,} "
          f"empty_rows={empty:,}", flush=True)

    # Phase 2: header + indptr, then every .idx in order, then every .val.
    t1 = time.time()
    indptr = np.zeros(nrow + 1, dtype=np.int64)
    pos = 0
    for r in results:
        cnt = np.fromfile(os.path.join(tmp, f"{r[0]:05d}.cnt"), dtype=np.int64)
        indptr[pos + 1: pos + 1 + cnt.size] = cnt
        pos += cnt.size
    np.cumsum(indptr, out=indptr)
    assert indptr[-1] == nnz, (indptr[-1], nnz)

    with open(args.out, "wb") as out:
        np.array([nrow, args.ncol, nnz], dtype=np.int64).tofile(out)
        indptr.tofile(out)
        for ext in (".idx", ".val"):
            for r in results:
                with open(os.path.join(tmp, f"{r[0]:05d}{ext}"), "rb") as part:
                    shutil.copyfileobj(part, out, 1 << 24)
    del indptr

    with open(args.out + ".pids", "w", encoding="utf-8") as out:
        for r in results:
            with open(os.path.join(tmp, f"{r[0]:05d}.pid"), encoding="utf-8") as part:
                shutil.copyfileobj(part, out, 1 << 24)

    expect = 24 + 8 * (nrow + 1) + 8 * nnz
    got = os.path.getsize(args.out)
    if got != expect:
        sys.exit(f"FATAL {args.out} is {got} bytes, expected {expect}")

    if not args.keep_tmp:
        shutil.rmtree(tmp)

    print(f"phase 2 done in {time.time() - t1:.0f}s")
    print(f"wrote {args.out} ({got / 2**30:.1f} GiB) and {args.out}.pids")


if __name__ == "__main__":
    main()
