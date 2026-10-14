#!/usr/bin/env python3
"""Convert Anserini SPLADE-encoded topics to a big-ann CSR binary.

Input: `topics.*.splade-pp-ed.tsv.gz` from castorini/eval (formerly
    anserini-tools), in the
    term-repetition impact format Lucene wants -- one line per query,

        <qid>\t<tok> <tok> <tok> ...

    where a token repeated n times carries integer weight n. E.g. query 2 of
    topics.msmarco-v2-passage.dev.splade-pp-ed.tsv.gz repeats "receptor" 2234
    times, so its weight on that term is 2234.

Output: the layout csr_matrix.h's CSRMatrix(path, ori=true) reads,
        int64 nrow, int64 ncol, int64 nnz,
        int64 indptr[nrow+1], int32 indices[nnz], float32 data[nnz]
    plus a `<out>.qids` sidecar: one query id per line, in row order. Row order
    is file order, which is the order the .gt file is indexed by.

Usage (needs `module load python` on Perlmutter):
    python3 convert_splade_topics_to_csr.py \
        --topics topics.msmarco-v2-passage.dev.splade-pp-ed.tsv.gz \
        --vocab bert_vocab.txt --out queries.dev.csr
"""
import argparse
import gzip
import io
import sys
from collections import Counter

import numpy as np

BERT_VOCAB_SIZE = 30522


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = {tok.rstrip("\n"): i for i, tok in enumerate(f)}
    if len(vocab) != BERT_VOCAB_SIZE:
        print(f"warning: {path} has {len(vocab)} tokens, expected {BERT_VOCAB_SIZE}",
              file=sys.stderr)
    return vocab


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True, help="topics.*.tsv[.gz]")
    ap.add_argument("--vocab", required=True, help="bert-base-uncased vocab.txt")
    ap.add_argument("--out", required=True, help="output .csr path")
    ap.add_argument("--ncol", type=int, default=BERT_VOCAB_SIZE)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every weight by this (default 1.0)")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)

    qids, counts, indices, values = [], [], [], []
    unknown = 0
    with open_maybe_gz(args.topics) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _, toks = line.partition("\t")
            row = []
            for tok, w in Counter(toks.split()).items():
                tid = vocab.get(tok)
                if tid is None:
                    unknown += 1
                    continue
                row.append((tid, w * args.scale))
            row.sort()          # CSR rows must be index-sorted

            qids.append(qid)
            counts.append(len(row))
            for tid, w in row:
                indices.append(tid)
                values.append(w)

    nrow = len(counts)
    nnz = len(indices)
    if nrow == 0:
        sys.exit(f"{args.topics}: no queries parsed")

    indptr = np.zeros(nrow + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    with open(args.out, "wb") as out:
        np.array([nrow, args.ncol, nnz], dtype=np.int64).tofile(out)
        indptr.tofile(out)
        np.asarray(indices, dtype=np.int32).tofile(out)
        np.asarray(values, dtype=np.float32).tofile(out)

    with open(args.out + ".qids", "w", encoding="utf-8") as out:
        out.write("\n".join(qids))
        out.write("\n")

    print(f"wrote {args.out}: nrow={nrow:,} ncol={args.ncol} nnz={nnz:,} "
          f"(mean {nnz / nrow:.1f}/row) unknown_tokens={unknown} scale={args.scale}")
    print(f"wrote {args.out}.qids")


if __name__ == "__main__":
    main()
