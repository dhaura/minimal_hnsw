"""kANNolo sparse-HNSW benchmark.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import numpy as np
from kannolo import SparsePlainHNSW

import bench_harness as bh

parser = argparse.ArgumentParser(description="kANNolo sparse benchmark.")
parser.add_argument("-input", required=True, help="Base CSR file (BigANN sparse format).")
parser.add_argument("-query", required=True, help="Query CSR file.")
parser.add_argument("-gt", required=True, help="Ground truth file.")
parser.add_argument("-csv", required=True, help="Results CSV to append to.")
parser.add_argument("-index", default=None,
                    help="If it exists, load it (search-only run). Else build and save here.")
parser.add_argument("-m", type=int, default=16,
                    help="Matched to SPARSE_HNSW/GrassRMA; was 32.")
parser.add_argument("-ef_construction", type=int, default=200)
parser.add_argument("-ef_list", default="10,20,50,100,200,400,800,1600,3200")
parser.add_argument("-early_exit_list", default="none",
                    help="Comma-separated early_exit_threshold values, or 'none'.")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-repeats", type=int, default=5)
parser.add_argument("-warmup", type=int, default=1)
args = parser.parse_args()

threads = bh.env_threads()
ef_list = [int(x) for x in args.ef_list.split(",") if x]
k = args.k

print(f"kANNolo | m={args.m} efC={args.ef_construction} threads={threads}", flush=True)

load_sec = 0.0
index_sec = 0.0

if args.index and os.path.exists(args.index):
    print(f"  loading existing index {args.index} (search-only run)", flush=True)
    with bh.phase("load index"):
        index = SparsePlainHNSW.load(args.index)
else:
    with bh.phase("load base vectors") as p_load:
        b_indptr, b_indices, b_data, _ = bh.read_bigann_csr(args.input)
    load_sec = p_load.sec

    with bh.phase("index") as p_index:
        index = SparsePlainHNSW.build_from_arrays(
            b_indices, b_data, b_indptr,
            m=args.m, ef_construction=args.ef_construction, metric="ip",
        )
    index_sec = p_index.sec

    if args.index:
        with bh.phase("save index"):
            index.save(args.index)

print(f"  peak RSS after indexing: {bh.peak_rss_gb():.1f} GB", flush=True)

q_indptr, q_indices, q_data, _ = bh.read_bigann_csr(args.query)
gt = bh.read_gt(args.gt)
n_queries = gt.shape[0]
if len(q_indptr) - 1 != n_queries:
    sys.exit(f"query file has {len(q_indptr) - 1} rows but gt has {n_queries}")

early_exits = [None if e.strip().lower() == "none" else float(e)
               for e in args.early_exit_list.split(",") if e.strip()]

print(f"  {n_queries} queries, k={k}, sweeping ef over {ef_list} "
      f"x early_exit over {early_exits}\n", flush=True)

index_bytes = bh.path_bytes(args.index)

rows = []
for ef in ef_list:
    if ef < k:
        print(f"  skipping ef={ef} (< k={k})", flush=True)
        continue

    for eet in early_exits:
        def run(eet=eet, ef=ef):
            kw = {} if eet is None else {"early_exit_threshold": eet}
            return index.batch_search(
                q_indices, q_data, q_indptr, k=k, ef_search=ef,
                num_threads=threads, **kw
            )

        (dists, ids), med, times = bh.timed_search(
            run, repeats=args.repeats, warmup=args.warmup,
            label=f"ef={ef} eet={eet}"
        )
        pred = np.asarray(ids).reshape(n_queries, k)
        ordered = bh.check_sorted_by_score(
            np.asarray(dists).reshape(n_queries, k), "kANNolo"
        )

        params = f"m={args.m} efC={args.ef_construction} ef={ef}"
        if eet is not None:
            params += f" eet={eet}"

        row = bh.make_row(
            model="kANNolo",
            params=params,
            threads=threads,
            gt=gt,
            pred=pred,
            k=k,
            index_sec=index_sec,
            load_sec=load_sec,
            convert_sec=0.0,
            search_times=times,
            index_bytes=index_bytes,
            results_ordered=ordered,
        )
        bh.print_point(row)
        rows.append(row)

bh.write_rows(args.csv, rows)
