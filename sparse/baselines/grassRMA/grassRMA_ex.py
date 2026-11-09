"""GrassRMA sparse-HNSW benchmark.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import numpy as np
import sparse_hnswlib

import bench_harness as bh

parser = argparse.ArgumentParser(description="GrassRMA sparse-HNSW benchmark.")
parser.add_argument("-n", type=int, required=True, help="Number of base vectors.")
parser.add_argument("-input", required=True, help="Base CSR file (BigANN sparse format).")
parser.add_argument("-query", required=True, help="Query CSR file.")
parser.add_argument("-gt", required=True, help="Ground truth file.")
parser.add_argument("-csv", required=True, help="Results CSV to append to.")
parser.add_argument("-M", type=int, default=16)
parser.add_argument("-ef_construction", type=int, default=200,
                    help="Matched to the other graph methods; was hardcoded to 1000.")
parser.add_argument("-ef_list", default="10,20,50,100,200,400,800,1600,3200")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-repeats", type=int, default=5)
parser.add_argument("-warmup", type=int, default=1)
parser.add_argument("-save_index", dest="save_index", default=None,
                    help="Optional path; written AFTER the sweep, never before.")
args = parser.parse_args()

threads = bh.env_threads()
ef_list = [int(x) for x in args.ef_list.split(",") if x]
k = args.k

print(f"GrassRMA | M={args.M} efC={args.ef_construction} threads={threads}", flush=True)

# --- load ----------------------------------------------------------------
with bh.phase("load (page-cache warm)") as p_load:
    with open(args.input, "rb") as f:
        while f.read(1 << 24):
            pass

# --- index ---------------------------------------------------------------
with bh.phase("index") as p_index:
    index = sparse_hnswlib.Index(space="ip", dim=16)
    index.init_index(
        max_elements=args.n,
        csr_path=args.input,
        ef_construction=args.ef_construction,
        M=args.M,
    )
    index.add_items(num_threads=threads)

print(f"  peak RSS after indexing: {bh.peak_rss_gb():.1f} GB", flush=True)

# --- queries + ground truth ----------------------------------------------
q_indptr, q_indices, q_data, _ = bh.read_bigann_csr(args.query)
gt = bh.read_gt(args.gt)
n_queries = gt.shape[0]
if len(q_indptr) - 1 != n_queries:
    sys.exit(f"query file has {len(q_indptr) - 1} rows but gt has {n_queries}")

print(f"  {n_queries} queries, k={k}, sweeping ef over {ef_list}\n", flush=True)

# --- sweep ---------------------------------------------------------------
rows = []
for ef in ef_list:
    if ef < k:
        print(f"  skipping ef={ef} (< k={k})", flush=True)
        continue
    index.set_ef(ef)

    def run():
        return index.knn_query(q_indptr, q_indices, q_data, k=k, num_threads=threads)

    (pred, dists), med, times = bh.timed_search(
        run, repeats=args.repeats, warmup=args.warmup, label=f"ef={ef}"
    )
    ordered = bh.check_sorted_by_score(dists, "GrassRMA")

    row = bh.make_row(
        model="GrassRMA",
        params=f"M={args.M} efC={args.ef_construction} ef={ef}",
        threads=threads,
        gt=gt,
        pred=np.asarray(pred),
        k=k,
        index_sec=p_index.sec,
        load_sec=p_load.sec,
        convert_sec=0.0,
        search_times=times,
        results_ordered=ordered,
    )
    bh.print_point(row)
    rows.append(row)

if args.save_index:
    with bh.phase("save index"):
        index.save_index(args.save_index)
    for r in rows:
        r.IndexBytes = bh.path_bytes(args.save_index)

bh.write_rows(args.csv, rows)
