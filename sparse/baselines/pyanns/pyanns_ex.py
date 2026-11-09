"""PyANNS sparse benchmark.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import numpy as np
import pyanns

import bench_harness as bh

parser = argparse.ArgumentParser(description="PyANNS sparse benchmark.")
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-input", required=True, help="Base CSR file (BigANN sparse format).")
parser.add_argument("-query", required=True)
parser.add_argument("-gt", required=True)
parser.add_argument("-csv", required=True)
parser.add_argument("-index", required=True, help="Index path passed to SparseGrapSearcher.")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-ef", type=int, default=80,
                    help="Single ef; ignored when -ef_list is given.")
parser.add_argument("-ef_list", default=None,
                    help="Comma-separated ef values to sweep.")
parser.add_argument("-budgets", default="0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5,0.8")
parser.add_argument("-R", type=int, default=32, help="Graph degree (our m=32).")
parser.add_argument("-L", type=int, default=200,
                    help="Build search-list size (our ef_construction=200).")
parser.add_argument("-repeats", type=int, default=5)
parser.add_argument("-warmup", type=int, default=1)
parser.add_argument("-no_rescore", action="store_true",
                    help="Skip the exact-IP re-ranking; RR@10 will be invalid.")
args = parser.parse_args()

threads = bh.env_threads()
k = args.k
budgets = [float(b) for b in args.budgets.split(",") if b]
ef_list = ([int(e) for e in args.ef_list.split(",") if e]
           if args.ef_list else [args.ef])

print(f"PyANNS | ef={','.join(map(str, ef_list))} threads={threads} "
      f"rescore={not args.no_rescore} R={args.R} L={args.L}", flush=True)

with bh.phase("load (page-cache warm)") as p_load:
    with open(args.input, "rb") as f:
        while f.read(1 << 24):
            pass

with bh.phase("index") as p_index:
    if not os.path.exists(args.index):
        print(f"  building graph with R={args.R} L={args.L} -> {args.index}",
              flush=True)
        graph = pyanns.SparseHNSWIndex(args.R, args.L).build(args.input)
        graph.save(args.index)
    else:
        print(f"  reusing existing graph {args.index}", flush=True)
    searcher = pyanns.SparseGrapSearcher(args.input, args.index)

print(f"  peak RSS after indexing: {bh.peak_rss_gb():.1f} GB", flush=True)

q_indptr, q_indices, q_data, _ = bh.read_bigann_csr(args.query)
gt = bh.read_gt(args.gt)
n_queries = gt.shape[0]

# --- exact-IP rescoring support -----------------------------------------
base = None
if not args.no_rescore:
    print("  loading base vectors for exact-IP re-ranking (not timed)...", flush=True)
    from scipy.sparse import csr_matrix
    b_indptr, b_indices, b_data, ncol = bh.read_bigann_csr(args.input)
    base = csr_matrix((b_data, b_indices, b_indptr), shape=(args.n, ncol))
    q_mat = csr_matrix(
        (q_data, q_indices, q_indptr), shape=(n_queries, ncol)
    )
    print(f"  base matrix ready ({base.nnz} nnz), peak RSS {bh.peak_rss_gb():.1f} GB",
          flush=True)


def rescore(pred: np.ndarray) -> np.ndarray:
    """Re-rank each row's candidates by exact inner product, descending."""
    if base is None:
        return pred
    out = np.empty_like(pred)
    for i in range(pred.shape[0]):
        cand = pred[i]
        valid = cand[cand >= 0]
        if valid.size == 0:
            out[i] = cand
            continue
        scores = np.asarray((base[valid] @ q_mat[i].T).todense()).ravel()
        order = np.argsort(-scores, kind="stable")
        ranked = valid[order]
        out[i, : ranked.size] = ranked
        out[i, ranked.size:] = -1
    return out


print(f"  {n_queries} queries, k={k}, {len(ef_list)}x{len(budgets)} "
      f"(ef, budget) points\n", flush=True)

index_bytes = bh.path_bytes(args.index)

rows = []
for ef in ef_list:
    searcher.set_ef(ef)
    for budget in budgets:
        def run(budget=budget):
            return searcher.search_batch(n_queries, q_indptr, q_indices, q_data, k, budget)

        raw, med, times = bh.timed_search(
            run, repeats=args.repeats, warmup=args.warmup,
            label=f"ef={ef} budget={budget}"
        )
        pred_raw = np.asarray(raw).reshape(n_queries, k)
        pred = rescore(pred_raw)

        row = bh.make_row(
            model="PyANNS",
            params=f"graph={os.path.basename(args.index)} ef={ef} budget={budget}"
                   f"{'' if not args.no_rescore else ' (unranked)'}",
            threads=threads,
            gt=gt,
            pred=pred,
            k=k,
            index_sec=p_index.sec,
            load_sec=p_load.sec,
            convert_sec=0.0,
            search_times=times,
            index_bytes=index_bytes,
            results_ordered=not args.no_rescore,
        )
        bh.print_point(row)
        if row.RR_at_10 < row.Recall - 1e-9:
            print("    WARNING: RR@k < recall@k even after re-ranking -- investigate.",
                  flush=True)
        rows.append(row)

bh.write_rows(args.csv, rows)
