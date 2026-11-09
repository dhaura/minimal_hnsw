"""SEISMIC benchmark.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import bench_harness as bh

parser = argparse.ArgumentParser(description="SEISMIC sparse benchmark.")
parser.add_argument("-input", required=True, help="Base CSR file (BigANN sparse format).")
parser.add_argument("-query", required=True, help="Query CSR file (BigANN sparse format).")
parser.add_argument("-gt", required=True, help="Ground truth file.")
parser.add_argument("-csv", required=True, help="Results CSV to append to.")
parser.add_argument("-bin_dir", required=True, help="Where the converted .bin files live.")
parser.add_argument("-index", default=None,
                    help="If it exists, load it (search-only run). Else build and save here.")
parser.add_argument("-n_postings", type=int, default=3000)
parser.add_argument("-summary_energy", type=float, default=0.5)
parser.add_argument("-centroid_fraction", type=float, default=0.2)
parser.add_argument("-min_cluster_size", type=int, default=2)
parser.add_argument("-max_fraction", type=float, default=6.0)
parser.add_argument("-doc_cut", type=int, default=15)
# Three settings hardcoded off/wrong until 2026-08-31:
#   nknn / n_knn   the kNN-graph refinement of "Pairing Clustered Inverted
#                  Indexes with k-NN Graphs" -- batch_search's n_knn argument
#                  was pinned to 0, so that variant was never measured.
#   sorted         docs/Guidelines.md gives sorted=False for MS MARCO; the
#                  wrapper passed True.
parser.add_argument("-nknn", type=int, default=0,
                    help="Build a kNN graph with this many neighbours per doc.")
parser.add_argument("-n_knn_list", default="0",
                    help="Comma-separated search-time n_knn values to sweep.")
parser.add_argument("-sorted", dest="sorted_scan", default="true",
                    help="true|false|both")
parser.add_argument("-rm_index", action="store_true",
                    help="Delete the saved index after recording its size.")
parser.add_argument("-k", type=int, default=10)
parser.add_argument("-repeats", type=int, default=5)
parser.add_argument("-warmup", type=int, default=1)
parser.add_argument("-sweep", default="3:0.9,5:0.9,10:0.9,10:0.8,20:0.8,20:0.7,30:0.7",
                    help="Comma-separated query_cut:heap_factor pairs.")
args = parser.parse_args()

threads = bh.env_threads()
os.environ.setdefault("RAYON_NUM_THREADS", str(threads))

import numpy as np
from seismic import SeismicIndexRaw

k = args.k
sweep = []
for pair in args.sweep.split(","):
    qc, hf = pair.split(":")
    sweep.append((int(qc), float(hf)))

print(f"SEISMIC | n_postings={args.n_postings} summary_energy={args.summary_energy} "
      f"centroid_fraction={args.centroid_fraction} min_cluster_size={args.min_cluster_size} "
      f"max_fraction={args.max_fraction} doc_cut={args.doc_cut} "
      f"threads={threads} (RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']})", flush=True)

os.makedirs(args.bin_dir, exist_ok=True)
base_bin = os.path.join(args.bin_dir, "base.bin")
query_bin = os.path.join(args.bin_dir, "queries.bin")

# --- convert -------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utilities"))
convert_sec = 0.0
todo = [(args.input, base_bin), (args.query, query_bin)]
missing = [(src, dst) for src, dst in todo if not os.path.exists(dst)]
if missing:
    from convert_csr_to_bin import convert_csr_to_bin  # noqa: E402
    with bh.phase("convert csr -> bin") as p_conv:
        for src, dst in missing:
            print(f"    {src} -> {dst}", flush=True)
            convert_csr_to_bin(src, dst)
    convert_sec = p_conv.sec
else:
    print("  .bin files already present; ConvertSec reported as 0 "
          "(re-run with a clean -bin_dir to measure it)", flush=True)

# --- load + index --------------------------------------------------------
load_sec = 0.0
index_sec = 0.0

if args.index and os.path.exists(args.index):
    print(f"  loading existing index {args.index} (search-only run)", flush=True)
    with bh.phase("load index"):
        index = SeismicIndexRaw.load(args.index)
else:
    # build() reads base_bin itself; warm the cache and charge it to LoadSec.
    with bh.phase("load (page-cache warm)") as p_load:
        with open(base_bin, "rb") as f:
            while f.read(1 << 24):
                pass
    load_sec = p_load.sec

    with bh.phase("index") as p_index:
        index = SeismicIndexRaw.build(
            base_bin, n_postings=args.n_postings, summary_energy=args.summary_energy,
            centroid_fraction=args.centroid_fraction,
            min_cluster_size=args.min_cluster_size,
            max_fraction=args.max_fraction, doc_cut=args.doc_cut
        )
    index_sec = p_index.sec

    if args.index:
        with bh.phase("save index"):
            index.save(args.index)

if args.nknn > 0:
    with bh.phase(f"build knn graph (nknn={args.nknn})") as p_knn:
        index.build_knn(args.nknn)
    print(f"  kNN graph: {args.nknn}/doc in {p_knn.sec:.1f}s", flush=True)
    # The kNN graph is part of the searchable index, so it counts toward IndexSec.
    index_sec += p_knn.sec

print(f"  peak RSS after indexing: {bh.peak_rss_gb():.1f} GB", flush=True)

gt = bh.read_gt(args.gt)
n_queries = gt.shape[0]
print(f"  {n_queries} queries, k={k}, {len(sweep)} sweep points\n", flush=True)

index_bytes = bh.path_bytes(args.index)
if args.rm_index and args.index and os.path.exists(args.index):
    os.remove(args.index)
    print(f"  removed {args.index} after recording {index_bytes} bytes", flush=True)

n_knns = [int(x) for x in args.n_knn_list.split(",") if x.strip()] if args.nknn > 0 else [0]
sorted_opts = {"true": [True], "false": [False],
               "both": [True, False]}[args.sorted_scan.strip().lower()]

rows = []
for query_cut, heap_factor in sweep:
    for n_knn in n_knns:
        for sorted_scan in sorted_opts:
            def run(qc=query_cut, hf=heap_factor, nk=n_knn, sc=sorted_scan):
                # num_threads is a no-op in this build; RAYON_NUM_THREADS rules.
                return index.batch_search(query_bin, k, qc, hf, nk, sc,
                                          num_threads=threads)

            results, med, times = bh.timed_search(
                run, repeats=args.repeats, warmup=args.warmup,
                label=f"qc={query_cut} hf={heap_factor} n_knn={n_knn} sorted={sorted_scan}",
            )

            # batch_search returns per-query lists of (score, doc_id), best first.
            pred = np.array(
                [[doc for _, doc in r] + [-1] * (k - len(r)) for r in results],
                dtype=np.int64)
            # Pad with -inf so short result lists still read as descending-by-score.
            scores = np.array(
                [[s for s, _ in r] + [-np.inf] * (k - len(r)) for r in results],
                dtype=np.float64)
            ordered = bh.check_sorted_by_score(scores, "SEISMIC")

            row = bh.make_row(
                model="SEISMIC",
                params=f"n_postings={args.n_postings} summary_energy={args.summary_energy} "
                       f"centroid_fraction={args.centroid_fraction} "
                       f"max_fraction={args.max_fraction} doc_cut={args.doc_cut} "
                       f"nknn={args.nknn} n_knn={n_knn} sorted={sorted_scan} "
                       f"query_cut={query_cut} heap_factor={heap_factor}",
                threads=threads,
                gt=gt,
                pred=pred,
                k=k,
                index_sec=index_sec,
                load_sec=load_sec,
                convert_sec=convert_sec,
                search_times=times,
                index_bytes=index_bytes,
                results_ordered=ordered,
            )
            bh.print_point(row)
            rows.append(row)

bh.write_rows(args.csv, rows)
