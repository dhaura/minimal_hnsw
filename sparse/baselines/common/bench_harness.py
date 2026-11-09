"""Shared measurement harness for the sparse-kNN comparison.

Definitions
----------------------
load_sec        wall time to get the vectors from disk into memory.
convert_sec     wall time for any one-off format conversion the method needs.
index_sec       wall time the actual index construction. Excludes disk I/O
                and excludes saving the index.
search_sec      Wall time for the whole query batch.
qps             n_queries / search_sec
amortized_us    search_sec / n_queries * 1e6. Inverse throughput. NOT latency.
latency_us      only set by a separate single-thread pass; genuine per-query
                latency.
peak_rss_gb     max RSS of this process, sampled after indexing and after search.
"""

from __future__ import annotations

import csv
import os
import resource
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def read_bigann_csr(path: str):
    """BigANN sparse CSR: int64 (nrow, ncol, nnz), int64 indptr, int32 indices,
    float32 data. Returns (indptr, indices, data, ncol).
    """
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype="int64", count=3)
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        indices = np.fromfile(f, dtype="int32", count=nnz)
        data = np.fromfile(f, dtype="float32", count=nnz)
    return indptr, indices, data, int(ncol)


def read_gt(path: str) -> np.ndarray:
    """BigANN ground truth: uint32 n, uint32 d, then n*d uint32 ids
    (float32 distances follow, unused). Returns the id matrix, shape (n, d)."""
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype="uint32", count=2)
        f.seek(8)
        I = np.fromfile(f, dtype="int32", count=int(n) * int(d)).reshape(int(n), int(d))
    return I


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    """|pred ∩ gt| / (n_queries * k), both truncated to k.
    """
    gt_k = gt[:, :k]
    hits = 0
    for g, p in zip(gt_k, pred[:, :k]):
        hits += len(np.intersect1d(g, p))
    return hits / (gt_k.shape[0] * k)


def rr_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Mean reciprocal rank of the first predicted id that is in the exact-NN
    ground truth."""
    gt_k = gt[:, :k]
    total = 0.0
    for g, p in zip(gt_k, pred[:, :k]):
        gs = set(g.tolist())
        for rank, item in enumerate(p[:k], start=1):
            if item in gs:
                total += 1.0 / rank
                break
    return total / gt_k.shape[0]


def check_sorted_by_score(dists: np.ndarray | None, name: str) -> bool:
    """Warn if a method returns results that are not ranked best-first.
    """
    if dists is None:
        return True
    d = np.asarray(dists)
    if d.ndim != 2 or d.shape[1] < 2:
        return True
    non_decreasing = bool(np.all(np.diff(d, axis=1) >= -1e-6))
    non_increasing = bool(np.all(np.diff(d, axis=1) <= 1e-6))
    ok = non_decreasing or non_increasing
    if not ok:
        print(
            f"  WARNING [{name}]: returned neighbours are not ordered by score. "
            "RR@k is not meaningful for this method until they are sorted.",
            flush=True,
        )
    return ok


# --------------------------------------------------------------------------
# Resource accounting
# --------------------------------------------------------------------------


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB. ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def path_bytes(path: str | None) -> int:
    """Total bytes of a saved index -- a file, or a directory tree."""
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


class phase:
    """Context manager timing one named phase.

        with phase("index") as p:
            build()
        p.sec  # elapsed

    Uses perf_counter (monotonic, highest resolution available).
    """

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.sec = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sec = time.perf_counter() - self._t0
        if self.verbose:
            print(f"  [{self.name}] {self.sec:.4f} s", flush=True)
        return False


def timed_search(
    fn: Callable[[], object],
    repeats: int = 5,
    warmup: int = 1,
    label: str = "",
):
    """Run `fn` `warmup` times untimed, then `repeats` times timed.

    Returns (last_result, median_sec, all_times).
    """
    for _ in range(max(0, warmup)):
        result = fn()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    if label:
        spread = (max(times) - min(times)) / med * 100 if med > 0 else 0.0
        print(
            f"  [{label}] median {med:.4f} s over {len(times)} runs "
            f"(spread {spread:.1f}%)",
            flush=True,
        )
    return result, med, times


# --------------------------------------------------------------------------
# Result rows
# --------------------------------------------------------------------------

SCHEMA = [
    "Model",
    "Params",
    "Threads",
    "Recall",
    "RR@10",
    "IndexSec",
    "LoadSec",
    "ConvertSec",
    "SearchSecMedian",
    "SearchSecRuns",
    "AmortizedUsPerQuery",
    "QPS",
    "LatencyUsSingleThread",
    "PeakRSSGB",
    "IndexBytes",
    "NumQueries",
    "K",
    "ResultsOrdered",
    "Host",
    "JobID",
]


@dataclass
class Row:
    Model: str
    Params: str
    Threads: int
    Recall: float
    RR_at_10: float
    IndexSec: float
    LoadSec: float
    ConvertSec: float
    SearchSecMedian: float
    SearchSecRuns: str
    AmortizedUsPerQuery: float
    QPS: float
    LatencyUsSingleThread: float | None
    PeakRSSGB: float
    IndexBytes: int
    NumQueries: int
    K: int
    ResultsOrdered: bool
    Host: str = field(default_factory=lambda: os.uname().nodename)
    JobID: str = field(default_factory=lambda: os.environ.get("SLURM_JOB_ID", ""))

    def as_csv_dict(self) -> dict:
        d = asdict(self)
        d["RR@10"] = d.pop("RR_at_10")
        return d


def make_row(
    model: str,
    params: str,
    threads: int,
    gt: np.ndarray,
    pred: np.ndarray,
    k: int,
    index_sec: float,
    load_sec: float,
    convert_sec: float,
    search_times: Sequence[float],
    index_bytes: int = 0,
    latency_us: float | None = None,
    results_ordered: bool = True,
) -> Row:
    n_queries = gt.shape[0]
    med = statistics.median(search_times)
    return Row(
        Model=model,
        Params=params,
        Threads=threads,
        Recall=recall_at_k(gt, pred, k),
        RR_at_10=rr_at_k(gt, pred, k),
        IndexSec=index_sec,
        LoadSec=load_sec,
        ConvertSec=convert_sec,
        SearchSecMedian=med,
        SearchSecRuns=";".join(f"{t:.6f}" for t in search_times),
        AmortizedUsPerQuery=med / n_queries * 1e6,
        QPS=n_queries / med,
        LatencyUsSingleThread=latency_us,
        PeakRSSGB=peak_rss_gb(),
        IndexBytes=index_bytes,
        NumQueries=n_queries,
        K=k,
        ResultsOrdered=results_ordered,
    )


def write_rows(csv_path: str, rows: Iterable[Row]) -> None:
    rows = list(rows)
    if not rows:
        return
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_dict())
    print(f"\nAppended {len(rows)} rows to {csv_path}", flush=True)


def print_point(row: Row) -> None:
    print(
        f"  {row.Params} | recall@{row.K} {row.Recall * 100:.4f}% "
        f"| RR {row.RR_at_10:.4f} | {row.SearchSecMedian:.4f} s "
        f"| {row.QPS:.2f} QPS @{row.Threads}T | peak {row.PeakRSSGB:.1f} GB",
        flush=True,
    )


def env_threads(default: int = 48) -> int:
    """Thread count the job was given. bench_env.sh sets these consistently."""
    for var in ("BENCH_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return default
