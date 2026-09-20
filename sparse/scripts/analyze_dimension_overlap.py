#!/usr/bin/env python3
"""Compare query/ground-truth dimension overlap before and after pruning.

For every query and each of its selected exact ground-truth neighbors, this
script computes

    overlap (%) = 100 * |query dimensions intersect document dimensions|
                        / |query dimensions|

The default denominator measures query-dimension coverage: the percentage of
active query dimensions also present in the ground-truth document.
``--denominator document`` provides the complement of the repository's
deadweight metric, while ``--denominator union`` provides Jaccard overlap.

The pruned version uses the same mass-ratio rule as sparse/prune.h: cast stored
weights to fp16, retain the fewest largest weights whose fp32 running sum
reaches alpha times the row sum, and restore dimension order.  Equal-weight
ties are broken by dimension ID to make the Python analysis deterministic;
C++ ``std::sort`` does not specify an order for ties.

By default all ten ground-truth neighbors are analyzed, producing one sample
per query/GT-document pair.  Use ``--gt-k 1`` for top-1-only analysis.

Examples
--------
  module load python/3.11-24.1.0
  python analyze_dimension_overlap.py
  python analyze_dimension_overlap.py --gt-k 1 --bins 50
  python analyze_dimension_overlap.py --datasets msmarco_full --msmarco-alpha 0.85
"""

import argparse
import csv
import gzip
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    style_axes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "dimension_overlap"

DATASET_DEFAULTS = {
    "msmarco_full": {
        "label": "MS MARCO (Full)",
        "directory": DATA_ROOT / "msmarco_full",
        "base": "base_full.csr",
        "queries": "queries.dev.csr",
        "gt": "base_full.dev.gt",
        "alpha": 0.85,
        # M=32, efC=400, ef=150, beta=4 reference result in
        # results/deadweight_efc/msmarco_full.
        "reference_recall": 98.3926,
    },
    "nq_splade": {
        "label": "Natural Questions (SPLADE)",
        "directory": DATA_ROOT / "nq_splade",
        "base": "base_nq.csr",
        "queries": "queries.test.csr",
        "gt": "base_nq.test.gt",
        "alpha": 0.85,
        # M=32, efC=400, ef=200, beta=4 reference result in
        # results/deadweight_efc/nqsplade.
        "reference_recall": 98.6066,
    },
}


class BigAnnCSR:
    """Read-only memory-mapped Big-ANN fp32 CSR matrix."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError("CSR matrix does not exist: {}".format(self.path))
        if self.path.stat().st_size < 24:
            raise ValueError("CSR file is too small for a header: {}".format(self.path))

        header = np.fromfile(str(self.path), dtype="<i8", count=3)
        self.nrow, self.ncol, self.nnz = (int(value) for value in header)
        if self.nrow < 0 or self.ncol < 0 or self.nnz < 0:
            raise ValueError("negative CSR header value in {}".format(self.path))

        pointer_offset = 24
        index_offset = pointer_offset + (self.nrow + 1) * 8
        value_offset = index_offset + self.nnz * 4
        expected_size = value_offset + self.nnz * 4
        if self.path.stat().st_size < expected_size:
            raise ValueError(
                "truncated CSR file {}: expected at least {:,} bytes".format(
                    self.path, expected_size
                )
            )

        self.indptr = np.memmap(
            str(self.path), dtype="<i8", mode="r", offset=pointer_offset,
            shape=(self.nrow + 1,),
        )
        self.indices = np.memmap(
            str(self.path), dtype="<i4", mode="r", offset=index_offset,
            shape=(self.nnz,),
        )
        self.data = np.memmap(
            str(self.path), dtype="<f4", mode="r", offset=value_offset,
            shape=(self.nnz,),
        )
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.nnz:
            raise ValueError("invalid CSR indptr endpoints in {}".format(self.path))

    def row(self, row_id):
        if row_id < 0 or row_id >= self.nrow:
            raise IndexError("row {} outside [0, {})".format(row_id, self.nrow))
        start = int(self.indptr[row_id])
        stop = int(self.indptr[row_id + 1])
        return self.indices[start:stop], self.data[start:stop]

    def close(self):
        for array in (self.data, self.indices, self.indptr):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class GroundTruth:
    """Memory-map the ID block of a Big-ANN ground-truth file."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file() or self.path.stat().st_size < 8:
            raise FileNotFoundError("ground-truth file is missing or invalid: {}".format(path))
        header = np.fromfile(str(self.path), dtype="<u4", count=2)
        self.nquery, self.k = (int(value) for value in header)
        id_bytes = self.nquery * self.k * 4
        if self.path.stat().st_size < 8 + id_bytes:
            raise ValueError("truncated ground-truth ID block: {}".format(self.path))
        self.ids = np.memmap(
            str(self.path), dtype="<u4", mode="r", offset=8,
            shape=(self.nquery, self.k),
        )

    def close(self):
        mmap = getattr(self.ids, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def mass_ratio_pruned_dimensions(indices, values, alpha):
    """Return dimensions retained by the repository's mass-ratio pruning."""
    if indices.size == 0 or alpha >= 1.0:
        return indices

    # CSRMatrix converts the source fp32 values to fp16 before prune.h sees
    # them. Convert back to fp32 for the same accumulation/comparison domain.
    weights = np.asarray(values, dtype=np.float16).astype(np.float32)
    dimensions = np.asarray(indices)
    total = np.cumsum(weights, dtype=np.float32)[-1]
    target = np.float32(alpha) * total

    # Descending weight, then ascending dimension as a deterministic tie-break.
    order = np.lexsort((dimensions, -weights))
    prefix = np.cumsum(weights[order], dtype=np.float32)
    kept = int(np.searchsorted(prefix, target, side="left")) + 1
    kept = min(kept, dimensions.size)
    return np.sort(dimensions[order[:kept]])


def overlap_percent(intersection, query_nnz, document_nnz, denominator):
    if denominator == "document":
        total = document_nnz
    elif denominator == "query":
        total = query_nnz
    else:
        total = query_nnz + document_nnz - intersection
    return 100.0 * intersection / total if total else float("nan")


def describe(values):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            "count": 0, "mean": float("nan"), "median": float("nan"),
            "p05": float("nan"), "p95": float("nan"), "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p05": float(np.percentile(valid, 5)),
        "p95": float(np.percentile(valid, 95)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def write_pair_csv(path, result):
    with gzip.open(str(path), "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "query_id", "gt_rank", "document_id", "query_nnz",
            "unpruned_document_nnz", "pruned_document_nnz",
            "unpruned_intersection_nnz", "pruned_intersection_nnz",
            "unpruned_overlap_pct", "pruned_overlap_pct",
        ])
        top_k = result["top_k"]
        for pair_index in range(result["doc_ids"].size):
            writer.writerow([
                pair_index // top_k,
                pair_index % top_k + 1,
                int(result["doc_ids"][pair_index]),
                int(result["query_nnz"][pair_index]),
                int(result["unpruned_doc_nnz"][pair_index]),
                int(result["pruned_doc_nnz"][pair_index]),
                int(result["unpruned_intersection"][pair_index]),
                int(result["pruned_intersection"][pair_index]),
                "{:.8f}".format(result["unpruned_pct"][pair_index]),
                "{:.8f}".format(result["pruned_pct"][pair_index]),
            ])


def analyze_dataset(spec, alpha, top_k, denominator, bins, output_dir):
    started = time.monotonic()
    directory = Path(spec["directory"]).expanduser().resolve()
    base_path = directory / spec["base"]
    query_path = directory / spec["queries"]
    gt_path = directory / spec["gt"]

    print("\n{}".format(spec["label"]))
    print("  base:    {}".format(base_path))
    print("  queries: {}".format(query_path))
    print("  GT:      {}".format(gt_path))
    print("  alpha:   {}".format(alpha))

    with BigAnnCSR(base_path) as base, BigAnnCSR(query_path) as queries, GroundTruth(gt_path) as gt:
        if base.ncol != queries.ncol:
            raise ValueError(
                "dimension mismatch for {}: base={} queries={}".format(
                    spec["slug"], base.ncol, queries.ncol
                )
            )
        if queries.nrow != gt.nquery:
            raise ValueError(
                "query/GT count mismatch for {}: {} vs {}".format(
                    spec["slug"], queries.nrow, gt.nquery
                )
            )
        if top_k > gt.k:
            raise ValueError(
                "--gt-k {} exceeds {}'s ground-truth k={}".format(
                    top_k, spec["slug"], gt.k
                )
            )

        doc_ids = np.asarray(gt.ids[:, :top_k]).reshape(-1).copy()
        if doc_ids.size and int(doc_ids.max()) >= base.nrow:
            raise ValueError("ground-truth document ID exceeds base row count")

        pair_count = queries.nrow * top_k
        query_nnz = np.empty(pair_count, dtype=np.uint32)
        unpruned_doc_nnz = np.empty(pair_count, dtype=np.uint32)
        pruned_doc_nnz = np.empty(pair_count, dtype=np.uint32)
        unpruned_intersection = np.empty(pair_count, dtype=np.uint32)
        pruned_intersection = np.empty(pair_count, dtype=np.uint32)
        unpruned_pct = np.empty(pair_count, dtype=np.float64)
        pruned_pct = np.empty(pair_count, dtype=np.float64)

        query_mask = np.zeros(base.ncol, dtype=np.bool_)
        for query_id in range(queries.nrow):
            query_dimensions, _ = queries.row(query_id)
            if query_dimensions.size:
                if int(query_dimensions.min()) < 0 or int(query_dimensions.max()) >= base.ncol:
                    raise ValueError("query {} has an out-of-range dimension".format(query_id))
                query_mask[query_dimensions] = True
            # Count unique active dimensions; CSR inputs are normally unique,
            # but the boolean mask makes the definition robust to duplicates.
            query_active = int(np.count_nonzero(query_mask))

            for rank in range(top_k):
                pair_index = query_id * top_k + rank
                document_id = int(gt.ids[query_id, rank])
                doc_dimensions, doc_values = base.row(document_id)
                if doc_dimensions.size and (
                    int(doc_dimensions.min()) < 0 or int(doc_dimensions.max()) >= base.ncol
                ):
                    raise ValueError("document {} has an out-of-range dimension".format(document_id))

                pruned_dimensions = mass_ratio_pruned_dimensions(
                    doc_dimensions, doc_values, alpha
                )
                full_overlap = int(np.count_nonzero(query_mask[doc_dimensions]))
                pruned_overlap = int(np.count_nonzero(query_mask[pruned_dimensions]))

                query_nnz[pair_index] = query_active
                unpruned_doc_nnz[pair_index] = doc_dimensions.size
                pruned_doc_nnz[pair_index] = pruned_dimensions.size
                unpruned_intersection[pair_index] = full_overlap
                pruned_intersection[pair_index] = pruned_overlap
                unpruned_pct[pair_index] = overlap_percent(
                    full_overlap, query_active, doc_dimensions.size, denominator
                )
                pruned_pct[pair_index] = overlap_percent(
                    pruned_overlap, query_active, pruned_dimensions.size, denominator
                )

            if query_dimensions.size:
                query_mask[query_dimensions] = False
            if (query_id + 1) % 500 == 0 or query_id + 1 == queries.nrow:
                print(
                    "  queries: {:,}/{:,}".format(query_id + 1, queries.nrow),
                    end="\r" if query_id + 1 < queries.nrow else "\n",
                    flush=True,
                )

        base_shape = (base.nrow, base.ncol)
        query_count = queries.nrow
        gt_available_k = gt.k

    unpruned_stats = describe(unpruned_pct)
    pruned_stats = describe(pruned_pct)
    result = {
        "slug": spec["slug"],
        "label": spec["label"],
        "base_path": base_path,
        "query_path": query_path,
        "gt_path": gt_path,
        "base_shape": base_shape,
        "query_count": query_count,
        "gt_available_k": gt_available_k,
        "top_k": top_k,
        "alpha": alpha,
        "reference_recall": spec.get("reference_recall") if abs(alpha - 0.85) < 1e-7 else None,
        "doc_ids": doc_ids,
        "query_nnz": query_nnz,
        "unpruned_doc_nnz": unpruned_doc_nnz,
        "pruned_doc_nnz": pruned_doc_nnz,
        "unpruned_intersection": unpruned_intersection,
        "pruned_intersection": pruned_intersection,
        "unpruned_pct": unpruned_pct,
        "pruned_pct": pruned_pct,
        "unpruned_stats": unpruned_stats,
        "pruned_stats": pruned_stats,
        "elapsed": time.monotonic() - started,
    }

    edges = np.linspace(0.0, 100.0, bins + 1)
    unpruned_counts, _ = np.histogram(unpruned_pct[np.isfinite(unpruned_pct)], bins=edges)
    pruned_counts, _ = np.histogram(pruned_pct[np.isfinite(pruned_pct)], bins=edges)
    result["edges"] = edges
    result["unpruned_counts"] = unpruned_counts
    result["pruned_counts"] = pruned_counts

    pair_csv = output_dir / "{}_pair_overlaps.csv.gz".format(spec["slug"])
    write_pair_csv(pair_csv, result)
    histogram_csv = output_dir / "{}_histogram.csv".format(spec["slug"])
    with histogram_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bin_lo_pct", "bin_hi_pct", "unpruned_count", "pruned_count"])
        for index in range(bins):
            writer.writerow([
                "{:.6f}".format(edges[index]),
                "{:.6f}".format(edges[index + 1]),
                int(unpruned_counts[index]),
                int(pruned_counts[index]),
            ])

    print(
        "  pairs: {:,}; mean overlap {:.2f}% unpruned -> {:.2f}% pruned; {:.2f}s".format(
            pair_count, unpruned_stats["mean"], pruned_stats["mean"], result["elapsed"]
        )
    )
    return result


def plot_results(results, denominator, log_y, output_path):
    figure, axes = plt.subplots(
        1, len(results), figsize=(7.1 * len(results), 5.4), squeeze=False,
        sharex=True,
    )
    figure.patch.set_facecolor(SURFACE)
    colors = (PALETTE[0], PALETTE[1])

    for index, result in enumerate(results):
        axis = axes[0, index]
        style_axes(axis)
        edges = result["edges"]
        full_counts = result["unpruned_counts"]
        pruned_counts = result["pruned_counts"]

        axis.stairs(
            full_counts, edges, color=colors[0], linewidth=2.0,
            fill=True, alpha=0.16,
            label="Unpruned",
        )
        axis.stairs(
            pruned_counts, edges, color=colors[1], linewidth=2.0,
            fill=True, alpha=0.16,
            label="Pruned, α={}".format(result["alpha"]),
        )
        axis.axvline(
            result["unpruned_stats"]["mean"], color=colors[0],
            linestyle=(0, (4, 3)), linewidth=1.4,
        )
        axis.axvline(
            result["pruned_stats"]["mean"], color=colors[1],
            linestyle=(0, (4, 3)), linewidth=1.4,
        )
        axis.set_xlim(0, 100)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("Dimension overlap (%)", color=INK_SECONDARY)
        axis.set_ylabel("Query-GT pair count", color=INK_SECONDARY)
        axis.set_title(result["label"], color=INK_PRIMARY, fontsize=13, pad=9)
        legend = axis.legend(frameon=False, fontsize=9, loc="upper right")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
        full_stats = result["unpruned_stats"]
        pruned_stats = result["pruned_stats"]
        stats_text = (
            "Unpruned: min {fmin:.2f}%  |  mean {fmean:.2f}%  |  max {fmax:.2f}%\n"
            "Pruned:    min {pmin:.2f}%  |  mean {pmean:.2f}%  |  max {pmax:.2f}%"
        ).format(
            fmin=full_stats["min"], fmean=full_stats["mean"], fmax=full_stats["max"],
            pmin=pruned_stats["min"], pmean=pruned_stats["mean"], pmax=pruned_stats["max"],
        )
        axis.text(
            0.02, 0.97, stats_text,
            transform=axis.transAxes, ha="left", va="top", fontsize=8.2,
            color=INK_SECONDARY, linespacing=1.45,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE,
                  "edgecolor": "none", "alpha": 0.88},
        )

    denominator_text = {
        "document": "|query intersect document| / |document| (deadweight complement)",
        "query": "|query intersect document| / |query| (query coverage)",
        "union": "|query intersect document| / |query union document| (Jaccard)",
    }[denominator]
    figure.suptitle(
        "Query-Ground-Truth Dimension Overlap",
        color=INK_PRIMARY, fontsize=17, fontweight="bold", y=0.985,
    )
    figure.text(
        0.01, 0.012,
        "One sample per query/GT-document pair. Overlap = 100 x {}. "
        "Dashed lines mark means.".format(denominator_text),
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    figure.tight_layout(rect=[0, 0.045, 1, 0.935])
    figure.savefig(str(output_path), dpi=200, facecolor=SURFACE)
    plt.close(figure)


def format_stats(name, stats):
    return [
        "{}:".format(name),
        "  samples: {:,}".format(stats["count"]),
        "  mean:    {:.6f}%".format(stats["mean"]),
        "  median:  {:.6f}%".format(stats["median"]),
        "  p05:     {:.6f}%".format(stats["p05"]),
        "  p95:     {:.6f}%".format(stats["p95"]),
        "  min:     {:.6f}%".format(stats["min"]),
        "  max:     {:.6f}%".format(stats["max"]),
    ]


def create_output_dir(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / "dimension_overlap_{}".format(stamp)
    suffix = 1
    while output.exists():
        output = root / "dimension_overlap_{}_{:02d}".format(stamp, suffix)
        suffix += 1
    output.mkdir()
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze query/ground-truth dimension overlap before and after pruning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASET_DEFAULTS),
        default=sorted(DATASET_DEFAULTS), help="datasets to analyze",
    )
    parser.add_argument(
        "--msmarco-dir", type=Path,
        default=DATASET_DEFAULTS["msmarco_full"]["directory"],
        help="directory containing MS MARCO base, queries, and GT",
    )
    parser.add_argument(
        "--nq-dir", type=Path,
        default=DATASET_DEFAULTS["nq_splade"]["directory"],
        help="directory containing NQ SPLADE base, queries, and GT",
    )
    parser.add_argument("--msmarco-alpha", type=float, default=0.85)
    parser.add_argument("--nq-alpha", type=float, default=0.85)
    parser.add_argument(
        "--gt-k", type=int, default=10,
        help="number of ground-truth neighbors analyzed per query",
    )
    parser.add_argument(
        "--denominator", choices=("document", "query", "union"),
        default="query", help="denominator used for overlap percentage",
    )
    parser.add_argument("--bins", type=int, default=100, help="number of 0-100%% bins")
    parser.add_argument("--log-y", action="store_true", help="use a logarithmic count axis")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="parent of the timestamped result directory",
    )
    args = parser.parse_args()
    if args.gt_k <= 0 or args.bins <= 0:
        parser.error("--gt-k and --bins must be positive")
    for name, alpha in (("--msmarco-alpha", args.msmarco_alpha), ("--nq-alpha", args.nq_alpha)):
        if not 0.0 < alpha <= 1.0:
            parser.error("{} must be in (0, 1]".format(name))
    return args


def main():
    args = parse_args()
    output_dir = create_output_dir(args.output_root)
    print("Output: {}".format(output_dir))

    specs = {}
    for slug in args.datasets:
        spec = dict(DATASET_DEFAULTS[slug])
        spec["slug"] = slug
        spec["directory"] = args.msmarco_dir if slug == "msmarco_full" else args.nq_dir
        specs[slug] = spec

    results = []
    try:
        for slug in args.datasets:
            alpha = args.msmarco_alpha if slug == "msmarco_full" else args.nq_alpha
            results.append(
                analyze_dataset(
                    specs[slug], alpha, args.gt_k, args.denominator,
                    args.bins, output_dir,
                )
            )

        figure_path = output_dir / "dimension_overlap_comparison.png"
        plot_results(results, args.denominator, args.log_y, figure_path)

        summary = [
            "Query-ground-truth dimension overlap analysis",
            "=============================================",
            "One sample per query/ground-truth-document pair.",
            "GT neighbors per query: {}".format(args.gt_k),
            "Overlap denominator: {}".format(args.denominator),
            "Pruning: fp16 mass-ratio pruning matching sparse/prune.h; deterministic dim-ID tie-break.",
            "Recall notes are reference HNSW runs; this script analyzes exact GT pairs and does not run ANN search.",
            "",
        ]
        for result in results:
            summary.extend([
                result["label"],
                "-" * len(result["label"]),
                "Base: {}".format(result["base_path"]),
                "Queries: {}".format(result["query_path"]),
                "Ground truth: {}".format(result["gt_path"]),
                "Base shape: {:,} x {:,}".format(*result["base_shape"]),
                "Queries: {:,}".format(result["query_count"]),
                "Pairs: {:,}".format(result["query_count"] * result["top_k"]),
                "Alpha: {}".format(result["alpha"]),
                "Reference ANN recall: {}".format(
                    "{:.4f}%".format(result["reference_recall"])
                    if result["reference_recall"] is not None
                    else "not supplied for this alpha"
                ),
                "Mean GT-document nnz: {:.6f} unpruned -> {:.6f} pruned".format(
                    float(np.mean(result["unpruned_doc_nnz"])),
                    float(np.mean(result["pruned_doc_nnz"])),
                ),
                "Elapsed seconds: {:.3f}".format(result["elapsed"]),
                "",
                *format_stats("Unpruned overlap", result["unpruned_stats"]),
                "",
                *format_stats("Pruned overlap", result["pruned_stats"]),
                "",
            ])
        summary.extend([
            "Outputs",
            "-------",
            figure_path.name,
            *["{}_histogram.csv".format(result["slug"]) for result in results],
            *["{}_pair_overlaps.csv.gz".format(result["slug"]) for result in results],
        ])
        summary_path = output_dir / "summary.txt"
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("\nSaved comparison figure and data to {}".format(output_dir))
        return 0
    except (OSError, ValueError, IndexError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
