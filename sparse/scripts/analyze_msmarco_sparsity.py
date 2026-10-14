#!/usr/bin/env python3
"""Analyze row- and column-wise sparsity of a large sparse dataset.

The default input may be either a matrix file or a directory.  For a
directory, a unique ``base*.npz`` or ``base*.csr`` matrix is selected while
query matrices are ignored.  Pass the matrix file directly if a directory has
multiple base matrices.

Supported formats
-----------------
* SciPy CSR/CSC ``.npz`` files.  To keep memory bounded, only the structural
  arrays (``indptr`` and ``indices``) are extracted to temporary files and
  memory-mapped; the usually much larger ``data`` array is never loaded.
* Big-ANN CSR binary files, including this repository's MS MARCO and NQ files:
  int64 nrow, int64 ncol, int64 nnz, int64 indptr[nrow+1],
  int32 indices[nnz], float32 data[nnz].  The arrays are memory-mapped.

Counts are *structural* nnz counts, matching SciPy's ``getnnz`` semantics.
That is, explicitly stored zero values count as entries.  If another input
format is used, adapt ``open_sparse_structure``; the analysis only needs a
compressed axis pointer and the indices along the other axis.

Examples
--------
    python analyze_msmarco_sparsity.py
    python analyze_msmarco_sparsity.py --input ../data/nq_splade
    python analyze_msmarco_sparsity.py --input ../data/my_dataset/base.npz
    python analyze_msmarco_sparsity.py --chunk-nnz 5000000 --bins 80
"""

import argparse
import io
import re
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

# Select a non-interactive backend before importing pyplot (important on HPC).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "data" / "msmarco_full"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "sparsity_analysis"


class SparseStructure:
    """Structural arrays needed to compute nnz distributions."""

    def __init__(self, shape, nnz, sparse_format, indptr, indices):
        # A small explicit class keeps this script usable with Python 3.6,
        # which is still the default Python executable on some HPC systems.
        self.shape = shape
        self.nnz = nnz
        self.format = sparse_format
        self.indptr = indptr
        self.indices = indices


def resolve_input(path: Path) -> Path:
    """Resolve a file or choose the base matrix from a dataset directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".npz", ".csr"}:
            raise ValueError(f"unsupported input extension: {path.suffix!r}")
        return path
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"input is neither a file nor a directory: {path}")

    # Prefer common exact names, then a unique base-prefixed matrix.  This
    # selects base_full.csr in MS MARCO and base_nq.csr in NQ SPLADE without
    # accidentally analyzing the much smaller query matrix.
    for name in ("base.npz", "base.csr", "base_full.npz", "base_full.csr"):
        candidate = path / name
        if candidate.is_file():
            return candidate

    candidates = sorted(path.glob("*.npz")) + sorted(path.glob("*.csr"))
    base_candidates = [
        candidate
        for candidate in candidates
        if candidate.stem.lower() == "base"
        or candidate.stem.lower().startswith("base_")
        or candidate.stem.lower().startswith("base-")
    ]
    if len(base_candidates) == 1:
        return base_candidates[0]
    if len(base_candidates) > 1:
        names = ", ".join(p.name for p in base_candidates)
        raise ValueError(
            f"multiple base matrices found in {path}; pass --input with one file: {names}"
        )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no .npz or .csr matrix found in {path}")
    names = ", ".join(p.name for p in candidates)
    raise ValueError(
        f"multiple matrices found in {path}; pass --input with one file: {names}"
    )


def dataset_identity(
    requested_input: Path, resolved_input: Path, override: str
) -> Tuple[str, str]:
    """Return human-readable and filesystem-safe names for the dataset."""
    if override:
        display_name = override.strip()
        if not display_name:
            raise ValueError("--dataset-name must not be blank")
    else:
        requested = requested_input.expanduser().resolve()
        # A directory name generally carries the dataset identity.  For a
        # directly supplied file, use its parent unless it has no useful name.
        display_name = requested.name if requested.is_dir() else resolved_input.parent.name
        if not display_name or display_name.lower() in {"data", "datasets"}:
            display_name = resolved_input.stem

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip("._-").lower()
    if not slug:
        slug = "dataset"
    return display_name, slug


def make_output_dir(root: Path, dataset_slug: str) -> Path:
    """Create and return a collision-resistant timestamped result directory."""
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{dataset_slug}_sparsity_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{dataset_slug}_sparsity_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _load_small_npy(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    """Load a small metadata .npy member from a SciPy sparse archive."""
    with archive.open(member) as src:
        return np.load(io.BytesIO(src.read()), allow_pickle=False)


def _npz_member(archive: zipfile.ZipFile, basename: str) -> str:
    """Find a member by basename, tolerating an optional archive prefix."""
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {basename!r} in sparse .npz; found {len(matches)}"
        )
    return matches[0]


@contextmanager
def open_npz_structure(path: Path, temp_parent: Path) -> Iterator[SparseStructure]:
    """Open a SciPy CSR/CSC .npz without loading all sparse data into RAM."""
    with zipfile.ZipFile(path, "r") as archive:
        format_member = _npz_member(archive, "format.npy")
        shape_member = _npz_member(archive, "shape.npy")
        raw_format = _load_small_npy(archive, format_member).item()
        if isinstance(raw_format, bytes):
            sparse_format = raw_format.decode("ascii")
        else:
            sparse_format = str(raw_format)
        sparse_format = sparse_format.lower()
        if sparse_format not in {"csr", "csc"}:
            raise ValueError(
                f"{path} stores {sparse_format!r}, but only CSR/CSC .npz files "
                "are supported. Convert it with matrix.tocsr() and scipy.sparse.save_npz()."
            )

        shape_array = _load_small_npy(archive, shape_member)
        if shape_array.size != 2:
            raise ValueError(f"invalid sparse shape in {path}: {shape_array!r}")
        shape = tuple(int(value) for value in shape_array)

        # Compressed .npz members cannot be memory-mapped in place.  Extract
        # only structure (not data.npy) to scratch space, then mmap the .npy
        # payloads.  Temporary files are deleted when analysis finishes.
        with tempfile.TemporaryDirectory(
            prefix="sparse_structure_", dir=temp_parent
        ) as temp_name:
            temp_dir = Path(temp_name)
            extracted = {}
            for basename in ("indptr.npy", "indices.npy"):
                member = _npz_member(archive, basename)
                destination = temp_dir / basename
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                extracted[basename] = destination

            indptr = np.load(extracted["indptr.npy"], mmap_mode="r", allow_pickle=False)
            indices = np.load(extracted["indices.npy"], mmap_mode="r", allow_pickle=False)
            if indptr.ndim != 1 or indices.ndim != 1:
                raise ValueError("sparse indptr and indices arrays must be one-dimensional")
            expected_ptrs = shape[0] + 1 if sparse_format == "csr" else shape[1] + 1
            if indptr.size != expected_ptrs:
                raise ValueError(
                    f"invalid {sparse_format.upper()} indptr length {indptr.size}; "
                    f"expected {expected_ptrs}"
                )
            yield SparseStructure(shape, int(indices.size), sparse_format, indptr, indices)


@contextmanager
def open_bigann_csr(path: Path) -> Iterator[SparseStructure]:
    """Memory-map a Big-ANN fp32 CSR binary file's structural arrays."""
    file_size = path.stat().st_size
    if file_size < 24:
        raise ValueError(f"{path} is too small to contain a Big-ANN CSR header")
    header = np.fromfile(path, dtype="<i8", count=3)
    nrow, ncol, nnz = (int(value) for value in header)
    if nrow < 0 or ncol < 0 or nnz < 0:
        raise ValueError(f"negative value in CSR header: {(nrow, ncol, nnz)}")

    indptr_offset = 3 * np.dtype("<i8").itemsize
    indices_offset = indptr_offset + (nrow + 1) * np.dtype("<i8").itemsize
    data_offset = indices_offset + nnz * np.dtype("<i4").itemsize
    expected_size = data_offset + nnz * np.dtype("<f4").itemsize
    if file_size < expected_size:
        raise ValueError(
            f"truncated Big-ANN CSR file: {file_size:,} bytes, expected at least "
            f"{expected_size:,} from its header"
        )

    indptr = np.memmap(
        path, dtype="<i8", mode="r", offset=indptr_offset, shape=(nrow + 1,)
    )
    indices = np.memmap(
        path, dtype="<i4", mode="r", offset=indices_offset, shape=(nnz,)
    )
    try:
        yield SparseStructure((nrow, ncol), nnz, "csr", indptr, indices)
    finally:
        # Release mmap file descriptors promptly on long-lived Python workers.
        del indices
        del indptr


@contextmanager
def open_sparse_structure(
    path: Path, temp_parent: Path
) -> Iterator[SparseStructure]:
    """Dispatch to the appropriate memory-efficient sparse reader."""
    if path.suffix.lower() == ".npz":
        with open_npz_structure(path, temp_parent) as matrix:
            yield matrix
    elif path.suffix.lower() == ".csr":
        with open_bigann_csr(path) as matrix:
            yield matrix
    else:  # resolve_input normally catches this; keep the API defensive.
        raise ValueError(f"unsupported sparse matrix extension: {path.suffix}")


def compressed_axis_distribution(
    indptr: np.ndarray, expected_nnz: int, chunk_items: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nnz value, frequency) pairs from a CSR/CSC pointer array."""
    if int(indptr[0]) != 0:
        raise ValueError(f"invalid indptr start: expected 0, found {int(indptr[0])}")
    if int(indptr[-1]) != expected_nnz:
        raise ValueError(
            f"indptr ends at {int(indptr[-1]):,}, but indices has "
            f"{expected_nnz:,} entries"
        )

    frequencies = {}  # type: Dict[int, int]
    item_count = indptr.size - 1
    for start in range(0, item_count, chunk_items):
        stop = min(start + chunk_items, item_count)
        # np.asarray keeps the mmap slice cheap; np.diff is the bounded buffer.
        counts = np.diff(np.asarray(indptr[start : stop + 1]))
        if counts.size and np.any(counts < 0):
            bad = start + int(np.flatnonzero(counts < 0)[0])
            raise ValueError(f"indptr is not monotonic near compressed item {bad}")
        values, occurrences = np.unique(counts, return_counts=True)
        for value, occurrence in zip(values, occurrences):
            key = int(value)
            frequencies[key] = frequencies.get(key, 0) + int(occurrence)
        print(
            f"  compressed-axis counts: {stop:,}/{item_count:,}",
            end="\r" if stop < item_count else "\n",
            flush=True,
        )

    values = np.array(sorted(frequencies), dtype=np.int64)
    occurrence_array = np.array([frequencies[int(v)] for v in values], dtype=np.int64)
    return values, occurrence_array


def indexed_axis_distribution(
    indices: np.ndarray, axis_size: int, chunk_nnz: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Count indices in bounded chunks, then return value/frequency pairs."""
    counts = np.zeros(axis_size, dtype=np.uint64)
    nnz = indices.size
    for start in range(0, nnz, chunk_nnz):
        stop = min(start + chunk_nnz, nnz)
        chunk = np.asarray(indices[start:stop])
        if chunk.size:
            low, high = int(chunk.min()), int(chunk.max())
            if low < 0 or high >= axis_size:
                raise ValueError(
                    f"index out of bounds in entries {start:,}:{stop:,}: "
                    f"range [{low}, {high}], axis size {axis_size:,}"
                )
            counts += np.bincount(chunk, minlength=axis_size).astype(
                np.uint64, copy=False
            )
        print(
            f"  indexed-axis counts: {stop:,}/{nnz:,}",
            end="\r" if stop < nnz else "\n",
            flush=True,
        )
    values, frequencies = np.unique(counts, return_counts=True)
    return values.astype(np.uint64, copy=False), frequencies.astype(np.int64, copy=False)


def weighted_quantile(values: np.ndarray, frequencies: np.ndarray, q: float) -> float:
    """Compute NumPy-style linear quantiles from a compact frequency table."""
    total = int(frequencies.sum())
    if total == 0:
        return float("nan")
    position = q * (total - 1)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    cumulative = np.cumsum(frequencies, dtype=np.int64)
    lower = float(values[np.searchsorted(cumulative, lower_rank, side="right")])
    upper = float(values[np.searchsorted(cumulative, upper_rank, side="right")])
    return lower + (position - lower_rank) * (upper - lower)


def distribution_stats(values: np.ndarray, frequencies: np.ndarray) -> Dict[str, float]:
    """Summarize a compact integer frequency distribution."""
    count = int(frequencies.sum())
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "zeros": 0,
        }
    weighted_sum = sum(int(v) * int(f) for v, f in zip(values, frequencies))
    zero_positions = np.flatnonzero(values == 0)
    zero_count = int(frequencies[zero_positions[0]]) if zero_positions.size else 0
    return {
        "count": count,
        "mean": weighted_sum / count,
        "median": weighted_quantile(values, frequencies, 0.5),
        "max": float(values[-1]),
        "min": float(values[0]),
        "p95": weighted_quantile(values, frequencies, 0.95),
        "p99": weighted_quantile(values, frequencies, 0.99),
        "zeros": zero_count,
    }


def plot_distribution(
    values: np.ndarray,
    frequencies: np.ndarray,
    *,
    axis_name: str,
    title: str,
    output_path: Path,
    bins: int,
    log_x: bool,
    log_y: bool,
) -> None:
    """Plot a weighted histogram directly from a compact frequency table."""
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    sample_count = int(frequencies.sum())
    mean = (
        sum(int(value) * int(frequency) for value, frequency in zip(values, frequencies))
        / sample_count
        if sample_count
        else float("nan")
    )
    values_to_plot = values
    frequencies_to_plot = frequencies
    zero_count = 0

    if log_x:
        positive = values > 0
        zero_count = int(frequencies[~positive].sum())
        values_to_plot = values[positive]
        frequencies_to_plot = frequencies[positive]

    if values_to_plot.size:
        low, high = float(values_to_plot[0]), float(values_to_plot[-1])
        if low == high:
            edges = np.array([max(0.0, low - 0.5), low + 0.5])
        elif log_x:
            edges = np.geomspace(low, high * (1.0 + 1e-9), bins + 1)
        elif high - low <= bins:
            edges = np.arange(low - 0.5, high + 1.5)
        else:
            edges = np.linspace(low, high, bins + 1)
        ax.hist(
            values_to_plot,
            bins=edges,
            weights=frequencies_to_plot,
            color="#2a78d6",
            edgecolor="white",
            linewidth=0.35,
        )
        if log_x and low != high:
            ax.set_xscale("log")

    # The mean is computed from the full distribution, including zero-nnz
    # items omitted from a logarithmic x-axis.
    if np.isfinite(mean) and (not log_x or mean > 0):
        ax.axvline(
            mean,
            color="#e34948",
            linestyle="--",
            linewidth=2.0,
            label=f"Mean = {mean:,.2f}",
            zorder=3,
        )
        ax.legend(frameon=False, loc="upper right")

    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(f"Non-zero elements per {axis_name}")
    ax.set_ylabel(f"Number of {axis_name}s")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if log_x and zero_count:
        ax.text(
            0.02,
            0.96,
            f"Zero-nnz {axis_name}s (not on log x-axis): {zero_count:,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_stats(name: str, stats: Dict[str, float]):
    """Format one distribution's statistics for the text report."""
    return [
        f"{name}:",
        f"  count:  {int(stats['count']):,}",
        f"  mean:   {stats['mean']:.6f}",
        f"  median: {stats['median']:.6f}",
        f"  max:    {stats['max']:.0f}",
        f"  min:    {stats['min']:.0f}",
        f"  p95:    {stats['p95']:.6f}",
        f"  p99:    {stats['p99']:.6f}",
        f"  zero-nnz count: {int(stats['zeros']):,}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze row- and column-wise nnz distributions of a sparse dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="SciPy CSR/CSC .npz, Big-ANN .csr, or directory containing the base matrix",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="name used in plot titles and output directory (inferred by default)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="parent directory for the timestamped result directory",
    )
    parser.add_argument(
        "--chunk-nnz",
        type=int,
        default=10_000_000,
        help="number of sparse indices processed per chunk",
    )
    parser.add_argument(
        "--chunk-items",
        type=int,
        default=1_000_000,
        help="number of rows/columns processed per indptr chunk",
    )
    parser.add_argument("--bins", type=int, default=100, help="maximum histogram bins")
    parser.add_argument(
        "--linear-y", action="store_true", help="use a linear instead of logarithmic y-axis"
    )
    args = parser.parse_args()
    if args.chunk_nnz <= 0 or args.chunk_items <= 0 or args.bins <= 0:
        parser.error("--chunk-nnz, --chunk-items, and --bins must be positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    try:
        input_path = resolve_input(args.input)
        dataset_name, dataset_slug = dataset_identity(
            args.input, input_path, args.dataset_name
        )
        output_dir = make_output_dir(args.output_root, dataset_slug)
        print(f"Dataset: {dataset_name}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_dir}")

        with open_sparse_structure(input_path, output_dir) as matrix:
            nrows, ncols = matrix.shape
            print(
                f"Matrix: format={matrix.format.upper()}, shape=({nrows:,}, {ncols:,}), "
                f"structural nnz={matrix.nnz:,}"
            )
            compressed_values, compressed_frequencies = compressed_axis_distribution(
                matrix.indptr, matrix.nnz, args.chunk_items
            )
            indexed_size = ncols if matrix.format == "csr" else nrows
            indexed_values, indexed_frequencies = indexed_axis_distribution(
                matrix.indices, indexed_size, args.chunk_nnz
            )

            if matrix.format == "csr":
                row_values, row_frequencies = compressed_values, compressed_frequencies
                column_values, column_frequencies = indexed_values, indexed_frequencies
            else:
                row_values, row_frequencies = indexed_values, indexed_frequencies
                column_values, column_frequencies = compressed_values, compressed_frequencies

        row_stats = distribution_stats(row_values, row_frequencies)
        column_stats = distribution_stats(column_values, column_frequencies)

        row_figure = output_dir / "row_nnz_distribution.png"
        column_figure = output_dir / "column_nnz_distribution.png"
        plot_distribution(
            row_values,
            row_frequencies,
            axis_name="row",
            title=f"{dataset_name}: row-wise nnz distribution",
            output_path=row_figure,
            bins=args.bins,
            log_x=False,
            log_y=not args.linear_y,
        )
        # Feature frequencies are typically highly skewed, so a logarithmic
        # x-axis exposes the full column distribution.  Zero-nnz columns are
        # reported in an annotation and in the text summary.
        plot_distribution(
            column_values,
            column_frequencies,
            axis_name="column",
            title=f"{dataset_name}: column-wise nnz distribution",
            output_path=column_figure,
            bins=args.bins,
            log_x=True,
            log_y=not args.linear_y,
        )

        elapsed = time.monotonic() - started
        summary_lines = [
            "Sparse matrix nnz analysis",
            "==========================",
            f"Dataset: {dataset_name}",
            f"Input: {input_path}",
            f"Sparse format: {matrix.format.upper()}",
            f"Shape: {nrows:,} rows x {ncols:,} columns",
            f"Structural nnz: {matrix.nnz:,}",
            "Definition: structural/stored entries (explicit stored zeros count as nnz)",
            f"Elapsed seconds: {elapsed:.3f}",
            "",
            *format_stats("Row-wise nnz", row_stats),
            "",
            *format_stats("Column-wise nnz", column_stats),
            "",
            "Figures:",
            f"  {row_figure.name}",
            f"  {column_figure.name}",
        ]
        summary_path = output_dir / "summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        print("\n".join(format_stats("Row-wise nnz", row_stats)))
        print("\n".join(format_stats("Column-wise nnz", column_stats)))
        print(f"Saved figures and summary to {output_dir}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
