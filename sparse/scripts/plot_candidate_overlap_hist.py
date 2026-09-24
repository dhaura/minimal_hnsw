#!/usr/bin/env python3
"""Plot query/document overlap split by candidate-queue insertion.

The input is the all-query histogram emitted by ``sparse_hnsw_demo_distlog``.
The standard output figure has one panel for documents added to the pending
expansion queue and one for documents rejected from it. For query-normalized
histograms, ``--retention-rate`` additionally plots the percentage retained
within each overlap bin: ``100 * added_count / count``.

Examples
--------
  python3 plot_candidate_overlap_hist.py \
      --indir results/candidate_overlap --dataset msmarco_full
  python3 plot_candidate_overlap_hist.py --hist path/to/run_hist.csv
"""

import argparse
import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, style_axes


COLORS = {
    "added": "#1baf7a",
    "not_added": "#eb6834",
}

DATASET_LABELS = {
    "msmarco_full": "MS MARCO (Full)",
    "nq_splade": "Natural Questions (SPLADE)",
}

FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_M(?P<M>\d+)_efc(?P<efc>\d+)_ef(?P<ef>\d+)_"
    r"a(?P<alpha>[\d.]+)_(?P<matrix>unpruned|pruned)"
    r"(?P<query>_query)?_hist\.csv$"
)

REQUIRED_COLUMNS = {
    "bin_lo",
    "bin_hi",
    "count",
    "added_count",
    "not_added_count",
}


def pretty_dataset(dataset):
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def discover(indir, dataset=None, matrix=None, normalization=None):
    runs = []
    pattern = os.path.join(os.path.abspath(indir), "**", "*_hist.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        match = FNAME_RE.match(os.path.basename(path))
        if not match:
            continue
        fields = match.groupdict()
        if dataset and fields["dataset"] != dataset:
            continue
        if matrix and fields["matrix"] != matrix:
            continue
        run_normalization = "query" if fields["query"] else "document"
        if normalization and run_normalization != normalization:
            continue
        runs.append(
            {
                "path": path,
                "dataset": fields["dataset"],
                "M": int(fields["M"]),
                "efc": int(fields["efc"]),
                "ef": int(fields["ef"]),
                "alpha": float(fields["alpha"]),
                "matrix": fields["matrix"],
                "normalization": run_normalization,
            }
        )
    return runs


def run_from_path(path):
    match = FNAME_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(
            "histogram filename must end in "
            "<dataset>_M<M>_efc<efC>_ef<ef>_a<alpha>_"
            "<unpruned|pruned>[_query]_hist.csv"
        )
    fields = match.groupdict()
    return {
        "path": os.path.abspath(path),
        "dataset": fields["dataset"],
        "M": int(fields["M"]),
        "efc": int(fields["efc"]),
        "ef": int(fields["ef"]),
        "alpha": float(fields["alpha"]),
        "matrix": fields["matrix"],
        "normalization": "query" if fields["query"] else "document",
    }


def load(run):
    frame = pd.read_csv(run["path"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "{} lacks candidate-split columns: {}. Re-run with the updated "
            "dist-log binary.".format(run["path"], ", ".join(sorted(missing)))
        )
    if not np.array_equal(
        frame["count"].to_numpy(),
        (frame["added_count"] + frame["not_added_count"]).to_numpy(),
    ):
        raise ValueError("{} has inconsistent split counts".format(run["path"]))

    frame["overlap_lo"] = frame["bin_lo"]
    frame["overlap_hi"] = frame["bin_hi"]
    frame["overlap_mid"] = (frame["overlap_lo"] + frame["overlap_hi"]) / 2.0
    frame = frame.sort_values("overlap_lo").reset_index(drop=True)

    run["df"] = frame
    run["added_total"] = int(frame["added_count"].sum())
    run["not_added_total"] = int(frame["not_added_count"].sum())
    run["total"] = run["added_total"] + run["not_added_total"]
    if run["total"] == 0:
        raise ValueError("{} is empty".format(run["path"]))

    if run["normalization"] == "query":
        result_path = run["path"].replace("_query_hist.csv", "_result.csv")
    else:
        result_path = run["path"].replace("_hist.csv", "_result.csv")
    run["recall"] = None
    if os.path.exists(result_path):
        try:
            run["recall"] = float(pd.read_csv(result_path)["recall"].iloc[-1])
        except (KeyError, ValueError, IndexError, pd.errors.EmptyDataError):
            pass
    return run


def low_overlap_count(frame, column, threshold):
    # The standard 100-bin output has an exact boundary at 10%. For unusual
    # bin counts, conservatively include only bins wholly below the threshold.
    return int(frame.loc[frame["overlap_hi"] <= threshold, column].sum())


def x_limit(frame, columns, keep=0.995):
    upper = 20.0
    for column in columns:
        counts = frame[column].to_numpy(dtype=np.float64)
        total = counts.sum()
        if total <= 0:
            continue
        index = min(int(np.searchsorted(np.cumsum(counts) / total, keep)), len(frame) - 1)
        upper = max(upper, float(frame["overlap_hi"].iloc[index]))
    return min(100.0, math.ceil(upper / 10.0) * 10.0)


def draw(run, outdir, threshold, gate_threshold=None,
         distance_calculated_only=False):
    frame = run["df"]
    groups = [
        ("added_count", "Added for future expansion", "added", run["added_total"]),
        ("not_added_count", "Not added", "not_added", run["not_added_total"]),
    ]
    xmax = x_limit(frame, [item[0] for item in groups])
    ymax = 0.0
    for column, _, _, total in groups:
        if total:
            ymax = max(ymax, float((100.0 * frame[column] / total).max()))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    widths = frame["overlap_hi"] - frame["overlap_lo"]

    for axis, (column, title, color_key, total) in zip(axes, groups):
        style_axes(axis)
        color = COLORS[color_key]
        shares = 100.0 * frame[column] / total if total else np.zeros(len(frame))
        axis.bar(
            frame["overlap_lo"], shares, width=widths, align="edge",
            color=color, alpha=0.38, edgecolor=color, linewidth=0.4,
        )
        axis.step(
            frame["overlap_mid"], shares, where="mid", color=color,
            linewidth=1.8, zorder=3,
        )
        axis.axvspan(0.0, threshold, color=INK_MUTED, alpha=0.10, zorder=0)
        axis.axvline(threshold, color=INK_MUTED, linewidth=1.0,
                     linestyle=(0, (4, 3)), zorder=2)
        axis.set_title(title, color=INK_PRIMARY, fontsize=13, pad=9)
        if run["normalization"] == "query":
            x_label = "Intersecting dimensions / query nnz (%)"
        else:
            x_label = "Intersecting dimensions / document nnz (%)"
        axis.set_xlabel(x_label, color=INK_SECONDARY)
        axis.set_xlim(0.0, xmax)
        axis.set_ylim(0.0, ymax * 1.16 if ymax else 1.0)

        low = low_overlap_count(frame, column, threshold)
        low_share = 100.0 * low / total if total else 0.0
        axis.text(
            0.98, 0.95,
            "{:,.0f} candidates\n{:.2f}% below {:.0f}% overlap".format(
                total, low_share, threshold
            ),
            transform=axis.transAxes, ha="right", va="top",
            color=INK_SECONDARY, fontsize=9,
        )

    axes[0].set_ylabel("Share within candidate class (%)", color=INK_SECONDARY)

    low_added = low_overlap_count(frame, "added_count", threshold)
    low_not_added = low_overlap_count(frame, "not_added_count", threshold)
    low_total = low_added + low_not_added
    low_add_rate = 100.0 * low_added / low_total if low_total else 0.0
    overall_add_rate = 100.0 * run["added_total"] / run["total"]

    recall = ""
    if run["recall"] is not None:
        recall = " · recall {:.2f}%".format(run["recall"])
    normalization_label = (
        "Query-Normalized" if run["normalization"] == "query"
        else "Document-Normalized"
    )
    fig.suptitle(
        "Candidate Queue Insertion vs. {} Overlap — {} ({})".format(
            normalization_label, pretty_dataset(run["dataset"]), run["matrix"]
        ),
        color=INK_PRIMARY, fontsize=16, fontweight="bold", y=0.985,
    )
    gate = ""
    if gate_threshold is not None:
        gate = " · query gate={:g}%".format(gate_threshold)
    population = ""
    if distance_calculated_only:
        population = " · distance-calculated docs only"
    fig.text(
        0.5, 0.925,
        "M={} · efC={} · ef={} · alpha={:g}{}{}{}".format(
            run["M"], run["efc"], run["ef"], run["alpha"], gate,
            population, recall
        ),
        ha="center", color=INK_MUTED, fontsize=9,
    )
    clipping = " Full 0–100% range shown."
    if xmax < 100.0:
        clipping = " x-axis clipped to 0–{:.0f}% (≥99.5% of each class shown).".format(xmax)
    fig.text(
        0.008, 0.014,
        "Below {:.0f}% overlap: {:,}/{:,} candidates added ({:.2f}%); "
        "overall add rate {:.2f}%.{}".format(
            threshold, low_added, low_total, low_add_rate, overall_add_rate, clipping
        ),
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.89])

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(run["path"]).replace("_hist.csv", "")
    output = os.path.join(outdir, stem + "_candidate_overlap.png")
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output, low_added, low_total, low_add_rate, overall_add_rate


def draw_retention_rate(run, outdir, threshold, gate_threshold=None,
                        distance_calculated_only=False):
    """Plot added_count/count within each query-normalized overlap bin."""
    if run["normalization"] != "query":
        raise ValueError(
            "retention-rate plots require a query-normalized histogram"
        )

    frame = run["df"]
    counts = frame["count"].to_numpy(dtype=np.float64)
    added = frame["added_count"].to_numpy(dtype=np.float64)
    rates = np.divide(
        100.0 * added,
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    widths = frame["overlap_hi"] - frame["overlap_lo"]
    xmax = x_limit(frame, ["count"])

    fig, axis = plt.subplots(figsize=(9.2, 5.3))
    fig.patch.set_facecolor(SURFACE)
    style_axes(axis)
    color = COLORS["added"]
    axis.bar(
        frame["overlap_lo"], rates, width=widths, align="edge",
        color=color, alpha=0.52, edgecolor=color, linewidth=0.45,
    )
    axis.axvspan(0.0, threshold, color=INK_MUTED, alpha=0.10, zorder=0)
    axis.axvline(
        threshold, color=INK_MUTED, linewidth=1.0,
        linestyle=(0, (4, 3)), zorder=2,
    )
    axis.set_xlabel(
        "Intersecting dimensions / query nnz (%)", color=INK_SECONDARY
    )
    axis.set_ylabel(
        "Candidates retained for future expansion (%)", color=INK_SECONDARY
    )
    axis.set_xlim(0.0, xmax)
    axis.set_ylim(0.0, 100.0)
    axis.set_yticks(np.arange(0.0, 101.0, 10.0))

    overall_rate = 100.0 * run["added_total"] / run["total"]
    axis.text(
        0.98, 0.95,
        "{:,} candidates\n{:.2f}% retained overall".format(
            run["total"], overall_rate
        ),
        transform=axis.transAxes, ha="right", va="top",
        color=INK_SECONDARY, fontsize=9,
    )

    fig.suptitle(
        "Candidate Retention Rate by Query Overlap — {} ({})".format(
            pretty_dataset(run["dataset"]), run["matrix"]
        ),
        color=INK_PRIMARY, fontsize=16, fontweight="bold", y=0.985,
    )
    gate = ""
    if gate_threshold is not None:
        gate = " · query gate={:g}%".format(gate_threshold)
    population = ""
    if distance_calculated_only:
        population = " · distance-calculated docs only"
    recall = ""
    if run["recall"] is not None:
        recall = " · recall {:.2f}%".format(run["recall"])
    fig.text(
        0.5, 0.925,
        "M={} · efC={} · ef={} · alpha={:g}{}{}{}".format(
            run["M"], run["efc"], run["ef"], run["alpha"], gate,
            population, recall
        ),
        ha="center", color=INK_MUTED, fontsize=9,
    )

    clipping = "Full 0–100% overlap range shown."
    if xmax < 100.0:
        clipping = (
            "Overlap axis clipped to 0–{:.0f}% "
            "(≥99.5% of candidates shown).".format(xmax)
        )
    fig.text(
        0.008, 0.014,
        "Each bar is 100 × added_count / count for that query-overlap bin. "
        + clipping,
        color=INK_MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.89])

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(run["path"]).replace("_hist.csv", "")
    output = os.path.join(
        outdir, stem + "_candidate_retention_rate.png"
    )
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hist", action="append", default=[],
                        help="specific histogram (repeatable)")
    parser.add_argument(
        "--indir", default=os.path.join(here, "results", "candidate_overlap"),
        help="recursively search this directory when --hist is omitted",
    )
    parser.add_argument("--outdir", default=None,
                        help="default: next to each histogram")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--matrix", choices=("unpruned", "pruned"), default=None)
    parser.add_argument("--normalization", choices=("document", "query"),
                        default=None)
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="low-overlap threshold in percent (default: 10)")
    parser.add_argument(
        "--gate-threshold", type=float, default=None,
        help="query-overlap gate used by the run; shown in the figure subtitle",
    )
    parser.add_argument(
        "--distance-calculated-only", action="store_true",
        help="annotate that gate-rejected/no-distance documents are excluded",
    )
    parser.add_argument(
        "--retention-rate", action="store_true",
        help=(
            "also plot added_count/count per bin for query-normalized "
            "histograms"
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.threshold < 100.0:
        parser.error("--threshold must be between 0 and 100")
    if args.gate_threshold is not None and not 0.0 < args.gate_threshold <= 100.0:
        parser.error("--gate-threshold must be in (0, 100]")

    if args.hist:
        runs = [run_from_path(path) for path in args.hist]
        if args.dataset:
            runs = [run for run in runs if run["dataset"] == args.dataset]
        if args.matrix:
            runs = [run for run in runs if run["matrix"] == args.matrix]
        if args.normalization:
            runs = [run for run in runs
                    if run["normalization"] == args.normalization]
    else:
        runs = discover(args.indir, args.dataset, args.matrix, args.normalization)
    if not runs:
        sys.exit("no candidate-overlap *_hist.csv files found")

    for run in runs:
        load(run)
        outdir = args.outdir or os.path.dirname(run["path"])
        output, low_added, low_total, low_rate, overall_rate = draw(
            run, outdir, args.threshold, args.gate_threshold,
            args.distance_calculated_only
        )
        print(
            "{} [{}; {}-normalized]: {:,}/{:,} candidates below {:.0f}% "
            "overlap added "
            "({:.2f}%); overall add rate {:.2f}%".format(
                run["dataset"], run["matrix"], run["normalization"],
                low_added, low_total,
                args.threshold, low_rate, overall_rate
            )
        )
        print("  ->", output)
        if args.retention_rate and run["normalization"] == "query":
            retention_output = draw_retention_rate(
                run, outdir, args.threshold, args.gate_threshold,
                args.distance_calculated_only,
            )
            print("  ->", retention_output)


if __name__ == "__main__":
    main()
