#!/usr/bin/env python3

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_and_average(csv_path):
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    counts = defaultdict(int)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["nfilter", "t1", "t2", "t3", "t4", "t5"]
        for name in required:
            if name not in reader.fieldnames:
                raise ValueError(f"Missing column: {name}")

        for row in reader:
            try:
                nfilter = int(row["nfilter"])
                values = [
                    float(row["t1"]),
                    float(row["t2"]),
                    float(row["t3"]),
                    float(row["t4"]),
                    float(row["t5"]),
                ]
            except (ValueError, TypeError):
                continue

            sums[nfilter] = [sums[nfilter][i] + values[i] for i in range(5)]
            counts[nfilter] += 1

    x_vals = sorted(counts.keys())
    y_vals = [[0.0 for _ in x_vals] for _ in range(5)]

    for idx, nfilter in enumerate(x_vals):
        count = counts[nfilter]
        if count == 0:
            continue
        for i in range(5):
            y_vals[i][idx] = sums[nfilter][i] / count

    return x_vals, y_vals


def plot_lines(x_vals, y_vals, output_path):
    plt.figure(figsize=(10, 6))
    labels = [
        "Filtering",
        "Matrix Copy",
        "GEMV",
        "Distance Calculation",
        "Candidate Updates",
    ]

    for i, label in enumerate(labels):
        plt.plot(x_vals, y_vals[i], label=label)

    plt.xlabel("nfilter")
    plt.ylabel("time (us)")
    plt.title("Average time per nfilter")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Plot average t1..t5 vs nfilter.")
    parser.add_argument(
        "--input",
        default="$SCRATCH/repos/minimal_hnsw/dense/output/timing_stats/mkl_stats.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="$SCRATCH/repos/minimal_hnsw/dense/output/timing_stats/plots/mkl_stats.png",
        help="Output plot path (PNG)",
    )

    args = parser.parse_args()
    x_vals, y_vals = load_and_average(args.input)
    if not x_vals:
        raise ValueError("No valid data found in the input CSV.")
    plot_lines(x_vals, y_vals, args.output)


if __name__ == "__main__":
    main()
