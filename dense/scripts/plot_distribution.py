#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

DIST_INSERTION_FILE = "layer0_distance_counts_insertion.txt"
CAND_INSERTION_FILE = "layer0_cand_elements_counts_insertion.txt"
MAX_HOPS_INSERTION_FILE = "layer0_max_hops_counts_insertion.txt"

DIST_SEARCH_FILE = "layer0_distance_counts_search.txt"
CAND_SEARCH_FILE = "layer0_cand_elements_counts_search.txt"
MAX_HOPS_SEARCH_FILE = "layer0_max_hops_counts_search.txt"


def load_counts(path: Path) -> list[int]:
    values: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(int(line))
    return values


def infer_metric(path: Path) -> str:
    name = path.name.lower()
    if "max_hops" in name:
        return "max_hops"
    if "cand_elements" in name:
        return "cand_elements"
    return "dist_calc"


def infer_phase(path: Path) -> str:
    name = path.name.lower()
    if "search" in name:
        return "search"
    if "insertion" in name:
        return "insertion"
    return "combined"


def plot_one(input_path: Path, output_path: Path, bins: int, metric_arg: str) -> None:
    values = load_counts(input_path)
    if not values:
        raise ValueError(f"No values found in {input_path}")

    mean_value = sum(values) / len(values)
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    std_value = math.sqrt(variance)

    metric = infer_metric(input_path) if metric_arg == "auto" else metric_arg
    phase = infer_phase(input_path)
    if metric == "cand_elements":
        metric_title = "Candidate Elements Count Distribution"
        x_label = "Candidate elements processed per layer-0 search"
    elif metric == "max_hops":
        metric_title = "Max Hops Distribution"
        x_label = "Max hops per layer-0 search"
    else:
        metric_title = "Distance Calculation Count Distribution"
        x_label = "Distance calculations per layer-0 search"

    if phase == "insertion":
        title = f"Layer-0 {metric_title} - Insertion Phase"
    elif phase == "search":
        title = f"Layer-0 {metric_title} - Search Phase"
    else:
        title = f"Layer-0 {metric_title}"

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins, alpha=0.8, rwidth=1.0, edgecolor="black", linewidth=0.6)
    plt.axvline(mean_value, color="red", linestyle="--", linewidth=2, label=f"mean={mean_value:.2f}")
    plt.axvline(mean_value - std_value, color="green", linestyle=":", linewidth=1.5, label=f"mean-std={mean_value - std_value:.2f}")
    plt.axvline(mean_value + std_value, color="green", linestyle=":", linewidth=1.5, label=f"mean+std={mean_value + std_value:.2f}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.text(
        0.98,
        0.95,
        f"n={len(values)}\nmean={mean_value:.2f}\nstd={std_value:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_difference(dist_path: Path, cand_path: Path, output_path: Path, bins: int, phase: str) -> None:
    dist_values = load_counts(dist_path)
    cand_values = load_counts(cand_path)
    if not dist_values:
        raise ValueError(f"No values found in {dist_path}")
    if not cand_values:
        raise ValueError(f"No values found in {cand_path}")
    if len(dist_values) != len(cand_values):
        raise ValueError(
            f"Mismatched lengths for not-considered-candidate values: {dist_path} has {len(dist_values)} values, "
            f"{cand_path} has {len(cand_values)} values"
        )

    wasted_values = [dist - cand for dist, cand in zip(dist_values, cand_values)]
    mean_value = sum(wasted_values) / len(wasted_values)
    variance = sum((x - mean_value) ** 2 for x in wasted_values) / len(wasted_values)
    std_value = math.sqrt(variance)

    if phase == "insertion":
        title = "Layer-0 Considered vs Not Considered Candidates - Insertion Phase"
    elif phase == "search":
        title = "Layer-0 Considered vs Not Considered Candidates - Search Phase"
    else:
        title = "Layer-0 Considered vs Not Considered Candidates"

    plt.figure(figsize=(9, 5))
    plt.hist(
        cand_values,
        bins=bins,
        color="green",
        alpha=0.45,
        label="considered candidates",
        rwidth=1.0,
        edgecolor="black",
        linewidth=0.6,
    )
    plt.hist(
        wasted_values,
        bins=bins,
        color="red",
        alpha=0.45,
        label="not considered candidates",
        rwidth=1.0,
        edgecolor="black",
        linewidth=0.6,
    )
    plt.title(title)
    plt.xlabel("Calculations per layer-0 search")
    plt.ylabel("Frequency")
    plt.text(
        0.98,
        0.95,
        f"n={len(wasted_values)}\nnot-considered mean={mean_value:.2f}\nnot-considered std={std_value:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution of layer-0 counts (distance calculations, candidate elements, or max hops)."
    )
    parser.add_argument("input", type=Path, help="Path to a text file, or a folder containing layer-0 count text files.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to <input_stem>_hist.png.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins.")
    parser.add_argument(
        "--metric",
        choices=["auto", "dist_calc", "cand_elements", "max_hops"],
        default="auto",
        help="Metric type for labels. Default: auto-detect from input filename.",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        dist_insert_path = args.input / DIST_INSERTION_FILE
        cand_insert_path = args.input / CAND_INSERTION_FILE
        max_hops_insert_path = args.input / MAX_HOPS_INSERTION_FILE
        dist_search_path = args.input / DIST_SEARCH_FILE
        cand_search_path = args.input / CAND_SEARCH_FILE
        max_hops_search_path = args.input / MAX_HOPS_SEARCH_FILE
        missing = [str(p) for p in (dist_insert_path, cand_insert_path, max_hops_insert_path, dist_search_path, cand_search_path, max_hops_search_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing expected files in folder: {', '.join(missing)}")

        output_dir = args.output if args.output is not None else args.input
        if output_dir.suffix.lower() == ".png":
            raise ValueError("When input is a folder, --output must be a folder (or omitted), not a .png file path.")

        plot_one(dist_insert_path, output_dir / f"{dist_insert_path.stem}_hist.png", args.bins, "dist_calc")
        plot_one(cand_insert_path, output_dir / f"{cand_insert_path.stem}_hist.png", args.bins, "cand_elements")
        plot_one(max_hops_insert_path, output_dir / f"{max_hops_insert_path.stem}_hist.png", args.bins, "max_hops")
        plot_difference(dist_insert_path, cand_insert_path, output_dir / "layer0_wasted_calculations_insertion_hist.png", args.bins, "insertion")
        plot_one(dist_search_path, output_dir / f"{dist_search_path.stem}_hist.png", args.bins, "dist_calc")
        plot_one(cand_search_path, output_dir / f"{cand_search_path.stem}_hist.png", args.bins, "cand_elements")
        plot_one(max_hops_search_path, output_dir / f"{max_hops_search_path.stem}_hist.png", args.bins, "max_hops")
        plot_difference(dist_search_path, cand_search_path, output_dir / "layer0_wasted_calculations_search_hist.png", args.bins, "search")
        return

    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    output_path = args.output if args.output is not None else args.input.with_name(f"{args.input.stem}_hist.png")
    plot_one(args.input, output_path, args.bins, args.metric)


if __name__ == "__main__":
    main()
