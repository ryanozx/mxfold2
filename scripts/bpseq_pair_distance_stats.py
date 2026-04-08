#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how common short-range and long-range RNA base pairs are in a BPSEQ dataset. "
            "The input is a .lst file containing BPSEQ paths, one per line."
        )
    )
    parser.add_argument("bpseq_list", type=Path, help="Path to the .lst file of BPSEQ paths")
    parser.add_argument(
        "--block-size",
        type=int,
        default=30,
        help="Block size used to compress nucleotide positions into block indices (default: 30)",
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=0,
        help="Ignore base pairs with |i-j| <= min_separation (default: 0)",
    )
    parser.add_argument(
        "--span-bins",
        type=str,
        default="30,60,120,240",
        help="Comma-separated base-span bin upper bounds, e.g. 30,60,120,240 (default: 30,60,120,240)",
    )
    parser.add_argument(
        "--block-span-bins",
        type=str,
        default="0,1,3,7",
        help=(
            "Comma-separated block-distance bin upper bounds, e.g. 0,1,3,7. "
            "Distance 0 means same block. (default: 0,1,3,7)"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON",
    )
    return parser.parse_args()


def parse_int_bins(raw: str) -> list[int]:
    bins = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        bins.append(int(field))
    if not bins:
        raise ValueError("Expected at least one bin boundary")
    bins = sorted(set(bins))
    return bins


def iter_bpseq_paths(list_path: Path):
    list_dir = list_path.resolve().parent.parent
    with list_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) == 1:
                candidate = Path(fields[0])
                yield candidate if candidate.is_absolute() else (list_dir / candidate)
            elif len(fields) == 2:
                # Keep compatibility with MXfold2 dataset lists that may contain two columns.
                candidate = Path(fields[0])
                if candidate.suffix.lower() == ".bpseq":
                    yield candidate if candidate.is_absolute() else (list_dir / candidate)
                else:
                    candidate = Path(fields[1])
                    if candidate.suffix.lower() == ".bpseq":
                        yield candidate if candidate.is_absolute() else (list_dir / candidate)
                    else:
                        raise ValueError(f"Could not find a BPSEQ path in line: {raw_line.rstrip()}")
            else:
                raise ValueError(f"Unsupported .lst line: {raw_line.rstrip()}")


def read_bpseq(bpseq_path: Path) -> tuple[int, list[tuple[int, int]]]:
    partners: list[int] = [0]
    with bpseq_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Expected 3 columns in {bpseq_path}, got: {raw_line.rstrip()}")
            idx, _base, partner = fields
            i = int(idx)
            j = int(partner)
            if i != len(partners):
                raise ValueError(f"Unexpected index order in {bpseq_path}: expected {len(partners)}, got {i}")
            partners.append(j)

    length = len(partners) - 1
    unique_pairs: list[tuple[int, int]] = []
    for i in range(1, length + 1):
        j = partners[i]
        if j > i:
            unique_pairs.append((i, j))
    return length, unique_pairs


def bucket_label(value: int, bins: list[int]) -> str:
    lower = 0
    for upper in bins:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{bins[-1]}"


def init_bucket_counts(bins: list[int]) -> dict[str, int]:
    labels = []
    lower = 0
    for upper in bins:
        labels.append(f"{lower}-{upper}")
        lower = upper + 1
    labels.append(f">{bins[-1]}")
    return {label: 0 for label in labels}


def format_fraction(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.min_separation < 0:
        raise ValueError("--min-separation must be non-negative")

    span_bins = parse_int_bins(args.span_bins)
    block_span_bins = parse_int_bins(args.block_span_bins)

    sequence_lengths: list[int] = []
    all_base_spans: list[int] = []
    all_block_spans: list[int] = []
    same_block_pairs = 0
    adjacent_block_pairs = 0
    filtered_pairs = 0
    total_pairs_before_filter = 0
    sequence_count = 0

    total_possible_block_cells_upper = 0
    total_possible_same_block_cells = 0
    total_possible_offdiag_block_cells_upper = 0
    total_positive_block_cells_upper = 0
    total_positive_same_block_cells = 0
    total_positive_offdiag_block_cells_upper = 0

    total_possible_block_cells_directed = 0
    total_positive_block_cells_directed = 0

    span_bucket_counts = init_bucket_counts(span_bins)
    block_span_bucket_counts = init_bucket_counts(block_span_bins)

    for bpseq_path in iter_bpseq_paths(args.bpseq_list):
        length, unique_pairs = read_bpseq(bpseq_path)
        sequence_count += 1
        sequence_lengths.append(length)
        total_pairs_before_filter += len(unique_pairs)
        num_blocks = math.ceil(length / args.block_size)
        total_possible_same_block_cells += num_blocks
        total_possible_offdiag_block_cells_upper += num_blocks * (num_blocks - 1) // 2
        total_possible_block_cells_upper += num_blocks * (num_blocks + 1) // 2
        total_possible_block_cells_directed += num_blocks * num_blocks

        positive_block_cells_upper: set[tuple[int, int]] = set()

        for i, j in unique_pairs:
            span = j - i
            if span <= args.min_separation:
                continue

            block_i = (i - 1) // args.block_size
            block_j = (j - 1) // args.block_size
            block_span = abs(block_j - block_i)
            positive_block_cells_upper.add((min(block_i, block_j), max(block_i, block_j)))

            filtered_pairs += 1
            all_base_spans.append(span)
            all_block_spans.append(block_span)
            span_bucket_counts[bucket_label(span, span_bins)] += 1
            block_span_bucket_counts[bucket_label(block_span, block_span_bins)] += 1

            if block_span == 0:
                same_block_pairs += 1
            if block_span == 1:
                adjacent_block_pairs += 1

        positive_same = sum(1 for bi, bj in positive_block_cells_upper if bi == bj)
        positive_upper = len(positive_block_cells_upper)
        positive_offdiag_upper = positive_upper - positive_same

        total_positive_same_block_cells += positive_same
        total_positive_offdiag_block_cells_upper += positive_offdiag_upper
        total_positive_block_cells_upper += positive_upper
        total_positive_block_cells_directed += positive_same + 2 * positive_offdiag_upper

    if sequence_count == 0:
        raise ValueError(f"No BPSEQ paths found in {args.bpseq_list}")

    summary = {
        "bpseq_list": str(args.bpseq_list.resolve()),
        "block_size": args.block_size,
        "min_separation": args.min_separation,
        "num_sequences": sequence_count,
        "sequence_length": {
            "min": min(sequence_lengths),
            "median": median(sequence_lengths),
            "mean": mean(sequence_lengths),
            "max": max(sequence_lengths),
        },
        "pairs": {
            "total_unique_pairs_before_filter": total_pairs_before_filter,
            "total_unique_pairs_after_filter": filtered_pairs,
        },
        "base_span": {
            "mean": mean(all_base_spans) if all_base_spans else 0.0,
            "median": median(all_base_spans) if all_base_spans else 0.0,
            "max": max(all_base_spans) if all_base_spans else 0,
            "bucket_counts": span_bucket_counts,
            "bucket_fractions": {
                label: format_fraction(count, filtered_pairs)
                for label, count in span_bucket_counts.items()
            },
        },
        "block_span": {
            "mean": mean(all_block_spans) if all_block_spans else 0.0,
            "median": median(all_block_spans) if all_block_spans else 0.0,
            "max": max(all_block_spans) if all_block_spans else 0,
            "bucket_counts": block_span_bucket_counts,
            "bucket_fractions": {
                label: format_fraction(count, filtered_pairs)
                for label, count in block_span_bucket_counts.items()
            },
            "same_block_pairs": same_block_pairs,
            "same_block_fraction": format_fraction(same_block_pairs, filtered_pairs),
            "adjacent_block_pairs": adjacent_block_pairs,
            "adjacent_block_fraction": format_fraction(adjacent_block_pairs, filtered_pairs),
            "same_or_adjacent_block_fraction": format_fraction(
                same_block_pairs + adjacent_block_pairs, filtered_pairs
            ),
        },
        "block_cell_sparsity": {
            "upper_triangle": {
                "possible_cells": total_possible_block_cells_upper,
                "positive_cells": total_positive_block_cells_upper,
                "positive_fraction": format_fraction(
                    total_positive_block_cells_upper, total_possible_block_cells_upper
                ),
                "same_block": {
                    "possible_cells": total_possible_same_block_cells,
                    "positive_cells": total_positive_same_block_cells,
                    "positive_fraction": format_fraction(
                        total_positive_same_block_cells, total_possible_same_block_cells
                    ),
                },
                "off_diagonal": {
                    "possible_cells": total_possible_offdiag_block_cells_upper,
                    "positive_cells": total_positive_offdiag_block_cells_upper,
                    "positive_fraction": format_fraction(
                        total_positive_offdiag_block_cells_upper,
                        total_possible_offdiag_block_cells_upper,
                    ),
                },
            },
            "directed_full_matrix": {
                "possible_cells": total_possible_block_cells_directed,
                "positive_cells": total_positive_block_cells_directed,
                "positive_fraction": format_fraction(
                    total_positive_block_cells_directed, total_possible_block_cells_directed
                ),
            },
        },
    }

    print("Dataset pair-distance summary")
    print(f"bpseq_list: {summary['bpseq_list']}")
    print(f"block_size: {args.block_size}")
    print(f"min_separation: {args.min_separation}")
    print(f"num_sequences: {sequence_count}")
    print(
        "sequence_length:"
        f" min={summary['sequence_length']['min']}"
        f" median={summary['sequence_length']['median']}"
        f" mean={summary['sequence_length']['mean']:.2f}"
        f" max={summary['sequence_length']['max']}"
    )
    print(
        "pairs:"
        f" before_filter={total_pairs_before_filter}"
        f" after_filter={filtered_pairs}"
    )

    print("\nBase span")
    print(
        f"mean={summary['base_span']['mean']:.2f} "
        f"median={summary['base_span']['median']} "
        f"max={summary['base_span']['max']}"
    )
    for label, count in summary["base_span"]["bucket_counts"].items():
        frac = summary["base_span"]["bucket_fractions"][label]
        print(f"  {label}: {count} ({frac:.4f})")

    print("\nBlock span")
    print(
        f"mean={summary['block_span']['mean']:.2f} "
        f"median={summary['block_span']['median']} "
        f"max={summary['block_span']['max']}"
    )
    print(
        f"same_block_fraction={summary['block_span']['same_block_fraction']:.4f} "
        f"adjacent_block_fraction={summary['block_span']['adjacent_block_fraction']:.4f} "
        f"same_or_adjacent_block_fraction={summary['block_span']['same_or_adjacent_block_fraction']:.4f}"
    )
    for label, count in summary["block_span"]["bucket_counts"].items():
        frac = summary["block_span"]["bucket_fractions"][label]
        print(f"  {label}: {count} ({frac:.4f})")

    upper = summary["block_cell_sparsity"]["upper_triangle"]
    same = upper["same_block"]
    offdiag = upper["off_diagonal"]
    directed = summary["block_cell_sparsity"]["directed_full_matrix"]

    print("\nBlock cell sparsity")
    print(
        f"upper_triangle_positive_fraction={upper['positive_fraction']:.6f} "
        f"({upper['positive_cells']}/{upper['possible_cells']})"
    )
    print(
        f"same_block_positive_fraction={same['positive_fraction']:.6f} "
        f"({same['positive_cells']}/{same['possible_cells']})"
    )
    print(
        f"off_diagonal_positive_fraction={offdiag['positive_fraction']:.6f} "
        f"({offdiag['positive_cells']}/{offdiag['possible_cells']})"
    )
    print(
        f"directed_full_matrix_positive_fraction={directed['positive_fraction']:.6f} "
        f"({directed['positive_cells']}/{directed['possible_cells']})"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
