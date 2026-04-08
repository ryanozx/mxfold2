#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a saved Sinkhorn mask PNG against BPSEQ ground truth. "
            "This expects a raw grayscale PNG export of the mask, not a screenshot with axes or a colormap."
        )
    )
    parser.add_argument("mask_png", type=Path, help="Path to the mask PNG")
    parser.add_argument(
        "bpseq",
        type=str,
        help="BPSEQ path or filename/stem to search for",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("."),
        help="Root directory to search when bpseq is given as a filename/stem (default: current directory)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Repeat-interleave factor if the PNG is block-level rather than nucleotide-level (default: 1)",
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=0,
        help="Ignore pairs with |i-j| <= min_separation when scoring the mask (default: 0)",
    )
    parser.add_argument(
        "--resize-if-needed",
        action="store_true",
        help="If dimensions still do not match after optional block upsampling, resize the PNG to sequence length",
    )
    parser.add_argument(
        "--print-ground-truth-map",
        action="store_true",
        help="Print the full binary ground-truth contact map to stdout",
    )
    parser.add_argument(
        "--save-ground-truth-map",
        type=Path,
        default=None,
        help="Save the binary ground-truth contact map as a grayscale PNG",
    )
    return parser.parse_args()


def resolve_bpseq_path(query: str, search_root: Path) -> Path:
    candidate = Path(query)
    if candidate.exists():
        return candidate.resolve()

    names = {query}
    if not query.endswith(".bpseq"):
        names.add(f"{query}.bpseq")

    matches = []
    for path in search_root.rglob("*.bpseq"):
        if path.name in names or path.stem == query:
            matches.append(path.resolve())

    if not matches:
        raise FileNotFoundError(f"Could not resolve BPSEQ file for '{query}' under {search_root}")
    if len(matches) > 1:
        joined = "\n".join(str(m) for m in matches[:10])
        raise RuntimeError(f"Multiple BPSEQ matches found for '{query}':\n{joined}")
    return matches[0]


def load_bpseq_matrix(bpseq_path: Path) -> Tuple[str, np.ndarray]:
    bases = []
    partners = []
    with bpseq_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Expected 3 columns in {bpseq_path}, got: {raw_line.rstrip()}")
            _, base, partner = fields
            pair = int(partner)
            bases.append(base)
            partners.append(pair)

    seq = "".join(bases)
    length = len(seq)
    matrix = np.zeros((length, length), dtype=np.float32)
    for i, j in enumerate(partners):
        if j > 0:
            matrix[i, j - 1] = 1.0
    return seq, matrix


def build_valid_pair_matrix(seq: str) -> np.ndarray:
    valid_pairs = {
        ("A", "U"),
        ("U", "A"),
        ("C", "G"),
        ("G", "C"),
        ("G", "U"),
        ("U", "G"),
    }
    length = len(seq)
    matrix = np.zeros((length, length), dtype=bool)
    for i, base_i in enumerate(seq):
        for j, base_j in enumerate(seq):
            matrix[i, j] = (base_i, base_j) in valid_pairs
    return matrix


def load_png_mask(mask_png: Path) -> np.ndarray:
    image = Image.open(mask_png).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Mask image is empty: {mask_png}")
    return arr / 255.0


def upsample_block_mask(mask: np.ndarray, seq_len: int, block_size: int) -> np.ndarray:
    if block_size <= 1:
        return mask
    expected_blocks = math.ceil(seq_len / block_size)
    expected_blocks_with_dummy = math.ceil((seq_len + 1) / block_size)
    if mask.shape != (expected_blocks, expected_blocks):
        if mask.shape != (expected_blocks_with_dummy, expected_blocks_with_dummy):
            return mask
    expanded = np.repeat(np.repeat(mask, block_size, axis=0), block_size, axis=1)
    if expanded.shape[0] >= seq_len + 1 and expanded.shape[1] >= seq_len + 1:
        if expanded.shape[0] == seq_len + 1 or expanded.shape[0] == expected_blocks_with_dummy * block_size:
            return expanded[: seq_len + 1, : seq_len + 1]
    return expanded[:seq_len, :seq_len]


def trim_dummy_index(mask: np.ndarray, seq_len: int) -> np.ndarray:
    if mask.shape == (seq_len + 1, seq_len + 1):
        return mask[1:, 1:]
    return mask


def resize_mask(mask: np.ndarray, seq_len: int) -> np.ndarray:
    image = Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = image.resize((seq_len, seq_len), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_binary_map_image(matrix: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((np.clip(matrix, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image.save(out_path)


def ensure_square(mask: np.ndarray) -> None:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"Mask must be a square 2D image, got shape {mask.shape}")


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    n = scores.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = ranks[labels.astype(bool)].sum()
    u = sum_ranks_pos - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    return float((precision * sorted_labels).sum() / pos)


def build_upper_triangle_mask(length: int, min_separation: int) -> np.ndarray:
    ii, jj = np.indices((length, length))
    return (jj > ii) & ((jj - ii) > min_separation)


def build_assignment_targets(gt: np.ndarray, min_separation: int) -> Tuple[np.ndarray, np.ndarray]:
    length = gt.shape[0]
    target_idx = np.arange(length, dtype=np.int64)
    paired_rows = np.zeros(length, dtype=bool)
    for i in range(length):
        true_partners = np.flatnonzero(gt[i] > 0.0)
        if true_partners.size == 0:
            continue
        true_j = int(true_partners[0])
        if abs(true_j - i) <= min_separation:
            continue
        target_idx[i] = true_j
        paired_rows[i] = True
    return target_idx, paired_rows


def row_argmax_partner_accuracy(mask: np.ndarray, gt: np.ndarray, min_separation: int) -> float:
    length = mask.shape[0]
    paired_rows = []
    hits = 0
    for i in range(length):
        true_partners = np.flatnonzero(gt[i] > 0.0)
        if true_partners.size == 0:
            continue
        true_j = int(true_partners[0])
        if abs(true_j - i) <= min_separation:
            continue
        row = mask[i].copy()
        invalid = np.abs(np.arange(length) - i) <= min_separation
        row[invalid] = -np.inf
        pred_j = int(np.argmax(row))
        paired_rows.append(i)
        if pred_j == true_j:
            hits += 1
    if not paired_rows:
        return float("nan")
    return hits / len(paired_rows)


def row_argmax_valid_fraction(mask: np.ndarray, valid_pair_matrix: np.ndarray, min_separation: int) -> float:
    length = mask.shape[0]
    valid_count = 0
    rows_used = 0
    for i in range(length):
        row = mask[i].copy()
        invalid = np.abs(np.arange(length) - i) <= min_separation
        row[invalid] = -np.inf
        pred_j = int(np.argmax(row))
        rows_used += 1
        if valid_pair_matrix[i, pred_j]:
            valid_count += 1
    if rows_used == 0:
        return float("nan")
    return valid_count / rows_used


def compare(mask: np.ndarray, gt: np.ndarray, seq: str, min_separation: int) -> dict:
    mask = mask.astype(np.float32, copy=False)
    gt = gt.astype(np.float32, copy=False)
    mask = np.clip(mask, 0.0, 1.0)
    mask = 0.5 * (mask + mask.T)
    valid_pair_matrix = build_valid_pair_matrix(seq)
    row_probs = mask / np.clip(mask.sum(axis=1, keepdims=True), 1e-8, None)
    target_idx, paired_rows = build_assignment_targets(gt, min_separation)
    unpaired_rows = ~paired_rows
    row_argmax = row_probs.argmax(axis=1)
    target_probs = row_probs[np.arange(mask.shape[0]), target_idx]
    assignment_hits = row_argmax == target_idx
    pair_or_null_valid = np.zeros(mask.shape[0], dtype=bool)
    for i, pred_j in enumerate(row_argmax):
        if pred_j == i:
            pair_or_null_valid[i] = bool(unpaired_rows[i])
        else:
            pair_or_null_valid[i] = bool(
                abs(int(pred_j) - i) > min_separation and valid_pair_matrix[i, pred_j]
            )

    upper = build_upper_triangle_mask(mask.shape[0], min_separation)
    scores = mask[upper]
    labels = gt[upper] > 0.0
    valid_labels = valid_pair_matrix[upper]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)

    metrics = {
        "num_positives": positives,
        "num_negatives": negatives,
        "score_mean_true_pairs": float(scores[labels].mean()) if positives else float("nan"),
        "score_mean_false_pairs": float(scores[~labels].mean()) if negatives else float("nan"),
        "mass_on_true_pairs": float(scores[labels].sum() / scores.sum()) if scores.sum() > 0 and positives else float("nan"),
        "score_mean_valid_pairs": float(scores[valid_labels].mean()) if int(valid_labels.sum()) else float("nan"),
        "score_mean_invalid_pairs": float(scores[~valid_labels].mean()) if int((~valid_labels).sum()) else float("nan"),
        "mass_on_valid_pairs": float(scores[valid_labels].sum() / scores.sum()) if scores.sum() > 0 and int(valid_labels.sum()) else float("nan"),
        "average_precision": average_precision(scores, labels),
        "roc_auc": auc_roc(scores, labels),
        "row_argmax_partner_accuracy": row_argmax_partner_accuracy(mask, gt, min_separation),
        "row_argmax_valid_pair_fraction": row_argmax_valid_fraction(mask, valid_pair_matrix, min_separation),
        "paired_rows": int(paired_rows.sum()),
        "unpaired_rows": int(unpaired_rows.sum()),
        "assignment_target_prob_mean": float(target_probs.mean()),
        "assignment_target_nll_mean": float((-np.log(np.clip(target_probs, 1e-8, 1.0))).mean()),
        "row_argmax_assignment_accuracy": float(assignment_hits.mean()),
        "row_argmax_pair_or_null_valid_fraction": float(pair_or_null_valid.mean()),
    }

    if int(paired_rows.sum()) > 0:
        metrics["paired_row_target_prob_mean"] = float(target_probs[paired_rows].mean())
        metrics["paired_row_argmax_assignment_accuracy"] = float(assignment_hits[paired_rows].mean())
    else:
        metrics["paired_row_target_prob_mean"] = float("nan")
        metrics["paired_row_argmax_assignment_accuracy"] = float("nan")

    if int(unpaired_rows.sum()) > 0:
        diag_probs = row_probs.diagonal()
        metrics["unpaired_row_diag_prob_mean"] = float(diag_probs[unpaired_rows].mean())
        metrics["unpaired_row_argmax_diagonal_accuracy"] = float((row_argmax[unpaired_rows] == np.flatnonzero(unpaired_rows)).mean())
    else:
        metrics["unpaired_row_diag_prob_mean"] = float("nan")
        metrics["unpaired_row_argmax_diagonal_accuracy"] = float("nan")

    if positives > 0:
        k = positives
        topk_idx = np.argpartition(scores, -k)[-k:]
        metrics["topk_precision_at_num_true_pairs"] = float(labels[topk_idx].mean())
        metrics["topk_valid_pair_fraction_at_num_true_pairs"] = float(valid_labels[topk_idx].mean())
    else:
        metrics["topk_precision_at_num_true_pairs"] = float("nan")
        metrics["topk_valid_pair_fraction_at_num_true_pairs"] = float("nan")

    return metrics


def main() -> None:
    args = parse_args()

    bpseq_path = resolve_bpseq_path(args.bpseq, args.search_root)
    seq, gt = load_bpseq_matrix(bpseq_path)

    mask = load_png_mask(args.mask_png)
    ensure_square(mask)
    mask = upsample_block_mask(mask, len(seq), args.block_size)
    mask = trim_dummy_index(mask, len(seq))

    if mask.shape != gt.shape:
        if args.resize_if_needed:
            mask = resize_mask(mask, len(seq))
        else:
            raise ValueError(
                f"Mask shape {mask.shape} does not match sequence length {len(seq)}. "
                "Use --block-size for block masks or --resize-if-needed as a fallback."
            )

    metrics = compare(mask, gt, seq, args.min_separation)

    print(f"mask_png: {args.mask_png.resolve()}")
    print(f"bpseq: {bpseq_path}")
    print(f"sequence_length: {len(seq)}")
    print(f"mask_shape: {mask.shape}")
    print(f"min_separation: {args.min_separation}")
    print("")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if args.print_ground_truth_map:
        print("")
        print("ground_truth_contact_map:")
        with np.printoptions(threshold=np.inf, linewidth=10_000):
            print(gt.astype(np.int8))

    if args.save_ground_truth_map is not None:
        save_binary_map_image(gt, args.save_ground_truth_map)
        print("")
        print(f"ground_truth_map_png: {args.save_ground_truth_map.resolve()}")


if __name__ == "__main__":
    main()
