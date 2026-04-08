import random

import torch
import torch.nn as nn


BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "T": 3,
}

CANONICAL_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("C", "G"),
    ("G", "C"),
    ("G", "U"),
    ("U", "G"),
    ("A", "T"),
    ("T", "A"),
    ("G", "T"),
    ("T", "G"),
}

CANONICAL_PAIR_MASK = torch.zeros((4, 4), dtype=torch.bool)
for a, b in CANONICAL_PAIRS:
    ia = BASE_TO_INDEX.get(a)
    ib = BASE_TO_INDEX.get(b)
    if ia is not None and ib is not None:
        CANONICAL_PAIR_MASK[ia, ib] = True


class PairFusionMLP(nn.Module):
    """Small pairwise classifier over frozen base-MXfold2 and Sinkhorn features."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_indicator_matrix(pairs, seq_len: int):
    """
    Convert a predicted/reference structure into a symmetric NxN pair-indicator matrix.

    The codebase uses a few representations for pairings:
    - a length-(N+1) partner vector with a dummy 0th entry
    - a 1D tensor in the same format
    - an explicit iterable of (i, j) base-pair tuples
    """
    mat = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    if isinstance(pairs, torch.Tensor):
        pairs = pairs.detach().cpu()

    # Partner-vector form: pairs[i] gives the paired partner of position i.
    if isinstance(pairs, torch.Tensor) and pairs.ndim == 1:
        for i in range(1, min(len(pairs), seq_len + 1)):
            j = int(pairs[i].item())
            if j > i and j <= seq_len:
                mat[i - 1, j - 1] = 1.0
                mat[j - 1, i - 1] = 1.0
        return mat

    if isinstance(pairs, torch.Tensor):
        pairs = pairs.tolist()

    # MXfold2 prediction can return a 1D Python list where index i stores the
    # paired partner of nucleotide i (with a dummy 0th entry). Handle that
    # representation before falling back to an explicit list of (i, j) pairs.
    if isinstance(pairs, (list, tuple)) and pairs and all(
        isinstance(x, (int, float)) for x in pairs
    ):
        for i in range(1, min(len(pairs), seq_len + 1)):
            j = int(pairs[i])
            if j > i and j <= seq_len:
                mat[i - 1, j - 1] = 1.0
                mat[j - 1, i - 1] = 1.0
        return mat

    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if 1 <= i <= seq_len and 1 <= j <= seq_len and i != j:
            a, b = min(i, j), max(i, j)
            mat[a - 1, b - 1] = 1.0
            mat[b - 1, a - 1] = 1.0
    return mat


def _base_onehot(base: str):
    """Return a compact 4D one-hot encoding for A/C/G/U(T)."""
    vec = [0.0, 0.0, 0.0, 0.0]
    idx = BASE_TO_INDEX.get(base.upper())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _seq_base_ids(seq: str) -> torch.Tensor:
    """Map a sequence to integer base ids, using -1 for unknown characters."""
    return torch.tensor([BASE_TO_INDEX.get(base.upper(), -1) for base in seq], dtype=torch.long)


def build_pairwise_features(
    seq: str,
    base_scores: torch.Tensor,
    sinkhorn_scores: torch.Tensor,
    base_pair_indicator: torch.Tensor,
    ref_pairs,
    min_separation: int = 4,
):
    """
    Build one training example per candidate pair (i, j).

    Each row combines frozen-model outputs with simple sequence priors:
    - base MXfold2 BPP score
    - Sinkhorn contact score
    - base-model predicted-pair indicator
    - normalized sequence distance
    - canonical-pair flag
    - one-hot identities of bases i and j

    The label is 1 iff (i, j) is a true pair in the reference BPSEQ.
    """
    seq = seq[1:] if len(seq) > 0 and seq[0] == "0" else seq
    seq_len = len(seq)
    if tuple(base_scores.shape) != (seq_len, seq_len):
        raise ValueError(f"Expected base_scores shape {(seq_len, seq_len)}, got {tuple(base_scores.shape)}")
    if tuple(sinkhorn_scores.shape) != (seq_len, seq_len):
        raise ValueError(f"Expected sinkhorn_scores shape {(seq_len, seq_len)}, got {tuple(sinkhorn_scores.shape)}")
    if tuple(base_pair_indicator.shape) != (seq_len, seq_len):
        raise ValueError(
            f"Expected base_pair_indicator shape {(seq_len, seq_len)}, got {tuple(base_pair_indicator.shape)}"
        )

    ref_indicator = pair_indicator_matrix(ref_pairs, seq_len)
    # Enumerate all valid upper-triangular candidate pairs in one shot instead
    # of walking them in nested Python loops.
    row_idx, col_idx = torch.triu_indices(seq_len, seq_len, offset=int(min_separation) + 1)
    if len(row_idx) == 0:
        return (
            torch.zeros((0, 13), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0, 2), dtype=torch.long),
        )

    pair_indices = torch.stack([row_idx, col_idx], dim=1)

    base_ids = _seq_base_ids(seq)
    base_onehot = torch.zeros((seq_len, 4), dtype=torch.float32)
    valid_base = base_ids >= 0
    if valid_base.any():
        base_onehot[valid_base, base_ids[valid_base]] = 1.0

    left_ids = base_ids[row_idx]
    right_ids = base_ids[col_idx]
    canonical = torch.zeros((len(row_idx),), dtype=torch.float32)
    valid_pair = (left_ids >= 0) & (right_ids >= 0)
    if valid_pair.any():
        canonical[valid_pair] = CANONICAL_PAIR_MASK[left_ids[valid_pair], right_ids[valid_pair]].to(torch.float32)

    distance = (col_idx - row_idx).to(torch.float32) / float(seq_len)
    features = torch.cat(
        [
            base_scores[row_idx, col_idx].to(torch.float32).unsqueeze(1),
            sinkhorn_scores[row_idx, col_idx].to(torch.float32).unsqueeze(1),
            base_pair_indicator[row_idx, col_idx].to(torch.float32).unsqueeze(1),
            distance.unsqueeze(1),
            canonical.unsqueeze(1),
            base_onehot[row_idx],
            base_onehot[col_idx],
        ],
        dim=1,
    )
    labels = ref_indicator[row_idx, col_idx].to(torch.float32)

    return (
        features,
        labels,
        pair_indices,
    )


def sample_pair_batch(features, labels, pair_indices, neg_ratio: float = 5.0):
    """
    Downsample negatives so BCE training is not dominated by non-pair examples.

    We keep all positives and sample at most neg_ratio * positives negatives.
    """
    if len(labels) == 0:
        return features, labels, pair_indices

    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if len(pos_idx) == 0 or len(neg_idx) == 0 or neg_ratio <= 0:
        return features, labels, pair_indices

    max_neg = min(len(neg_idx), int(max(1, round(len(pos_idx) * float(neg_ratio)))))
    perm = torch.randperm(len(neg_idx))[:max_neg]
    keep_idx = torch.cat([pos_idx, neg_idx[perm]], dim=0)
    keep_idx = keep_idx[torch.randperm(len(keep_idx))]
    return features[keep_idx], labels[keep_idx], pair_indices[keep_idx]


def scatter_pair_probabilities(seq_len: int, pair_indices: torch.Tensor, probs: torch.Tensor):
    """Scatter per-pair probabilities back into a symmetric NxN score matrix."""
    mat = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    for (i, j), p in zip(pair_indices.tolist(), probs.tolist()):
        mat[i, j] = float(p)
        mat[j, i] = float(p)
    return mat
