import torch
import matplotlib.pyplot as plt
import numpy as np

def visualise_attention(image_filename: str, ground_truth_filename: str):
    attn_list = torch.load(image_filename, weights_only=True)
    
    attn_numpy = []
    for attn in attn_list:
        if attn is None:
            attn_numpy.append(None)
            continue
        # e.g. attn.shape = (1, H, L, L)
        a = attn.detach().cpu().numpy()
        if a.ndim == 4:  # batch dimension first
            a = a[0]  # index batch dimension
        attn_numpy.append(a)  # shape now (H, L, L)

    bases, gt_matrix = load_bpseq(ground_truth_filename)
    base_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    base_indices = np.array([base_map.get(b, -1) for b in bases])
    # One-hot encode: shape (seq_len, 4)
    onehot = np.eye(len(base_map), dtype=int)[base_indices.clip(0, 3)]
    # Transpose to (4, seq_len) so rows = base categories
    base_onehot = onehot.T  # shape (4, seq_len)
    
    for i, layer_attn in enumerate(attn_numpy):
        if layer_attn is None:
            print(f"Layer {i}: no attention stored")
            continue
        H, _, _ = layer_attn.shape
        
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
        axs = axs.flatten()
        # assert H == 8

        for idx in range(H):
            ax = axs[idx]
            mat = layer_attn[idx]
            ax.imshow(mat, aspect="auto", cmap="viridis_r")
            ax.set_title(f"Layer {i + 1}: Head {idx}")

        ax = axs[H]
        avg_mat = layer_attn.mean(axis=0)
        ax.imshow(avg_mat, aspect="auto", cmap="viridis_r")
        ax.set_title(f"Layer {i + 1}: Average Head")
        print(f"Layer {i + 1} average head stats:", avg_mat.min(), avg_mat.max(), avg_mat.mean())
        
        plt.suptitle(f"Layer {i + 1}: Attention")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.1)
    
    fig_gt, ax_gt = plt.subplots(figsize=(6, 6))
    ax_gt.imshow(gt_matrix, aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
    ax_gt.set_title("Ground truth")
    fig_gt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    fig_base, ax_base = plt.subplots(figsize=(6, 2))
    ax_base.imshow(base_onehot, aspect="auto", cmap="Greys", interpolation="none")
    # Put row labels for bases
    ax_base.set_yticks([0, 1, 2, 3])
    ax_base.set_yticklabels(["A", "C", "G", "U"])
    # No x tick labels
    ax_base.set_xticks([])
    ax_base.set_title("Bases")
    fig_base.tight_layout()

    plt.show(block=True)

def load_bpseq(bpseq_filename):
    """Reads a .bpseq file into base sequence and pairing matrix."""
    bases = []
    pairs = []
    with open(bpseq_filename) as f:
        for line in f:
            if not line.strip():
                continue
            _, base, pair = line.strip().split()
            bases.append(base)
            pairs.append(int(pair))
    n = len(bases)
    pairing_matrix = np.zeros((n, n))
    for i, j in enumerate(pairs):
        if j != 0:
            pairing_matrix[i, j - 1] = 1
    return bases, pairing_matrix

visualise_attention("../attn_weights_262974_TS0-canonicals/bpRNA_RFAM_19252.attn.pt", "../data/bpRNA_dataset-canonicals/TS0/bpRNA_RFAM_19252.bpseq")