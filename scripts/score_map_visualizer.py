#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _load_dump(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _crop_dummy(mat: torch.Tensor, seq_len: int | None):
    if seq_len is None:
        return mat
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        return mat
    if tuple(mat.shape) == (seq_len + 1, seq_len + 1):
        return mat[1:, 1:]
    return mat


def _select_maps(payload):
    maps = {}
    for key in ("base_param_maps", "fused_param_maps"):
        block = payload.get(key)
        if isinstance(block, dict):
            maps.update(block)
    for key in ("sinkhorn_mask", "sinkhorn_scores", "base_scores", "fused_scores"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            maps[key] = value
    return maps


def main():
    parser = argparse.ArgumentParser(description="Visualize dumped score maps from predict.py")
    parser.add_argument("dump", type=Path, help="path to a .score_maps.pt dump file")
    parser.add_argument("--keys", nargs="*", default=None, help="optional specific map keys to plot")
    parser.add_argument("--crop-dummy", action="store_true", help="crop the dummy 0th row/column when present")
    parser.add_argument("--save-dir", type=Path, default=None, help="optional directory to save PNGs instead of only showing them")
    parser.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap (default: viridis)")
    args = parser.parse_args()

    payload = _load_dump(args.dump)
    seq = payload.get("sequence")
    seq_len = len(seq) if isinstance(seq, str) else None
    maps = _select_maps(payload)

    if args.keys is not None and len(args.keys) > 0:
        maps = {k: v for k, v in maps.items() if k in set(args.keys)}

    if not maps:
        raise ValueError("No visualizable score maps found in the dump.")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for key, mat in maps.items():
        if not isinstance(mat, torch.Tensor):
            continue
        plot_mat = mat
        if args.crop_dummy:
            plot_mat = _crop_dummy(plot_mat, seq_len)
        plot_np = plot_mat.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(plot_np, aspect="auto", cmap=args.cmap)
        ax.set_title(key)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        if args.save_dir is not None:
            out = args.save_dir / f"{args.dump.stem}.{key.replace('.', '_')}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
        else:
            plt.show(block=False)
            plt.pause(0.1)

    if args.save_dir is None:
        plt.show()


if __name__ == "__main__":
    main()
