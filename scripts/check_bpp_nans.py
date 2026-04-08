#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def scan_bpp(path: Path) -> tuple[int, int, tuple[int, ...] | None]:
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception as exc:
        print(f"ERROR {path}: failed to read ({exc})")
        return -1, -1, None

    if arr.ndim == 0:
        arr = np.array([[float(arr)]], dtype=np.float32)
    elif arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    return nan_count, inf_count, arr.shape


def iter_lst_entries(lst_path: Path):
    with lst_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield raw_line, line.split()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read a .lst file of BPSEQ paths, check the corresponding .bpp files by stem, "
            "and write a filtered .lst containing only entries with finite BPPs."
        )
    )
    parser.add_argument("lst_file", type=Path, help="Input .lst file containing BPSEQ paths")
    parser.add_argument("bpp_dir", type=Path, help="Directory containing cached .bpp files")
    parser.add_argument("output_lst", type=Path, help="Output .lst file containing only good entries")
    parser.add_argument(
        "--bad-lst",
        type=Path,
        default=None,
        help="Optional output .lst file containing only bad entries",
    )
    parser.add_argument(
        "--show-clean",
        action="store_true",
        help="Also print clean entries as they are checked",
    )
    args = parser.parse_args()

    if not args.lst_file.exists():
        raise FileNotFoundError(f".lst file does not exist: {args.lst_file}")
    if not args.bpp_dir.exists():
        raise FileNotFoundError(f"BPP directory does not exist: {args.bpp_dir}")
    if not args.bpp_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {args.bpp_dir}")

    total_entries = 0
    good_entries = 0
    bad_entries = 0
    total_nans = 0
    total_infs = 0

    args.output_lst.parent.mkdir(parents=True, exist_ok=True)
    bad_handle = None
    if args.bad_lst is not None:
        args.bad_lst.parent.mkdir(parents=True, exist_ok=True)
        bad_handle = args.bad_lst.open("w")

    with args.output_lst.open("w") as good_out:
        try:
            for raw_line, parts in iter_lst_entries(args.lst_file):
                total_entries += 1
                bpseq_path = Path(parts[0])
                bpp_path = args.bpp_dir / f"{bpseq_path.stem}.bpp"

                if not bpp_path.exists():
                    bad_entries += 1
                    print(f"BAD {bpseq_path} missing_bpp={bpp_path}")
                    if bad_handle is not None:
                        bad_handle.write(raw_line)
                    continue

                nan_count, inf_count, shape = scan_bpp(bpp_path)
                if nan_count < 0 or inf_count > 0 or nan_count > 0:
                    bad_entries += 1
                    total_nans += max(nan_count, 0)
                    total_infs += max(inf_count, 0)
                    print(f"BAD {bpseq_path} bpp={bpp_path} shape={shape} nan={nan_count} inf={inf_count}")
                    if bad_handle is not None:
                        bad_handle.write(raw_line)
                    continue

                good_entries += 1
                good_out.write(raw_line)
                if args.show_clean:
                    print(f"OK  {bpseq_path} bpp={bpp_path} shape={shape}")
        finally:
            if bad_handle is not None:
                bad_handle.close()

    print(
        f"Processed {total_entries} entries; "
        f"good_entries={good_entries} bad_entries={bad_entries} "
        f"total_nans={total_nans} total_infs={total_infs}"
    )
    print(f"Wrote good entries to {args.output_lst}")
    if args.bad_lst is not None:
        print(f"Wrote bad entries to {args.bad_lst}")


if __name__ == "__main__":
    main()
