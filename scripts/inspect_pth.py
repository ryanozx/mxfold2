#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


def format_shape(shape) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def summarize_value(value) -> str:
    if torch.is_tensor(value):
        return (
            f"Tensor shape={format_shape(value.shape)} "
            f"dtype={value.dtype} device={value.device}"
        )
    if isinstance(value, Mapping):
        return f"{type(value).__name__} with {len(value)} keys"
    if isinstance(value, (str, bytes)):
        return f"{type(value).__name__} value={value!r}"
    if isinstance(value, Sequence):
        return f"{type(value).__name__} length={len(value)}"
    return f"{type(value).__name__} value={value!r}"


def resolve_state_dict(obj, explicit_key: str | None):
    if explicit_key is not None:
        if not isinstance(obj, Mapping):
            raise KeyError(f"Top-level object is not a mapping, cannot use key {explicit_key!r}")
        return explicit_key, obj[explicit_key]

    if isinstance(obj, Mapping):
        for candidate in ("model_state_dict", "state_dict"):
            if candidate in obj and isinstance(obj[candidate], Mapping):
                return candidate, obj[candidate]
        if all(isinstance(k, str) for k in obj.keys()) and all(
            torch.is_tensor(v) for v in obj.values()
        ):
            return None, obj

    return None, None


def print_top_level(obj) -> None:
    print(f"Top-level type: {type(obj).__name__}")
    if not isinstance(obj, Mapping):
        print(f"Summary: {summarize_value(obj)}")
        return

    print(f"Top-level keys: {len(obj)}")
    for key, value in obj.items():
        print(f"  {key}: {summarize_value(value)}")


def print_state_dict(state_dict, prefix: str | None, contains: str | None, limit: int | None) -> None:
    items = list(state_dict.items())
    if prefix:
        items = [(k, v) for k, v in items if k.startswith(prefix)]
    if contains:
        items = [(k, v) for k, v in items if contains in k]

    print(f"state_dict entries: {len(items)}")
    shown = items if limit is None else items[:limit]
    for key, value in shown:
        print(f"  {key}: {summarize_value(value)}")

    if limit is not None and len(items) > limit:
        print(f"... {len(items) - limit} more entries not shown")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch .pth/.pt checkpoint without loading it into a model."
    )
    parser.add_argument("path", type=Path, help="Path to the .pth/.pt file")
    parser.add_argument(
        "--state-dict-key",
        type=str,
        default=None,
        help="Explicit top-level key containing the state_dict, e.g. model_state_dict",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only show state_dict keys starting with this prefix",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Only show state_dict keys containing this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of matching state_dict entries to print; use -1 for all",
    )
    parser.add_argument(
        "--top-only",
        action="store_true",
        help="Only print the top-level checkpoint structure",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"File not found: {args.path}")

    obj = torch.load(args.path, map_location="cpu")
    print_top_level(obj)

    if args.top_only:
        return

    state_dict_key, state_dict = resolve_state_dict(obj, args.state_dict_key)
    if state_dict is None:
        print("No state_dict detected.")
        return

    print()
    if state_dict_key is None:
        print("Using top-level object as state_dict")
    else:
        print(f"Using state_dict from top-level key: {state_dict_key}")

    limit = None if args.limit < 0 else args.limit
    print_state_dict(state_dict, args.prefix, args.contains, limit)


if __name__ == "__main__":
    main()
