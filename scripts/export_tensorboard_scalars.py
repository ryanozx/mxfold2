#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all TensorBoard scalar data under a log directory into one CSV file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="TensorBoard log directory or a single TensorBoard event file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-tensor-scalars",
        action="store_true",
        help="Also export scalar summaries stored in the tensors collection.",
    )
    return parser.parse_args()


def find_run_dirs(input_path: Path):
    if input_path.is_file():
        return [input_path.parent]

    event_files = sorted(input_path.rglob("events.out.tfevents.*"))
    run_dirs = sorted({event_file.parent for event_file in event_files})
    return run_dirs


def relative_run_name(root: Path, run_dir: Path):
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        return run_dir.name

    return "." if rel == Path(".") else rel.as_posix()


def read_scalar_rows(run_dir: Path, run_name: str, include_tensor_scalars: bool):
    acc = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    acc.Reload()

    rows = []

    for tag in acc.Tags().get("scalars", []):
        for event in acc.Scalars(tag):
            rows.append(
                {
                    "run": run_name,
                    "tag": tag,
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                }
            )

    if include_tensor_scalars:
        for tag in acc.Tags().get("tensors", []):
            for event in acc.Tensors(tag):
                value = tensor_util.make_ndarray(event.tensor_proto)
                if value.size != 1:
                    continue
                rows.append(
                    {
                        "run": run_name,
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": float(value.reshape(-1)[0]),
                    }
                )

    return rows


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    root = input_path.parent if input_path.is_file() else input_path
    run_dirs = find_run_dirs(input_path)
    if not run_dirs:
        raise FileNotFoundError(
            f"No TensorBoard event files found under: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run", "tag", "step", "wall_time", "value"],
        )
        writer.writeheader()

        for run_dir in run_dirs:
            run_name = relative_run_name(root, run_dir)
            rows = read_scalar_rows(
                run_dir=run_dir,
                run_name=run_name,
                include_tensor_scalars=args.include_tensor_scalars,
            )
            rows.sort(key=lambda row: (row["tag"], row["step"], row["wall_time"]))
            writer.writerows(rows)
            total_rows += len(rows)

    print(
        f"Exported {total_rows} scalar points from {len(run_dirs)} run directories to {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
