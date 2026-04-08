#!/usr/bin/env python3

import argparse
from pathlib import Path

PREDICT_MODEL_SCALAR_ARGS = (
    'max_helix_length',
    'embed_size',
    'dilation',
    'num_lstm_layers',
    'num_lstm_units',
    'num_transformer_layers',
    'num_transformer_hidden_units',
    'num_transformer_att',
    'dropout_rate',
    'fc_dropout_rate',
    'num_att',
    'pair_join',
    'num_sinkhorn_iterations',
    'sinkhorn_block_size',
    'sinkhorn_temp',
    'sinkhorn_alpha',
    'sinkhorn_temp_length_center',
    'sinkhorn_temp_length_slope',
    'sinkhorn_pooling',
    'sinkhorn_gating',
    'sinkhorn_diag_logit_penalty',
)

PREDICT_MODEL_MULTI_ARGS = (
    'num_filters',
    'filter_size',
    'pool_size',
    'num_paired_filters',
    'paired_filter_size',
    'num_hidden_units',
)

PREDICT_MODEL_BOOL_ARGS = (
    'no_split_lr',
    'use_sparse_sinkhorn',
    'sinkhorn_use_slack',
    'sinkhorn_real_only_normalization',
)


def _option_name(name: str) -> str:
    return "--" + name.replace("_", "-")


BASE_MODEL_OPTIONS = {"--model", "--param"}
BASE_MODEL_OPTIONS.update(_option_name(name) for name in PREDICT_MODEL_SCALAR_ARGS)
BASE_MODEL_OPTIONS.update(_option_name(name) for name in PREDICT_MODEL_MULTI_ARGS)
BASE_MODEL_OPTIONS.update(_option_name(name) for name in PREDICT_MODEL_BOOL_ARGS)

ENSEMBLE_MAP = {"--model": "--ensemble-model", "--param": "--ensemble-sinkhorn-param"}
ENSEMBLE_MAP.update(
    {
        _option_name(name): _option_name("ensemble_" + name)
        for name in (
            list(PREDICT_MODEL_SCALAR_ARGS)
            + list(PREDICT_MODEL_MULTI_ARGS)
            + list(PREDICT_MODEL_BOOL_ARGS)
        )
    }
)


def read_argfile(path: Path):
    tokens = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


def parse_entries(path: Path):
    tokens = read_argfile(path)
    entries = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            entries.append((token, tokens[i + 1]))
            i += 2
        else:
            entries.append((token, None))
            i += 1
    return entries


def write_entries(path: Path, entries):
    with path.open("w") as f:
        for option, value in entries:
            f.write(f"{option}\n")
            if value is not None:
                f.write(f"{value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create an ensemble predict @conf from a base-model conf and a SinkhornContact conf."
    )
    parser.add_argument("base_conf", type=Path, help="saved conf for the base folding model")
    parser.add_argument("sinkhorn_conf", type=Path, help="saved conf for the SinkhornContact model")
    parser.add_argument("output_conf", type=Path, help="path to write the merged ensemble conf")
    parser.add_argument("--base-weight", type=float, default=None, help="optional ensemble base weight")
    parser.add_argument("--pair-threshold", type=float, default=None, help="optional ensemble pair threshold")
    parser.add_argument(
        "--decoder",
        choices=("reciprocal", "matching"),
        default=None,
        help="optional ensemble decoder",
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=None,
        help="optional shared minimum separation for pair decoding",
    )
    args = parser.parse_args()

    base_entries = parse_entries(args.base_conf)
    sinkhorn_entries = parse_entries(args.sinkhorn_conf)

    base_model = next((value for option, value in base_entries if option == "--model"), None)
    sinkhorn_model = next((value for option, value in sinkhorn_entries if option == "--model"), None)
    if base_model == "SinkhornContact":
        raise ValueError("Base conf must describe the primary folding model, not SinkhornContact.")
    if sinkhorn_model is not None and sinkhorn_model != "SinkhornContact":
        raise ValueError("Sinkhorn conf must describe a SinkhornContact model.")

    merged = []
    for option, value in base_entries:
        if option in BASE_MODEL_OPTIONS:
            merged.append((option, value))

    for option, value in sinkhorn_entries:
        mapped = ENSEMBLE_MAP.get(option)
        if mapped is not None:
            if value is None and option in {_option_name(name) for name in PREDICT_MODEL_BOOL_ARGS}:
                value = "true"
            merged.append((mapped, value))

    if args.base_weight is not None:
        merged.append(("--ensemble-base-weight", str(args.base_weight)))
    if args.pair_threshold is not None:
        merged.append(("--ensemble-pair-threshold", str(args.pair_threshold)))
    if args.decoder is not None:
        merged.append(("--ensemble-decoder", args.decoder))
    if args.min_separation is not None:
        merged.append(("--sinkhorn-min-separation", str(args.min_separation)))

    write_entries(args.output_conf, merged)


if __name__ == "__main__":
    main()
