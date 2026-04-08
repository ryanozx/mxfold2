import random
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset
from .pair_fusion import (
    PairFusionMLP,
    build_pairwise_features,
    pair_indicator_matrix,
    sample_pair_batch,
    scatter_pair_probabilities,
)
from .predict import Predict

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _add_time(stats, key: str, dt: float):
    if stats is not None:
        stats[key] = stats.get(key, 0.0) + float(dt)


def _format_timing(stats) -> str:
    if not stats:
        return ""
    parts = [f"{k}={v:.3f}s" for k, v in sorted(stats.items())]
    return " ".join(parts)


def _build_fusion_checkpoint(
    model,
    args,
    input_dim: int,
    best_threshold: float,
    best_val_loss: float,
    *,
    optimizer=None,
    epoch=None,
    train_loss=None,
    val_loss=None,
    global_step=None,
):
    """Save only the learned fusion head and reload metadata, not the frozen upstream models."""
    state = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_threshold": best_threshold,
        "decoder": args.decoder,
        "min_separation": args.min_separation,
        "base_conf": str(args.base_conf),
        "sinkhorn_conf": str(args.sinkhorn_conf),
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = int(epoch)
    if train_loss is not None:
        state["train_loss"] = float(train_loss)
    if val_loss is not None:
        state["val_loss"] = float(val_loss)
    if global_step is not None:
        state["global_step"] = int(global_step)
    return state


def _average_loss_on_records(model, records, device, criterion, timing=None):
    """Compute mean BCE loss on precomputed records without decoding."""
    t0 = time.perf_counter()
    total_loss = 0.0
    total_examples = 0
    for payload in records:
        features = payload["features"]
        labels = payload["labels"]
        if len(labels) == 0:
            continue
        with torch.no_grad():
            t = time.perf_counter()
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            _add_time(timing, "val_loss_forward_s", time.perf_counter() - t)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    _add_time(timing, "val_loss_total_s", time.perf_counter() - t0)
    return total_loss / max(1, total_examples)


def _make_predict_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@')
    sub = parser.add_subparsers()
    Predict.add_args(sub)
    return parser


def _load_predict_args(conf_path: Path, dummy_input: str = "dummy.bpseq"):
    parser = _make_predict_parser()
    return parser.parse_args(["predict", dummy_input, f"@{conf_path}"])


def _load_frozen_model(conf_path: Path, device):
    args = _load_predict_args(conf_path)
    predictor = Predict()
    model, _ = predictor.build_model(args)
    param = Path(args.param)
    if not param.exists():
        param = conf_path.parent / param
    state = torch.load(param, map_location="cpu")
    state = predictor._remap_state_dict_for_model(model, state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return predictor, model, args


def _compute_record(
    header,
    seq,
    ref_pairs,
    base_predictor,
    base_model,
    sink_predictor,
    sink_model,
    min_separation: int,
    timing=None,
):
    t0 = time.perf_counter()
    with torch.no_grad():
        t = time.perf_counter()
        _, _, base_pairs, _, bpps = base_model([seq], return_partfunc=True)
        _add_time(timing, "base_forward_s", time.perf_counter() - t)

        t = time.perf_counter()
        sink_out = sink_model([seq], return_aux=True)
        _add_time(timing, "sink_forward_s", time.perf_counter() - t)

        t = time.perf_counter()
        base_scores = base_predictor._real_pair_scores_from_bpp(bpps[0], len(seq))
        sink_scores = sink_predictor._real_pair_scores_from_sinkhorn_mask(sink_out["sinkhorn_mask"][0]).cpu()
        base_indicator = pair_indicator_matrix(base_pairs[0], len(seq))
        _add_time(timing, "record_postprocess_s", time.perf_counter() - t)

        t = time.perf_counter()
        features, labels, pair_indices = build_pairwise_features(
            seq=seq,
            base_scores=base_scores.cpu(),
            sinkhorn_scores=sink_scores,
            base_pair_indicator=base_indicator,
            ref_pairs=ref_pairs,
            min_separation=min_separation,
        )
        _add_time(timing, "feature_build_s", time.perf_counter() - t)

    _add_time(timing, "compute_record_total_s", time.perf_counter() - t0)

    return {
        "header": header,
        "seq": seq[1:] if len(seq) > 0 and seq[0] == "0" else seq,
        "ref_pairs": ref_pairs.detach().cpu() if isinstance(ref_pairs, torch.Tensor) else ref_pairs,
        "features": features,
        "labels": labels,
        "pair_indices": pair_indices,
    }


def _materialize_records(dataset, base_predictor, base_model, sink_predictor, sink_model, min_separation: int, timing=None):
    """Precompute frozen-model features once so repeated threshold sweeps are cheap."""
    t0 = time.perf_counter()
    records = []
    for header, seq, ref_pairs, _ in dataset:
        records.append(
            _compute_record(
                header,
                seq,
                ref_pairs,
                base_predictor,
                base_model,
                sink_predictor,
                sink_model,
                min_separation=min_separation,
                timing=timing,
            )
        )
    _add_time(timing, "materialize_records_s", time.perf_counter() - t0)
    return records


def _evaluate_records(model, records, device, threshold: float, decoder: str, min_separation: int, timing=None):
    t0 = time.perf_counter()
    totals = [0, 0, 0, 0]
    for payload in records:
        features = payload["features"].to(device)
        pair_indices = payload["pair_indices"]
        seq = payload["seq"]
        ref_pairs = payload["ref_pairs"]
        with torch.no_grad():
            t = time.perf_counter()
            logits = model(features) if len(features) > 0 else torch.zeros((0,), device=device)
            probs = torch.sigmoid(logits).detach().cpu()
            _add_time(timing, "eval_mlp_s", time.perf_counter() - t)

        t = time.perf_counter()
        score_matrix = scatter_pair_probabilities(len(seq), pair_indices, probs)
        bp, _ = Predict._decode_real_pair_scores(
            score_matrix,
            decoder=decoder,
            min_separation=min_separation,
            pair_threshold=threshold,
        )
        _add_time(timing, "eval_decode_s", time.perf_counter() - t)

        t = time.perf_counter()
        tp, tn, fp, fn = compare_bpseq(ref_pairs, bp)
        _add_time(timing, "eval_compare_s", time.perf_counter() - t)
        totals[0] += tp
        totals[1] += tn
        totals[2] += fp
        totals[3] += fn
    _add_time(timing, "evaluate_records_total_s", time.perf_counter() - t0)
    return totals, accuracy(*totals)


def _evaluate_dataset(model, dataset, base_predictor, base_model, sink_predictor, sink_model, device, threshold: float, decoder: str, min_separation: int):
    records = _materialize_records(
        dataset,
        base_predictor,
        base_model,
        sink_predictor,
        sink_model,
        min_separation=min_separation,
    )
    return _evaluate_records(model, records, device, threshold, decoder, min_separation)


def _write_predictions(model, records, device, threshold: float, decoder: str, min_separation: int, result_path=None, bpseq_dir=None, timing=None):
    res_fn = open(result_path, "w") if result_path is not None else None
    bpseq_dir = Path(bpseq_dir) if bpseq_dir is not None else None
    if bpseq_dir is not None:
        bpseq_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    totals = [0, 0, 0, 0]
    for payload in records:
        header = payload["header"]
        seq = payload["seq"]
        ref_pairs = payload["ref_pairs"]
        features = payload["features"].to(device)
        pair_indices = payload["pair_indices"]

        with torch.no_grad():
            t = time.perf_counter()
            logits = model(features) if len(features) > 0 else torch.zeros((0,), device=device)
            probs = torch.sigmoid(logits).detach().cpu()
            _add_time(timing, "predict_mlp_s", time.perf_counter() - t)
        t = time.perf_counter()
        score_matrix = scatter_pair_probabilities(len(seq), pair_indices, probs)
        bp, sc = Predict._decode_real_pair_scores(
            score_matrix,
            decoder=decoder,
            min_separation=min_separation,
            pair_threshold=threshold,
        )
        _add_time(timing, "predict_decode_s", time.perf_counter() - t)

        if bpseq_dir is not None:
            t = time.perf_counter()
            out = bpseq_dir / f"{Path(header).stem}.bpseq"
            with out.open("w") as f:
                print(f"# {header} (s={float(sc.item()):.4f}, 0.00000s)", file=f)
                for i in range(1, len(bp)):
                    print(f"{i}\t{seq[i-1]}\t{bp[i]}", file=f)
            _add_time(timing, "predict_write_bpseq_s", time.perf_counter() - t)

        if res_fn is not None:
            t = time.perf_counter()
            x = compare_bpseq(ref_pairs, bp)
            totals[0] += x[0]
            totals[1] += x[1]
            totals[2] += x[2]
            totals[3] += x[3]
            vals = [header, len(seq), 0.0, float(sc.item())] + list(x) + list(accuracy(*x))
            res_fn.write(", ".join(str(v) for v in vals) + "\n")
            _add_time(timing, "predict_write_result_s", time.perf_counter() - t)

    if res_fn is not None:
        res_fn.close()
    _add_time(timing, "write_predictions_total_s", time.perf_counter() - t0)
    return totals


class PairFusionCLI:
    @staticmethod
    def cmd_train(args, conf=None):
        del conf
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
        writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir is not None and SummaryWriter is not None else None

        train_dataset = BPseqDataset(str(args.train_input))
        valid_dataset = BPseqDataset(str(args.valid_input))
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")
        if args.log_dir is not None:
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)

        base_predictor, base_model, _ = _load_frozen_model(Path(args.base_conf), device)
        sink_predictor, sink_model, _ = _load_frozen_model(Path(args.sinkhorn_conf), device)
        setup_timing = {} if args.log_timing else None
        valid_records = _materialize_records(
            valid_dataset,
            base_predictor,
            base_model,
            sink_predictor,
            sink_model,
            min_separation=args.min_separation,
            timing=setup_timing,
        )
        if args.log_timing:
            print(f"[timing] validation_materialize {_format_timing(setup_timing)}")

        input_dim = None
        for header, seq, ref_pairs, _ in train_dataset:
            sample = _compute_record(
                header,
                seq,
                ref_pairs,
                base_predictor,
                base_model,
                sink_predictor,
                sink_model,
                min_separation=args.min_separation,
            )
            if sample["features"].ndim == 2 and sample["features"].shape[1] > 0:
                input_dim = int(sample["features"].shape[1])
                break
        if input_dim is None:
            raise ValueError("Could not infer fusion input dimension from the training dataset.")

        model = PairFusionMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        start_epoch = 1
        global_step = 0
        best_val_loss = float("inf")
        best_state = None
        best_threshold = args.threshold
        if args.resume is not None:
            resume_path = Path(args.resume)
            resume_state = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(resume_state["model_state_dict"])
            if "optimizer_state_dict" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
            best_threshold = float(resume_state.get("best_threshold", best_threshold))
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            global_step = int(resume_state.get("global_step", 0))
            print(
                f"resumed fusion training from {resume_path} "
                f"at epoch={start_epoch} best_val_loss={best_val_loss:.6f}"
            )

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_timing = {} if args.log_timing else None
            train_indices = list(range(len(train_dataset)))
            random.shuffle(train_indices)
            train_loss = 0.0
            train_steps = 0
            for idx in train_indices:
                header, seq, ref_pairs, _ = train_dataset[idx]
                payload = _compute_record(
                    header,
                    seq,
                    ref_pairs,
                    base_predictor,
                    base_model,
                    sink_predictor,
                    sink_model,
                    min_separation=args.min_separation,
                    timing=epoch_timing,
                )
                features = payload["features"]
                labels = payload["labels"]
                pair_indices = payload["pair_indices"]
                t = time.perf_counter()
                features, labels, pair_indices = sample_pair_batch(features, labels, pair_indices, neg_ratio=args.neg_ratio)
                _add_time(epoch_timing, "train_sample_batch_s", time.perf_counter() - t)
                if len(labels) == 0:
                    continue
                t = time.perf_counter()
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                _add_time(epoch_timing, "train_opt_step_s", time.perf_counter() - t)
                loss_value = float(loss.item())
                train_loss += loss_value
                train_steps += 1
                global_step += 1

                if writer is not None and args.log_interval > 0 and (global_step % args.log_interval == 0):
                    writer.add_scalar("train/step_loss", loss_value, global_step)
                    writer.add_scalar("train/running_loss", train_loss / max(1, train_steps), global_step)

            model.eval()
            avg_train_loss = train_loss / max(1, train_steps)
            val_loss_timing = {} if args.log_timing else None
            val_loss = _average_loss_on_records(
                model,
                valid_records,
                device=device,
                criterion=criterion,
                timing=val_loss_timing,
            )

            if writer is not None:
                writer.add_scalar("train/loss", avg_train_loss, epoch)
                writer.add_scalar("valid/loss", val_loss, epoch)

            print(f"epoch={epoch} train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f}")
            if args.log_timing:
                print(f"[timing] train_epoch {_format_timing(epoch_timing)}")
                print(f"[timing] val_loss {_format_timing(val_loss_timing)}")
            improved = val_loss < best_val_loss

            epoch_state = _build_fusion_checkpoint(
                model,
                args,
                input_dim=input_dim,
                best_threshold=best_threshold,
                best_val_loss=min(best_val_loss, val_loss),
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                global_step=global_step,
            )
            if args.log_dir is not None:
                epoch_ckpt = Path(args.log_dir) / f"checkpoint_epoch_{epoch}.pth"
                torch.save(epoch_state, epoch_ckpt)

            if improved:
                best_val_loss = val_loss

            if improved:
                best_state = _build_fusion_checkpoint(
                    model,
                    args,
                    input_dim=input_dim,
                    best_threshold=best_threshold,
                    best_val_loss=best_val_loss,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    global_step=global_step,
                )
                torch.save(best_state, args.output_param)

        if writer is not None:
            writer.close()
        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")
        print(f"saved best fusion model to {args.output_param} with threshold={best_threshold:.3f} val_loss={best_val_loss:.6f}")

    @staticmethod
    def cmd_predict(args, conf=None):
        del conf
        device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
        checkpoint = torch.load(args.param, map_location="cpu")
        model = PairFusionMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dataset = BPseqDataset(str(args.input))
        base_predictor, base_model, _ = _load_frozen_model(Path(args.base_conf), device)
        sink_predictor, sink_model, _ = _load_frozen_model(Path(args.sinkhorn_conf), device)
        threshold = args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5)
        decoder = args.decoder if args.decoder is not None else checkpoint.get("decoder", "matching")
        min_separation = args.min_separation if args.min_separation is not None else checkpoint.get("min_separation", 4)
        predict_timing = {} if args.log_timing else None
        records = _materialize_records(
            dataset,
            base_predictor,
            base_model,
            sink_predictor,
            sink_model,
            min_separation=min_separation,
            timing=predict_timing,
        )

        totals = _write_predictions(
            model,
            records,
            device=device,
            threshold=threshold,
            decoder=decoder,
            min_separation=min_separation,
            result_path=args.result,
            bpseq_dir=args.bpseq,
            timing=predict_timing,
        )
        if args.result is not None:
            metrics = accuracy(*totals)
            sen, ppv, f1, mcc = metrics
            print(f"sen={sen:.4f} ppv={ppv:.4f} f1={f1:.4f} mcc={mcc:.4f}")
        if args.log_timing:
            print(f"[timing] predict {_format_timing(predict_timing)}")

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('pair_fusion', help='learned pairwise fusion on frozen base and Sinkhorn models')
        pair_sub = subparser.add_subparsers(dest='pair_fusion_cmd', required=True)

        train = pair_sub.add_parser("train", help="train a learned pairwise fusion head on frozen base and Sinkhorn models")
        train.add_argument("train_input", type=str)
        train.add_argument("--valid-input", required=True, type=str)
        train.add_argument("--base-conf", required=True, type=str)
        train.add_argument("--sinkhorn-conf", required=True, type=str)
        train.add_argument("--output-param", required=True, type=str)
        train.add_argument("--epochs", type=int, default=10)
        train.add_argument("--lr", type=float, default=1e-3)
        train.add_argument("--hidden-dim", type=int, default=32)
        train.add_argument("--dropout", type=float, default=0.1)
        train.add_argument("--neg-ratio", type=float, default=5.0)
        train.add_argument("--decoder", choices=("reciprocal", "matching"), default="matching")
        train.add_argument("--threshold", type=float, default=0.4)
        train.add_argument("--min-separation", type=int, default=4)
        train.add_argument("--gpu", type=int, default=-1)
        train.add_argument("--seed", type=int, default=0)
        train.add_argument("--log-dir", type=str, default=None)
        train.add_argument("--resume", type=str, default=None)
        train.add_argument("--log-interval", type=int, default=100)
        train.add_argument("--log-timing", action="store_true")
        train.set_defaults(func=cls.cmd_train)

        pred = pair_sub.add_parser("predict", help="run a trained pairwise fusion head")
        pred.add_argument("input", type=str)
        pred.add_argument("--base-conf", required=True, type=str)
        pred.add_argument("--sinkhorn-conf", required=True, type=str)
        pred.add_argument("--param", required=True, type=str)
        pred.add_argument("--threshold", type=float, default=None)
        pred.add_argument("--decoder", choices=("reciprocal", "matching"), default=None)
        pred.add_argument("--min-separation", type=int, default=None)
        pred.add_argument("--bpseq", type=str, default=None)
        pred.add_argument("--result", type=str, default=None)
        pred.add_argument("--gpu", type=int, default=-1)
        pred.add_argument("--log-timing", action="store_true")
        pred.set_defaults(func=cls.cmd_predict)
