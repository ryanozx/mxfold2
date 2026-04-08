import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from argparse import ArgumentTypeError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx

from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset, FastaDataset
from .fold.contact import SinkhornContactFold
from .fold.mix import MixedFold
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold
from .fold.layers import TransformerLayer, JOIN_MAP, JoinType


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


def _parse_optional_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ArgumentTypeError(f"Expected a boolean value, got: {value}")


class Predict:
    def __init__(self):
        self.test_loader = None
        self.ensemble_sinkhorn_model = None

    @staticmethod
    def _decode_real_pair_scores_thresholded(
        scores: torch.Tensor,
        min_separation: int = 4,
        pair_threshold: float = 0.5,
    ):
        if scores.dim() != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError(f"Expected square 2D pair score matrix, got {tuple(scores.shape)}")

        l = int(scores.shape[0])
        if l == 0:
            return [0], torch.tensor(0.0, device=scores.device)

        threshold = float(pair_threshold)
        candidate_cols = []
        candidate_scores = []
        for i in range(l):
            row = scores[i].clone()
            row[i] = 0.0
            lo = max(0, i - int(min_separation))
            hi = min(l, i + int(min_separation) + 1)
            row[lo:hi] = 0.0

            score, j0 = torch.max(row, dim=0)
            if float(score.item()) >= threshold:
                candidate_cols.append(int(j0.item()) + 1)
                candidate_scores.append(float(score.item()))
            else:
                candidate_cols.append(0)
                candidate_scores.append(0.0)

        bp = [0] * (l + 1)
        accepted_scores = []
        for i in range(1, l + 1):
            j = candidate_cols[i - 1]
            if j <= 0 or j > l or j == i:
                continue
            if candidate_cols[j - 1] != i:
                continue
            if bp[i] != 0 or bp[j] != 0:
                continue
            bp[i] = j
            bp[j] = i
            accepted_scores.append(min(candidate_scores[i - 1], candidate_scores[j - 1]))

        score = float(sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0
        return bp, torch.tensor(score, device=scores.device)

    @staticmethod
    def _decode_real_pair_scores_matching(
        scores: torch.Tensor,
        min_separation: int = 4,
        pair_threshold: float = 0.5,
    ):
        if scores.dim() != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError(f"Expected square 2D pair score matrix, got {tuple(scores.shape)}")

        l = int(scores.shape[0])
        if l == 0:
            return [0], torch.tensor(0.0, device=scores.device)

        threshold = float(pair_threshold)
        graph = nx.Graph()
        graph.add_nodes_from(range(1, l + 1))

        for i in range(l):
            for j in range(i + 1, l):
                if abs(j - i) <= int(min_separation):
                    continue
                weight = float(scores[i, j].item())
                if weight < threshold:
                    continue
                graph.add_edge(i + 1, j + 1, weight=weight)

        matching = nx.max_weight_matching(graph, maxcardinality=False, weight='weight')
        bp = [0] * (l + 1)
        accepted_scores = []
        for i, j in matching:
            if i == j:
                continue
            bp[i] = j
            bp[j] = i
            accepted_scores.append(float(graph[i][j]['weight']))

        score = float(sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0
        return bp, torch.tensor(score, device=scores.device)

    @staticmethod
    def _decode_real_pair_scores(
        scores: torch.Tensor,
        decoder: str = "matching",
        min_separation: int = 4,
        pair_threshold: float = 0.5,
    ):
        if decoder == "reciprocal":
            return Predict._decode_real_pair_scores_thresholded(
                scores,
                min_separation=min_separation,
                pair_threshold=pair_threshold,
            )
        if decoder == "matching":
            return Predict._decode_real_pair_scores_matching(
                scores,
                min_separation=min_separation,
                pair_threshold=pair_threshold,
            )
        raise ValueError(f"Unsupported decoder: {decoder}")

    @staticmethod
    def _real_pair_scores_from_sinkhorn_mask(mask: torch.Tensor):
        if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError(f"Expected square 2D Sinkhorn mask, got {tuple(mask.shape)}")
        if int(mask.shape[0]) <= 1:
            return torch.zeros((0, 0), device=mask.device, dtype=mask.dtype)
        real = mask[1:, 1:]
        real = torch.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * (real + real.transpose(0, 1))

    @staticmethod
    def _real_pair_scores_from_bpp(bpp, seq_len: int):
        scores = torch.as_tensor(bpp, dtype=torch.float32)
        if scores.dim() != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError(f"Expected square 2D BPP matrix, got {tuple(scores.shape)}")

        if tuple(scores.shape) == (seq_len + 1, seq_len + 1):
            scores = scores[1:, 1:]
        elif tuple(scores.shape) != (seq_len, seq_len):
            raise ValueError(
                f"Expected BPP matrix with shape {(seq_len, seq_len)} or {(seq_len + 1, seq_len + 1)}, "
                f"got {tuple(scores.shape)}"
            )

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores.clamp(0.0, 1.0)
        scores = torch.triu(scores)
        return scores + scores.transpose(0, 1)

    @staticmethod
    def _fuse_pair_scores(base_scores: torch.Tensor, sinkhorn_scores: torch.Tensor, base_weight: float):
        if base_scores.shape != sinkhorn_scores.shape:
            raise ValueError(
                f"Cannot fuse score matrices with different shapes: "
                f"{tuple(base_scores.shape)} vs {tuple(sinkhorn_scores.shape)}"
            )
        lam = float(base_weight)
        return lam * base_scores + (1.0 - lam) * sinkhorn_scores

    @staticmethod
    def _apply_sinkhorn_bias_to_param(
        param,
        masks,
        bias_scale: float,
        min_separation: int = 4,
    ):
        def _bias_one(target_param, mask):
            bias = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            if bias.dim() != 2 or bias.shape[0] != bias.shape[1]:
                raise ValueError(f"Expected square 2D Sinkhorn mask, got {tuple(bias.shape)}")

            # Keep the model's internal dummy 0th position. Suppress diagonal and short-range
            # contacts so the Sinkhorn bias only nudges plausible long-range pair scores.
            n = int(bias.shape[0])
            for i in range(n):
                lo = max(0, i - int(min_separation))
                hi = min(n, i + int(min_separation) + 1)
                bias[i, lo:hi] = 0.0
            bias = 0.5 * (bias + bias.transpose(0, 1))
            bias = float(bias_scale) * bias

            paired_keys = (
                'score_basepair',
                'score_helix_stacking',
                'score_mismatch_external',
                'score_mismatch_hairpin',
                'score_mismatch_internal',
                'score_mismatch_multi',
            )
            for key in paired_keys:
                if key in target_param:
                    target_param[key] = target_param[key] + bias.to(
                        device=target_param[key].device,
                        dtype=target_param[key].dtype,
                    )
            return target_param

        fused = []
        for p, mask in zip(param, masks):
            p_out = dict(p)
            if 'positional' in p_out:
                positional = dict(p_out['positional'])
                p_out['positional'] = _bias_one(positional, mask)
            else:
                p_out = _bias_one(p_out, mask)
            fused.append(p_out)
        return fused

    @staticmethod
    def _decode_sinkhorn_contact_bp(mask: torch.Tensor, min_separation: int = 4):
        if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError(f"Expected square 2D Sinkhorn mask, got {tuple(mask.shape)}")

        n = int(mask.shape[0])
        if n <= 1:
            return [0], torch.tensor(0.0, device=mask.device)

        # The standalone model keeps the internal dummy 0th position. Decode only real
        # nucleotides 1..L and require reciprocal argmax agreement to form a base pair.
        real = mask[1:, 1:]
        l = int(real.shape[0])
        row_argmax = torch.argmax(real, dim=1) + 1
        bp = [0] * (l + 1)
        accepted_scores = []

        for i in range(1, l + 1):
            j = int(row_argmax[i - 1].item())
            if j <= 0 or j > l or j == i:
                continue
            if abs(j - i) <= int(min_separation):
                continue
            if int(row_argmax[j - 1].item()) != i:
                continue
            bp[i] = j
            accepted_scores.append(float(real[i - 1, j - 1].item()))

        score = float(sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0
        return bp, torch.tensor(score, device=mask.device)

    @staticmethod
    def _decode_sinkhorn_contact_bp_thresholded(
        mask: torch.Tensor,
        min_separation: int = 4,
        pair_threshold: float = 0.5,
    ):
        scores = Predict._real_pair_scores_from_sinkhorn_mask(mask)
        return Predict._decode_real_pair_scores_thresholded(
            scores,
            min_separation=min_separation,
            pair_threshold=pair_threshold,
        )

    @staticmethod
    def _decode_sinkhorn_contact_bp_matching(
        mask: torch.Tensor,
        min_separation: int = 4,
        pair_threshold: float = 0.5,
    ):
        scores = Predict._real_pair_scores_from_sinkhorn_mask(mask)
        return Predict._decode_real_pair_scores_matching(
            scores,
            min_separation=min_separation,
            pair_threshold=pair_threshold,
        )

    @staticmethod
    def _bpseq_to_simple_dotbracket(bp):
        chars = ['.'] * (len(bp) - 1)
        for i in range(1, len(bp)):
            j = int(bp[i])
            if j > i:
                chars[i - 1] = '('
                chars[j - 1] = ')'
        return ''.join(chars)

    @staticmethod
    def _remap_state_dict_for_model(model, state_dict):
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        if isinstance(model, SinkhornContactFold):
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith("zuker.net."):
                    remapped[key.removeprefix("zuker.")] = value
                else:
                    remapped[key] = value
            return remapped
        return state_dict

    @staticmethod
    def _get_prefixed_arg(args, name, prefix=None, fallback=None):
        if prefix:
            value = getattr(args, f"{prefix}{name}", None)
            if value is not None:
                return value
        return getattr(args, name) if fallback is None else fallback

    @staticmethod
    def _tensor_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: Predict._tensor_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Predict._tensor_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(Predict._tensor_to_cpu(v) for v in value)
        return value

    @staticmethod
    def _extract_square_param_maps(param_dict, prefix=""):
        maps = {}
        for key, value in param_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                maps.update(Predict._extract_square_param_maps(value, prefix=full_key))
            elif isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[0] == value.shape[1]:
                maps[full_key] = value.detach().cpu()
        return maps

    @staticmethod
    def _save_score_map_dump(output_dir, header, seq, payload):
        os.makedirs(output_dir, exist_ok=True)
        fn_base = os.path.splitext(os.path.basename(header))[0]
        out = {"header": header, "sequence": seq}
        out.update(Predict._tensor_to_cpu(payload))
        torch.save(out, os.path.join(output_dir, f"{fn_base}.score_maps.pt"))

    def predict(self, output_bpseq=None, output_bpp=None, result=None, use_constraint=False, output_attn_dir=None, output_score_map_dir=None):
        res_fn = open(result, 'w') if result is not None else None
        self.model.eval()
        if self.ensemble_sinkhorn_model is not None:
            self.ensemble_sinkhorn_model.eval()
        if isinstance(self.model, SinkhornContactFold) and output_bpp is not None:
            raise ValueError("SinkhornContactFold does not support --bpp output")
        if isinstance(self.model, SinkhornContactFold) and use_constraint:
            raise ValueError("SinkhornContactFold does not support constrained prediction")
        if self.ensemble_sinkhorn_model is not None and output_bpp is not None:
            raise ValueError("Late-fusion prediction does not support --bpp output")
        if self.ensemble_sinkhorn_model is not None and use_constraint:
            raise ValueError("Late-fusion prediction does not support constrained prediction")
        if self.ensemble_sinkhorn_model is not None and output_attn_dir is not None:
            raise ValueError("Late-fusion prediction does not support --output-attn-dir")
        if output_attn_dir is not None:
            os.makedirs(output_attn_dir, exist_ok = True)
            print("will return attn weights")
        with torch.no_grad():
            return_attn_weights = output_attn_dir is not None
            for batch in self.test_loader:
                score_map_records = None
                if len(batch) == 4:
                    headers, seqs, refs, _ = batch
                else:
                    headers, seqs, refs = batch
                start = time.time()
                ensemble_mode = getattr(self, "ensemble_mode", "late_fusion")
                if self.ensemble_sinkhorn_model is not None and ensemble_mode == "late_fusion":
                    scs = []
                    preds = []
                    bps = []
                    decoder = getattr(self, "ensemble_decoder", "matching")
                    base_weight = getattr(self, "ensemble_base_weight", 0.5)
                    threshold = getattr(self, "ensemble_pair_threshold", 0.5)
                    scs_base, preds_base, bps_base, pfs, bpps = self.model(
                        seqs,
                        return_partfunc=True,
                        return_attn_weights=False,
                    )
                    sinkhorn_out = self.ensemble_sinkhorn_model(seqs, return_aux=True)
                    masks = sinkhorn_out["sinkhorn_mask"]
                    score_map_records = []
                    for seq, bpp, mask in zip(seqs, bpps, masks):
                        base_scores = self._real_pair_scores_from_bpp(bpp, len(seq))
                        sinkhorn_scores = self._real_pair_scores_from_sinkhorn_mask(mask).to(base_scores.device)
                        fused_scores = self._fuse_pair_scores(base_scores, sinkhorn_scores, base_weight)
                        bp, sc = self._decode_real_pair_scores(
                            fused_scores,
                            decoder=decoder,
                            min_separation=getattr(self, "sinkhorn_min_separation", 4),
                            pair_threshold=threshold,
                        )
                        scs.append(sc)
                        preds.append(self._bpseq_to_simple_dotbracket(bp))
                        bps.append(bp)
                        score_map_records.append({
                            "mode": "late_fusion",
                            "base_scores": base_scores,
                            "sinkhorn_mask": mask,
                            "sinkhorn_scores": sinkhorn_scores,
                            "fused_scores": fused_scores,
                        })
                    pfs = bpps = [None] * len(preds)
                elif self.ensemble_sinkhorn_model is not None and ensemble_mode == "zuker_bias":
                    if isinstance(self.model, SinkhornContactFold):
                        raise ValueError("Zuker-side fusion requires the primary model to be a Zuker/MXfold2 model")
                    base_out = self.model(seqs, return_param=True, return_attn_weights=False)
                    scs_base, preds_base, bps_base, base_param = base_out
                    sinkhorn_out = self.ensemble_sinkhorn_model(seqs, return_aux=True)
                    masks = sinkhorn_out["sinkhorn_mask"]
                    fused_param = self._apply_sinkhorn_bias_to_param(
                        base_param,
                        masks,
                        bias_scale=getattr(self, "ensemble_zuker_bias_scale", 1.0),
                        min_separation=getattr(self, "sinkhorn_min_separation", 4),
                    )
                    scs, preds, bps = self.model(
                        seqs,
                        param=fused_param,
                        return_attn_weights=False,
                    )
                    pfs = bpps = [None] * len(preds)
                    return_attn_weights = False
                    score_map_records = []
                    for base_p, mask, fused_p in zip(base_param, masks, fused_param):
                        score_map_records.append({
                            "mode": "zuker_bias",
                            "base_param_maps": self._extract_square_param_maps(base_p),
                            "sinkhorn_mask": mask,
                            "sinkhorn_scores": self._real_pair_scores_from_sinkhorn_mask(mask),
                            "fused_param_maps": self._extract_square_param_maps(fused_p),
                        })
                elif isinstance(self.model, SinkhornContactFold):
                    out = self.model(seqs, return_aux=True)
                    masks = out["sinkhorn_mask"]
                    scs = []
                    preds = []
                    bps = []
                    decoder = getattr(self, "sinkhorn_decoder", "matching")
                    score_map_records = []
                    for seq, mask in zip(seqs, masks):
                        if decoder == "reciprocal":
                            bp, sc = self._decode_sinkhorn_contact_bp_thresholded(
                                mask,
                                min_separation=getattr(self, "sinkhorn_min_separation", 4),
                                pair_threshold=getattr(self, "sinkhorn_pair_threshold", 0.5),
                            )
                        elif decoder == "matching":
                            bp, sc = self._decode_sinkhorn_contact_bp_matching(
                                mask,
                                min_separation=getattr(self, "sinkhorn_min_separation", 4),
                                pair_threshold=getattr(self, "sinkhorn_pair_threshold", 0.5),
                            )
                        else:
                            raise ValueError(f"Unsupported SinkhornContact decoder: {decoder}")
                        scs.append(sc)
                        preds.append(self._bpseq_to_simple_dotbracket(bp))
                        bps.append(bp)
                        score_map_records.append({
                            "mode": "sinkhorn_contact",
                            "sinkhorn_mask": mask,
                            "sinkhorn_scores": self._real_pair_scores_from_sinkhorn_mask(mask),
                        })
                    pfs = bpps = [None] * len(preds)
                elif output_bpp is None:
                    dump_params = None
                    if output_score_map_dir is not None:
                        _, _, _, dump_params = self.model(seqs, return_param=True, return_attn_weights=False)
                    if use_constraint:
                        scs, preds, bps = self.model(seqs, constraint=refs, return_attn_weights=return_attn_weights)
                    else:
                        scs, preds, bps = self.model(seqs, return_attn_weights=return_attn_weights)
                    if dump_params is not None:
                        score_map_records = [
                            {
                                "mode": "base_model",
                                "base_param_maps": self._extract_square_param_maps(p),
                            }
                            for p in dump_params
                        ]
                    pfs = bpps = [None] * len(preds)
                else:
                    dump_params = None
                    if output_score_map_dir is not None:
                        _, _, _, dump_params = self.model(seqs, return_param=True, return_attn_weights=False)
                    if use_constraint:
                        scs, preds, bps, pfs, bpps = self.model(seqs, return_partfunc=True, constraint=refs, return_attn_weights=return_attn_weights)
                    else:
                        scs, preds, bps, pfs, bpps = self.model(seqs, return_partfunc=True, return_attn_weights=return_attn_weights)
                    if dump_params is not None:
                        score_map_records = []
                        for p, bpp, seq in zip(dump_params, bpps, seqs):
                            score_map_records.append({
                                "mode": "base_model_with_bpp",
                                "base_param_maps": self._extract_square_param_maps(p),
                                "base_scores": self._real_pair_scores_from_bpp(bpp, len(seq)),
                            })
                elapsed_time = time.time() - start

                # retrieve attention maps if produced
                batch_attn = None
                encoder = None
                if return_attn_weights:
                    if isinstance(self.model, ZukerFold):
                        net = getattr(self.model, "net", None)
                        encoder = getattr(net, "encoder", None)
                    elif isinstance(self.model, MixedFold):
                        zuker = getattr(self.model, "zuker", None)
                        net = getattr(zuker, "net", None)
                        encoder = getattr(net, "encoder", None)
                    elif isinstance(self.model, SinkhornContactFold):
                        net = getattr(self.model, "net", None)
                        encoder = getattr(net, "encoder", None)
                    
                    if encoder is not None and hasattr(encoder, '_attn_maps'):
                        batch_attn = encoder._attn_maps


                for header, seq, ref, sc, pred, bp, pf, bpp in zip(headers, seqs, refs, scs, preds, bps, pfs, bpps):
                    if output_bpseq is None:
                        print('>'+header)
                        print(seq)
                        print(pred, f'({sc:.1f})')
                    elif output_bpseq == "stdout":
                        print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)')
                        for i in range(1, len(bp)):
                            print(f'{i}\t{seq[i-1]}\t{bp[i]}')
                    else:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpseq, fn+".bpseq")
                        with open(fn, "w") as f:
                            print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)', file=f)
                            for i in range(1, len(bp)):
                                print(f'{i}\t{seq[i-1]}\t{bp[i]}', file=f)
                    if res_fn is not None:
                        x = compare_bpseq(ref, bp)
                        x = [header, len(seq), elapsed_time, sc.item()] + list(x) + list(accuracy(*x))
                        res_fn.write(', '.join([str(v) for v in x]) + "\n")
                    if output_bpp is not None:
                        bpp = np.triu(bpp)
                        bpp = bpp + bpp.T
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpp, fn+".bpp")
                        np.savetxt(fn, bpp, fmt='%.5f')

                    if return_attn_weights and batch_attn is not None:
                        save_maps = [attn_map.detach().cpu() if attn_map is not None else None for attn_map in batch_attn]
                        
                        fn_base = os.path.splitext(os.path.basename(header))[0]
                        torch.save(save_maps, os.path.join(output_attn_dir, f"{fn_base}.attn.pt"))

                if output_score_map_dir is not None and score_map_records is not None:
                    for header, seq, record in zip(headers, seqs, score_map_records):
                        self._save_score_map_dump(output_score_map_dir, header, seq, record)
                
                if return_attn_weights and 'encoder' in locals() and encoder is not None:
                    if isinstance(encoder, TransformerLayer):
                        for layer in encoder.layers:
                            layer._attn_weights = None
                    
                    if hasattr(encoder, "_attn_maps"):
                        encoder._attn_maps = None




    def build_model(self, args, prefix=None, default_model=None):
        model_name = self._get_prefixed_arg(args, 'model', prefix=prefix, fallback=default_model)
        if model_name == 'Turner':
            if args.param != '' and not prefix:
                return RNAFold(), {}
            else:
                from . import param_turner2004
                return RNAFold(param_turner2004), {}

        config = {
            'max_helix_length': self._get_prefixed_arg(args, 'max_helix_length', prefix=prefix),
            'embed_size' : self._get_prefixed_arg(args, 'embed_size', prefix=prefix),
            'num_filters': self._get_prefixed_arg(args, 'num_filters', prefix=prefix) if self._get_prefixed_arg(args, 'num_filters', prefix=prefix) is not None else (96,),
            'filter_size': self._get_prefixed_arg(args, 'filter_size', prefix=prefix) if self._get_prefixed_arg(args, 'filter_size', prefix=prefix) is not None else (5,),
            'pool_size': self._get_prefixed_arg(args, 'pool_size', prefix=prefix) if self._get_prefixed_arg(args, 'pool_size', prefix=prefix) is not None else (1,),
            'dilation': self._get_prefixed_arg(args, 'dilation', prefix=prefix), 
            'num_lstm_layers': self._get_prefixed_arg(args, 'num_lstm_layers', prefix=prefix), 
            'num_lstm_units': self._get_prefixed_arg(args, 'num_lstm_units', prefix=prefix),
            'num_transformer_layers': self._get_prefixed_arg(args, 'num_transformer_layers', prefix=prefix),
            'num_transformer_hidden_units': self._get_prefixed_arg(args, 'num_transformer_hidden_units', prefix=prefix),
            'num_transformer_att': self._get_prefixed_arg(args, 'num_transformer_att', prefix=prefix),
            'num_hidden_units': self._get_prefixed_arg(args, 'num_hidden_units', prefix=prefix) if self._get_prefixed_arg(args, 'num_hidden_units', prefix=prefix) is not None else (32,),
            'num_paired_filters': self._get_prefixed_arg(args, 'num_paired_filters', prefix=prefix),
            'paired_filter_size': self._get_prefixed_arg(args, 'paired_filter_size', prefix=prefix),
            'dropout_rate': self._get_prefixed_arg(args, 'dropout_rate', prefix=prefix),
            'fc_dropout_rate': self._get_prefixed_arg(args, 'fc_dropout_rate', prefix=prefix),
            'num_att': self._get_prefixed_arg(args, 'num_att', prefix=prefix),
            'pair_join': JOIN_MAP.get(self._get_prefixed_arg(args, 'pair_join', prefix=prefix).lower(), JoinType.CAT),
            'no_split_lr': self._get_prefixed_arg(args, 'no_split_lr', prefix=prefix),
            'use_sparse_sinkhorn': self._get_prefixed_arg(args, 'use_sparse_sinkhorn', prefix=prefix),
            'num_sinkhorn_iterations': self._get_prefixed_arg(args, 'num_sinkhorn_iterations', prefix=prefix),
            'sinkhorn_temp': self._get_prefixed_arg(args, 'sinkhorn_temp', prefix=prefix),
            'sinkhorn_block_size': self._get_prefixed_arg(args, 'sinkhorn_block_size', prefix=prefix),
            'sinkhorn_alpha': self._get_prefixed_arg(args, 'sinkhorn_alpha', prefix=prefix),
            'sinkhorn_temp_length_center': self._get_prefixed_arg(args, 'sinkhorn_temp_length_center', prefix=prefix),
            'sinkhorn_temp_length_slope': self._get_prefixed_arg(args, 'sinkhorn_temp_length_slope', prefix=prefix),
            'sinkhorn_pooling': self._get_prefixed_arg(args, 'sinkhorn_pooling', prefix=prefix),
            'sinkhorn_gating': self._get_prefixed_arg(args, 'sinkhorn_gating', prefix=prefix),
            'sinkhorn_diag_logit_penalty': self._get_prefixed_arg(args, 'sinkhorn_diag_logit_penalty', prefix=prefix),
            'sinkhorn_use_slack': self._get_prefixed_arg(args, 'sinkhorn_use_slack', prefix=prefix),
            'sinkhorn_real_only_normalization': self._get_prefixed_arg(args, 'sinkhorn_real_only_normalization', prefix=prefix),
        }

        if model_name == 'Zuker':
            model = ZukerFold(model_type=ZukerFold.ZukerType.M, **config)

        elif model_name == 'ZukerC':
            model = ZukerFold(model_type=ZukerFold.ZukerType.C, **config)

        elif model_name == 'ZukerL':
            model = ZukerFold(model_type=ZukerFold.ZukerType.L, **config)

        elif model_name == 'ZukerS':
            model = ZukerFold(model_type=ZukerFold.ZukerType.S, **config)

        elif model_name == 'Mix':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, **config)

        elif model_name == 'MixC':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, model_type=ZukerFold.ZukerType.C, **config)

        elif model_name == 'SinkhornContact':
            model = SinkhornContactFold(**config)

        else:
            raise('not implemented')

        return model, config


    def run(self, args, conf=None):
        test_dataset = FastaDataset(args.input)
        if len(test_dataset) == 0:
            test_dataset = BPseqDataset(args.input)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        self.model, _ = self.build_model(args)
        if args.param != '':
            param = Path(args.param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location='cpu')
            p = self._remap_state_dict_for_model(self.model, p)
            self.model.load_state_dict(p)
        self.sinkhorn_min_separation = getattr(args, "sinkhorn_min_separation", 4)
        self.sinkhorn_pair_threshold = getattr(args, "sinkhorn_pair_threshold", 0.5)
        self.sinkhorn_decoder = getattr(args, "sinkhorn_decoder", "matching")
        self.ensemble_mode = getattr(args, "ensemble_mode", "late_fusion")
        self.ensemble_base_weight = getattr(args, "ensemble_base_weight", 0.5)
        self.ensemble_pair_threshold = (
            args.ensemble_pair_threshold
            if getattr(args, "ensemble_pair_threshold", None) is not None
            else self.sinkhorn_pair_threshold
        )
        self.ensemble_decoder = (
            args.ensemble_decoder
            if getattr(args, "ensemble_decoder", None) is not None
            else self.sinkhorn_decoder
        )
        self.ensemble_zuker_bias_scale = getattr(args, "ensemble_zuker_bias_scale", 1.0)
        if args.ensemble_sinkhorn_param != '':
            if isinstance(self.model, SinkhornContactFold):
                raise ValueError("Late fusion expects the primary --model to be the base folding model, not SinkhornContact")
            ensemble_model_name = getattr(args, "ensemble_model", None) or 'SinkhornContact'
            self.ensemble_sinkhorn_model, _ = self.build_model(
                args,
                prefix='ensemble_',
                default_model=ensemble_model_name,
            )
            ensemble_param = Path(args.ensemble_sinkhorn_param)
            if not ensemble_param.exists() and conf is not None:
                ensemble_param = Path(conf).parent / ensemble_param
            p = torch.load(ensemble_param, map_location='cpu')
            p = self._remap_state_dict_for_model(self.ensemble_sinkhorn_model, p)
            self.ensemble_sinkhorn_model.load_state_dict(p)
        if getattr(args, "ablate_sinkhorn", False):
            net = getattr(self.model, "net", None)
            if net is not None and getattr(net, "use_sparse_sinkhorn", False):
                gate_scale = getattr(net, "sinkhorn_gate_scale", None)
                if gate_scale is not None:
                    with torch.no_grad():
                        gate_scale.zero_()

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
            if self.ensemble_sinkhorn_model is not None:
                self.ensemble_sinkhorn_model.to(torch.device("cuda", args.gpu))

        output_attn_dir = args.output_attn_dir
        self.predict(
            output_bpseq=args.bpseq,
            output_bpp=args.bpp,
            result=args.result,
            use_constraint=args.use_constraint,
            output_attn_dir=output_attn_dir,
            output_score_map_dir=args.output_score_map_dir,
        )


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('predict', help='predict')
        # input
        subparser.add_argument('input', type=str,
                            help='FASTA-formatted file or list of BPseq files')

        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--use-constraint', default=False, action='store_true')
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')
        subparser.add_argument('--bpp', type=str, default=None,
                            help='output the base-pairing probability matrix to the specified directory')
        subparser.add_argument('--output-attn-dir', type=str, default=None, 
                               help='directory to save per-layer attention maps (torch.pt) during prediction')
        subparser.add_argument('--output-score-map-dir', type=str, default=None,
                               help='directory to save per-sample score-map dumps (.pt) for base paired score tensors, Sinkhorn masks, and fused matrices')
        subparser.add_argument('--ensemble-sinkhorn-param', type=str, default='',
                            help='optional checkpoint for a frozen SinkhornContact model to late-fuse with the primary model')
        subparser.add_argument('--ensemble-mode', choices=('late_fusion', 'zuker_bias'), default='late_fusion',
                            help='how to combine the primary model with the frozen SinkhornContact model (default: late_fusion)')
        subparser.add_argument('--ensemble-base-weight', type=float, default=0.5,
                            help='weight on the primary model in late fusion; SinkhornContact gets 1 - weight (default: 0.5)')
        subparser.add_argument('--ensemble-pair-threshold', type=float, default=None,
                            help='pair threshold used when decoding the fused matrix; defaults to --sinkhorn-pair-threshold')
        subparser.add_argument('--ensemble-decoder', choices=('reciprocal', 'matching'), default=None,
                            help='decoder used on the fused matrix; defaults to --sinkhorn-decoder')
        subparser.add_argument('--ensemble-zuker-bias-scale', type=float, default=1.0,
                            help='additive scale applied to the Sinkhorn mask before biasing base paired scores in ensemble-mode=zuker_bias (default: 1.0)')


        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'SinkhornContact'), default='Turner', 
                            help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'SinkhornContact')")
        gparser.add_argument('--ensemble-model', choices=('SinkhornContact',), default=None,
                            help="late-fusion secondary model type; currently only 'SinkhornContact' is supported")
        gparser.add_argument('--max-helix-length', type=int, default=30, 
                        help='the maximum length of helices (default: 30)')
        gparser.add_argument('--embed-size', type=int, default=0,
                        help='the dimention of embedding (default: 0 == onehot)')
        gparser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--filter-size', type=int, action='append',
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN (default: 1)')
        gparser.add_argument('--dilation', type=int, default=0, 
                        help='Use the dilated convolution (default: 0)')
        gparser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers (default: 0)')
        gparser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units (default: 0)')
        gparser.add_argument('--num-transformer-layers', type=int, default=0,
                        help='the number of the transformer layers (default: 0)')
        gparser.add_argument('--num-transformer-hidden-units', type=int, default=2048,
                        help='the number of the hidden units of each transformer layer (default: 2048)')
        gparser.add_argument('--num-transformer-att', type=int, default=8,
                        help='the number of the attention heads of each transformer layer (default: 8)')
        gparser.add_argument('--num-paired-filters', type=int, action='append', default=[],
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--paired-filter-size', type=int, action='append', default=[],
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers (default: 32)')
        gparser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the CNN and LSTM units (default: 0.0)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--num-att', type=int, default=0,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear', 'JoinType.CAT', 'JoinType.ADD', 'JoinType.MUL', 'JoinType.BILINEAR'), default='cat', 
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear', 'JoinType.CAT', 'JoinType.ADD', 'JoinType.MUL', 'JoinType.BILINEAR') (default: 'cat')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')

        gparser.add_argument('--use-sparse-sinkhorn', default=False, action='store_true')
        gparser.add_argument('--num-sinkhorn-iterations', type=int, default=10)
        gparser.add_argument('--sinkhorn-block-size', type=int, default=30)
        gparser.add_argument('--sinkhorn-temp', type=float, default=0.1)
        gparser.add_argument('--sinkhorn-alpha', type=float, default=0.5,
                        help='strength of sigmoid length-based temperature scaling (default: 0.5)')
        gparser.add_argument('--sinkhorn-temp-length-center', type=float, default=8.0,
                        help='block-count midpoint of sigmoid temperature scaling (default: 8.0)')
        gparser.add_argument('--sinkhorn-temp-length-slope', type=float, default=3.0,
                        help='block-count slope of sigmoid temperature scaling (default: 3.0)')
        gparser.add_argument('--sinkhorn-pooling', choices=('attention', 'mean'), default='attention',
                        help='how blocks are pooled before Sinkhorn bilinear scoring (default: attention)')
        gparser.add_argument('--sinkhorn-gating', choices=('residual', 'multiplicative'), default='residual',
                        help='how the Sinkhorn mask is applied to pair features (default: residual)')
        gparser.add_argument('--sinkhorn-diag-logit-penalty', type=float, default=0.0,
                        help='additive penalty subtracted from diagonal logits just before Sinkhorn iterations (default: 0.0)')
        gparser.add_argument('--sinkhorn-use-slack', default=False, action='store_true',
                        help='append a learned slack row/column to the block-level Sinkhorn and crop it out before gating')
        gparser.add_argument('--sinkhorn-real-only-normalization', default=False, action='store_true',
                        help='normalize only the real rows/columns when slack is enabled, leaving slack as residual capacity')
        gparser.add_argument('--sinkhorn-min-separation', type=int, default=4,
                        help='minimum nucleotide separation when decoding SinkhornContactFold masks into BPSEQ pairs (default: 4)')
        gparser.add_argument('--sinkhorn-pair-threshold', type=float, default=0.5,
                        help='minimum Sinkhorn score required to decode a non-null pair in SinkhornContactFold prediction (default: 0.5)')
        gparser.add_argument('--sinkhorn-decoder', choices=('reciprocal', 'matching'), default='matching',
                        help='how SinkhornContactFold masks are decoded into pairs: thresholded reciprocal matches or thresholded maximum-weight matching (default: matching)')
        gparser.add_argument('--ablate-sinkhorn', default=False, action='store_true',
                        help='load the same checkpoint but zero the Sinkhorn residual gate scale at inference')
        gparser.add_argument('--ensemble-max-helix-length', type=int, default=None,
                        help='late-fusion secondary model max helix length; defaults to the primary model value')
        gparser.add_argument('--ensemble-embed-size', type=int, default=None,
                        help='late-fusion secondary model embedding size; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-filters', type=int, action='append', default=None,
                        help='late-fusion secondary model CNN filters; defaults to the primary model value')
        gparser.add_argument('--ensemble-filter-size', type=int, action='append', default=None,
                        help='late-fusion secondary model CNN filter sizes; defaults to the primary model value')
        gparser.add_argument('--ensemble-pool-size', type=int, action='append', default=None,
                        help='late-fusion secondary model CNN pool sizes; defaults to the primary model value')
        gparser.add_argument('--ensemble-dilation', type=int, default=None,
                        help='late-fusion secondary model dilation; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-lstm-layers', type=int, default=None,
                        help='late-fusion secondary model LSTM layers; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-lstm-units', type=int, default=None,
                        help='late-fusion secondary model LSTM units; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-transformer-layers', type=int, default=None,
                        help='late-fusion secondary model transformer layers; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-transformer-hidden-units', type=int, default=None,
                        help='late-fusion secondary model transformer hidden units; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-transformer-att', type=int, default=None,
                        help='late-fusion secondary model transformer heads; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-paired-filters', type=int, action='append', default=None,
                        help='late-fusion secondary model paired CNN filters; defaults to the primary model value')
        gparser.add_argument('--ensemble-paired-filter-size', type=int, action='append', default=None,
                        help='late-fusion secondary model paired CNN filter sizes; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-hidden-units', type=int, action='append', default=None,
                        help='late-fusion secondary model hidden units; defaults to the primary model value')
        gparser.add_argument('--ensemble-dropout-rate', type=float, default=None,
                        help='late-fusion secondary model dropout rate; defaults to the primary model value')
        gparser.add_argument('--ensemble-fc-dropout-rate', type=float, default=None,
                        help='late-fusion secondary model FC dropout rate; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-att', type=int, default=None,
                        help='late-fusion secondary model attention heads; defaults to the primary model value')
        gparser.add_argument('--ensemble-pair-join', choices=('cat', 'add', 'mul', 'bilinear', 'JoinType.CAT', 'JoinType.ADD', 'JoinType.MUL', 'JoinType.BILINEAR'), default=None,
                        help='late-fusion secondary model pair join; defaults to the primary model value')
        gparser.add_argument('--ensemble-no-split-lr', type=_parse_optional_bool, nargs='?', const=True, default=None,
                        help='late-fusion secondary model no-split-lr boolean; defaults to the primary model value')
        gparser.add_argument('--ensemble-use-sparse-sinkhorn', type=_parse_optional_bool, nargs='?', const=True, default=None,
                        help='late-fusion secondary model sparse Sinkhorn boolean; defaults to the primary model value')
        gparser.add_argument('--ensemble-num-sinkhorn-iterations', type=int, default=None,
                        help='late-fusion secondary model Sinkhorn iterations; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-block-size', type=int, default=None,
                        help='late-fusion secondary model Sinkhorn block size; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-temp', type=float, default=None,
                        help='late-fusion secondary model Sinkhorn temperature; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-alpha', type=float, default=None,
                        help='late-fusion secondary model Sinkhorn alpha; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-temp-length-center', type=float, default=None,
                        help='late-fusion secondary model Sinkhorn temperature length center; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-temp-length-slope', type=float, default=None,
                        help='late-fusion secondary model Sinkhorn temperature length slope; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-pooling', choices=('attention', 'mean'), default=None,
                        help='late-fusion secondary model Sinkhorn pooling; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-gating', choices=('residual', 'multiplicative'), default=None,
                        help='late-fusion secondary model Sinkhorn gating; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-diag-logit-penalty', type=float, default=None,
                        help='late-fusion secondary model diagonal logit penalty; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-use-slack', type=_parse_optional_bool, nargs='?', const=True, default=None,
                        help='late-fusion secondary model slack boolean; defaults to the primary model value')
        gparser.add_argument('--ensemble-sinkhorn-real-only-normalization', type=_parse_optional_bool, nargs='?', const=True, default=None,
                        help='late-fusion secondary model real-only normalization boolean; defaults to the primary model value')

        subparser.set_defaults(func = lambda args, conf: Predict().run(args, conf))
