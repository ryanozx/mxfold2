import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Loss:
    # Assumption: Batch size = 1
    loss: torch.Tensor
    pred_score: torch.Tensor
    ref_score: torch.Tensor
    margin_loss: torch.Tensor
    score_loss: torch.Tensor
    l1_loss: torch.Tensor

    def __str__(self):
        return f"Loss = {self.loss.item():.6g}  \
        Energy Gap = {self.calc_energy_gap():.6g} \
        Margin Loss = {self.margin_loss.item():.6g} \
        Score Loss = {self.score_loss.item():.6g} \
        L1 Loss = {self.l1_loss.item():.6g}"
    
    def calc_energy_gap(self):
        return self.pred_score.item() - self.ref_score.item()

class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq, pairs, fname=None) -> Loss:
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        margin_loss = (pred - ref)
        sl_loss = torch.zeros_like(margin_loss)
        loss = margin_loss + sl_loss
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss).any():
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        l1_loss = torch.zeros_like(margin_loss)
        
        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                l1_loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss
        return Loss(loss = loss, 
                    margin_loss = margin_loss, 
                    pred_score = pred, ref_score = ref,
                    score_loss = sl_loss, 
                    l1_loss = l1_loss)


class StructuredLossWithTurner(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., sl_weight=1., verbose=False):
        super(StructuredLossWithTurner, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.verbose = verbose
        from .fold.rnafold import RNAFold
        from . import param_turner2004
        if getattr(self.model, "turner", None):
            self.turner = self.model.turner
        else:
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)
        self.gap_turner_vals = {}
        self.gap_turner_cache_hits = 0
        self.gap_turner_cache_misses = 0
        self.gap_turner_forward_calls = 0
        self.gap_turner_debug_interval = 100

    @staticmethod
    def _normalize_fnames(fname, batch_size):
        if fname is None:
            return [None] * batch_size
        if isinstance(fname, str):
            return [fname]
        if isinstance(fname, (list, tuple)):
            out = list(fname)
            if len(out) == batch_size:
                return out
            if batch_size == 1 and len(out) > 0:
                return [out[0]]
            return out
        return [str(fname)]

    @staticmethod
    def _select_constraint(pairs, idx):
        if isinstance(pairs, torch.Tensor):
            if pairs.ndim == 0:
                return pairs
            return pairs[idx:idx + 1]
        if isinstance(pairs, (list, tuple)):
            return [pairs[idx]]
        return pairs

    def forward(self, seq, pairs, fname=None) -> Loss:
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        mix_free, _, _ = self.model(seq, param=param, constraint=None, reference=None)
        ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
        gap_mix = (ref - mix_free)
        
        with torch.no_grad():
            fnames = self._normalize_fnames(fname, len(seq))
            gap_turner_list = []
            batch_hits = 0
            batch_misses = 0
            for i, seq_i in enumerate(seq):
                cache_key = fnames[i] if i < len(fnames) else None
                cached_val = self.gap_turner_vals.get(cache_key) if cache_key is not None else None

                if cached_val is None:
                    batch_misses += 1
                    self.gap_turner_cache_misses += 1
                    seq_batch = [seq_i]
                    pair_batch = self._select_constraint(pairs, i)
                    ref2_i, _, _ = self.turner(seq_batch, constraint=pair_batch, max_internal_length=None)
                    turner_free_i, _, _ = self.turner(seq_batch, constraint=None, max_internal_length=None)
                    gap_i = (ref2_i - turner_free_i).reshape(-1)[0]
                    if cache_key is not None:
                        self.gap_turner_vals[cache_key] = float(gap_i.item())
                    gap_turner_list.append(gap_i.to(device=pred.device, dtype=pred.dtype))
                else:
                    batch_hits += 1
                    self.gap_turner_cache_hits += 1
                    gap_turner_list.append(pred.new_tensor(cached_val))
            gap_turner = torch.stack(gap_turner_list)
            self.gap_turner_forward_calls += 1
            if (
                self.verbose
                and self.gap_turner_debug_interval > 0
                and self.gap_turner_forward_calls % self.gap_turner_debug_interval == 0
            ):
                total = self.gap_turner_cache_hits + self.gap_turner_cache_misses
                hit_rate = (self.gap_turner_cache_hits / total) if total > 0 else 0.0
                print(
                    "[gap_turner_cache] "
                    f"calls={self.gap_turner_forward_calls} "
                    f"batch_hits={batch_hits} batch_misses={batch_misses} "
                    f"total_hits={self.gap_turner_cache_hits} total_misses={self.gap_turner_cache_misses} "
                    f"hit_rate={hit_rate:.3f} cache_size={len(self.gap_turner_vals)}"
                )



        l = torch.tensor([len(s) for s in seq], device=pred.device)
        margin_loss = (pred - ref) 
        criterion_loss = torch.nn.HuberLoss(delta=0.1)
        sl_loss = self.sl_weight * criterion_loss(gap_mix, gap_turner)

        loss = margin_loss + sl_loss

        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss).any():
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        l1_loss = torch.zeros_like(margin_loss)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                l1_loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss

        return Loss(
            loss = loss,
            margin_loss = margin_loss,
            pred_score = pred,
            ref_score = ref,
            score_loss = sl_loss,
            l1_loss = l1_loss
        )
