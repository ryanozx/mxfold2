import copy
import time
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
    aux_loss: torch.Tensor
    l1_loss: torch.Tensor

    def __str__(self):
        return f"Loss = {self.loss.item():.6g}  \
        Energy Gap = {self.calc_energy_gap():.6g} \
        Margin Loss = {self.margin_loss.item():.6g} \
        Score Loss = {self.score_loss.item():.6g} \
        Aux Loss = {self.aux_loss.item():.6g} \
        L1 Loss = {self.l1_loss.item():.6g}"
    
    def calc_energy_gap(self):
        return self.pred_score.item() - self.ref_score.item()

class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., margin_weight=1.0, verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.margin_weight = margin_weight
        self.verbose = verbose
        self._last_timing = None

    @staticmethod
    def _include_param_in_l1(name: str, param: torch.Tensor) -> bool:
        if param is None or not param.requires_grad:
            return False
        if name.startswith("turner."):
            return False
        return True

    def compute_normalized_l1_loss(self, reference: torch.Tensor) -> torch.Tensor:
        total_abs = torch.zeros((), device=reference.device, dtype=torch.float64)

        for name, p in self.model.named_parameters():
            if not self._include_param_in_l1(name, p):
                continue
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite parameter in L1 term: {name}")
            total_abs = total_abs + torch.sum(torch.abs(p).to(dtype=torch.float64))

        return (self.l1_weight * (total_abs)).to(dtype=reference.dtype)



    def forward(self, seq, pairs, fname=None, bpp_target=None) -> Loss:
        timing = {
            "batch_size": len(seq),
            "pred_forward_s": 0.0,
            "pred_forward_detail": None,
            "ref_forward_s": 0.0,
            "ref_forward_detail": None,
            "l1_s": 0.0,
            "total_s": 0.0,
            # Keep compatibility with the existing train-time timing logger.
            "pred_mix_s": 0.0,
            "pred_mix_detail": None,
            "mix_free_s": 0.0,
            "mix_free_detail": None,
            "ref_mix_s": 0.0,
            "ref_mix_detail": None,
            "gap_turner_total_s": 0.0,
            "turner_constrained_s": 0.0,
            "turner_free_s": 0.0,
            "turner_cache_hits": 0,
            "turner_cache_misses": 0,
        }
        total_start = time.perf_counter()

        if self.margin_weight == 0.0:
            t0 = time.perf_counter()
            timing["pred_forward_s"] = time.perf_counter() - t0
            timing["pred_forward_detail"] = None
            timing["pred_mix_s"] = timing["pred_forward_s"]
            timing["pred_mix_detail"] = None
            zero = torch.zeros((), device=next(self.model.parameters()).device)
            margin_loss = zero
            sl_loss = zero
            pred = zero
            ref = zero
            pred_s = None
            ref_s = None
            aux_loss = zero
            loss = aux_loss
        else:
            t0 = time.perf_counter()
            pred_out = self.model(seq, return_param=True, return_aux=False, reference=pairs,
                                    loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                    loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
            pred, pred_s, _, param = pred_out
            timing["pred_forward_s"] = time.perf_counter() - t0
            timing["pred_forward_detail"] = copy.deepcopy(getattr(self.model, "_last_forward_timing", None))
            timing["pred_mix_s"] = timing["pred_forward_s"]
            timing["pred_mix_detail"] = timing["pred_forward_detail"]

            t0 = time.perf_counter()
            ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
            timing["ref_forward_s"] = time.perf_counter() - t0
            timing["ref_forward_detail"] = copy.deepcopy(getattr(self.model, "_last_forward_timing", None))
            timing["ref_mix_s"] = timing["ref_forward_s"]
            timing["ref_mix_detail"] = timing["ref_forward_detail"]

            raw_margin_loss = (pred - ref)
            margin_loss = self.margin_weight * raw_margin_loss
            sl_loss = torch.zeros_like(margin_loss)
            aux_loss = torch.zeros_like(margin_loss)
            loss = margin_loss + sl_loss + aux_loss
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
            t0 = time.perf_counter()
            l1_loss = self.compute_normalized_l1_loss(margin_loss)
            timing["l1_s"] = time.perf_counter() - t0

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss
        timing["total_s"] = time.perf_counter() - total_start
        self._last_timing = timing

        return Loss(loss = loss, 
                    margin_loss = margin_loss, 
                    pred_score = pred, ref_score = ref,
                    score_loss = sl_loss, 
                    aux_loss = aux_loss,
                    l1_loss = l1_loss)


class SinkhornContactLoss(nn.Module):
    def __init__(self, model, l1_weight=0., l2_weight=0., verbose=False,
                sinkhorn_aux_weight=0., sinkhorn_aux_use_bpp=False, sinkhorn_aux_pos_weight=1.0,
                sinkhorn_aux_neg_weight=0.0, sinkhorn_aux_diag_weight=None, sinkhorn_aux_slack_self_weight=None,
                sinkhorn_aux_true_pair_weight=None, sinkhorn_aux_true_unpaired_weight=None,
                sinkhorn_aux_false_pair_weight=0.0, sinkhorn_aux_false_unpaired_weight=0.0,
                sinkhorn_aux_min_separation=4, sinkhorn_aux_stack_bonus=0.0):
        super(SinkhornContactLoss, self).__init__()
        # This loss is dedicated to the standalone Sinkhorn contact-map model:
        # there is no structured Zuker margin term here, only direct supervision
        # of the Sinkhorn matrix plus optional parameter regularization.
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose
        self.sinkhorn_aux_weight = sinkhorn_aux_weight
        self.sinkhorn_aux_use_bpp = sinkhorn_aux_use_bpp
        self.sinkhorn_aux_pos_weight = sinkhorn_aux_pos_weight
        self.sinkhorn_aux_neg_weight = sinkhorn_aux_neg_weight
        self.sinkhorn_aux_diag_weight = (
            sinkhorn_aux_neg_weight
            if sinkhorn_aux_diag_weight is None
            else sinkhorn_aux_diag_weight
        )
        self.sinkhorn_aux_slack_self_weight = (
            sinkhorn_aux_neg_weight
            if sinkhorn_aux_slack_self_weight is None
            else sinkhorn_aux_slack_self_weight
        )
        self.sinkhorn_aux_true_pair_weight = (
            sinkhorn_aux_pos_weight
            if sinkhorn_aux_true_pair_weight is None
            else sinkhorn_aux_true_pair_weight
        )
        self.sinkhorn_aux_true_unpaired_weight = (
            self.sinkhorn_aux_diag_weight
            if sinkhorn_aux_true_unpaired_weight is None
            else sinkhorn_aux_true_unpaired_weight
        )
        self.sinkhorn_aux_false_pair_weight = sinkhorn_aux_false_pair_weight
        self.sinkhorn_aux_false_unpaired_weight = sinkhorn_aux_false_unpaired_weight
        self.sinkhorn_aux_min_separation = sinkhorn_aux_min_separation
        self.sinkhorn_aux_stack_bonus = sinkhorn_aux_stack_bonus
        self._last_timing = None

    @staticmethod
    def _include_param_in_l1(name: str, param: torch.Tensor) -> bool:
        if param is None or not param.requires_grad:
            return False
        return True

    def compute_normalized_l1_loss(self, reference: torch.Tensor) -> torch.Tensor:
        # Keep L1 simple in this branch: sum absolute values over all trainable
        # parameters and scale by l1_weight.
        total_abs = torch.zeros((), device=reference.device, dtype=torch.float64)
        for name, p in self.model.named_parameters():
            if not self._include_param_in_l1(name, p):
                continue
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite parameter in L1 term: {name}")
            total_abs = total_abs + torch.sum(torch.abs(p).to(dtype=torch.float64))
        return (self.l1_weight * total_abs).to(dtype=reference.dtype)

    def _pair_stack_weight(self, pair_vec: torch.Tensor, i: int, j: int) -> float:
        # Optional bonus for positives that sit inside a stacked helix pattern.
        bonus = float(self.sinkhorn_aux_stack_bonus)
        if bonus <= 0.0:
            return 1.0
        stacked = False
        if i + 1 < pair_vec.numel() and j - 1 >= 0 and int(pair_vec[i + 1].item()) == (j - 1):
            stacked = True
        if i - 1 >= 0 and j + 1 < pair_vec.numel() and int(pair_vec[i - 1].item()) == (j + 1):
            stacked = True
        return 1.0 + (bonus if stacked else 0.0)

    @staticmethod
    def _align_pairs_tensor(pairs, batch_size, target_len, device):
        # Convert BPSEQ-style partner indices into a dense (B, N) tensor aligned with
        # the Sinkhorn mask. If supervision is length N-1, pad a dummy 0th position so
        # it matches the model's internal 1-based indexing convention.
        if isinstance(pairs, torch.Tensor):
            pair_tensor = pairs.to(device=device)
        else:
            pair_tensor = torch.as_tensor(pairs, device=device)
        if pair_tensor.ndim == 1:
            pair_tensor = pair_tensor.unsqueeze(0)
        elif pair_tensor.ndim > 2:
            pair_tensor = pair_tensor.reshape(batch_size, -1)
        if pair_tensor.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} pair rows, got {pair_tensor.shape[0]}")
        if pair_tensor.shape[1] == target_len:
            return pair_tensor.to(dtype=torch.long)
        if pair_tensor.shape[1] == target_len - 1:
            out = torch.zeros((batch_size, target_len), device=device, dtype=torch.long)
            out[:, 1:] = pair_tensor.to(dtype=torch.long)
            return out
        raise ValueError(
            f"Cannot align pair tensor of shape {tuple(pair_tensor.shape)} to mask length {target_len}"
        )

    @staticmethod
    def _align_bpp_tensor(bpp_target, batch_size, target_len, device):
        # Teacher BPPs are expected to already be square matrices on the same index
        # convention as the model. This helper just normalizes device/dtype/batch shape.
        if bpp_target is None:
            return None
        if isinstance(bpp_target, torch.Tensor):
            bpp_tensor = bpp_target.to(device=device, dtype=torch.float32)
        else:
            bpp_tensor = torch.as_tensor(bpp_target, device=device, dtype=torch.float32)
        if bpp_tensor.numel() == 0:
            return None
        if bpp_tensor.ndim == 2:
            bpp_tensor = bpp_tensor.unsqueeze(0)
        if bpp_tensor.ndim != 3:
            raise ValueError(f"Expected BPP tensor with 2 or 3 dims, got shape {tuple(bpp_tensor.shape)}")
        if bpp_tensor.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} BPP rows, got {bpp_tensor.shape[0]}")
        if bpp_tensor.shape[1] != target_len or bpp_tensor.shape[2] != target_len:
            raise ValueError(
                f"Expected BPP tensor of shape ({batch_size}, {target_len}, {target_len}), got {tuple(bpp_tensor.shape)}"
            )
        return bpp_tensor

    @staticmethod
    def _zero_near_diagonal(matrix: torch.Tensor, min_sep: int) -> torch.Tensor:
        # Mask out self-pairs and short-range local contacts that are disallowed by the
        # current minimum-separation assumption.
        if min_sep < 0:
            return matrix
        n = matrix.shape[-1]
        idx = torch.arange(n, device=matrix.device)
        keep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > min_sep
        return matrix * keep.to(dtype=matrix.dtype)

    @staticmethod
    def _aggregate_square_to_blocks(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
        # When Sinkhorn runs on blocks, convert token-level supervision to block-level
        # supervision by summing all token-token mass inside each block pair.
        if block_size <= 1:
            return matrix
        batch_size, n, _ = matrix.shape
        num_blocks = (n + block_size - 1) // block_size
        out = matrix.new_zeros((batch_size, num_blocks, num_blocks))
        for bi in range(num_blocks):
            i0 = bi * block_size
            i1 = min(i0 + block_size, n)
            for bj in range(num_blocks):
                j0 = bj * block_size
                j1 = min(j0 + block_size, n)
                out[:, bi, bj] = matrix[:, i0:i1, j0:j1].sum(dim=(1, 2))
        return out

    def _block_row_assignment_aux_loss(self, pair_tensor, block_mask, block_size, reference, *, use_slack: bool = False):
        # Hard block target:
        # each source block chooses one destination block based on where its true token
        # partners land after coarse aggregation. Optional slack handles partnerless blocks.
        batch_size, total_cols, _ = block_mask.shape
        num_blocks = total_cols - 1 if use_slack else total_cols
        row_probs = block_mask / block_mask.sum(dim=2, keepdim=True).clamp_min(1e-8)
        log_row_probs = row_probs.clamp_min(1e-8).log()
        target_idx = torch.arange(num_blocks, device=block_mask.device).unsqueeze(0).expand(batch_size, -1).clone()
        row_weights = torch.zeros((batch_size, num_blocks), device=block_mask.device, dtype=block_mask.dtype)
        min_sep = int(self.sinkhorn_aux_min_separation)
        slack_idx = num_blocks if use_slack else None

        for b in range(batch_size):
            pair_vec = pair_tensor[b]
            block_pair_counts = torch.zeros((num_blocks, num_blocks), device=block_mask.device, dtype=block_mask.dtype)
            for i in range(pair_vec.numel()):
                if i == 0 and pair_vec.numel() > 1:
                    continue
                j = int(pair_vec[i].item())
                if j <= 0 or j >= pair_vec.numel():
                    continue
                if abs(j - i) <= min_sep:
                    continue
                pair_weight = self._pair_stack_weight(pair_vec, i, j)
                bi = (i - 1) // block_size
                bj = (j - 1) // block_size
                if 0 <= bi < num_blocks and 0 <= bj < num_blocks:
                    block_pair_counts[bi, bj] += pair_weight

            for bi in range(num_blocks):
                row_counts = block_pair_counts[bi]
                max_count, bj = row_counts.max(dim=0)
                if float(max_count.item()) > 0.0:
                    target_idx[b, bi] = int(bj.item())
                    row_weights[b, bi] = float(self.sinkhorn_aux_pos_weight)
                else:
                    target_idx[b, bi] = slack_idx if use_slack else bi
                    row_weights[b, bi] = float(self.sinkhorn_aux_neg_weight)

        main_weight_sum = row_weights.sum()
        if main_weight_sum.item() <= 0.0 and not use_slack:
            return torch.zeros_like(reference)

        picked_log_probs = log_row_probs.gather(2, target_idx.unsqueeze(-1)).squeeze(-1)
        numer = -(picked_log_probs * row_weights).sum()
        denom = main_weight_sum

        if use_slack and float(self.sinkhorn_aux_slack_self_weight) > 0.0:
            slack_row_log_probs = log_row_probs[:, slack_idx, :]
            slack_self_log_prob = slack_row_log_probs[:, slack_idx]
            slack_weight = torch.full((batch_size,), float(self.sinkhorn_aux_slack_self_weight), device=block_mask.device, dtype=block_mask.dtype)
            numer = numer - (slack_self_log_prob * slack_weight).sum()
            denom = denom + slack_weight.sum()

        if denom.item() <= 0.0:
            return torch.zeros_like(reference)
        aux = numer / denom
        return (float(self.sinkhorn_aux_weight) * aux).to(dtype=reference.dtype)

    def _bpp_kl_aux_loss(self, bpp_target, mask, block_size, reference):
        # Soft-target mode:
        # teacher BPP rows are normalized into partner distributions, Sinkhorn rows are
        # normalized the same way, and we average KL over rows with non-zero teacher mass.
        batch_size, n, _ = mask.shape
        bpp_tensor = self._align_bpp_tensor(bpp_target, batch_size, n, mask.device)
        if bpp_tensor is None:
            return torch.zeros_like(reference)
        target = self._zero_near_diagonal(bpp_tensor, int(self.sinkhorn_aux_min_separation)).clamp_min(0.0)
        pred = mask.clamp_min(0.0)
        if block_size not in (None, 1):
            block_size = int(block_size)
            target = self._aggregate_square_to_blocks(target, block_size)
            pred = self._aggregate_square_to_blocks(pred, block_size)
        target_row_sum = target.sum(dim=2, keepdim=True)
        valid_rows = target_row_sum.squeeze(-1) > 0
        if not valid_rows.any():
            return torch.zeros_like(reference)
        target_probs = target / target_row_sum.clamp_min(1e-8)
        pred_probs = pred / pred.sum(dim=2, keepdim=True).clamp_min(1e-8)
        kl = (
            target_probs.clamp_min(1e-8)
            * (target_probs.clamp_min(1e-8).log() - pred_probs.clamp_min(1e-8).log())
        ).sum(dim=2)
        aux = kl[valid_rows].mean()
        return (float(self.sinkhorn_aux_weight) * aux).to(dtype=reference.dtype)

    def _token_structured_aux_loss(self, pair_tensor, mask, reference):
        # Token-level structured mode:
        # - paired rows reward mass on the true partner and can penalize diagonal/null mass
        # - unpaired rows reward diagonal/null mass and can penalize off-diagonal pair mass
        row_probs = mask / mask.sum(dim=2, keepdim=True).clamp_min(1e-8)
        batch_size, n, _ = row_probs.shape
        min_sep = int(self.sinkhorn_aux_min_separation)

        numer = torch.zeros((), device=mask.device, dtype=mask.dtype)
        denom = torch.zeros((), device=mask.device, dtype=mask.dtype)

        for b in range(batch_size):
            pair_vec = pair_tensor[b]
            for i in range(pair_vec.numel()):
                if i == 0 and pair_vec.numel() == n:
                    continue
                row = row_probs[b, i]
                diag_prob = row[i].clamp_min(1e-8)
                j = int(pair_vec[i].item())
                is_true_pair = (0 < j < n and abs(j - i) > min_sep)

                if is_true_pair:
                    pair_prob = row[j].clamp_min(1e-8)
                    pair_weight = float(self.sinkhorn_aux_true_pair_weight) * self._pair_stack_weight(pair_vec, i, j)
                    if pair_weight > 0.0:
                        numer = numer - pair_weight * pair_prob.log()
                        denom = denom + pair_weight

                    false_unpaired_weight = float(self.sinkhorn_aux_false_unpaired_weight)
                    if false_unpaired_weight > 0.0:
                        numer = numer - false_unpaired_weight * (1.0 - diag_prob).clamp_min(1e-8).log()
                        denom = denom + false_unpaired_weight
                else:
                    true_unpaired_weight = float(self.sinkhorn_aux_true_unpaired_weight)
                    if true_unpaired_weight > 0.0:
                        numer = numer - true_unpaired_weight * diag_prob.log()
                        denom = denom + true_unpaired_weight

                    false_pair_weight = float(self.sinkhorn_aux_false_pair_weight)
                    if false_pair_weight > 0.0:
                        offdiag_mass = (1.0 - diag_prob).clamp(0.0, 1.0 - 1e-8)
                        numer = numer - false_pair_weight * (1.0 - offdiag_mass).clamp_min(1e-8).log()
                        denom = denom + false_pair_weight

        if denom.item() <= 0.0:
            return torch.zeros_like(reference)
        aux = numer / denom
        return (float(self.sinkhorn_aux_weight) * aux).to(dtype=reference.dtype)

    def compute_sinkhorn_aux_loss(self, pairs, reference: torch.Tensor, sinkhorn_aux=None, bpp_target=None) -> torch.Tensor:
        if self.sinkhorn_aux_weight <= 0.0:
            return torch.zeros_like(reference)

        # The standalone contact model exposes the Sinkhorn output directly.
        # Training supervises that matrix itself rather than a downstream DP score.
        mask = None if sinkhorn_aux is None else sinkhorn_aux.get("sinkhorn_mask")
        if mask is None or mask.numel() == 0 or mask.dim() != 3 or mask.shape[0] == 0:
            return torch.zeros_like(reference)

        block_size = None if sinkhorn_aux is None else sinkhorn_aux.get("sinkhorn_block_size")
        if self.sinkhorn_aux_use_bpp:
            # Optional soft-target mode: match row-normalized teacher BPPs with KL.
            return self._bpp_kl_aux_loss(bpp_target, mask, block_size, reference)

        batch_size, n, _ = mask.shape
        pair_tensor = self._align_pairs_tensor(pairs, batch_size, n, mask.device)
        if block_size not in (None, 1):
            # Block-level mode supervises the block matrix directly. If a slack row/column is
            # present, rows with no true partner can be assigned to that dustbin target.
            block_mask_with_slack = None if sinkhorn_aux is None else sinkhorn_aux.get("sinkhorn_block_mask_with_slack")
            num_blocks = (n + int(block_size) - 1) // int(block_size)
            use_slack = (
                block_mask_with_slack is not None
                and block_mask_with_slack.numel() > 0
                and block_mask_with_slack.dim() == 3
                and block_mask_with_slack.shape[1] == block_mask_with_slack.shape[2]
                and block_mask_with_slack.shape[1] == num_blocks + 1
            )
            effective_block_mask = block_mask_with_slack
            if effective_block_mask is None or effective_block_mask.numel() == 0 or effective_block_mask.dim() != 3:
                return torch.zeros_like(reference)
            return self._block_row_assignment_aux_loss(
                pair_tensor,
                effective_block_mask,
                int(block_size),
                reference,
                use_slack=use_slack,
            )

        return self._token_structured_aux_loss(pair_tensor, mask, reference)

    def forward(self, seq, pairs, fname=None, bpp_target=None) -> Loss:
        start = time.perf_counter()
        batch_size = len(seq)
        device = next(self.model.parameters()).device
        zero = torch.zeros((batch_size,), device=device, dtype=torch.float32)

        # The standalone contact model returns the Sinkhorn mask directly, so the total training
        # objective is just:
        #   Sinkhorn auxiliary supervision + optional L1 regularization.
        sinkhorn_aux = self.model(seq, return_aux=True)
        aux_loss = self.compute_sinkhorn_aux_loss(
            pairs,
            zero,
            sinkhorn_aux=sinkhorn_aux,
            bpp_target=bpp_target,
        )

        l1_loss = torch.zeros_like(zero)
        if self.l1_weight > 0.0:
            l1_loss = self.compute_normalized_l1_loss(zero)

        loss = aux_loss + l1_loss
        self._last_timing = {
            "batch_size": batch_size,
            "contact_aux_only_s": time.perf_counter() - start,
        }
        return Loss(
            loss=loss,
            pred_score=zero,
            ref_score=zero,
            margin_loss=torch.zeros_like(zero),
            score_loss=torch.zeros_like(zero),
            aux_loss=aux_loss,
            l1_loss=l1_loss,
        )


class StructuredLossWithTurner(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., margin_weight=1.0, sl_weight=1., verbose=False):
        super(StructuredLossWithTurner, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.margin_weight = margin_weight
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
        self._last_timing = None

    @staticmethod
    def _include_param_in_l1(name: str, param: torch.Tensor) -> bool:
        if param is None or not param.requires_grad:
            return False
        if name.startswith("turner."):
            return False
        return True

    def compute_normalized_l1_loss(self, reference: torch.Tensor) -> torch.Tensor:
        total_abs = torch.zeros((), device=reference.device, dtype=torch.float64)
        total_numel = 0

        for name, p in self.model.named_parameters():
            if not self._include_param_in_l1(name, p):
                continue
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite parameter in L1 term: {name}")
            total_abs = total_abs + torch.sum(torch.abs(p).to(dtype=torch.float64))
            total_numel += p.numel()

        if total_numel == 0:
            return torch.zeros_like(reference)

        return (self.l1_weight * (total_abs / total_numel)).to(dtype=reference.dtype)


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

    def forward(self, seq, pairs, fname=None, bpp_target=None) -> Loss:
        timing = {
            "batch_size": len(seq),
            "pred_mix_s": 0.0,
            "pred_mix_detail": None,
            "mix_free_s": 0.0,
            "mix_free_detail": None,
            "ref_mix_s": 0.0,
            "ref_mix_detail": None,
            "gap_turner_total_s": 0.0,
            "turner_constrained_s": 0.0,
            "turner_free_s": 0.0,
            "turner_cache_hits": 0,
            "turner_cache_misses": 0,
            "total_s": 0.0,
        }
        total_start = time.perf_counter()

        if self.margin_weight == 0.0 and self.sl_weight == 0.0:
            t0 = time.perf_counter()
            timing["pred_mix_s"] = time.perf_counter() - t0
            timing["pred_mix_detail"] = None
            zero = torch.zeros((), device=next(self.model.parameters()).device)
            pred = zero
            ref = zero
            pred_s = None
            ref_s = None
            margin_loss = zero
            sl_loss = zero
            aux_loss = zero
            loss = aux_loss
        else:
            t0 = time.perf_counter()
            pred_out = self.model(seq, return_param=True, return_aux=False, reference=pairs,
                                    loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                    loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
            pred, pred_s, _, param = pred_out
            timing["pred_mix_s"] = time.perf_counter() - t0
            timing["pred_mix_detail"] = copy.deepcopy(getattr(self.model, "_last_forward_timing", None))

            t0 = time.perf_counter()
            mix_free, _, _ = self.model(seq, param=param, constraint=None, reference=None)
            timing["mix_free_s"] = time.perf_counter() - t0
            timing["mix_free_detail"] = copy.deepcopy(getattr(self.model, "_last_forward_timing", None))

            t0 = time.perf_counter()
            ref, ref_s, _ = self.model(seq, param=param, constraint=pairs, max_internal_length=None)
            timing["ref_mix_s"] = time.perf_counter() - t0
            timing["ref_mix_detail"] = copy.deepcopy(getattr(self.model, "_last_forward_timing", None))
            gap_mix = (ref - mix_free)
            with torch.no_grad():
                gap_turner_start = time.perf_counter()
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
                        t0 = time.perf_counter()
                        ref2_i, _, _ = self.turner(seq_batch, constraint=pair_batch, max_internal_length=None)
                        timing["turner_constrained_s"] += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        turner_free_i, _, _ = self.turner(seq_batch, constraint=None)
                        timing["turner_free_s"] += time.perf_counter() - t0
                        gap_i = (ref2_i - turner_free_i).reshape(-1)[0]
                        if cache_key is not None:
                            self.gap_turner_vals[cache_key] = float(gap_i.item())
                        gap_turner_list.append(gap_i.to(device=pred.device, dtype=pred.dtype))
                    else:
                        batch_hits += 1
                        self.gap_turner_cache_hits += 1
                        gap_turner_list.append(pred.new_tensor(cached_val))
                gap_turner = torch.stack(gap_turner_list)
                timing["gap_turner_total_s"] = time.perf_counter() - gap_turner_start
                timing["turner_cache_hits"] = batch_hits
                timing["turner_cache_misses"] = batch_misses
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

            raw_margin_loss = (pred - ref)
            margin_loss = self.margin_weight * raw_margin_loss
            criterion_loss = torch.nn.HuberLoss(delta=0.1)
            sl_loss = self.sl_weight * criterion_loss(gap_mix, gap_turner)
            aux_loss = torch.zeros_like(margin_loss)

            loss = margin_loss + sl_loss + aux_loss

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
            l1_loss = self.compute_normalized_l1_loss(margin_loss)

        # if self.l2_weight > 0.0:
        #     l2_reg = torch.zeros_like(margin_loss)
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum(p ** 2)
        #     l2_loss += self.l2_weight * torch.sqrt(l2_reg)

        loss += l1_loss
        timing["total_s"] = time.perf_counter() - total_start
        self._last_timing = timing

        return Loss(
            loss = loss,
            margin_loss = margin_loss,
            pred_score = pred,
            ref_score = ref,
            score_loss = sl_loss,
            aux_loss = aux_loss,
            l1_loss = l1_loss
        )
