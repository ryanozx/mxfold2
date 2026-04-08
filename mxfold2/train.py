import os
import random
import time
from pathlib import Path
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BPseqDataset
from .fold.contact import SinkhornContactFold
from .fold.mix import MixedFold
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold
from .fold.layers import JoinType, JOIN_MAP, init_weights, init_heads
from .loss import Loss, SinkhornContactLoss, StructuredLoss, StructuredLossWithTurner

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

max_norm = 1000.0
epsilon = 1e-8

class TrainStage(Enum):
    PRE_UNSCALE = 1
    PRE_CLIP = 2
    POST_CLIP = 3

class Train:
    step = 0
    HEAD_GROUP_PREFIXES = ("fc_paired", "fc_unpaired", "fc_length/")
    HEAD_PARAM_PREFIXES = ("net.fc_paired.", "net.fc_unpaired.", "fc_length.")
    SINKHORN_GROUP_NAME = "sparse_sinkhorn"
    SINKHORN_PARAM_KEYWORDS = ("sparse_sinkhorn", "sinkhorn_mask_proj", "sinkhorn_gate_scale", "contact_logit_proj")
    EMBEDDING_PARAM_PREFIXES = (
        "zuker.net.embedding.",
        "net.embedding.",
    )
    CNN_PARAM_PREFIXES = (
        "zuker.net.encoder.conv.",
        "net.encoder.conv.",
    )
    LSTM_PARAM_PREFIXES = (
        "zuker.net.encoder.pre_lstm_ln.",
        "zuker.net.encoder.lstm.",
        "zuker.net.encoder.lstm_ln.",
        "net.encoder.pre_lstm_ln.",
        "net.encoder.lstm.",
        "net.encoder.lstm_ln.",
    )
    SINKHORN_LENGTH_BUCKETS = (
        (100, "len_le_100"),
        (200, "len_101_200"),
        (None, "len_gt_200"),
    )

    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None
        # self.scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    @staticmethod
    def _resolve_zuker_model(model):
        if isinstance(model, ZukerFold):
            return model
        if isinstance(model, MixedFold):
            return model.zuker
        if hasattr(model, "net"):
            return model
        return None

    @classmethod
    def _is_sinkhorn_param_name(cls, name: str) -> bool:
        return any(keyword in name for keyword in cls.SINKHORN_PARAM_KEYWORDS)

    @classmethod
    def _is_cnn_param_name(cls, name: str) -> bool:
        return name.startswith(cls.CNN_PARAM_PREFIXES)

    @classmethod
    def _is_embedding_param_name(cls, name: str) -> bool:
        return name.startswith(cls.EMBEDDING_PARAM_PREFIXES)

    @classmethod
    def _is_lstm_param_name(cls, name: str) -> bool:
        return name.startswith(cls.LSTM_PARAM_PREFIXES)

    @classmethod
    def _is_head_param_name(cls, name: str) -> bool:
        return name.startswith(cls.HEAD_PARAM_PREFIXES)

    @classmethod
    def _sinkhorn_length_bucket(cls, seq_len: int) -> str:
        lower_bound = 0
        for upper_bound, bucket_name in cls.SINKHORN_LENGTH_BUCKETS:
            if upper_bound is None:
                return bucket_name
            if lower_bound < seq_len <= upper_bound:
                return bucket_name
            lower_bound = upper_bound
        return cls.SINKHORN_LENGTH_BUCKETS[-1][1]

    def _log_sinkhorn_scalar(self, tag: str, value: float, step: int, seq_len: int):
        if self.writer is None:
            return

        self.writer.add_scalar(tag, value, step)
        bucket_name = self._sinkhorn_length_bucket(seq_len)
        self.writer.add_scalar(
            f"{tag}/by_length/{bucket_name}",
            value,
            step,
        )

    def _log_sinkhorn_block_metrics(
        self,
        tag_prefix: str,
        block_img: torch.Tensor,
        step: int,
        seq_len: int,
        *,
        log_image: bool = False,
    ):
        if self.writer is None or block_img is None or block_img.numel() == 0:
            return

        block_img = block_img.detach().to(dtype=torch.float32)
        block_finite = torch.isfinite(block_img)
        if not block_finite.any():
            return

        block_img = torch.where(
            block_finite,
            block_img.clamp(0.0, 1.0),
            torch.zeros_like(block_img),
        )

        # logs the image
        if log_image:
            self.writer.add_image(
                f"{tag_prefix}_mask",
                block_img.unsqueeze(0),
                step,
            )

        # mae of the block with its transpose - measures the symmetry mae
        # we expect the mae to be as small as possible due to the pre-sinkhorn symmetrisation;
        # failures here indicate that the sinkhorn iterations caused the divergence
        block_symmetry_mae = float(
            (block_img - block_img.transpose(0, 1)).abs().mean().item()
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_symmetry_mae",
            block_symmetry_mae,
            step,
            seq_len,
        )

        row_sums = block_img.sum(dim=1)
        col_sums = block_img.sum(dim=0)
        row_sum_err = (row_sums - 1.0).abs()
        col_sum_err = (col_sums - 1.0).abs()

        # these stats show whether the sinkhorn converges; the row sum should be as close to 1 as possible
        # failures here are indicative of non-convergence, causes include bad initial logits and insufficient sinkhorn iterations
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_row_sum_mean",
            float(row_sums.mean().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_row_sum_mae_from_1",
            float(row_sum_err.mean().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_row_sum_max_abs_err_from_1",
            float(row_sum_err.max().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_col_sum_mean",
            float(col_sums.mean().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_col_sum_mae_from_1",
            float(col_sum_err.mean().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_col_sum_max_abs_err_from_1",
            float(col_sum_err.max().item()),
            step,
            seq_len,
        )

        # mean of the diagonal - we want this to be as close to zero as possible by design so that 
        # the model is allocating values to the slack/other cells
        # a failure here indicates that we're not penalising diagonals well enough
        row_mass = block_img.sum(dim=1, keepdim=True).clamp_min(1e-8)
        row_norm = block_img / row_mass
        diag_vals = row_norm.diagonal()
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_diag_prob_mean",
            float(diag_vals.mean().item()),
            step,
            seq_len,
        )

        # how big the largest value in each row is; indicates level of certainty about positions
        row_top1 = row_norm.max(dim=1).values
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_row_top1_mean",
            float(row_top1.mean().item()),
            step,
            seq_len,
        )

        if row_norm.shape[1] > 1:
            # how big the gap is between the largest value in the row vs the second largest value in the row
            # indicates the level of confidence that the model has in whether a block has a single partner
            # But: RNA can have several structures with similar energy levels; the model would be correct if it
            # is uncertain.
            top2 = row_norm.topk(k=2, dim=1).values
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_row_top1_gap_mean",
                float((top2[:, 0] - top2[:, 1]).mean().item()),
                step,
                seq_len,
            )
            # we are using Shannon entropy to measure whether the mask is getting sharp (mass is concentrated in
            # a few positions per row)
            row_entropy = -(
                row_norm.clamp_min(1e-8)
                * row_norm.clamp_min(1e-8).log()
            ).sum(dim=1)
            row_entropy = row_entropy / np.log(row_norm.shape[1])
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_row_entropy_mean",
                float(row_entropy.mean().item()),
                step,
                seq_len,
            )

        row_argmax = row_norm.argmax(dim=1)
        row_ids = torch.arange(
            row_argmax.numel(),
            device=row_argmax.device,
        )

        # the percentage of rows in which the diagonal contains the maximum value; ideally this should be as close
        # to zero as possible
        diag_argmax = row_argmax == row_ids
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_diag_argmax_rate",
            float(diag_argmax.to(dtype=torch.float32).mean().item()),
            step,
            seq_len,
        )

        # whether pairs agree on each other being the partner, this value should be as close to 1 as possible
        # we have no post-sinkhorn symmetrisation, so this is not guaranteed and depends on the pre-Sinkhorn logits
        reciprocal = row_argmax[row_argmax] == row_ids
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_reciprocal_argmax_rate",
            float(reciprocal.to(dtype=torch.float32).mean().item()),
            step,
            seq_len,
        )


        argmax_counts = torch.bincount(row_argmax, minlength=row_argmax.numel())
        max_collision_fraction = argmax_counts.max().to(dtype=torch.float32) / max(row_argmax.numel(), 1)
        # fraction of rows that pile onto the single most popular argmax column; high values for this metric
        # indicate the formation of attention sinks. an ideal case is that this fraction is 1 / num_blocks
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_max_argmax_collision_fraction",
            float(max_collision_fraction.item()),
            step,
            seq_len,
        )

        incoming_col_mass = row_norm.sum(dim=0) / max(row_norm.shape[0], 1)
        # fraction of rows that pile onto the single most popular column based on their masses, again used
        # to see if attention sinks exist 
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_max_incoming_col_mass",
            float(incoming_col_mass.max().item()),
            step,
            seq_len,
        )

    def _log_sinkhorn_slack_metrics(
        self,
        tag_prefix: str,
        block_img_with_slack: torch.Tensor,
        num_real_blocks: int,
        step: int,
        seq_len: int,
        pairs=None,
        block_size: int = 1,
    ):
        if (
            self.writer is None
            or block_img_with_slack is None
            or block_img_with_slack.numel() == 0
            or num_real_blocks <= 0
            or block_img_with_slack.shape[0] <= num_real_blocks
            or block_img_with_slack.shape[1] <= num_real_blocks
        ):
            return

        block_img_with_slack = block_img_with_slack.detach().to(dtype=torch.float32)
        finite = torch.isfinite(block_img_with_slack)
        if not finite.any():
            return

        block_img_with_slack = torch.where(
            finite,
            block_img_with_slack.clamp(0.0, 1.0),
            torch.zeros_like(block_img_with_slack),
        )

        row_mass = block_img_with_slack.sum(dim=1, keepdim=True).clamp_min(1e-8)
        row_norm = block_img_with_slack / row_mass
        slack_idx = num_real_blocks

        real_rows = row_norm[:num_real_blocks]
        real_to_slack = real_rows[:, slack_idx]
        # how much is being allocated to the slack column per row (how much do real rows want no real partner?)
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_real_row_to_slack_prob_mean",
            float(real_to_slack.mean().item()),
            step,
            seq_len,
        )

        # fraction of rows whose top choice is slack
        row_argmax = real_rows.argmax(dim=1)
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_real_row_argmax_to_slack_rate",
            float((row_argmax == slack_idx).to(dtype=torch.float32).mean().item()),
            step,
            seq_len,
        )

        # total incoming probability mass that the slack column receives from real rows
        incoming_from_real = real_rows.sum(dim=0) / max(num_real_blocks, 1)
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_slack_incoming_col_mass_from_real_rows",
            float(incoming_from_real[slack_idx].item()),
            step,
            seq_len,
        )

        # probability that the slack row assigns to the slack column itself (the slack block should
        # not have a partner and thus point to itself i.e. this value should be high)
        slack_row = row_norm[slack_idx]
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_slack_row_self_prob",
            float(slack_row[slack_idx].item()),
            step,
            seq_len,
        )
        # average probability from the slack row back into real columns - if this is high, the slack
        # row is spreading mass back over real blocks
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_slack_row_to_real_prob_mean",
            float(slack_row[:num_real_blocks].mean().item()),
            step,
            seq_len,
        )

        if pairs is None:
            return

        pair_tensor = pairs
        if not isinstance(pair_tensor, torch.Tensor):
            pair_tensor = torch.as_tensor(pair_tensor)
        pair_tensor = pair_tensor.detach().to(device=block_img_with_slack.device, dtype=torch.long)
        if pair_tensor.ndim == 1:
            pair_tensor = pair_tensor.unsqueeze(0)
        if pair_tensor.ndim > 2:
            pair_tensor = pair_tensor.reshape(pair_tensor.shape[0], -1)
        if pair_tensor.shape[0] == 0:
            return

        sample_pairs = pair_tensor[0]
        target_len = num_real_blocks * max(int(block_size), 1)
        if sample_pairs.numel() == target_len - 1:
            aligned_pairs = torch.zeros(target_len, device=sample_pairs.device, dtype=torch.long)
            aligned_pairs[1:] = sample_pairs
        elif sample_pairs.numel() == target_len:
            aligned_pairs = sample_pairs
        else:
            return

        real_partner_mask = aligned_pairs[1:] > 0
        if int(block_size) > 1:
            partner_by_block = []
            for bi in range(num_real_blocks):
                start = bi * int(block_size)
                end = min(start + int(block_size), real_partner_mask.numel())
                partner_by_block.append(bool(real_partner_mask[start:end].any().item()))
            partner_by_block = torch.tensor(
                partner_by_block,
                device=real_to_slack.device,
                dtype=torch.bool,
            )
        else:
            partner_by_block = real_partner_mask[:num_real_blocks]

        no_partner_by_block = ~partner_by_block

        if partner_by_block.any():
            partner_real_to_slack = real_to_slack[partner_by_block]
            partner_row_argmax = row_argmax[partner_by_block]
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_partner_row_to_slack_prob_mean",
                float(partner_real_to_slack.mean().item()),
                step,
                seq_len,
            )
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_partner_row_argmax_to_slack_rate",
                float((partner_row_argmax == slack_idx).to(dtype=torch.float32).mean().item()),
                step,
                seq_len,
            )

        if no_partner_by_block.any():
            no_partner_real_to_slack = real_to_slack[no_partner_by_block]
            no_partner_row_argmax = row_argmax[no_partner_by_block]
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_no_partner_row_to_slack_prob_mean",
                float(no_partner_real_to_slack.mean().item()),
                step,
                seq_len,
            )
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_no_partner_row_argmax_to_slack_rate",
                float((no_partner_row_argmax == slack_idx).to(dtype=torch.float32).mean().item()),
                step,
                seq_len,
            )

    def _log_sinkhorn_pooling_metrics(
        self,
        tag_prefix: str,
        attn_weights: torch.Tensor,
        valid_mask: torch.Tensor,
        step: int,
        seq_len: int,
    ):
        if (
            self.writer is None
            or attn_weights is None
            or valid_mask is None
            or attn_weights.numel() == 0
            or valid_mask.numel() == 0
        ):
            return

        attn_weights = attn_weights.detach().to(dtype=torch.float32)
        valid_mask = valid_mask.detach().to(dtype=torch.bool)
        if attn_weights.dim() != 2 or valid_mask.dim() != 2:
            return

        valid_counts = valid_mask.sum(dim=1)
        valid_rows = valid_counts > 0
        if not valid_rows.any():
            return

        attn_weights = attn_weights[valid_rows]
        valid_mask = valid_mask[valid_rows]
        valid_counts = valid_counts[valid_rows]
        masked_weights = attn_weights.masked_fill(~valid_mask, 0.0)

        self._log_sinkhorn_scalar(
            f"{tag_prefix}_valid_tokens_mean",
            float(valid_counts.to(dtype=torch.float32).mean().item()),
            step,
            seq_len,
        )

        top1 = masked_weights.max(dim=1).values
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_top1_mean",
            float(top1.mean().item()),
            step,
            seq_len,
        )

        multi_token_rows = valid_counts > 1
        if multi_token_rows.any():
            top2 = masked_weights[multi_token_rows].topk(k=2, dim=1).values
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_top1_gap_mean",
                float((top2[:, 0] - top2[:, 1]).mean().item()),
                step,
                seq_len,
            )

            entropy = -(
                masked_weights[multi_token_rows].clamp_min(1e-8)
                * masked_weights[multi_token_rows].clamp_min(1e-8).log()
            ).sum(dim=1)
            entropy = entropy / valid_counts[multi_token_rows].to(dtype=torch.float32).log().clamp_min(1e-8)
            self._log_sinkhorn_scalar(
                f"{tag_prefix}_entropy_mean",
                float(entropy.mean().item()),
                step,
                seq_len,
            )

        argmax_idx = masked_weights.argmax(dim=1)
        rel_pos = argmax_idx.to(dtype=torch.float32) / (valid_counts.to(dtype=torch.float32) - 1.0).clamp_min(1.0)
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_argmax_relpos_mean",
            float(rel_pos.mean().item()),
            step,
            seq_len,
        )
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_argmax_relpos_std",
            float(rel_pos.std(unbiased=False).item()),
            step,
            seq_len,
        )

        edge_hits = (argmax_idx == 0) | (argmax_idx == (valid_counts - 1))
        self._log_sinkhorn_scalar(
            f"{tag_prefix}_edge_argmax_rate",
            float(edge_hits.to(dtype=torch.float32).mean().item()),
            step,
            seq_len,
        )

    def log_model_init_stats(self, tag):
        """
        prints initial param values
        the objective of these stats are to verify the initialisation of the model

        - what it should do:
            - for each group, print the initialisation strategy and then the stats for the layers in the group
        - concerns:
            - how should it report layers that are not part of any group/has default initialisation? 
        """
        print(f"[init] {tag}")
        for name, p in self._resolve_zuker_model(self.model).named_parameters():
            if p is None or p.numel() == 0:
                print(f"[init] {name} empty")
                continue
            p_det = p.detach()
            finite = torch.isfinite(p_det)
            if finite.any():
                vals = p_det[finite]
                mean = float(vals.mean().item())
                if vals.numel() > 1:
                    std = float(vals.std(unbiased=False).item())
                else:
                    std = 0.0
                p_min = float(vals.min().item())
                p_max = float(vals.max().item())
            else:
                mean = std = p_min = p_max = float("nan")
            nan = int(torch.isnan(p_det).sum().item())
            inf = int(torch.isinf(p_det).sum().item())
            print(f"[init] {name} mean={mean:.4g} std={std:.4g} min={p_min:.4g} max={p_max:.4g} nan={nan} inf={inf}")

    def log_grad_stats(self, step: int, stage: TrainStage = TrainStage.PRE_UNSCALE):
        """
        prints gradient statistics for each named parameter. this does not write into tensorboard

        why we need it: i am trying to track down why the signal does not successfully propagate backward. printing stats of each layer helps me identify the following:
        - what the values are pre-and-post unscaling
        - what the values are pre-and-post clip
        - max - min: see whether weight layers are actually learning stuff rather than just being noise
        - mean: if the mean is near zero, it means this layer isn't contributing anything
        - nan and inf: self-explanatory, alerts us if there is an exploding gradient or underflow
        - 
        """
        total_norm_sq = 0.0
        match stage:
            case TrainStage.PRE_UNSCALE:
                stage_name = "pre_unscale"
            case TrainStage.PRE_CLIP:
                stage_name = "pre_clip"
            case TrainStage.POST_CLIP:
                stage_name = "post_clip"
                
        for name, p in self._resolve_zuker_model(self.model).named_parameters():
            if p is None or p.grad is None:
                print(f"[grad] [{stage_name}] {name} grad=None")
                continue
            g_det = p.grad.detach()
            if g_det.numel() == 0:
                print(f"[grad] [{stage_name}] {name} grad=empty")
                continue
            finite = torch.isfinite(g_det)
            if finite.any():
                vals = g_det[finite]
                g_min = float(vals.min().item())
                g_max = float(vals.max().item())
                g_mean = float(vals.mean().item())
            else:
                g_min = g_max = g_mean = float("nan")
            g_nan = int(torch.isnan(g_det).sum().item())
            g_inf = int(torch.isinf(g_det).sum().item())
            print(f"[grad] [{stage_name}] {name} mean={g_mean:.4g} min={g_min:.4g} max={g_max:.4g} nan={g_nan} inf={g_inf}")
            total_norm_sq += g_det.norm().item() ** 2
        print(f"[grad] [{stage_name}] total_norm={total_norm_sq ** 0.5:.4g}")

    def log_grad_group_metrics(self, step: int):
        """
        logs grad norms for each group

        goal: comparing total norms across different groups allows us to determine whether updates are flowing through the model
        """
        if self.writer is None:
            return

        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            norm_sq = 0.0
            nan_count = 0
            inf_count = 0
            n_with_grad = 0
            n_params = 0
            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                g = p.grad.detach()
                norm_sq += g.norm().item() ** 2
                nan_count += int(torch.isnan(g).sum().item())
                inf_count += int(torch.isinf(g).sum().item())
                n_with_grad += 1
                n_params += g.numel()
            
            grad_norm = norm_sq ** 0.5
            self.writer.add_scalar(f"train/health/grad_norm/{group_name}", grad_norm, step)

            if n_params > 0:
                grad_rms = grad_norm / (n_params ** 0.5)
                self.writer.add_scalar(f"train/health/normalised_grad_norm/{group_name}", grad_rms, step)

    @staticmethod
    def _append_unique_params(dest, params, seen, exclude_ids=None, require_grad=True):
        if exclude_ids is None:
            exclude_ids = set()
        for p in params:
            if p is None:
                continue
            if require_grad and not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen or pid in exclude_ids:
                continue
            seen.add(pid)
            dest.append(p)

    def _collect_optimizer_group_params(self, group_predicate, exclude_ids=None):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return []

        params = []
        seen = set()
        for group in optimizer.param_groups:
            group_name = group.get("name", "")
            if not group_predicate(group_name):
                continue
            self._append_unique_params(
                params,
                group.get("params", []),
                seen,
                exclude_ids=exclude_ids,
            )
        return params

    def _collect_module_params(self, modules, exclude_ids=None):
        params = []
        seen = set()
        for module in modules:
            if module is None:
                continue
            self._append_unique_params(
                params,
                module.parameters(),
                seen,
                exclude_ids=exclude_ids,
            )
        return params

    def _collect_named_model_params(self, name_predicate, exclude_ids=None, require_grad=True):
        zuker_model = self._resolve_zuker_model(self.model)
        if zuker_model is None:
            return []

        params = []
        seen = set()
        for name, p in zuker_model.named_parameters():
            if not name_predicate(name):
                continue
            self._append_unique_params(
                params,
                [p],
                seen,
                exclude_ids=exclude_ids,
                require_grad=require_grad,
            )
        return params

    def _iter_head_params(self):
        head_params = self._collect_optimizer_group_params(
            lambda name: name.startswith(self.HEAD_GROUP_PREFIXES)
        )
        if head_params:
            return head_params

        return self._collect_named_model_params(
            lambda name: name.startswith(self.HEAD_PARAM_PREFIXES)
        )
    
    def freeze_heads(self):
        # Collect by name prefixes so we can still find the same params after freezing.
        seen = set()
        zuker_model = self._resolve_zuker_model(self.model)
        if zuker_model is None:
            return
        for name, p in zuker_model.named_parameters():
            if p is None:
                continue
            if not name.startswith(self.HEAD_PARAM_PREFIXES):
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            p.requires_grad = False

    def unfreeze_heads(self):
        seen = set()
        zuker_model = self._resolve_zuker_model(self.model)
        if zuker_model is None:
            return
        for name, p in zuker_model.named_parameters():
            if p is None:
                continue
            if not name.startswith(self.HEAD_PARAM_PREFIXES):
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            p.requires_grad = True

    def _iter_encoder_params(self):
        zuker_model = self._resolve_zuker_model(self.model)
        if zuker_model is None:
            return []

        excluded = {id(p) for p in self._iter_head_params()}
        excluded.update(id(p) for p in self._iter_sinkhorn_params())
        return self._collect_module_params(
            [zuker_model.net.embedding, zuker_model.net.encoder],
            exclude_ids=excluded,
        )

    def _iter_sinkhorn_params(self):
        sinkhorn_params = self._collect_optimizer_group_params(
            lambda name: name == self.SINKHORN_GROUP_NAME
        )
        if sinkhorn_params:
            return sinkhorn_params

        zuker_model = self._resolve_zuker_model(self.model)
        if (
            zuker_model is None
            or not hasattr(zuker_model, "net")
            or not hasattr(zuker_model.net, "sparse_sinkhorn")
        ):
            return sinkhorn_params

        modules = [zuker_model.net.sparse_sinkhorn]
        if hasattr(zuker_model.net, "sinkhorn_mask_proj"):
            modules.append(zuker_model.net.sinkhorn_mask_proj)

        params = self._collect_module_params(modules)
        gate_scale = getattr(zuker_model.net, "sinkhorn_gate_scale", None)
        if gate_scale is not None and gate_scale.requires_grad:
            params.append(gate_scale)
        return params

    def apply_granular_freeze(self, *, freeze_embedding=False, freeze_cnn=False, freeze_lstm=False, freeze_heads=False, freeze_sinkhorn=False):
        total_params = 0
        trainable_params = 0
        frozen_parts = []
        for name, p in self.model.named_parameters():
            if p is None:
                continue
            total_params += p.numel()
            should_freeze = (
                (freeze_embedding and self._is_embedding_param_name(name))
                or (freeze_cnn and self._is_cnn_param_name(name))
                or (freeze_lstm and self._is_lstm_param_name(name))
                or (freeze_heads and self._is_head_param_name(name))
                or (freeze_sinkhorn and self._is_sinkhorn_param_name(name))
            )
            p.requires_grad = not should_freeze
            if p.requires_grad:
                trainable_params += p.numel()

        if freeze_embedding:
            frozen_parts.append("embedding")
        if freeze_cnn:
            frozen_parts.append("cnn")
        if freeze_lstm:
            frozen_parts.append("lstm")
        if freeze_heads:
            frozen_parts.append("heads")
        if freeze_sinkhorn:
            frozen_parts.append("sinkhorn")
        frozen_label = ", ".join(frozen_parts) if frozen_parts else "nothing"
        print(
            f"[setup] applied granular freeze ({frozen_label}); "
            f"trainable={trainable_params} frozen={total_params - trainable_params}"
        )

    @staticmethod
    def _remap_state_dict_for_model(model, state_dict):
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Standalone SinkhornContact reuses the old MixC encoder/paired stack but exposes
        # them under `net.*` rather than `zuker.net.*`. Remap those checkpoint keys so
        # partial loading can recover the pretrained CNN/LSTM/head weights.
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
    def load_filtered_state_dict(model, state_dict, key_predicate=None):
        state_dict = Train._remap_state_dict_for_model(model, state_dict)

        model_state = model.state_dict()
        filtered_state = {}
        skipped_missing = []
        skipped_shape = []

        for key, value in state_dict.items():
            if key_predicate is not None and not key_predicate(key):
                continue
            if key not in model_state:
                skipped_missing.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value

        incompatible = model.load_state_dict(filtered_state, strict=False)
        return {
            "loaded_keys": sorted(filtered_state.keys()),
            "skipped_missing": skipped_missing,
            "skipped_shape": skipped_shape,
            "missing_after_load": list(incompatible.missing_keys),
            "unexpected_after_load": list(incompatible.unexpected_keys),
        }

    @classmethod
    def load_partial_state_dict(cls, model, state_dict):
        return cls.load_filtered_state_dict(model, state_dict)

    @classmethod
    def load_sinkhorn_state_dict(cls, model, state_dict):
        return cls.load_filtered_state_dict(
            model,
            state_dict,
            key_predicate=cls._is_sinkhorn_param_name,
        )

    @staticmethod
    def _print_load_info(prefix: str, load_info: dict, *, max_items: int | None = None):
        def _iter_items(items):
            return items if max_items is None else items[:max_items]

        print(
            f"{prefix}: "
            f"loaded={len(load_info['loaded_keys'])} "
            f"skipped_missing={len(load_info['skipped_missing'])} "
            f"skipped_shape={len(load_info['skipped_shape'])} "
            f"missing_after_load={len(load_info['missing_after_load'])} "
            f"unexpected_after_load={len(load_info['unexpected_after_load'])}"
        )
        for key in _iter_items(load_info["skipped_missing"]):
            print(f"{prefix} skipped missing key: {key}")
        for key, src_shape, dst_shape in _iter_items(load_info["skipped_shape"]):
            print(f"{prefix} skipped shape mismatch: {key} src={src_shape} dst={dst_shape}")
        for key in _iter_items(load_info["missing_after_load"]):
            print(f"{prefix} missing after load: {key}")
        for key in _iter_items(load_info["unexpected_after_load"]):
            print(f"{prefix} unexpected after load: {key}")

    @staticmethod
    def compute_grad_norm(params) -> float:
        norm_sq = 0.0
        for p in params:
            if p is None or p.grad is None:
                continue
            g = p.grad.detach()
            norm_sq += g.norm().item() ** 2
        return norm_sq ** 0.5

    @staticmethod
    def compute_param_std(params) -> float:
        total = 0
        sum_x = 0.0
        sum_x2 = 0.0
        for p in params:
            if p is None:
                continue
            t = p.detach().reshape(-1)
            finite = torch.isfinite(t)
            if not finite.any():
                continue
            vals = t[finite].to(dtype=torch.float64)
            total += vals.numel()
            sum_x += vals.sum().item()
            sum_x2 += (vals * vals).sum().item()
        if total <= 1:
            return 0.0
        mean = sum_x / total
        var = max(sum_x2 / total - mean * mean, 0.0)
        return var ** 0.5

    def log_param_std_metrics(self, step: int):
        model_params = [p for p in self.model.parameters() if p is not None and p.requires_grad]
        model_std = self.compute_param_std(model_params)
        if self.writer is not None:
            self.writer.add_scalar("train/param_std/all", model_std, step)

        head_params = []
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            std = self.compute_param_std(group["params"])
            if self.writer is not None:
                self.writer.add_scalar(f"train/param_std/{group_name}", std, step)
            if group_name.startswith("fc_"):
                head_params.extend(group["params"])

        head_std = None
        if head_params:
            head_std = self.compute_param_std(head_params)
            if self.writer is not None:
                self.writer.add_scalar("train/param_std/head", head_std, step)

        if self.verbose:
            parts = [f"[param_std] step={step} all={model_std:.4g}"]
            if head_std is not None:
                parts.append(f"head={head_std:.4g}")
            print(" ".join(parts))

    def log_lr_metrics(self, step: int, scheduler_metric: float | None = None, source: str = "optimizer"):
        if getattr(self, "optimizer", None) is None:
            return

        lr_values = []
        lr_parts = []
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            group_lr = float(group["lr"])
            lr_values.append(group_lr)
            lr_parts.append(f"{group_name}={group_lr:.3e}")
            if self.writer is not None:
                self.writer.add_scalar(f"train/lr/{group_name}", group_lr, step)

        if lr_values and self.writer is not None:
            self.writer.add_scalar("train/lr/min", min(lr_values), step)
            self.writer.add_scalar("train/lr/max", max(lr_values), step)

        metric_part = ""
        if scheduler_metric is not None:
            metric_part = f" metric={scheduler_metric:.6f}"
        print(f"[lr] epoch={step} source={source}{metric_part} lrs: {', '.join(lr_parts)}")

    def log_head_bias_mean_metrics(self, step: int):
        head_ids = {id(p) for p in self._iter_head_params()}
        means = []

        for name, p in self.model.named_parameters():
            if p is None or not p.requires_grad:
                continue
            if id(p) not in head_ids:
                continue
            if not name.endswith("bias"):
                continue

            t = p.detach().reshape(-1)
            finite = torch.isfinite(t)
            if finite.any():
                mean = float(t[finite].mean().item())
                means.append(mean)
            else:
                mean = float("nan")

            if self.writer is not None:
                self.writer.add_scalar(f"train/head_bias_mean/{name.replace('.', '/')}", mean, step)

        if means:
            agg_mean = float(np.mean(means))
            if self.writer is not None:
                self.writer.add_scalar("train/head_bias_mean/aggregate", agg_mean, step)
            if self.verbose:
                print(f"[head_bias_mean] step={step} aggregate={agg_mean:.4g}")

    def _resolve_sinkhorn_layer(self):
        zuker_model = self._resolve_zuker_model(self.model)
        if zuker_model is None:
            return None
        if hasattr(zuker_model, "sparse_sinkhorn"):
            return getattr(zuker_model, "sparse_sinkhorn", None)
        if hasattr(zuker_model, "net") and hasattr(zuker_model.net, "sparse_sinkhorn"):
            return getattr(zuker_model.net, "sparse_sinkhorn", None)
        return None

    def compute_sinkhorn_temp(self, epoch: int) -> float | None:
        start_temp = getattr(self, "sinkhorn_temp_start", None)
        end_temp = getattr(self, "sinkhorn_temp_end", None)
        if start_temp is None or end_temp is None:
            return None

        start_epoch = max(int(self.sinkhorn_temp_anneal_start_epoch), 1)
        end_epoch = max(int(self.sinkhorn_temp_anneal_end_epoch), start_epoch)

        if epoch <= start_epoch:
            return float(start_temp)
        if epoch >= end_epoch:
            return float(end_temp)
        ratio = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
        return float(start_temp + ratio * (end_temp - start_temp))

    def set_sinkhorn_temp(self, epoch: int):
        sinkhorn_layer = self._resolve_sinkhorn_layer()
        if sinkhorn_layer is None:
            return None

        temp = self.compute_sinkhorn_temp(epoch)
        if temp is None:
            temp = float(sinkhorn_layer.temp)
        sinkhorn_layer.temp = float(temp)

        if self.writer is not None:
            self.writer.add_scalar("train/representations/sparse_sinkhorn_temp", float(temp), epoch)
        if self.verbose:
            print(f"[sinkhorn] epoch={epoch} temp={temp:.4g}")
        return float(temp)

    def train(self, epoch, lr=1e-4):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, num = 0, 0
        running_loss, running_margin_loss, running_sl_loss, running_aux_loss, running_l1_loss, n_running_loss = 0, 0, 0, 0, 0, 0
        running_energy_gap = 0
        start = time.time()

        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for batch_idx, (fnames, seqs, pairs, bpp_targets) in enumerate(self.train_loader):
                if self.verbose:
                    print()
                    print("Step: {}, {}".format(self.step, fnames))
                    self.step += 1

                global_step: int = (epoch - 1) * n_dataset + batch_idx                
                curr_lrs = ", ".join(
                    f"{group.get('name', f'group_{i}')}={group['lr']:.2e}"
                    for i, group in enumerate(self.optimizer.param_groups)
                )

                # 3. Log the status every few batches (to avoid bloating the log file)
                if self.verbose and batch_idx % 10 == 0:
                    print(f"[lr_log] Epoch: {epoch}, Batch: {batch_idx}, Global Step: {global_step}, LR: {curr_lrs}")
                
                n_batch = len(seqs)
                # torch.cuda.reset_peak_memory_stats()
                self.optimizer.zero_grad()

                
                loss_calc : Loss = self.loss_fn(seqs, pairs, fname=fnames, bpp_target=bpp_targets)
                # yes, batch size is 1, but we use .sum() to future-proof
                loss = loss_calc.loss.sum()
                margin_loss = loss_calc.margin_loss.sum()
                sl_loss = loss_calc.score_loss.sum()
                aux_loss = loss_calc.aux_loss.sum()
                l1_loss = loss_calc.l1_loss.sum()

                if hasattr(self.loss_fn, "_last_timing") and (self.verbose or n_dataset <= 5):
                    timing = self.loss_fn._last_timing
                    if timing is not None:
                        pred_forward_s = timing.get("pred_forward_s", timing.get("pred_mix_s", 0.0))
                        ref_forward_s = timing.get("ref_forward_s", timing.get("ref_mix_s", 0.0))
                        mix_free_s = timing.get("mix_free_s", 0.0)
                        l1_s = timing.get("l1_s", 0.0)
                        print(
                            "[timing] "
                            f"loss_total={timing['total_s']:.3f}s "
                            f"pred_forward={pred_forward_s:.3f}s "
                            f"mix_free={mix_free_s:.3f}s "
                            f"ref_forward={ref_forward_s:.3f}s "
                            f"l1={l1_s:.3f}s "
                            f"gap_turner={timing.get('gap_turner_total_s', 0.0):.3f}s "
                            f"turner_hits={timing.get('turner_cache_hits', 0)} "
                            f"turner_misses={timing.get('turner_cache_misses', 0)}"
                        )
                        detail_labels = (
                            ("pred_forward_detail", "pred_forward"),
                            ("mix_free_detail", "mix_free"),
                            ("ref_forward_detail", "ref_forward"),
                        )
                        for label, display_name in detail_labels:
                            detail = timing.get(label)
                            if detail is None:
                                legacy_label = {
                                    "pred_forward_detail": "pred_mix_detail",
                                    "ref_forward_detail": "ref_mix_detail",
                                }.get(label, label)
                                detail = timing.get(legacy_label)
                            if detail is None:
                                continue
                            print(
                                f"[timing] {display_name} "
                                f"total={detail['total_s']:.3f}s "
                                f"make_param={detail['make_param_s']:.3f}s "
                                f"zuker_make_param={detail.get('zuker_make_param_s', 0.0):.3f}s "
                                f"cpu_copy={detail['cpu_copy_s']:.3f}s "
                                f"clear_count={detail['clear_count_s']:.3f}s "
                                f"cast={detail['cast_contiguous_s']:.3f}s "
                                f"predict={detail['predict_s']:.3f}s "
                                f"diff_score={detail['diff_score_s']:.3f}s"
                            )
                        if timing.get("gap_turner_total_s", 0.0) > 0.0:
                            print(
                                "[timing] turner_calls "
                                f"constrained={timing.get('turner_constrained_s', 0.0):.3f}s "
                                f"free={timing.get('turner_free_s', 0.0):.3f}s"
                            )

                if self.writer is not None and batch_idx % 100 == 0:
                    zuker_model = self._resolve_zuker_model(self.model)
                    sinkhorn_layer = self._resolve_sinkhorn_layer()
                    if (
                        zuker_model is not None
                        and sinkhorn_layer is not None
                    ):
                        mask = getattr(zuker_model, "_last_sinkhorn_mask", None)
                        if mask is None and hasattr(zuker_model, "net"):
                            mask = getattr(zuker_model.net, "_last_sinkhorn_mask", None)
                        if mask is not None and mask.numel() > 0:
                            seq_len = len(seqs[0]) if len(seqs) > 0 else 0
                            if sinkhorn_layer is not None and hasattr(sinkhorn_layer, "_last_effective_temp"):
                                self._log_sinkhorn_scalar(
                                    "train/representations/sparse_sinkhorn_temp_effective",
                                    float(sinkhorn_layer._last_effective_temp),
                                    global_step,
                                    seq_len,
                                )
                            block_pooled = getattr(sinkhorn_layer, "_last_block_pooled", None)
                            if block_pooled is not None and block_pooled.numel() > 0:
                                block_pooled = block_pooled[0].detach().to(dtype=torch.float32)
                                finite_pooled = torch.isfinite(block_pooled)
                                if finite_pooled.any():
                                    finite_pooled_vals = block_pooled[finite_pooled]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pooled_min",
                                        float(finite_pooled_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pooled_max",
                                        float(finite_pooled_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pooled_mean",
                                        float(finite_pooled_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pooled_range",
                                        float((finite_pooled_vals.max() - finite_pooled_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            block_projected = getattr(sinkhorn_layer, "_last_block_projected", None)
                            if block_projected is not None and block_projected.numel() > 0:
                                block_projected = block_projected[0].detach().to(dtype=torch.float32)
                                finite_projected = torch.isfinite(block_projected)
                                if finite_projected.any():
                                    finite_projected_vals = block_projected[finite_projected]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_projected_min",
                                        float(finite_projected_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_projected_max",
                                        float(finite_projected_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_projected_mean",
                                        float(finite_projected_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_projected_range",
                                        float((finite_projected_vals.max() - finite_projected_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            block_raw_bilinear_logits = getattr(
                                sinkhorn_layer,
                                "_last_block_raw_bilinear_logits",
                                None,
                            )
                            if block_raw_bilinear_logits is not None and block_raw_bilinear_logits.numel() > 0:
                                block_raw_bilinear_logits = block_raw_bilinear_logits[0].detach().to(dtype=torch.float32)
                                finite_raw_bilinear_logits = torch.isfinite(block_raw_bilinear_logits)
                                if finite_raw_bilinear_logits.any():
                                    finite_raw_bilinear_logit_vals = block_raw_bilinear_logits[finite_raw_bilinear_logits]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_raw_bilinear_logit_min",
                                        float(finite_raw_bilinear_logit_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_raw_bilinear_logit_max",
                                        float(finite_raw_bilinear_logit_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_raw_bilinear_logit_mean",
                                        float(finite_raw_bilinear_logit_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_raw_bilinear_logit_range",
                                        float((finite_raw_bilinear_logit_vals.max() - finite_raw_bilinear_logit_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            block_pre_sinkhorn_logits = getattr(
                                sinkhorn_layer,
                                "_last_block_pre_sinkhorn_logits",
                                None,
                            )
                            if block_pre_sinkhorn_logits is not None and block_pre_sinkhorn_logits.numel() > 0:
                                block_pre_sinkhorn_logits = block_pre_sinkhorn_logits[0].detach().to(dtype=torch.float32)
                                finite_pre_sinkhorn_logits = torch.isfinite(block_pre_sinkhorn_logits)
                                if finite_pre_sinkhorn_logits.any():
                                    finite_pre_sinkhorn_logit_vals = block_pre_sinkhorn_logits[finite_pre_sinkhorn_logits]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_sinkhorn_logit_min",
                                        float(finite_pre_sinkhorn_logit_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_sinkhorn_logit_max",
                                        float(finite_pre_sinkhorn_logit_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_sinkhorn_logit_mean",
                                        float(finite_pre_sinkhorn_logit_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_sinkhorn_logit_range",
                                        float((finite_pre_sinkhorn_logit_vals.max() - finite_pre_sinkhorn_logit_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            block_pre_loop_logits = getattr(
                                sinkhorn_layer,
                                "_last_block_pre_loop_logits",
                                None,
                            )
                            if block_pre_loop_logits is not None and block_pre_loop_logits.numel() > 0:
                                block_pre_loop_logits = block_pre_loop_logits[0].detach().to(dtype=torch.float32)
                                finite_pre_loop_logits = torch.isfinite(block_pre_loop_logits)
                                if finite_pre_loop_logits.any():
                                    finite_pre_loop_logit_vals = block_pre_loop_logits[finite_pre_loop_logits]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_loop_logit_min",
                                        float(finite_pre_loop_logit_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_loop_logit_max",
                                        float(finite_pre_loop_logit_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_loop_logit_mean",
                                        float(finite_pre_loop_logit_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_pre_loop_logit_range",
                                        float((finite_pre_loop_logit_vals.max() - finite_pre_loop_logit_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            block_pool_attn_weights = getattr(
                                sinkhorn_layer,
                                "_last_block_pool_attn_weights",
                                None,
                            )
                            block_pool_valid_mask = getattr(
                                sinkhorn_layer,
                                "_last_block_pool_valid_mask",
                                None,
                            )
                            if (
                                block_pool_attn_weights is not None
                                and block_pool_valid_mask is not None
                                and block_pool_attn_weights.numel() > 0
                                and block_pool_valid_mask.numel() > 0
                            ):
                                self._log_sinkhorn_pooling_metrics(
                                    "train/representations/sparse_sinkhorn_pool_attn",
                                    block_pool_attn_weights[0],
                                    block_pool_valid_mask[0],
                                    global_step,
                                    seq_len,
                                )
                            block_logits = getattr(sinkhorn_layer, "_last_block_logits", None)
                            if block_logits is not None and block_logits.numel() > 0:
                                block_logits = block_logits[0].detach().to(dtype=torch.float32)
                                finite_logits = torch.isfinite(block_logits)
                                if finite_logits.any():
                                    finite_logit_vals = block_logits[finite_logits]
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_logit_min",
                                        float(finite_logit_vals.min().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_logit_max",
                                        float(finite_logit_vals.max().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_logit_mean",
                                        float(finite_logit_vals.mean().item()),
                                        global_step,
                                        seq_len,
                                    )
                                    self._log_sinkhorn_scalar(
                                        "train/representations/sparse_sinkhorn_block_logit_range",
                                        float((finite_logit_vals.max() - finite_logit_vals.min()).item()),
                                        global_step,
                                        seq_len,
                                    )
                            self._log_sinkhorn_scalar(
                                "train/representations/sparse_sinkhorn_seq_len",
                                float(seq_len),
                                global_step,
                                seq_len,
                            )
                            mask_img = mask[0].detach().to(dtype=torch.float32)
                            finite = torch.isfinite(mask_img)
                            if finite.any():
                                finite_vals = mask_img[finite]
                                mask_img = torch.where(
                                    finite,
                                    mask_img.clamp(0.0, 1.0),
                                    torch.zeros_like(mask_img),
                                )
                                self._log_sinkhorn_scalar(
                                    "train/representations/sparse_sinkhorn_mask_min",
                                    float(finite_vals.min().item()),
                                    global_step,
                                    seq_len,
                                )
                                self._log_sinkhorn_scalar(
                                    "train/representations/sparse_sinkhorn_mask_max",
                                    float(finite_vals.max().item()),
                                    global_step,
                                    seq_len,
                                )
                                self._log_sinkhorn_scalar(
                                    "train/representations/sparse_sinkhorn_mask_mean",
                                    float(finite_vals.mean().item()),
                                    global_step,
                                    seq_len,
                                )

                        if (
                            sinkhorn_layer is not None
                            and mask is not None
                            and mask.numel() > 0
                        ):
                            block_size = max(int(sinkhorn_layer.block_size), 1)
                            block_img = getattr(sinkhorn_layer, "_last_block_mask", None)
                            if block_img is None:
                                block_img = mask[:, ::block_size, ::block_size]
                            block_img = block_img[0]
                            self._log_sinkhorn_block_metrics(
                                "train/representations/sparse_sinkhorn_block",
                                block_img,
                                global_step,
                                seq_len,
                                log_image=True,
                            )
                            block_img_with_slack = getattr(sinkhorn_layer, "_last_block_mask_with_slack", None)
                            num_real_blocks = getattr(sinkhorn_layer, "_last_num_blocks", None)
                            if (
                                block_img_with_slack is not None
                                and num_real_blocks is not None
                                and block_img_with_slack.numel() > 0
                            ):
                                self._log_sinkhorn_block_metrics(
                                    "train/representations/sparse_sinkhorn_block_full",
                                    block_img_with_slack[0],
                                    global_step,
                                    seq_len,
                                    log_image=False,
                                )
                                self._log_sinkhorn_slack_metrics(
                                    "train/representations/sparse_sinkhorn_block",
                                    block_img_with_slack[0],
                                    int(num_real_blocks),
                                    global_step,
                                    seq_len,
                                    pairs=pairs,
                                    block_size=block_size,
                                )

                loss_total += loss.item()
                num += n_batch
                if loss.item() > 0.:
                    loss.backward()
                    if self.verbose and batch_idx % 100 == 0:
                        print(f"[step] {loss}")
                        # self.log_grad_stats(global_step, TrainStage.PRE_UNSCALE)
                    # Clip Evoformer and head modules separately.
                    if self.verbose and batch_idx % 100 == 0:
                        self.log_grad_stats(global_step, TrainStage.PRE_CLIP)
                    if self.writer is not None and batch_idx % 100 == 0:
                        self.log_grad_group_metrics(global_step)

                    # head_params = self._iter_head_params()
                    # pre_clip_head_norm = self.compute_grad_norm(head_params)
                    # clipping_factor = min(1.0, self.max_head_grad_norm / max(pre_clip_head_norm, epsilon))
                    # clipped_head_norm = self.clip_head_grad_norm_(self.max_head_grad_norm)
                    # post_clip_head_norm = self.compute_grad_norm(head_params)

                    # THROWAWAY
                    head_params = self._iter_head_params()
                    encoder_params = self._iter_encoder_params()
                    sinkhorn_params = self._iter_sinkhorn_params()
                    torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=self.max_encoder_grad_norm)
                    torch.nn.utils.clip_grad_norm_(sinkhorn_params, max_norm=self.max_sinkhorn_grad_norm)
                    torch.nn.utils.clip_grad_norm_(head_params, max_norm=self.max_head_grad_norm)

                    if self.verbose and batch_idx % 100 == 0:
                        self.log_grad_stats(global_step, TrainStage.POST_CLIP)

                    if self.writer is not None and batch_idx % 100 == 0:
                        # self.writer.add_scalar("train/health/head/total_norm_pre_clip", pre_clip_head_norm, global_step)
                        # self.writer.add_scalar("train/health/head/total_norm_post_clip", post_clip_head_norm, global_step)
                        # self.writer.add_scalar("train/health/head/clipping_factor", clipping_factor, global_step)
                        
                        # self.writer.add_scalar("train/health/Numerical/Grad_Scale", scale_factor, global_step)
                        #print(
                        #    f"[clip] head_total_norm_pre={pre_clip_head_norm:.4g} "
                        #    f"head_total_norm_post={post_clip_head_norm:.4g} "
                        #    f"head_clip_factor={clipping_factor:.4g} "
                        #    f"head_reported_norm={clipped_head_norm:.4g} "
                        #    f"head_max_norm={self.max_head_grad_norm:.4g}"
                        #)
                        self.log_param_std_metrics(global_step)
                        self.log_head_bias_mean_metrics(global_step)

                        # TODO: clean this up
                        zuker_model = self._resolve_zuker_model(self.model)
                        if zuker_model is not None and hasattr(zuker_model, "_score_range") and hasattr(zuker_model, "_score_mean"):
                            score_paired_range, score_unpaired_range = zuker_model._score_range
                            score_paired_mean, score_unpaired_mean = zuker_model._score_mean
                            if self.writer is not None and batch_idx % 100 == 0:
                                self.writer.add_scalar("train/summary/score_ratios", (score_paired_range + epsilon) / (score_unpaired_range + epsilon), global_step)
                                self.writer.add_scalar("train/representations/contrast/score_paired_range", score_paired_range, global_step)
                                self.writer.add_scalar("train/representations/contrast/score_unpaired_range", score_unpaired_range, global_step)
                                self.writer.add_scalar("train/representations/contrast/score_paired_mean", score_paired_mean, global_step)
                                if score_unpaired_mean is not None:
                                    self.writer.add_scalar("train/representations/contrast/score_unpaired_mean", score_unpaired_mean, global_step)

                    self.optimizer.step()
                
                # print("peak GB", torch.cuda.max_memory_allocated() / 1e9)

                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

                running_loss += loss.item()
                running_margin_loss += margin_loss.item()
                running_sl_loss += sl_loss.item()
                running_aux_loss += aux_loss.item()
                running_l1_loss += l1_loss.item()
                running_energy_gap += loss_calc.calc_energy_gap()
                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    running_margin_loss /= n_running_loss
                    running_sl_loss /= n_running_loss
                    running_aux_loss /= n_running_loss
                    running_l1_loss /= n_running_loss
                    running_energy_gap /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/summary/loss", running_loss, global_step)
                        self.writer.add_scalar("train/summary/margin_loss", running_margin_loss, global_step)
                        self.writer.add_scalar("train/summary/sl_loss", running_sl_loss, global_step)
                        self.writer.add_scalar("train/summary/sinkhorn_aux_loss", running_aux_loss, global_step)
                        self.writer.add_scalar("train/summary/l1_loss", running_l1_loss, global_step)
                        self.writer.add_scalar("train/summary/energy_gap", running_energy_gap,global_step)
                        self.writer.add_scalar("train/summary/margin_sl_ratio", (sl_loss.item() + epsilon) / (margin_loss.item() + epsilon), global_step)
                    running_loss, running_margin_loss, running_sl_loss, running_aux_loss, running_l1_loss, running_energy_gap = 0, 0, 0, 0, 0, 0
                    n_running_loss = 0
        elapsed_time = time.time() - start
        if self.verbose:
            print()
        print('Train Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))
        if self.writer is not None:
            self.writer.flush()
        return loss_total / num


    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, num = 0, 0
        aux_total = 0
        start = time.time()
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs, bpp_targets in self.test_loader:
                n_batch = len(seqs)
                loss_calc = self.loss_fn(seqs, pairs, fname=fnames, bpp_target=bpp_targets)
                loss = loss_calc.loss
                loss_total += loss.item()
                aux_total += loss_calc.aux_loss.sum().item()
                num += n_batch
                pbar.set_postfix(test_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

        elapsed_time = time.time() - start
        if self.writer is not None:
            self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
            self.writer.add_scalar("test/sinkhorn_aux_loss", aux_total / num, epoch * n_dataset)
        print('Test Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))
        if self.writer is not None:
            self.writer.flush()
        return loss_total / num


    def save_checkpoint(self, outdir, epoch):
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if getattr(self, "scheduler", None) is not None else None,
        }, filename)


    def resume_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if (
            getattr(self, "scheduler", None) is not None
            and checkpoint.get('scheduler_state_dict') is not None
        ):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return epoch


    def build_model(self, args):
        if args.model == 'Turner':
            return RNAFold(), {}

        config = {
            'max_helix_length': args.max_helix_length,
            'embed_size' : args.embed_size,
            'num_filters': args.num_filters if args.num_filters is not None else (96,),
            'filter_size': args.filter_size if args.filter_size is not None else (5,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation, 
            'num_lstm_layers': args.num_lstm_layers, 
            'num_lstm_units': args.num_lstm_units,
            'num_transformer_layers': args.num_transformer_layers,
            'num_transformer_hidden_units': args.num_transformer_hidden_units,
            'num_transformer_att': args.num_transformer_att,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (32,),
            'num_paired_filters': args.num_paired_filters,
            'paired_filter_size': args.paired_filter_size,
            'dropout_rate': args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'num_att': args.num_att,
            'pair_join': JOIN_MAP.get(args.pair_join.lower(), JoinType.CAT),
            'no_split_lr': args.no_split_lr,
            'use_sparse_sinkhorn': args.use_sparse_sinkhorn,
            'num_sinkhorn_iterations': args.num_sinkhorn_iterations,
            'sinkhorn_temp': args.sinkhorn_temp,
            'sinkhorn_block_size': args.sinkhorn_block_size,
            'sinkhorn_alpha': args.sinkhorn_alpha,
            'sinkhorn_temp_length_center': args.sinkhorn_temp_length_center,
            'sinkhorn_temp_length_slope': args.sinkhorn_temp_length_slope,
            'sinkhorn_pooling': args.sinkhorn_pooling,
            'sinkhorn_gating': args.sinkhorn_gating,
            'sinkhorn_diag_logit_penalty': args.sinkhorn_diag_logit_penalty,
            'sinkhorn_use_slack': args.sinkhorn_use_slack,
            'sinkhorn_real_only_normalization': args.sinkhorn_real_only_normalization,
        }

        if args.model == 'Zuker':
            model = ZukerFold(model_type=ZukerFold.ZukerType.M, **config)

        elif args.model == 'ZukerC':
            model = ZukerFold(model_type=ZukerFold.ZukerType.C, **config)

        elif args.model == 'ZukerL':
            model = ZukerFold(model_type=ZukerFold.ZukerType.L, **config)

        elif args.model == 'ZukerS':
            model = ZukerFold(model_type=ZukerFold.ZukerType.S, **config)

        elif args.model == 'Mix':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, **config)

        elif args.model == 'MixC':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, model_type=ZukerFold.ZukerType.C, **config)

        elif args.model == 'SinkhornContact':
            model = SinkhornContactFold(**config)

        else:
            raise('not implemented')

        return model, config


    def build_optimizer(self, optimizer, model, lr, l2_weight, args=None):
        head_mult = getattr(args, "head_lr_mult", 1.0) if args is not None else 1.0
        encoder_mult = getattr(args, "encoder_lr_mult", 1.0) if args is not None else 1.0
        sinkhorn_mult = getattr(args, "sinkhorn_lr_mult", 1.0) if args is not None else 1.0

        print(f"[setup] head lr mult: {head_mult}")

        param_groups = None
        param_groups = []
        used = set()
        zuker_model = self._resolve_zuker_model(model)

        def add_group(params, group_lr, group_mult, group_name):
            group_params = []
            for p in params:
                if p is None or not p.requires_grad:
                    continue
                pid = id(p)
                if pid in used:
                    continue
                used.add(pid)
                group_params.append(p)
            if group_params:
                param_groups.append({
                    "params": group_params,
                    "lr": group_lr,
                    "lr_mult": group_mult,
                    "name": group_name
                })

        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "encoder") and hasattr(zuker_model.net.encoder, "conv"):
            add_group(zuker_model.net.encoder.conv.parameters(), lr * encoder_mult, encoder_mult, "conv")
        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "encoder") and hasattr(zuker_model.net.encoder, "lstm"):
            add_group(zuker_model.net.encoder.lstm.parameters(), lr * encoder_mult, encoder_mult, "lstm")
        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "embedding"):
            add_group(zuker_model.net.embedding.parameters(), lr * encoder_mult, encoder_mult, "embedding")
        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "sparse_sinkhorn"):
            sinkhorn_params = list(zuker_model.net.sparse_sinkhorn.parameters())
            if hasattr(zuker_model.net, "sinkhorn_mask_proj"):
                sinkhorn_params.extend(list(zuker_model.net.sinkhorn_mask_proj.parameters()))
            gate_scale = getattr(zuker_model.net, "sinkhorn_gate_scale", None)
            if gate_scale is not None:
                sinkhorn_params.append(gate_scale)
            add_group(sinkhorn_params, lr * sinkhorn_mult, sinkhorn_mult, "sparse_sinkhorn")
        elif hasattr(model, "sparse_sinkhorn"):
            sinkhorn_params = list(model.sparse_sinkhorn.parameters())
            if hasattr(model, "contact_logit_proj"):
                sinkhorn_params.extend(list(model.contact_logit_proj.parameters()))
            add_group(sinkhorn_params, lr * sinkhorn_mult, sinkhorn_mult, "sparse_sinkhorn")
        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "fc_paired"):
            add_group(zuker_model.net.fc_paired.parameters(), lr * head_mult, head_mult, "fc_paired")
        if zuker_model is not None and hasattr(zuker_model, "net") and hasattr(zuker_model.net, "fc_unpaired") and zuker_model.net.fc_unpaired is not None:
            add_group(zuker_model.net.fc_unpaired.parameters(), lr * head_mult, head_mult, "fc_unpaired")
        if zuker_model is not None and hasattr(zuker_model, "fc_length"):
            for key in zuker_model.fc_length:
                add_group(zuker_model.fc_length[key].parameters(), lr * head_mult, head_mult, f"fc_length/{key}")
        add_group(model.parameters(), lr, 1.0, "remaining")
        
        if optimizer == 'Adam':
            return optim.Adam(param_groups if param_groups is not None else model.parameters(),
                              lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(param_groups if param_groups is not None else model.parameters(),
                               lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(param_groups if param_groups is not None else model.parameters(),
                                 lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(param_groups if param_groups is not None else model.parameters(),
                             nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
            #return optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(param_groups if param_groups is not None else model.parameters(),
                              lr=lr, weight_decay=l2_weight)
        else:
            raise('not implemented')

    def build_scheduler(self, optimizer, args):
        scheduler_name = getattr(args, "lr_scheduler", "none")
        if scheduler_name == "none":
            return None
        if scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience,
                threshold=args.lr_scheduler_threshold,
                min_lr=args.lr_scheduler_min_lr,
            )
        raise ValueError(f"Unknown lr scheduler: {scheduler_name}")


    def build_loss_function(self, loss_func, model, args):
        if loss_func != 'sinkhorn_contact' and float(args.sinkhorn_aux_weight) > 0.0:
            raise ValueError(
                "sinkhorn_aux_weight is only supported with --loss-func sinkhorn_contact in the current refactor"
            )
        if loss_func == 'hinge':
            return StructuredLoss(model, verbose=self.verbose,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            margin_weight=args.margin_weight)
        if loss_func == 'hinge_mix':
            return StructuredLossWithTurner(model, verbose=self.verbose,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            margin_weight=args.margin_weight, sl_weight=args.score_loss_weight)
        if loss_func == 'sinkhorn_contact':
            return SinkhornContactLoss(model, verbose=self.verbose,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sinkhorn_aux_weight=args.sinkhorn_aux_weight,
                            sinkhorn_aux_use_bpp=args.sinkhorn_aux_use_bpp,
                            sinkhorn_aux_pos_weight=args.sinkhorn_aux_pos_weight,
                            sinkhorn_aux_neg_weight=args.sinkhorn_aux_neg_weight,
                            sinkhorn_aux_diag_weight=args.sinkhorn_aux_diag_weight,
                            sinkhorn_aux_slack_self_weight=args.sinkhorn_aux_slack_self_weight,
                            sinkhorn_aux_true_pair_weight=args.sinkhorn_aux_true_pair_weight,
                            sinkhorn_aux_true_unpaired_weight=args.sinkhorn_aux_true_unpaired_weight,
                            sinkhorn_aux_false_pair_weight=args.sinkhorn_aux_false_pair_weight,
                            sinkhorn_aux_false_unpaired_weight=args.sinkhorn_aux_false_unpaired_weight,
                            sinkhorn_aux_min_separation=args.sinkhorn_aux_min_separation,
                            sinkhorn_aux_stack_bonus=args.sinkhorn_aux_stack_bonus)
        else:
            raise('not implemented')


    def save_config(self, file, config):
        with open(file, 'w') as f:
            for k, v in config.items():
                k = '--' + k.replace('_', '-')
                if type(v) is bool: # pylint: disable=unidiomatic-typecheck
                    if v:
                        f.write('{}\n'.format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write('{}\n{}\n'.format(k, vv))
                else:
                    f.write('{}\n{}\n'.format(k, v))


    def run(self, args, conf=None):
        self.disable_progress_bar = args.disable_progress_bar
        self.verbose = args.verbose
        self.max_head_grad_norm = args.max_head_grad_norm
        self.max_encoder_grad_norm = args.max_encoder_grad_norm
        self.max_sinkhorn_grad_norm = args.max_sinkhorn_grad_norm
        self.writer = None
        if args.log_dir is not None and 'SummaryWriter' in globals():
            self.writer = SummaryWriter(log_dir=args.log_dir)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # dataset preparation
        train_dataset = BPseqDataset(args.input, bpp_dir=args.bpp_dir)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input, bpp_dir=args.test_bpp_dir if args.test_bpp_dir else args.bpp_dir)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model, config = self.build_model(args)
        config.update({ 'model': args.model, 'param': args.param })

        self.init_model_weights(args)

        if self.verbose:
            self.log_model_init_stats("post_init")
        
        if args.init_param != '':
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
            p = torch.load(init_param)
            if args.init_param_partial:
                load_info = self.load_partial_state_dict(self.model, p)
                self._print_load_info("[setup] partially loaded init weights", load_info)
            else:
                if isinstance(p, dict) and 'model_state_dict' in p:
                    p = p['model_state_dict']
                self.model.load_state_dict(p)
            if self.verbose:
                self.log_model_init_stats("post_load")

        if args.init_sinkhorn_param != '':
            init_sinkhorn_param = Path(args.init_sinkhorn_param)
            if not init_sinkhorn_param.exists() and conf is not None:
                init_sinkhorn_param = Path(conf) / init_sinkhorn_param
            p = torch.load(init_sinkhorn_param)
            load_info = self.load_sinkhorn_state_dict(self.model, p)
            self._print_load_info("[setup] loaded sinkhorn init weights", load_info)
            if self.verbose:
                for key in load_info["loaded_keys"][:200]:
                    print(f"[setup] loaded sinkhorn key: {key}")
                self.log_model_init_stats("post_sinkhorn_load")

        granular_freeze_flags = (
            int(bool(args.freeze_embedding))
            +
            int(bool(args.freeze_cnn))
            + int(bool(args.freeze_lstm))
            + int(bool(args.freeze_heads))
            + int(bool(args.freeze_sinkhorn))
        )
        if granular_freeze_flags > 0:
            self.apply_granular_freeze(
                freeze_embedding=args.freeze_embedding,
                freeze_cnn=args.freeze_cnn,
                freeze_lstm=args.freeze_lstm,
                freeze_heads=args.freeze_heads,
                freeze_sinkhorn=args.freeze_sinkhorn,
            )

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
        self.optimizer = self.build_optimizer(args.optimizer, self.model, args.lr, args.l2_weight, args)
        self.scheduler = self.build_scheduler(self.optimizer, args)
        self.loss_fn = self.build_loss_function(args.loss_func, self.model, args)
        self.sinkhorn_temp_start = args.sinkhorn_temp if args.sinkhorn_temp_start is None else args.sinkhorn_temp_start
        self.sinkhorn_temp_end = self.sinkhorn_temp_start if args.sinkhorn_temp_end is None else args.sinkhorn_temp_end
        self.sinkhorn_temp_anneal_start_epoch = args.sinkhorn_temp_anneal_start_epoch
        self.sinkhorn_temp_anneal_end_epoch = (
            args.sinkhorn_temp_anneal_end_epoch if args.sinkhorn_temp_anneal_end_epoch > 0 else args.epochs
        )

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        """
        freeze_head_epochs = 10
        if checkpoint_epoch < freeze_head_epochs:
            self.freeze_heads()
        else:
            self.unfreeze_heads()
        """

        for epoch in range(checkpoint_epoch + 1, args.epochs + 1):
            """
            if epoch == freeze_head_epochs + 1:
                self.unfreeze_heads()
            """
            self.set_sinkhorn_temp(epoch)
            train_loss = self.train(epoch, args.lr)
            scheduler_metric = train_loss
            if self.test_loader is not None:
                scheduler_metric = self.test(epoch)
            if self.scheduler is not None:
                self.scheduler.step(scheduler_metric)
                self.log_lr_metrics(epoch, scheduler_metric=scheduler_metric, source="scheduler")
            else:
                self.log_lr_metrics(epoch, source="optimizer")
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch)

        if args.param is not None:
            torch.save(self.model.state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)
        
        if self.writer is not None:
            self.writer.close()

        return self.model
    
    def init_model_weights(self, args):
        # CHANGE: weight initialisation
        model = self._resolve_zuker_model(self.model)
        if model is None:
            return

        encoder = getattr(model.net, "encoder", None)
        if encoder is not None:
            # Keep encoder conv and norm layers off PyTorch defaults.
            if getattr(encoder, "conv", None) is not None:
                encoder.conv.apply(init_weights)
            if getattr(encoder, "pre_lstm_ln", None) is not None:
                encoder.pre_lstm_ln.apply(init_weights)
            if getattr(encoder, "lstm_ln", None) is not None:
                encoder.lstm_ln.apply(init_weights)

        # orthogonal initialisation of the lstm to avoid exploding/vanishing gradients
        lstm = getattr(encoder, "lstm", None) if encoder is not None else None
        if lstm is not None:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    # Gate-wise Xavier for input projections: i, f, g, o
                    for block in param.chunk(4, 0):
                        nn.init.xavier_uniform_(block)

                elif "weight_hh" in name:
                    # Gate-wise orthogonal for recurrent projections
                    for block in param.chunk(4, 0):
                        nn.init.orthogonal_(block)

                elif "bias_ih" in name:
                    nn.init.zeros_(param)
                    hidden = param.shape[0] // 4
                    # PyTorch gate order is i, f, g, o
                    param.data[hidden:2 * hidden].fill_(1.0)

                elif "bias_hh" in name:
                    nn.init.zeros_(param)

        model.net.fc_paired.apply(init_heads)
        # model.net.premsa.apply(init_heads)

        if model.net.fc_unpaired is not None:
            model.net.fc_unpaired.apply(init_heads)

        if hasattr(model, 'fc_length'):
            for key in model.fc_length:
                model.fc_length[key].apply(init_heads)


    #TODO: Replace command line arguments with yaml file
    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of the list of BPSEQ-formatted files')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='',
                            help='the file name of the initial parameters')
        subparser.add_argument('--bpp-dir', type=str, default='',
                            help='directory containing cached .bpp targets keyed by BPSEQ filename stem')
        subparser.add_argument('--test-bpp-dir', type=str, default='',
                            help='directory containing cached .bpp targets for the test split; defaults to --bpp-dir')
        subparser.add_argument('--init-sinkhorn-param', type=str, default='',
                            help='checkpoint file whose Sinkhorn-specific weights will be loaded on top of the base init')
        subparser.add_argument('--init-param-partial', default=False, action='store_true',
                            help='load only init checkpoint weights whose names and shapes match the current model')
        subparser.add_argument('--freeze-embedding', default=False, action='store_true',
                            help='freeze only the embedding layer')
        subparser.add_argument('--freeze-cnn', default=False, action='store_true',
                            help='freeze only the encoder CNN stack')
        subparser.add_argument('--freeze-lstm', default=False, action='store_true',
                            help='freeze only the encoder LSTM stack and its surrounding norms')
        subparser.add_argument('--freeze-heads', default=False, action='store_true',
                            help='freeze the paired/unpaired/length prediction heads')
        subparser.add_argument('--freeze-sinkhorn', default=False, action='store_true',
                            help='freeze the sinkhorn module and sinkhorn-specific projection/gating parameters')

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')
        subparser.add_argument('--verbose', action='store_true',
                            help='enable verbose outputs for debugging')
        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=1.,
                            help='the weight for score loss for hinge_mix loss (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.001,
                            help='the learning rate for optimizer (default: 0.001)')
        gparser.add_argument('--encoder-lr-mult', type=float, default=1.0,
                            help='learning-rate multiplier for encoder parameters (default: 1.0)')
        gparser.add_argument('--head-lr-mult', type=float, default=0.01,
                            help='learning-rate multiplier for scoring head parameters (default: 0.01)')
        gparser.add_argument('--sinkhorn-lr-mult', type=float, default=1.0,
                            help='learning-rate multiplier for sinkhorn parameters (default: 1.0)')
        gparser.add_argument('--lr-scheduler', choices=('none', 'plateau'), default='plateau',
                            help='learning-rate scheduler type (default: plateau)')
        gparser.add_argument('--lr-scheduler-factor', type=float, default=0.5,
                            help='multiplicative factor for plateau scheduler (default: 0.5)')
        gparser.add_argument('--lr-scheduler-patience', type=int, default=10,
                            help='epochs with no improvement before lowering LR (default: 10)')
        gparser.add_argument('--lr-scheduler-threshold', type=float, default=1e-3,
                            help='minimum relative improvement for plateau scheduler (default: 1e-3)')
        gparser.add_argument('--lr-scheduler-min-lr', type=float, default=1e-6,
                            help='lower bound for scheduled learning rates (default: 1e-6)')
        gparser.add_argument('--max-head-grad-norm', type=float, default=10,
                            help='max gradient norm for head clipping (default: 1.0)')
        gparser.add_argument('--max-encoder-grad-norm', type=float, default=1.0,
                            help='max gradient norm for encoder clipping (default: 1.0)')
        gparser.add_argument('--max-sinkhorn-grad-norm', type=float, default=1.0,
                            help='max gradient norm for sinkhorn clipping (default: 1.0)')
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix', 'sinkhorn_contact'), default='hinge',
                            help="loss fuction ('hinge', 'hinge_mix', 'sinkhorn_contact') ")
        gparser.add_argument('--loss-pos-paired', type=float, default=0.5,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
        gparser.add_argument('--loss-neg-paired', type=float, default=0.005,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--margin-weight', type=float, default=1.0,
                            help='weight applied to the structured margin term (default: 1.0)')
        gparser.add_argument('--sinkhorn-temp-anneal-start-epoch', type=int, default=1,
                        help='epoch to start linearly annealing Sinkhorn temperature (default: 1)')
        gparser.add_argument('--sinkhorn-temp-anneal-end-epoch', type=int, default=0,
                        help='epoch to finish annealing Sinkhorn temperature; 0 means last epoch')
        gparser.add_argument('--sinkhorn-temp-start', type=float, default=None,
                        help='starting temperature for Sinkhorn annealing; defaults to --sinkhorn-temp')
        gparser.add_argument('--sinkhorn-temp-end', type=float, default=None,
                        help='final temperature for Sinkhorn annealing; defaults to start temperature')
        gparser.add_argument('--sinkhorn-aux-weight', type=float, default=0.0,
                        help='weight for row-wise Sinkhorn auxiliary assignment loss (default: 0.0)')
        gparser.add_argument('--sinkhorn-aux-use-bpp', default=False, action='store_true',
                        help='use cached BPP targets with row-normalized KL instead of hard BPSEQ row targets')
        gparser.add_argument('--sinkhorn-aux-pos-weight', type=float, default=1.0,
                        help='row weight for paired nucleotides in the Sinkhorn auxiliary loss (default: 1.0)')
        gparser.add_argument('--sinkhorn-aux-neg-weight', type=float, default=0.0,
                        help='row weight for unpaired/diagonal targets in the Sinkhorn auxiliary loss (default: 0.0)')
        gparser.add_argument('--sinkhorn-aux-diag-weight', type=float, default=None,
                        help='row weight for diagonal null targets in standalone Sinkhorn contact training; defaults to --sinkhorn-aux-neg-weight')
        gparser.add_argument('--sinkhorn-aux-true-pair-weight', type=float, default=None,
                        help='reward weight for putting paired rows on their true partner; defaults to --sinkhorn-aux-pos-weight')
        gparser.add_argument('--sinkhorn-aux-true-unpaired-weight', type=float, default=None,
                        help='reward weight for putting unpaired rows on the null target; defaults to --sinkhorn-aux-diag-weight')
        gparser.add_argument('--sinkhorn-aux-false-pair-weight', type=float, default=0.0,
                        help='penalty weight for putting unpaired rows on off-diagonal pair mass in standalone Sinkhorn contact training')
        gparser.add_argument('--sinkhorn-aux-false-unpaired-weight', type=float, default=0.0,
                        help='penalty weight for putting paired rows on the null target in standalone Sinkhorn contact training')
        gparser.add_argument('--sinkhorn-aux-slack-self-weight', type=float, default=None,
                        help='weight for explicitly supervising the slack row to map to the slack column; defaults to --sinkhorn-aux-neg-weight')
        gparser.add_argument('--sinkhorn-aux-min-separation', type=int, default=4,
                        help='minimum off-diagonal separation for paired targets in the Sinkhorn auxiliary loss (default: 4)')
        gparser.add_argument('--sinkhorn-aux-stack-bonus', type=float, default=0.0,
                        help='extra per-pair weight added for stacked pairs when building the Sinkhorn auxiliary target (default: 0.0)')
        gparser.add_argument('--sinkhorn-real-only-normalization', default=False, action='store_true',
                        help='normalize only the real rows/columns when slack is enabled, leaving slack as residual capacity')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'SinkhornContact'), default='Turner', 
                            help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'SinkhornContact')")
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

        subparser.set_defaults(func = lambda args, conf: Train().run(args, conf))
