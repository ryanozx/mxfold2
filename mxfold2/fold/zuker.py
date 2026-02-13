import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, EvoformerNet, NeuralNet


class ZukerFold(AbstractFold):
    def __init__(self, model_type="M", max_helix_length=30, **kwargs):
        super(ZukerFold, self).__init__(interface.predict_zuker, interface.partfunc_zuker)

        exclude_diag = True
        if model_type == "S":
            n_out_paired_layers = 1
            n_out_unpaired_layers = 1
        elif model_type == "M":
            n_out_paired_layers = 2
            n_out_unpaired_layers = 1
        elif model_type == "L":
            n_out_paired_layers = 5
            n_out_unpaired_layers = 4
        elif model_type == "C":
            n_out_paired_layers = 3
            n_out_unpaired_layers = 0
            exclude_diag = False
        else:
            raise("not implemented")

        self.model_type = model_type
        self.max_helix_length = max_helix_length
        """
        self.net = NeuralNet(**kwargs, 
            n_out_paired_layers=n_out_paired_layers,
            n_out_unpaired_layers=n_out_unpaired_layers,
            exclude_diag=exclude_diag)
        """
        
        self.net = EvoformerNet(
            seq_input_dim = 6, 
            msa_input_dim = 6 + 1,
            seq_embed_dim = 64,
            pair_embed_dim = 64,
            n_out_paired_layers=n_out_paired_layers,
            n_out_unpaired_layers=n_out_unpaired_layers,
            exclude_diag=exclude_diag)

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })


    def forward(self, seq, **kwargs):
        return super(ZukerFold, self).forward(seq, max_helix_length=self.max_helix_length, **kwargs)


    def make_param(self, seq, **kwargs):
        # seq has the shape (B, L, N); B = 1
        device = next(self.parameters()).device
        score_paired, score_unpaired = self.net(seq, **kwargs)
        # score_paired has shape (B, L, L, n_out_paired_layers), while score_unpaired has shape (B, L, n_out_unpaired_layers)
        # print(f"score shapes: {score_paired.shape} {score_unpaired.shape if score_unpaired is not None else None}")
        B, N, _, _ = score_paired.shape

        def _stat_line(name, t):
            t_det = t.detach()
            finite = torch.isfinite(t_det)
            finite_count = int(finite.sum().item())
            nan_count = int(torch.isnan(t_det).sum().item())
            inf_count = int(torch.isinf(t_det).sum().item())
            if finite_count > 0:
                t_f = t_det[finite]
                t_min = float(t_f.min().item())
                t_max = float(t_f.max().item())
                t_mean = float(t_f.mean().item())
            else:
                t_min = float("nan")
                t_max = float("nan")
                t_mean = float("nan")
            print(
                f"[stat] {name}: shape={tuple(t_det.shape)} "
                f"dtype={t_det.dtype} min={t_min:.6g} max={t_max:.6g} "
                f"mean={t_mean:.6g} nan={nan_count} inf={inf_count}"
            )

        if score_paired.dim() == 4:
            for c in range(score_paired.shape[-1]):
                _stat_line(f"score_paired[{c}]", score_paired[:, :, :, c])
        else:
            _stat_line("score_paired", score_paired)

        if score_unpaired is not None:
            if score_unpaired.dim() == 3:
                for c in range(score_unpaired.shape[-1]):
                    _stat_line(f"score_unpaired[{c}]", score_unpaired[:, :, c])
            else:
                _stat_line("score_unpaired", score_unpaired)

        # Normalize score_unpaired *per model_type*.
        # Contract:
        #   - For S/M/C: we need a raw per-position tensor of shape (B, N)
        #   - For L:     we need raw per-position channels of shape (B, N, 4)
        if self.model_type in ("S", "M", "C"):
            # No learned unpaired channels in C; S/M typically have exactly 1.
            if score_unpaired is None:
                score_unpaired = torch.zeros((B, N), device=device, dtype=torch.float32)
            else:
                # If model returned (B, N, 1), squeeze to (B, N)
                if score_unpaired.dim() == 3:
                    score_unpaired = score_unpaired[:, :, 0]
                score_unpaired = score_unpaired.to(device=device, dtype=torch.float32)

        elif self.model_type == "L":
            # L uses 4 distinct unpaired channels.
            if score_unpaired is None:
                score_unpaired = torch.zeros((B, N, 4), device=device, dtype=torch.float32)
            else:
                score_unpaired = score_unpaired.to(device=device, dtype=torch.float32)

        else:
            raise("not implemented")

        def unpair_interval(su):
            # su is expected to be (B, N) on the same device/dtype we want the result.
            su = su.view(B, 1, N)
            ones = torch.ones(B, N, 1, device=su.device, dtype=su.dtype)
            su = torch.bmm(ones, su)  # (B, N, N)
            # Keep only i<=j region, consistent with interval-style bookkeeping.
            tri = torch.triu(torch.ones_like(su))
            su = torch.bmm(torch.triu(su), tri)
            return su

        if self.model_type == "S":
            score_basepair = score_paired[:, :, :, 0] # (B, N, N)
            score_unpaired = unpair_interval(score_unpaired.contiguous())
            score_helix_stacking = torch.zeros((B, N, N), device=device)
            score_mismatch_external = score_helix_stacking
            score_mismatch_internal = score_helix_stacking
            score_mismatch_multi = score_helix_stacking
            score_mismatch_hairpin = score_helix_stacking
            score_base_hairpin = score_unpaired
            score_base_internal = score_unpaired
            score_base_multi = score_unpaired
            score_base_external = score_unpaired

        elif self.model_type == "M":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
            score_unpaired = unpair_interval(score_unpaired)
            score_base_hairpin = score_unpaired
            score_base_internal = score_unpaired
            score_base_multi = score_unpaired
            score_base_external = score_unpaired

        elif self.model_type == "L":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 2] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 3] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 4] # (B, N, N)
            score_base_hairpin = unpair_interval(score_unpaired[:, :, 0].contiguous())
            score_base_internal = unpair_interval(score_unpaired[:, :, 1].contiguous())
            score_base_multi = unpair_interval(score_unpaired[:, :, 2].contiguous())
            score_base_external = unpair_interval(score_unpaired[:, :, 3].contiguous())

        elif self.model_type == 'C':
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
            score_unpaired = unpair_interval(score_unpaired.contiguous()) # (B, N, N)
            score_base_hairpin = score_unpaired
            score_base_internal = score_unpaired
            score_base_multi = score_unpaired
            score_base_external = score_unpaired

        else:
            raise("not implemented")

        param = [ { 
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param(),
            'score_helix_length': self.fc_length['score_helix_length'].make_param()
        } for i in range(B) ]

        return param
