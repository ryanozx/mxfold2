import torch
import torch.nn as nn

from .layers import NeuralNet


class PairwiseSinkhornLayer(nn.Module):
    def __init__(
        self,
        num_iterations: int,
        temp: float = 0.1,
        diag_logit_penalty: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.temp = float(temp)
        self.diag_logit_penalty = float(diag_logit_penalty)
        self.block_size = 1
        self.use_slack = False
        self.real_only_normalization = False
        self._last_effective_temp = float(temp)
        self._last_block_logits = None
        self._last_pre_loop_block_logits = None
        self._last_block_mask = None
        self._last_block_mask_with_slack = None
        self._last_block_mask_with_slack_live = None
        self._last_num_blocks = None
        self._last_block_pooled = None
        self._last_block_projected = None
        self._last_block_raw_bilinear_logits = None
        self._last_block_pre_sinkhorn_logits = None

    def forward(self, pair_logits: torch.Tensor) -> torch.Tensor:
        if pair_logits.dim() == 4:
            if pair_logits.shape[-1] != 1:
                raise ValueError(
                    f"Expected pair logits with last dim 1, got shape {tuple(pair_logits.shape)}"
                )
            pair_logits = pair_logits[..., 0]
        if pair_logits.dim() != 3:
            raise ValueError(f"Expected pair logits with shape (B, N, N), got {tuple(pair_logits.shape)}")

        logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        self._last_block_logits = logits.detach()

        if self.diag_logit_penalty != 0.0:
            n = logits.shape[-1]
            eye = torch.eye(n, device=logits.device, dtype=logits.dtype).unsqueeze(0)
            logits = logits + eye * self.diag_logit_penalty

        temp = max(float(self.temp), 1e-8)
        log_alpha = logits / temp
        self._last_effective_temp = temp
        self._last_pre_loop_block_logits = log_alpha.detach()
        self._last_block_pre_sinkhorn_logits = log_alpha.detach()

        for _ in range(max(self.num_iterations, 1)):
            log_alpha = log_alpha - torch.logsumexp(log_alpha + 1e-12, dim=2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha + 1e-12, dim=1, keepdim=True)

        mask = log_alpha.exp()
        self._last_block_mask = mask.detach()
        self._last_block_mask_with_slack = None
        self._last_block_mask_with_slack_live = None
        self._last_num_blocks = int(mask.shape[-1])
        return mask


class SinkhornContactFold(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        net_kwargs = dict(kwargs)
        net_kwargs.pop("max_helix_length")
        num_sinkhorn_iterations = net_kwargs.pop("num_sinkhorn_iterations", 10)
        sinkhorn_temp = net_kwargs.pop("sinkhorn_temp", 0.1)
        sinkhorn_diag_logit_penalty = net_kwargs.pop("sinkhorn_diag_logit_penalty", 0.0)
        net_kwargs["use_sparse_sinkhorn"] = False
        net_kwargs["n_out_paired_layers"] = 3
        net_kwargs["n_out_unpaired_layers"] = 0
        net_kwargs["exclude_diag"] = False

        self.net = NeuralNet(**net_kwargs)
        self.contact_logit_proj = nn.Linear(3, 1)
        self.sparse_sinkhorn = PairwiseSinkhornLayer(
            num_sinkhorn_iterations,
            temp=sinkhorn_temp,
            diag_logit_penalty=sinkhorn_diag_logit_penalty,
        )
        self._last_sinkhorn_mask = None
        self._score_range = (0.0, 0.0)
        self._score_mean = (0.0, None)

    def forward(self, seq, return_aux: bool = False, **kwargs):
        score_paired, _ = self.net(seq, **kwargs)
        pair_logits = self.contact_logit_proj(score_paired).squeeze(-1)
        mask = self.sparse_sinkhorn(pair_logits)
        self._last_sinkhorn_mask = mask.detach()

        finite = torch.isfinite(pair_logits)
        if finite.any():
            vals = pair_logits[finite]
            self._score_range = (float(vals.max().item() - vals.min().item()), 0.0)
            self._score_mean = (float(vals.mean().item()), None)
        else:
            self._score_range = (0.0, 0.0)
            self._score_mean = (0.0, None)

        if return_aux:
            return {
                "contact_logits": pair_logits,
                "sinkhorn_mask": mask,
                "sinkhorn_block_size": 1,
                "sinkhorn_block_mask_with_slack": None,
            }
        return {
            "contact_logits": pair_logits,
            "sinkhorn_mask": mask,
        }
