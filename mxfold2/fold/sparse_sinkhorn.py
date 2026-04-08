import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseSinkhornLayer(nn.Module):
    def __init__(
        self,
        block_size: int,
        num_iterations: int,
        num_hidden_states: int,
        temp: float = 0.1,
        alpha: float = 0.5,
        temp_length_center: float = 8.0,
        temp_length_slope: float = 3.0,
        pooling: str = "attention",
        diag_logit_penalty: float = 0.0,
        use_slack: bool = False,
        real_only_normalization: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_iterations = num_iterations
        self.temp = temp
        self.W = nn.Parameter(torch.Tensor(num_hidden_states, num_hidden_states))
        self.alpha = alpha
        self.temp_length_center = temp_length_center
        self.temp_length_slope = temp_length_slope
        self.pooling = str(pooling).lower()
        self.diag_logit_penalty = float(diag_logit_penalty)
        self.use_slack = bool(use_slack)
        self.real_only_normalization = bool(real_only_normalization)
        self._last_effective_temp = float(temp)
        self._last_num_blocks = None
        self._last_block_mask = None
        self._last_block_mask_with_slack = None
        self._last_block_mask_with_slack_live = None
        self._last_block_pooled = None
        self._last_block_projected = None
        self._last_block_pool_attn_weights = None
        self._last_block_pool_valid_mask = None
        self._last_block_raw_bilinear_logits = None
        self._last_block_pre_sinkhorn_logits = None
        self._last_block_pre_loop_logits = None
        self._last_block_logits = None
        self.block_projected_norm = nn.LayerNorm(num_hidden_states)
        if self.pooling not in {"attention", "mean"}:
            raise ValueError(f"Unknown Sinkhorn pooling mode: {pooling}")
        if self.use_slack:
            self.slack_embed = nn.Parameter(torch.zeros(num_hidden_states))
        if self.pooling == "attention":
            self.pool_attn = nn.Sequential(
                nn.LayerNorm(num_hidden_states),
                nn.Linear(num_hidden_states, num_hidden_states // 2),
                nn.Tanh(),
                nn.Linear(num_hidden_states // 2, 1, bias=False)
            )
        else:
            self.pool_attn = None
        nn.init.xavier_uniform_(self.W)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _length_adjusted_temp(self, num_blocks: int) -> float:
        slope = max(float(self.temp_length_slope), 1e-6)
        center = float(self.temp_length_center)

        # Normalise the sigmoid so a single block keeps the base temperature,
        # while longer sequences smoothly transition toward stronger cooling.
        u0 = self._sigmoid((1.0 - center) / slope)
        u = self._sigmoid((float(num_blocks) - center) / slope)
        denom = max(1.0 - u0, 1e-8)
        length_factor = min(max((u - u0) / denom, 0.0), 1.0)
        scale = 1.0 + float(self.alpha) * length_factor
        return float(self.temp) / scale
        
    def forward(self, x: torch.Tensor):
        # x: Tensor of dimensions (B, N, no. of hidden states = D)

        # Step 1: Blockwise-pooling operation (into blocks of block_size); pooling operation takes the mean of embeddings within each block
        # Step 2: Produce a 2D interaction matrix of blocks using bilinear product
        # Step 3: Iteratively perform sinkhorn normalisation
        # Step 4: Blow up the block-based mask back into the original dimensions (either using kronecker product or "nearest neighbour" expansion)

        # Output: Mask of dimension (B, N, N)
        B, L, D = x.shape

        num_blocks = math.ceil(L / self.block_size)
        self._last_num_blocks = num_blocks
        padded_size = num_blocks * self.block_size
        pad = padded_size - L

        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad), 'constant', 0)

        x_blocks = x.view(B, num_blocks, self.block_size, D)
        if self.block_size == 1:
            self._last_block_pool_attn_weights = None
            self._last_block_pool_valid_mask = None
            x_pooled = x_blocks.squeeze(2)
        elif self.pooling == "mean":
            valid_pos = torch.ones((B, L), device=x.device, dtype=x.dtype)

            if pad > 0:
                valid_pos = F.pad(valid_pos, (0, pad))
            valid_pos = valid_pos.view(B, num_blocks, self.block_size, 1)
            denom = valid_pos.sum(dim=2).clamp_min(1.0)
            self._last_block_pool_attn_weights = None
            self._last_block_pool_valid_mask = valid_pos.squeeze(-1).detach()
            x_pooled = (x_blocks * valid_pos).sum(dim=2) / denom
        else:
            valid_pos = torch.ones((B, L), device=x.device, dtype=x.dtype)

            if pad > 0:
                valid_pos = F.pad(valid_pos, (0, pad))
            valid_pos = valid_pos.view(B, num_blocks, self.block_size, 1)

            attn_scores = self.pool_attn(x_blocks).squeeze(-1)
            valid_mask = valid_pos.squeeze(-1) > 0
            attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=2)
            attn_weights = attn_weights.masked_fill(~valid_mask, 0.0)
            self._last_block_pool_attn_weights = attn_weights.detach()
            self._last_block_pool_valid_mask = valid_mask.detach()

            x_pooled = torch.sum(x_blocks * attn_weights.unsqueeze(-1), dim=2)

        self._last_block_pooled = x_pooled.detach()
        x_pooled_for_sinkhorn = x_pooled
        if self.use_slack:
            slack_embed = self.slack_embed.view(1, 1, D).expand(B, 1, D)
            x_pooled_for_sinkhorn = torch.cat((x_pooled_for_sinkhorn, slack_embed), dim=1)

        x_transformed = torch.matmul(x_pooled_for_sinkhorn, self.W)
        self._last_block_projected = x_transformed[:, :num_blocks].detach()
        x_transformed = self.block_projected_norm(x_transformed)
        logits = torch.bmm(x_transformed, x_pooled_for_sinkhorn.transpose(1, 2)) / math.sqrt(D)
        temp_eff = self._length_adjusted_temp(num_blocks)
        self._last_effective_temp = float(temp_eff)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        self._last_block_raw_bilinear_logits = logits[:, :num_blocks, :num_blocks].detach()

        # sparse sinkhorn converges quickly if the logits are not too extreme
        logit_mu = logits.mean(dim=(1, 2), keepdim=True)
        logit_sigma = logits.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-4)
        c = 10.0
        logits = (logits - logit_mu) / torch.sqrt(logit_sigma)
        logits = c * torch.tanh(logits / c)
        self._last_block_pre_sinkhorn_logits = logits[:, :num_blocks, :num_blocks].detach()
        logits = logits / temp_eff
        if self.diag_logit_penalty != 0.0:
            full_blocks = logits.shape[1]
            diag = torch.eye(full_blocks, device=logits.device, dtype=logits.dtype).unsqueeze(0)
            if self.use_slack:
                diag[:, -1, -1] = 0.0
            logits = logits - self.diag_logit_penalty * diag
        self._last_block_pre_loop_logits = logits[:, :num_blocks, :num_blocks].detach()


        if self.real_only_normalization and self.use_slack:
            real_slice = slice(0, num_blocks)
            for _ in range(self.num_iterations):
                # Normalize only real rows and real columns; slack row/column act as residual capacity.
                row_log_norm = torch.logsumexp(logits[:, real_slice, :] + 1e-12, dim=2, keepdim=True)
                logits[:, real_slice, :] = logits[:, real_slice, :] - row_log_norm
                col_log_norm = torch.logsumexp(logits[:, :, real_slice] + 1e-12, dim=1, keepdim=True)
                logits[:, :, real_slice] = logits[:, :, real_slice] - col_log_norm
        else:
            for _ in range(self.num_iterations):
                # row normalisation then column normalisation
                logits = logits - torch.logsumexp(logits + 1e-12, dim=2, keepdim=True)
                logits = logits - torch.logsumexp(logits + 1e-12, dim=1, keepdim=True)

        full_mask = torch.exp(logits.clamp(max=10))
        self._last_block_mask_with_slack = full_mask.detach()
        self._last_block_mask_with_slack_live = full_mask
        mask = full_mask[:, :num_blocks, :num_blocks]
        self._last_block_logits = logits[:, :num_blocks, :num_blocks].detach()
        self._last_block_mask = mask.detach()
        # upsample/kronecker product here
        full_mask = mask.repeat_interleave(self.block_size, dim=1).repeat_interleave(self.block_size, dim=2)
        full_mask = full_mask[:, :L, :L]

        return full_mask
