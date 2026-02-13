import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Linear

# Taken from DRFold
class PreMSA(nn.Module):
    """
    PreMSA is the *input embedding module* in the model.
    
    It takes a raw RNA sequence and one hot / MSA encoding and produces:
      1) sequence_embeddings: a per-position vector for each nucleotide
      2) pairwise_embeddings: a matrix of vectors encoding relationships between all pairs of positions
    
    These two outputs (m, z) are the input features for the transformer/Evoformer part of the network.
    """
    # How far the model explicitly distinguishes pairwise distances
    MAX_RELATIVE_DISTANCE = 32

    # If i and j differ by more than 32, we clamp them to 32
    NUM_RELATIVE_BINS = MAX_RELATIVE_DISTANCE * 2 + 1

    MAX_SEQUENCE_LENGTH = 2000

    # How many bits to use when encoding absolute position
    ABSOLUTE_POS_BITS = 14

    def __init__(self, seq_input_dim: int, msa_input_dim: int, seq_embed_dim: int, pair_embed_dim: int):
        """
        Args:
        - seq_input_dim: dimension of the raw sequence input features
                         (e.g., one-hot size for nucleotides)
        - msa_input_dim: dimension of the MSA features (if using MSA)
        - seq_embed_dim: desired resulting sequence embedding dimension
        - pair_embed_dim: desired resulting pairwise embedding dimension
        """
        super(PreMSA,self).__init__()

        # These are fixed positional encodings that don’t change during training.
        # They get projected into learned spaces later.
        self.relative_distance_onehots = self._build_relative_distance_table().float()
        self.absolute_position_bits   = self._build_absolute_position_table()

        # These learn how to turn raw inputs into useful representations.
        self.msa_projector   = Linear(msa_input_dim, seq_embed_dim)
        self.seq_query_proj  = Linear(seq_input_dim, pair_embed_dim)
        self.seq_key_proj    = Linear(seq_input_dim, pair_embed_dim)
        self.seq_feat_proj   = Linear(seq_input_dim, seq_embed_dim)

        # Positional encoders get mapped into the appropriate embedding space
        self.relative_pos_proj = Linear(PreMSA.NUM_RELATIVE_BINS, pair_embed_dim)
        self.absolute_pos_proj = Linear(PreMSA.ABSOLUTE_POS_BITS, seq_embed_dim)

    def _build_absolute_position_table(self, max_len=MAX_SEQUENCE_LENGTH):
        """
        Build fixed absolute positional features for indices 0..max_len-1.
        
        We write each index in binary (e.g., 5 → 0101 in bits) so that
        the model can learn how absolute position matters.
        """
        positions = torch.arange(max_len)
        # Expand to bits so that bit i corresponds to the i-th power of 2
        bits = (((positions[:, None] &
                  (1 << np.arange(PreMSA.ABSOLUTE_POS_BITS))) > 0).float())
        return bits  # shape (max_len, ABSOLUTE_POS_BITS)

    def _build_relative_distance_table(self, max_len=MAX_SEQUENCE_LENGTH):
        """
        Build fixed relative positional features for all pairs of positions.

        This table contains one-hot encoded distances between positions i and j,
        clamped so anything bigger than +/- MAX_RELATIVE_DISTANCE is binned together.
        """
        pos_indices = torch.arange(max_len)
        raw_diffs = (pos_indices[None, :] - pos_indices[:, None])
        clipped = raw_diffs.clamp(-PreMSA.MAX_RELATIVE_DISTANCE,
                                  PreMSA.MAX_RELATIVE_DISTANCE)
        bins  = clipped + PreMSA.MAX_RELATIVE_DISTANCE
        return F.one_hot(bins, PreMSA.NUM_RELATIVE_BINS)     
    
    def forward(self, seq, msa):
        """
        Args:
        - seq: Tensor of shape (L, seq_input_dim)
                     e.g., one-hot of nucleotide identity
        - msa: Tensor of shape (N, L, msa_input_dim)
                     N = number of rows in the MSA (often 1 or 2 here)

        Returns:
        - sequence_embeddings: Tensor (N, L, seq_embed_dim)
        - pairwise_embeddings : Tensor (L, L, pair_embed_dim)
        """

        # Ensure the positional tables live on the same device as the input
        device = msa.device
        if self.relative_distance_onehots.device != device:
            self.relative_distance_onehots = self.relative_distance_onehots.to(device)
        if self.absolute_position_bits.device != device:
            self.absolute_position_bits = self.absolute_position_bits.to(device)

        _, length, _ = msa.shape

        # ---- BUILD SEQUENCE EMBEDDING ----

        # (L, seq_embed_dim): project raw sequence features
        seq_encoded = self.seq_feat_proj(seq)
        # (N, L, seq_embed_dim): project MSA features
        msa_encoded = self.msa_projector(msa)
        # (L, seq_embed_dim): get absolute position bit encodings for actual length
        abs_pos_for_len = self.absolute_pos_proj(self.absolute_position_bits[:length])
        # Add them together so that positional and raw sequence info are combined
        seq_embed = msa_encoded + seq_encoded[None, :, :] + abs_pos_for_len[None, :, :]

        # ---- BUILD PAIRWISE EMBEDDING ----

        # Query and key projections of sequence features
        query = self.seq_query_proj(seq)  # (L, pair_embed_dim)
        key   = self.seq_key_proj  (seq)  # (L, pair_embed_dim)
        # Outer sum to combine every pair i,j
        pair_embed = query[None, :, :] + key[:, None, :]
        # Add relative positional embedding for each (i,j)
        pair_embed += self.relative_pos_proj(
            self.relative_distance_onehots[:length, :length]
        )

        return seq_embed, pair_embed
        