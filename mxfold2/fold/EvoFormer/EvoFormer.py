import os
import torch
from torch import nn
from torch.nn import functional as F
from .EvoPair import *
from .EvoMSA import *
from torch.utils.checkpoint import checkpoint

def _mem_log(tag, msa_feats=None, pair_feats=None):
    if os.getenv("MXFOLD2_MEMLOG") != "1":
        return
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if msa_feats is not None and pair_feats is not None:
        m_shape = tuple(msa_feats.shape)
        p_shape = tuple(pair_feats.shape)
        print(f"[mem] {tag} alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB msa={m_shape} pair={p_shape}")
    else:
        print(f"[mem] {tag} alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")


class EvoformerBlock(nn.Module):
    """
    A single Evoformer block refines:
      1) an MSA representation (like sequence features over rows)
      2) a pairwise representation (like distance/contact features between positions)
    """
    def __init__(self, seq_feat_dim: int, pair_feat_dim: int, enable_checkpoint=False):
        """
        Args:
          seq_feat_dim  : hidden dimension for sequence/MSA features
          pair_feat_dim : hidden dimension for pairwise features
          enable_checkpoint : if True, use PyTorch checkpointing to save memory
        """
        super(EvoformerBlock,self).__init__()
        self.enable_checkpoint = enable_checkpoint

        # ---- MSA (sequence alignment) track modules ----
        # Row-wise attention (attend across positions within each MSA row)
        self.msa_row_attn  = MSARow(seq_feat_dim, pair_feat_dim)
        # Column-wise attention (attend across rows at the same position)
        self.msa_col_attn  = MSACol(seq_feat_dim)
        # Transition (simple feed-forward refinement)
        self.msa_transition = MSATrans(seq_feat_dim)
        # Outer Product Mean: convert MSA features into a pairwise update
        self.msa_to_pair    = MSAOPM(seq_feat_dim, pair_feat_dim)

        # ---- Pair (2D) track modules ----
        # Triangle/outgoing update
        self.pair_tri_out  = TriOut(pair_feat_dim)
        # Triangle/incoming update
        self.pair_tri_in   = TriIn(pair_feat_dim)
        # Triangle self-attention starting node
        self.pair_tri_start = TriAttStart(pair_feat_dim)
        # Triangle self-attention ending node
        self.pair_tri_end   = TriAttEnd(pair_feat_dim)
        # Pair feed-forward transition
        self.pair_transition = PairTrans(pair_feat_dim)

    def layerfunc_msa_row(self,m,z):
        return self.msa_row(m,z) + m
    def layerfunc_msa_col(self,m):
        return self.msa_col(m) + m
    def layerfunc_msa_trans(self,m):
        return self.msa_trans(m) + m
    def layerfunc_msa_opm(self,m,z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self,z):
        return self.pair_triout(z) + z
    def layerfunc_pair_triin(self,z):
        return self.pair_triin(z) + z
    def layerfunc_pair_tristart(self,z):
        return self.pair_tristart(z) + z
    def layerfunc_pair_triend(self,z):
        return self.pair_triend(z) + z      
    def layerfunc_pair_trans(self,z):
        return self.pair_trans(z) + z  
    
    def forward(self, msa_feats, pair_feats):
        """
        Updates the msa_feats and pair_feats in-place with residual connections.

        Args:
          msa_feats  : Tensor (N, L, seq_feat_dim)
          pair_feats : Tensor (L, L, pair_feat_dim)

        Returns:
          (updated_msa_feats, updated_pair_feats)
        """

        _mem_log("EvoformerBlock:start", msa_feats, pair_feats)
        # ===== MSA (sequence) updates =====
        # Row-wise attention with pairwise bias
        msa_feats = msa_feats + self.msa_row_attn(msa_feats, pair_feats)
        _mem_log("EvoformerBlock:after_msa_row", msa_feats, pair_feats)
        # Column-wise attention
        msa_feats = msa_feats + self.msa_col_attn(msa_feats)
        _mem_log("EvoformerBlock:after_msa_col", msa_feats, pair_feats)
        # Feed-forward transition
        msa_feats = msa_feats + self.msa_transition(msa_feats)
        _mem_log("EvoformerBlock:after_msa_trans", msa_feats, pair_feats)

        # ===== Inject MSA info into the pair representation =====
        pair_feats = pair_feats + self.msa_to_pair(msa_feats)
        _mem_log("EvoformerBlock:after_msa_opm", msa_feats, pair_feats)

        # ===== Pair (2D) updates =====
        # Triangular outgoing update
        pair_feats = pair_feats + self.pair_tri_out(pair_feats)
        _mem_log("EvoformerBlock:after_tri_out", msa_feats, pair_feats)
        # Triangular incoming update
        pair_feats = pair_feats + self.pair_tri_in(pair_feats)
        _mem_log("EvoformerBlock:after_tri_in", msa_feats, pair_feats)
        # Triangle self-attention (start node viewpoint)
        pair_feats = pair_feats + self.pair_tri_start(pair_feats)
        _mem_log("EvoformerBlock:after_tri_start", msa_feats, pair_feats)
        # Triangle self-attention (end node viewpoint)
        pair_feats = pair_feats + self.pair_tri_end(pair_feats)
        _mem_log("EvoformerBlock:after_tri_end", msa_feats, pair_feats)
        # Pairwise transition feed-forward
        pair_feats = pair_feats + self.pair_transition(pair_feats)
        _mem_log("EvoformerBlock:after_pair_trans", msa_feats, pair_feats)

        return msa_feats, pair_feats

class EvoformerStack(nn.Module):
    """
    A stack of multiple Evoformer blocks.
    Each block refines both the MSA and pairwise representations.

    This implementation uses checkpointing in chunks
    so that deeper stacks can run with less GPU memory.
    """

    NUM_LAYERS = 6

    def __init__(self, seq_feat_dim: int, pair_feat_dim: int, use_checkpoints=True):
        """
        Args:
          seq_feat_dim  : dimension for sequence/MSA features
          pair_feat_dim : dimension for pairwise features
          use_checkpoints : whether to chunk and checkpoint to reduce memory
        """
        super(EvoformerStack, self).__init__()

        self.use_checkpoints = use_checkpoints

        # Build the real stack of EvoformerBlocks
        self.blocks = nn.ModuleList([
            EvoformerBlock(seq_feat_dim, pair_feat_dim, enable_checkpoint=use_checkpoints)
            for _ in range(EvoformerStack.NUM_LAYERS)
        ])

    def _run_block_range(self, msa_emb, pair_emb, start, end):
        for idx in range(start, end):
            msa_emb, pair_emb = self.blocks[idx](msa_emb, pair_emb)
        return msa_emb, pair_emb
    
    def forward(self, msa_feats, pair_feats):
        """
        Runs the Evoformer stack in several sub-chunks.
        This reduces peak memory via checkpointing.
        """
        if not self.use_checkpoints:
            # Simple loop without checkpointing
            for block in self.blocks:
                msa_feats, pair_feats = block(msa_feats, pair_feats)
            return msa_feats, pair_feats
        
        chunks = [(0, 3), (3, 6)]

        for (chunk_inclusive_start, chunk_exclusive_end) in chunks:
            msa_feats, pair_feats = checkpoint(
                self._run_block_range, msa_feats, pair_feats, chunk_inclusive_start, chunk_exclusive_end, use_reentrant=False
                )

        return msa_feats, pair_feats
