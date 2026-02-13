from collections import defaultdict
from enum import Enum

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .embedding import OneHotEmbedding, SparseEmbedding
from .transformer import TransformerLayer
from .EvoFormer.PreMSA import PreMSA
from .EvoFormer.EvoFormer import EvoformerStack

class CNNLayer(nn.Module):
    def __init__(self, n_in: int, 
                 num_filters: tuple[int] = (128,), filter_size: tuple[int] = (7,), pool_size: tuple[int] = (1,), 
                 dilation: int = 1, dropout_rate: float = 0.0, enable_resnet: bool = False):
        super(CNNLayer, self).__init__()
        assert len(num_filters) == len(filter_size) == len(pool_size)

        self.enable_resnet = enable_resnet
        self.n_out = num_filters[-1]
        self.net = nn.ModuleList()

        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append( 
                nn.Sequential( 
                    nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2**dilation, padding=2**dilation*(ksize//2)),
                    nn.MaxPool1d(p, stride=1, padding=p//2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out), # same as LayerNorm?
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate)
                )
            )
            n_in = n_out


    def forward(self, x: torch.Tensor):
        """
        input: sequence of shape (B, n_in, N) (assume B = 1, n_in = 4)

        output: sequence of shape (B, n_last_num_filters, N) 
        """
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.enable_resnet and x.shape[1]==x_a.shape[1] else x_a
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_in: int, 
            num_filters: tuple[int] = (256,), filter_size: tuple[int] = (7,), pool_size: tuple[int] = (1,), dilation: int = 0, # cnn parameters
            num_lstm_layers: int = 0, num_lstm_units: int = 0, # lstm parameters
            num_att: int = 0, # att parameters
            dropout_rate: float = 0.0, # dropout rate is shared among all three components
            enable_resnet: bool = True):

        super(CNNLSTMEncoder, self).__init__()
        self.enable_resnet = enable_resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1],)
        assert len(num_filters) == len(filter_size)

        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1],)
        assert len(num_filters) == len(pool_size)
        
        # if num_lstm_units is defined, a LSTM layer will be created if there are none
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        enable_cnn = len(num_filters) > 0 and num_filters[0] > 0
        enable_lstm = num_lstm_layers > 0
        enable_att = num_att > 0

        self.conv = self.lstm = self.att = None

        if enable_cnn:
            self.conv = CNNLayer(n_in = n_in, 
                                 num_filters = num_filters, filter_size = filter_size, pool_size = pool_size, 
                                 dilation = dilation, dropout_rate = dropout_rate, enable_resnet = self.enable_resnet)
            self.n_out = n_in = num_filters[-1]

        if enable_lstm:
            assert num_lstm_units > 0
            self.lstm = nn.LSTM(input_size = n_in, hidden_size = num_lstm_units, num_layers=num_lstm_layers, batch_first=True, bidirectional=True, 
                            dropout = dropout_rate if num_lstm_layers>1 else 0)
            self.n_out = n_in = num_lstm_units * 2
            self.lstm_ln = nn.LayerNorm(self.n_out)
            self.dropout = nn.Dropout(p=dropout_rate)

        if enable_att:
            assert self.n_out % num_att == 0
            self.att = nn.MultiheadAttention(embed_dim = self.n_out, num_heads = num_att, dropout=dropout_rate)


    def forward(self, x: torch.Tensor, **kwargs):
        """
        input: embeddings of shape (B, n_in, N) (assume B = 1)

        output: sequence representation of shape (B, N, n_out) (value of n_out depends on what the last layer is)
        """
        # CNN to help capture local context
        if self.conv is not None:
            x = self.conv(x) # (B, conv_C, N)

        x = torch.transpose(x, 1, 2) # (B, N, conv_C)

        if self.lstm is not None:
            x_a, _ = self.lstm(x) # (B, N, n_lstm_out = num_lstm_units * 2)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a)) # (B, N, n_lstm_out)
            x = x + x_a if self.enable_resnet and x.shape[2]==x_a.shape[2] else x_a # resnet occurs only if n_in == 2 * num_lstm_units

        if self.att is not None:
            x = torch.transpose(x, 0, 1) # (N, B, n_in)
            x_a, _ = self.att(x, x, x)
            x = x + x_a
            x = torch.transpose(x, 0, 1) # (B, n_in, N)

        return x

class JoinType(Enum):
    CAT = 1
    ADD = 2
    MUL = 3
    BILINEAR = 4

JOIN_MAP = {
    "cat": JoinType.CAT,
    "add": JoinType.ADD,
    "mul": JoinType.MUL,
    "bilinear": JoinType.BILINEAR,
}

class Transform2D(nn.Module):
    """
    Takes two matching sequence representation and combines them to form a pairwise 2D pair representation
    """

    def __init__(self, join: JoinType = JoinType.CAT):
        super(Transform2D, self).__init__()
        self.join = join


    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor):
        """
        Inputs:
        x_l, x_r: Sequence representations of shape (B, N, C)

        Outputs:
        Pairwise representation of shape (B, N, N, C)
        """
        assert(x_l.shape == x_r.shape)

        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        
        match self.join:
            case JoinType.CAT:
                x = torch.cat((x_l, x_r), dim = 3) # (B, N, N, C * 2)
            case JoinType.ADD:
                x = x_l + x_r # (B, N, N, C)
            case JoinType.MUL:
                x = x_l * x_r # (B, N, N, C)

        return x


class PairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, exclude_diag=True, enable_resnet=True):
        super(PairedLayer, self).__init__()

        self.enable_resnet = enable_resnet        
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        assert len(filters) == len(ksize)

        self.conv = nn.ModuleList()
        for n_conv_out, kernel_size in zip(filters, ksize):
            self.conv.append(
                nn.Sequential( 
                    nn.Conv2d(in_channels = n_in, out_channels = n_conv_out, kernel_size = kernel_size, padding=kernel_size//2), 
                    nn.GroupNorm(num_groups = 1, num_channels = n_conv_out),
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = n_conv_out

        fc = []
        for n_fc_out in fc_layers:
            fc += [
                nn.Linear(n_in, n_fc_out), 
                nn.LayerNorm(n_fc_out),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate) ]
            n_in = n_fc_out
        fc += [ nn.Linear(n_in, n_out) ]
        self.fc = nn.Sequential(*fc)


    def forward(self, x: torch.Tensor):
        """
        input: pair representation of shape (B, N, N, n_in) (assume B = 1)

        output: learned pair scores of shape (B, N, N, n_out)
        """
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        # separate upper and lower triangles
        x_u = torch.triu(x.view(B*C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.view(B*C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B*2, C, N, N)
        # apply convolution
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.enable_resnet and x.shape[1]==x_a.shape[1] else x_a # (B*2, n_last_conv_out, N, N)
            # seems like it could be possible to enable residual connections only for certain layers of conv (i.e. not all or none)

        # combine triangles together by masking and summing
        x_u, x_l = torch.split(x, B, dim=0) # (B, n_last_conv_out, N, N) * 2
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).view(B*N*N, -1) # (B * N * N, n_last_conv_out)
        x = self.fc(x)
        return x.view(B, N, N, -1) # (B, N, N, n_paired_layer_out)


class UnpairedLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, 
                 filters: tuple[int] = (), ksize: tuple[int] = (), fc_layers: tuple[int] = (), 
                 dropout_rate: float = 0.0, enable_resnet: bool = True):
        super(UnpairedLayer, self).__init__()

        self.enable_resnet = enable_resnet
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)
        assert len(filters) == len(ksize)

        self.conv = nn.ModuleList()
        for n_conv_out, kernel_size in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(in_channels = n_in, out_channels = n_conv_out, kernel_size = kernel_size, padding = kernel_size // 2), 
                    nn.GroupNorm(1, n_conv_out),
                    nn.CELU(), 
                    nn.Dropout(p = dropout_rate) ) )
            n_in = n_conv_out

        fc = []
        for n_fc_out in fc_layers:
            fc += [
                nn.Linear(in_features = n_in, out_features = n_fc_out), 
                nn.LayerNorm(n_fc_out),
                nn.CELU(), 
                nn.Dropout(p = dropout_rate)]
            n_in = n_fc_out
        fc += [ nn.Linear(n_in, n_out) ] # , nn.LayerNorm(n_out) ]
        self.fc = nn.Sequential(*fc)


    def forward(self, x: torch.Tensor):
        """
        input: sequence representation of shape (B, N, n_in) (assume B = 1)

        output: learned pair scores of shape (B, N, n_out)
        """
        B, N, _ = x.shape
        x = x.transpose(1, 2) # (B, n_in, N)

        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a

        x = x.transpose(1, 2).view(B*N, -1) # (B * N, n_last_conv_out)
        x = self.fc(x)
        return x.view(B, N, -1) # (B, N, n_out)


class LengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        l = []
        for m in layers:
            l += [ nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate) ]
            n = m
        l += [ nn.Linear(n, 1) ]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(k<=i ,l<=j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)


    def forward(self, x): 
        return self.net(x)


    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class NeuralNet(nn.Module):
    def __init__(self, embed_size: int = 0,
            num_filters: tuple[int] = (96,), filter_size: tuple[int] = (5,), pool_size: tuple[int] = (1,), dilation: int = 0, # CNNLSTMEncoder - CNN
            num_lstm_layers: int = 0, num_lstm_units: int = 0, # CNNLSTMEncoder - LSTM
            num_att: int = 0, # CNNLSTMEncoder - attention
            num_transformer_layers: int = 0, num_transformer_hidden_units: int = 2048, num_transformer_att: int = 8, # TransformerEncoder
            no_split_lr: bool = False, 
            pair_join: JoinType = JoinType.CAT,
            num_paired_filters: tuple[int] = (), paired_filter_size: tuple[int] = (),
            num_hidden_units: tuple[int] = (32,), dropout_rate: float = 0.0, fc_dropout_rate: float = 0.0, 
            exclude_diag: bool = True, n_out_paired_layers: int =0, n_out_unpaired_layers: int = 0):

        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if num_transformer_layers==0:
            self.encoder = CNNLSTMEncoder(n_in = n_in,
                num_filters = num_filters, filter_size = filter_size, pool_size = pool_size, dilation = dilation, num_att = num_att,
                num_lstm_layers = num_lstm_layers, num_lstm_units = num_lstm_units, dropout_rate = dropout_rate)
        else:
            self.encoder = TransformerLayer(n_in = n_in, n_head = num_transformer_att, 
                            n_hidden = num_transformer_hidden_units, 
                            n_layers = num_transformer_layers, dropout = dropout_rate)
            
        n_in = self.encoder.n_out

        if self.pair_join != JoinType.BILINEAR:
            self.transform2d = Transform2D(join = pair_join)

            n_in_paired = n_in // 2 if pair_join != JoinType.CAT else n_in
            if self.no_split_lr:
                n_in_paired *= 2

            self.fc_paired = PairedLayer(n_in = n_in_paired, n_out = n_out_paired_layers,
                                    filters = num_paired_filters, ksize = paired_filter_size,
                                    exclude_diag = exclude_diag,
                                    fc_layers = num_hidden_units, dropout_rate = fc_dropout_rate)
            if n_out_unpaired_layers > 0:
                self.fc_unpaired = UnpairedLayer(n_in = n_in, n_out = n_out_unpaired_layers,
                                        filters = num_paired_filters, ksize = paired_filter_size,
                                        fc_layers = num_hidden_units, dropout_rate = fc_dropout_rate)
            else:
                self.fc_unpaired = None

        else:
            n_in_paired = n_in // 2 if not self.no_split_lr else n_in
            self.bilinear = nn.Bilinear(n_in_paired, n_in_paired, n_out_paired_layers)
            self.linear = nn.Linear(n_in, n_out_unpaired_layers)


    def forward(self, seq, **kwargs):
        """
        seq: Sequence representation of dimension (B, L, N)
        """
        device = next(self.parameters()).device
        # create embeddings and encodings
        # prepend '0' to the sequence for 1-based indexing in Zuker
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x, **kwargs)


        if self.no_split_lr:
            x_l, x_r = x, x
        else:
            x_l = x[:, :, 0::2]
            x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis

        if self.pair_join != 'bilinear':
            x_lr = self.transform2d(x_l, x_r)

            score_paired = self.fc_paired(x_lr)
            if self.fc_unpaired is not None:
                score_unpaired = self.fc_unpaired(x)
            else:
                score_unpaired = None

            return score_paired, score_unpaired

        else:
            B, N, C = x_l.shape
            x_l = x_l.view(B, N, 1, C).expand(B, N, N, C).reshape(B*N*N, -1)
            x_r = x_r.view(B, 1, N, C).expand(B, N, N, C).reshape(B*N*N, -1)
            score_paired = self.bilinear(x_l, x_r).view(B, N, N, -1)
            score_unpaired = self.linear(x)

            return score_paired, score_unpaired
        
class EvoformerNet(nn.Module):
    """
    EvoformerNet: 
      - Takes a batch of RNA sequences
      - Produces paired and unpaired folding scores
      - Can be plugged into folding scoring heads (e.g., for MXFold2 style outputs)
    """

    def __init__(self,
                 seq_input_dim: int,
                 msa_input_dim: int,
                 seq_embed_dim: int,      # embedding dimension for sequence track
                 pair_embed_dim: int,
                 n_out_paired_layers: int = 0,
                 n_out_unpaired_layers: int = 0,
                 num_paired_filters = (),
                 paired_filter_size = (),
                 exclude_diag = True,
                 num_hidden_units = (32,),
                 fc_dropout_rate: float = 0.0):    # embedding dimension for pair track
        super(EvoformerNet, self).__init__()

        self.seq_embed_dim = seq_embed_dim
        self.pair_embed_dim = pair_embed_dim
        self.n_out_paired_layers = n_out_paired_layers
        self.n_out_unpaired_layers = n_out_unpaired_layers

        # PreMSA transforms raw sequence + MSA into initial m (seq) and z (pair) embeddings
        self.premsa = PreMSA(seq_input_dim, msa_input_dim, seq_embed_dim, pair_embed_dim)

        # Evoformer stack refines these
        self.evoformer = EvoformerStack(seq_embed_dim, pair_embed_dim)

        self.pair_layernorm = nn.LayerNorm(pair_embed_dim)
        self.msa_layernorm = nn.LayerNorm(seq_embed_dim)

        # fully connected layers to compute folding scores from the pair representation output from Evoformer
        self.fc_paired = PairedLayer(pair_embed_dim, n_out_paired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    exclude_diag=exclude_diag,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
        if n_out_unpaired_layers > 0:
            self.fc_unpaired = UnpairedLayer(seq_embed_dim, n_out_unpaired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
        else:
            self.fc_unpaired = None

    def forward(self, seq_batch):
        """
        Args:
          seq_batch: list of RNA strings (length B = 1)

        Returns:
          score_paired: Tensor of pair scores of shape (B, L, L, n_out_paired_layers)
          score_unpaired: Tensor of unpaired nucleotide scores of shape (B, L, n_out_unpaired_layers)
        """

        # NOTE: Embedding input must be one-hot + MSA format like DRFold
        # Build fake MSA: two identical copies per sequence
        device = next(self.parameters()).device

        def _mem_log(tag):
            if os.getenv("MXFOLD2_MEMLOG") != "1":
                return
            if not torch.cuda.is_available():
                return
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[mem] {tag} alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")

        # Convert seq_batch to one-hot
        # PreMSA expects msa shape (N, L, msa_input_dim) and seq shape (L, seq_input_dim)
        # We can build msa and seq for each item in batch

        seq = seq_batch[0]

        L = len(seq)

        one_hot_seq = self.one_hot_encode_seq('0' + seq).to(device) # shape (L + 1, seq_input_dim)

        # 2) build fake MSA by repeating sequence
        mask_channel = torch.zeros(L + 1, 1, device=one_hot_seq.device)

        # print(one_hot_seq.shape, mask_channel.shape)

        # combine
        msa = torch.cat([
            one_hot_seq.unsqueeze(0).repeat(2,1,1),  # (2, L + 1, 6)
            mask_channel.unsqueeze(0).repeat(2,1,1)  # (2, L + 1, 1)
        ], dim=-1)  # -> (2, L + 1, 7)

        # 3) PreMSA → initial seq & pair embeddings
        # _mem_log("EvoformerNet:pre_premsa")
        m, z = self.premsa(one_hot_seq, msa) # m: (2,L + 1,seq_embed_dim), z: (L + 1,L + 1,pair_embed_dim)
        # _mem_log("EvoformerNet:post_premsa")

        # Pass through Evoformer
        refined_m, refined_z = self.evoformer(m, z)
        # _mem_log("EvoformerNet:post_evoformer")

        refined_z = self.pair_layernorm(refined_z)
        refined_m = self.msa_layernorm(refined_m)

        score_paired = self.fc_paired(refined_z.unsqueeze(0))
        # _mem_log("EvoformerNet:post_fc_paired")

        # CHANGE: cap mismatch so that mismatches are not rewarded too much
        score_paired_stacking = score_paired[..., 0:1]
        score_paired_mismatch = torch.celu(score_paired[..., 1:2])
        SQUASH_VAL = 1.0
        score_paired = torch.cat([score_paired_stacking, score_paired_mismatch], dim=-1) / SQUASH_VAL

        if self.fc_unpaired != None:
            # the MSA was constructed by duplicating the sequence, but we can only pass
            # in one sequence into the unpaired layer
            # CHANGE: use softplus to constrain sign to ensure that unpairs are penalised
            score_unpaired = -F.softplus(self.fc_unpaired(refined_m[0].unsqueeze(0))) / SQUASH_VAL
        else:
            B = score_paired.shape[0]
            score_unpaired = torch.zeros((B, L + 1), device=score_paired.device, dtype=score_paired.dtype)

        return score_paired, score_unpaired

    def one_hot_encode_seq(self, seq):
        """
        Convert an RNA sequence to one-hot tensor (L x 6)
        """
        mapping = defaultdict(lambda: 5, {'0': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4})
        L = len(seq)
        one_hot = torch.zeros(L, 6, dtype=torch.float32)
        for i, ch in enumerate(seq):
            idx = mapping.get(ch.upper(), 5)
            one_hot[i, idx] = 1.0
        return one_hot

# improved initialisation
def init_weights(m):
    if hasattr(m, 'is_residual_output') and m.is_residual_output:
        inner = getattr(m, 'linear', None)
        if isinstance(inner, torch.nn.Linear):
            nn.init.constant_(inner.weight, 0)
            if inner.bias is not None:
                nn.init.zeros_(inner.bias)
            return
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
        if hasattr(m, 'is_residual_output') and m.is_residual_output:
            nn.init.constant_(m.weight, 0)
        else:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="leaky_relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

def init_heads(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
