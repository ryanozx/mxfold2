from typing import Optional
from torch import Tensor
from torch.nn import TransformerEncoderLayer

class TransformerEncoderPatchedLayer(TransformerEncoderLayer):
    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        need_weights: bool = False
    ) -> Tensor:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False
        )
        if need_weights:
            self._attn_weights = attn_weights
        else:
            self._attn_weights = None
        return self.dropout1(attn_out)
    
    def forward(
        self,
        src: Tensor,
        return_attn: bool = False
    )  -> Tensor:
        if not return_attn:
            return super().forward(src, src_mask=None, src_key_padding_mask=None)

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), attn_mask=None, key_padding_mask=None, need_weights = True
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, attn_mask=None, key_padding_mask=None, need_weights = True)
            )
            x = self.norm2(x + self._ff_block(x))
        return x