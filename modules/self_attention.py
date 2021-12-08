# standard library improrts
import math
from typing import Optional

# third party imports
import torch
import torch.nn as nn
from torch import Tensor

# local applicaiton imports
from .config import TransformerConfig


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        kwargs = {'device': config.device, 'dtype': config.dtype}
        self.num_attention_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_size
        self.hidden_size = config.hidden_size

        self.query_projection = nn.Linear(self.hidden_size, self.all_head_size, **kwargs)
        self.key_projection = nn.Linear(self.hidden_size, self.all_head_size, **kwargs)
        self.value_projection = nn.Linear(self.hidden_size, self.all_head_size, **kwargs)
        self.weight_matrix = nn.Linear(self.hidden_size, self.all_head_size, **kwargs)

        self.dropout = nn.Dropout(config.dropout)

    def _project(self, query: Tensor, key: Tensor, value: Tensor):
        query_emb_size, key_emb_size, value_emb_size = query.shape[-1], key.shape[-1], value.shape[-1]
        assert query_emb_size == self.hidden_size, f'Excepting query embedding size is {self.hidden_size}, but got {query_emb_size}'
        assert key_emb_size == self.hidden_size, f'Excepting key embedding size is {self.hidden_size}, but got {key_emb_size}'
        assert value_emb_size == self.hidden_size, f'Excepting value embedding size is {self.hidden_size}, but got {value_emb_size}'
        
        return self.query_projection(query), self.key_projection(key), self.value_projection(value)

    def _change_for_multihead_shape(self, x: Tensor):
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.head_size)
        x = x.view(*new_shape)
        return x.transpose(1, 2)

    def _dot_product_attention(self,
                                query: Tensor,
                                key: Tensor,
                                value: Tensor,
                                attention_mask: Optional[Tensor] = None):
        """
        query (B, Nh, Nt, E)  where B is batch size, Nh is number of head, Nt is the target sequence length, and E is embedding dimension
        key (B, Nh, Ns, E) where B is batch size, Nh is number of head, Ns is the source sequence length, and E is embedding dimension
        value (B, Nh, Ns, E) where B is batch size, Nh is number of head, Ns is the source sequence length, and E is embedding dimension
        attention_mask: either a 3D tensor of shape (B, Nt, Ns) or a 2D tensor of shape (Nt, Ns)

        Output have shape (B, Nh, Nt, E)
        """
        batch_size, number_of_heads, tgt_length, emb_size = query.shape
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(emb_size)
        if attention_mask is not None:
            attention_scores += attention_mask
        attention_scores = nn.Softmax(dim=-1)(attention_scores)
        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, value)
        return output

    def forward(self, 
                query: Tensor, 
                key: Tensor, 
                value: Tensor,
                attention_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):

        batch_size, tgt_length, hidden_size = query.shape
        _, src_length, _ = key.shape

        assert hidden_size == self.hidden_size, f'Expecting hidden_size is {self.hidden_size}, but got {hidden_size}'

        q, k, v = self._project(query, key, value)

        if attention_mask is not None:
            assert attention_mask.dtype == torch.bool or attention_mask.dtype == torch.int, f'Only bool and int are supported for attention_mask, not {attention_mask.dtype}'
            if attention_mask.dtype == torch.int:
                attention_mask = attention_mask.to(torch.bool)

            if attention_mask.dim() == 2:
                correct_shape = (tgt_length, src_length)
                if attention_mask.shape != correct_shape:
                    raise RuntimeError(f'Expecting the shape of 2D attention_mask is {correct_shape}, but got {attention_mask.shape}')
                attention_mask = attention_mask.view(1, 1, tgt_length, src_length)
            
            elif attention_mask.dim() == 3:
                correct_shape = (batch_size, tgt_length, src_length)
                if attention_mask.shape != correct_shape:
                    raise RuntimeError(f'Expecting the shape of 3D attention_mask is {correct_shape}, but got {attention_mask.shape}')
                attention_mask = attention_mask.view(batch_size, 1, tgt_length, src_length)
            else:
                raise RuntimeError(f"Attention_mask's dimension {attention_mask.dim()} is not supported")

        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool or key_padding_mask.dtype == torch.int, f'Only bool and int are supported for key_padding_mask, not {key_padding_mask.dtype}'
            correct_shape = (batch_size, src_length)
            if key_padding_mask.shape != correct_shape:
                raise RuntimeError(f'Expecting the shape of key_padding_mask is {correct_shape}, but got {key_padding_mask.shape}')
            if key_padding_mask.dtype == torch.int:
                key_padding_mask = key_padding_mask.to(torch.bool)

        # change shape to (batch_size, number_of_heads, sequence_length, emb_size)
        q = self._change_for_multihead_shape(q)
        k = self._change_for_multihead_shape(k)
        v = self._change_for_multihead_shape(v)
        
        # merge key padding mask and attention mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_length).expand(-1, self.num_attention_heads, -1, -1)
            if attention_mask is None:
                attention_mask = key_padding_mask
            else:
                attention_mask = attention_mask.logical_or(key_padding_mask)

        if attention_mask is not None:
            new_attention_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            new_attention_mask = new_attention_mask.masked_fill(attention_mask, float('-inf'))
            attention_mask = new_attention_mask
        
        attention_output = self._dot_product_attention(q, k, v, attention_mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, tgt_length, hidden_size)
        attention_output = self.weight_matrix(attention_output)

        return attention_output




