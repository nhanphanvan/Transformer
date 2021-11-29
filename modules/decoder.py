from typing import Optional
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from config import TransformerConfig
from self_attention import SelfAttention
from utils import _get_activation_function, _get_clones

class TransformerDecoderLayer(nn.Module):
        
    def __init__(self, config: TransformerConfig):
        super().__init__()
        kwargs = {'device': config.device, 'dtype': config.dtype}
        self.self_attention = SelfAttention(config)
        self.cross_attention = SelfAttention(config)
        self.linear1 = nn.Linear(config.hidden_size, config.feedforward_size, **kwargs)
        self.linear2 = nn.Linear(config.feedforward_size, config.hidden_size, **kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.activation = _get_activation_function(config.activation)
        self.norm_first = config.norm_first

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
    
    def _self_attention_block(self,
                              x: Tensor,
                              attention_mask: Optional[Tensor],
                              key_padding_mask: Optional[Tensor]) -> Tensor:

        x = self.self_attention(x, x, x, attention_mask=attention_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _cross_attention_block(self,
                               x: Tensor,
                               memory: Tensor,
                               attention_mask: Optional[Tensor],
                               key_padding_mask: Optional[Tensor]) -> Tensor:
        
        x = self.cross_attention(x, memory, memory, attention_mask=attention_mask, key_padding_mask=key_padding_mask)
        return self.dropout2(x)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        """
            FFN(x) = max(0, x.W_1 + b_1).W_2 + b_2
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
            N: batch size, S: source sequence length, T: target sequence length, E: feature number
            src: (N, S, E)
            tgt : (N, T, E)
            memory: (N, S, E)
            src_mask: (S, S)
            tgt_mask: (T, T)
            memory_mask: (T, S)
            src_key_padding_mask: (N, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """
        x = tgt
        if self.norm_first:
            x = x + self._self_attention_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._cross_attention_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._feedforward_block(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attention_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._cross_attention_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._feedforward_block(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_decoder_layers, decoder_norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        self.num_decoder_layer = num_decoder_layers
        self.decoder_norm = decoder_norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
            N: batch size, S: source sequence length, T: target sequence length, E: feature number
            src: (N, S, E)
            tgt : (N, T, E)
            memory: (N, S, E)
            src_mask: (S, S)
            tgt_mask: (T, T)
            memory_mask: (T, S)
            src_key_padding_mask: (N, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, 
                      memory, 
                      tgt_mask=tgt_mask, 
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        
        if self.decoder_norm is not None:
            x = self.decoder_norm(x)

        return x