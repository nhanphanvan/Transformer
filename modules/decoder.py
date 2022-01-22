# standard library improrts
from typing import Optional

# third party imports
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# local applicaiton imports
from .config import TransformerConfig
from .self_attention import SelfAttention
from .utils import _get_activation_function, _get_clones


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
            Args:
            tgt:
                the sequence to the decoder. Have shape (N, T, E).
            memory:
                the sequence from the last layer of the encoder. Have shape (N, S, E).
            tgt_mask (*optional*):
                the additive mask for the tgt sequence. Have shape (T, T).
            memory_mask (*optional*):
                the additive mask for the encoder output. Have shape (T, S).
            tgt_key_padding_mask (*optional*):
                the mask for tgt keys per batch. Have shape (N, T).
            memory_key_padding_mask (*optional*):
                the mask for memory keys per batch. Have shape (N, S).
        """
        if self.norm_first:
            tgt = tgt + self._self_attention_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask)
            tgt = tgt + self._cross_attention_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask)
            tgt = tgt + self._feedforward_block(self.norm3(tgt))
        else:
            tgt = self.norm1(tgt + self._self_attention_block(tgt, tgt_mask, tgt_key_padding_mask))
            tgt = self.norm2(tgt + self._cross_attention_block(tgt, memory, memory_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._feedforward_block(tgt))

        return tgt


class TransformerDecoder(nn.Module):
    r"""
    Args:
        decoder_layer:
            an instance of TransformerDecoderLayer class.
        num_decoder_layers:
            The number of sub-decoder in decoder block.
        decoder_norm (*Optional*):
            the layer normalization component.
        output_hidden_states (`bool`, *optional*, default = False):
            If `True`, return the hidden states of all layers in encoder and decoder.
        apply_layer_norm (`bool`, *optional*, default = False):
            Use if `output_hidden_state` = `True`. If `True`, apply LayerNorm for all
            hidden states.
    """

    def __init__(self, decoder_layer, num_decoder_layers,
                 decoder_norm=None, output_hidden_states=False, apply_layer_norm=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        self.num_decoder_layer = num_decoder_layers
        self.decoder_norm = decoder_norm
        self.output_hidden_states = output_hidden_states
        self.apply_layer_norm = apply_layer_norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
            N: batch size, S: source sequence length, T: target sequence length, E: feature number
            Args:
            tgt:
                the sequence to the decoder. Have shape (N, T, E).
            memory:
                the sequence from the last layer of the encoder. Have shape (N, S, E).
            tgt_mask (*optional*):
                the additive mask for the tgt sequence. Have shape (T, T).
            memory_mask (*optional*):
                the additive mask for the encoder output. Have shape (T, S).
            tgt_key_padding_mask (*optional*):
                the mask for tgt keys per batch. Have shape (N, T).
            memory_key_padding_mask (*optional*):
                the mask for memory keys per batch. Have shape (N, S).
        """
        hidden_states = (tgt,) if self.output_hidden_states else None 
        for layer in self.layers:
            tgt = layer(tgt, 
                      memory, 
                      tgt_mask=tgt_mask, 
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
            if self.output_hidden_states:
                hidden_states += (self.decoder_norm(tgt),) if self.apply_layer_norm and self.decoder_norm is not None else (tgt,)
        
        if self.decoder_norm is not None:
            tgt = self.decoder_norm(tgt)
        if self.output_hidden_states:
            hidden_states += (tgt,)

        return tuple(element for element in [tgt, hidden_states] if element is not None)