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


class TransformerEncoderLayer(nn.Module):
    r"""
    Args:
        config:
            an instance of TransformerConfig class.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        kwargs = {'device': config.device, 'dtype': config.dtype}
        self.self_attention = SelfAttention(config)
        self.linear1 = nn.Linear(config.hidden_size, config.feedforward_size, **kwargs)
        self.linear2 = nn.Linear(config.feedforward_size, config.hidden_size, **kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
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

    def _feedforward_block(self, x: Tensor) -> Tensor:
        """
            FFN(x) = max(0, x.W_1 + b_1).W_2 + b_2
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
            N: batch size, S: source sequence length, T: target sequence length, E: feature number
            Args:
            src:
                the sequence to the encoder. Have shape (N, S, E).
            src_mask (*optional*):
                the additive mask for the src sequence. Have shape (S, S).
            src_key_padding_mask (*optional*):
                the mask for src keys per batch. Have shape (N, S).
        """       
        if self.norm_first:
            src = src + self._self_attention_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._feedforward_block(self.norm2(src))
        else:
            src = self.norm1(src + self._self_attention_block(src, src_mask, src_key_padding_mask))
            src = self.norm2(src + self._feedforward_block(src))

        return src


class TransformerEncoder(nn.Module):
    r"""
    Args:
        encoder_layer:
            an instance of TransformerEncoderLayer class.
        num_encoder_layers:
            The number of sub-encoder in encoder block.
        encoder_norm (*Optional*):
            the layer normalization component.
        output_hidden_states (`bool`, *optional*, default = False):
            If `True`, return the hidden states of all layers in encoder and decoder.
        apply_layer_norm (`bool`, *optional*, default = False):
            Use if `output_hidden_state` = `True`. If `True`, apply LayerNorm for all
            hidden states.
    """

    def __init__(self, encoder_layer, num_encoder_layers,
                 encoder_norm=None, output_hidden_states=False, apply_layer_norm=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.encoder_norm = encoder_norm
        self.output_hidden_states = output_hidden_states
        self.apply_layer_norm = apply_layer_norm

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
            N: batch size, S: source sequence length, T: target sequence length, E: feature number
            Args:
            src:
                the sequence to the encoder. Have shape (N, S, E).
            src_mask (*optional*):
                the additive mask for the src sequence. Have shape (S, S).
            src_key_padding_mask (*optional*):
                the mask for src keys per batch. Have shape (N, S).
        """
        hidden_states = (src,) if self.output_hidden_states else None 
        for layer in self.layers:
            src = layer(src,
                      src_mask=src_mask,
                      src_key_padding_mask=src_key_padding_mask)
            if self.output_hidden_states:
                hidden_states += (self.encoder_norm(src),) if self.apply_layer_norm and self.encoder_norm is not None else (src,)

        if self.encoder_norm is not None:
            src = self.encoder_norm(src)
        if self.output_hidden_states:
            hidden_states += (src,)

        return tuple(element for element in [src, hidden_states] if element is not None)
        