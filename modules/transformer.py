# standard library improrts
from typing import Optional
import collections

# third party imports
from torch import Tensor
import torch.nn as nn

# local applicaiton imports
from .config import TransformerConfig
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoder import TransformerDecoder, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feedforward_size = config.feedforward_size
        self.num_attention_heads = config.num_attention_heads
        self.output_hidden_states = config.output_hidden_states
        self.apply_layer_norm = config.apply_layer_norm
        kwargs = {'device': config.device, 'dtype': config.dtype}

        encoder_layer = TransformerEncoderLayer(config=config)
        encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                          num_encoder_layers=config.num_encoder_layers, 
                                          encoder_norm=encoder_norm, 
                                          output_hidden_states=config.output_hidden_states,
                                          apply_layer_norm=config.apply_layer_norm)
        decoder_layer = TransformerDecoderLayer(config=config)
        decoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          num_decoder_layers=config.num_decoder_layers,
                                          decoder_norm=decoder_norm,
                                          output_hidden_states=config.output_hidden_states,
                                          apply_layer_norm=config.apply_layer_norm)
        self._reset_parameters()
        
    def forward(self, 
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        N: batch size, S: source sequence length, T: target sequence length, E: number of features
        Args:
            src:
                the sequence to the encoder. Have shape (N, S, E).
            tgt:
                the sequence to the decoder. Have shape (N, T, E).
            src_mask (*optional*):
                the additive mask for the src sequence. Have shape (S, S).
            tgt_mask (*optional*):
                the additive mask for the tgt sequence. Have shape (T, T).
            memory_mask (*optional*):
                the additive mask for the encoder output. Have shape (T, S).
            src_key_padding_mask (*optional*):
                the mask for src keys per batch. Have shape (N, S).
            tgt_key_padding_mask (*optional*):
                the mask for tgt keys per batch. Have shape (N, T).
            memory_key_padding_mask (*optional*):
                the mask for memory keys per batch. Have shape (N, S).
        
        Note:
            Only bool and int are supported for attention_mask.
                If `bool` is provided, positions with `True` are not allowed to attend,
                while `False` values will be unchanged.
                If `int` is provided, the non-zero positions are not allowed to attend,
                while the zero positions will be unchanged.
            Only bool and int are supported for padding_mask.
                If `bool` is provided, positions with `True` will be ignored while the
                position with `False` will be unchanged.
                If `int` is provided, the non-zero positions will be ignored while the
                zero positions will be unchanged.
        """       
        if src.shape[0] != tgt.shape[0]:
            raise RuntimeError('The batch size of src and tgt must be equal')
        if src.shape[-1] != tgt.shape[-1]:
            raise RuntimeError('The hidden size of src and tgt must be equal')

        encoder_outputs = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = encoder_outputs[0]
        decoder_outputs = self.decoder(tgt, 
                              memory=memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = decoder_outputs[0]
        if self.output_hidden_states:
            TransformerOutput = collections.namedtuple('TransformerOutput',
                                                      ['output', 'encoder_hidden_states', 'decoder_hidden_states'])
            return TransformerOutput(output, encoder_outputs[1], decoder_outputs[1])
        return output,

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)