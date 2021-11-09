from typing import Optional
from torch import Tensor
import torch.nn as nn
from config import TransformerConfig
from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feedforward_size = config.feedforward_size
        self.num_attention_heads = config.num_attention_heads
        
        encoder_layer = TransformerEncoderLayer(config=config)
        encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                          num_encoder_layers=config.num_encoder_layers, 
                                          encoder_norm=encoder_norm)
        decoder_layer = TransformerDecoderLayer(config=config)
        decoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          num_decoder_layers=config.num_decoder_layers,
                                          decoder_norm=decoder_norm)
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

        if src.shape[0] != tgt.shape[0]:
            raise RuntimeError('The batch size of src and tgt must be equal')
        if src.shape[-1] != tgt.shape[-1]:
            raise RuntimeError('The hidden size of src and tgt must be equal')

        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder.forward(tgt, 
                                      memory=memory,
                                      tgt_mask=tgt_mask,
                                      memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)