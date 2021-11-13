# standard library improrts

# third party imports
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# local applicaiton imports
from config import TransformerConfig
from embedding import PositionalEmbedding, TransformerEmbedding
from transformer import Transformer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.transformer = Transformer(config=config)
        self.positional_embedding = PositionalEmbedding(config.hidden_size, config.dropout, config.max_sequence_length)
        self.src_embedding = TransformerEmbedding(config.hidden_size, config.src_vocab_size, config.src_padding_id)
        self.tgt_embedding = TransformerEmbedding(config.hidden_size, config.tgt_vocab_size, config.tgt_padding_id)
        self.generator = nn.Linear(config.hidden_size, config.tgt_vocab_size)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        src_emb = self.positional_embedding(self.src_embedding(src))
        tgt_emb = self.positional_embedding(self.tgt_embedding(tgt))
        outputs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        return self.generator(outputs)

    def encode(self, 
               src: Tensor, 
               src_mask: Optional[Tensor] = None,
               src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        src_emb = self.positional_embedding(self.src_embedding(src))
        return self.transformer.encoder(src_emb, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self,
              tgt: Tensor,
              memory: Tensor,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt_emb = self.positional_embedding(self.tgt_embedding(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
