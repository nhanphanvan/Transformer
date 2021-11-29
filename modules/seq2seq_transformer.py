# standard library improrts

# third party imports
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# local applicaiton imports
from config import TransformerConfig
from embedding import PositionalEmbedding, TransformerEmbedding, BertEmbedding
from transformer import Transformer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        kwargs = {'device': config.device, 'dtype': config.dtype}
        self.bert_embedding = config.bert_embedding
        self.transformer = Transformer(config=config)
        self.generator = nn.Linear(config.hidden_size, config.tgt_vocab_size, **kwargs)
        if self.bert_embedding:
            self.src_embedding = BertEmbedding(config.hidden_size, config.src_vocab_size, config.src_padding_id, config.max_sequence_length, 
                                               config.layer_norm_eps, config.dropout, config.type_vocab_size, **kwargs)
            self.tgt_embedding = BertEmbedding(config.hidden_size, config.tgt_vocab_size, config.tgt_padding_id, config.max_sequence_length,
                                               config.layer_norm_eps, config.dropout, config.type_vocab_size, **kwargs)
        else:
            self.positional_embedding = PositionalEmbedding(config.hidden_size, config.dropout, config.max_sequence_length)
            self.src_embedding = TransformerEmbedding(config.hidden_size, config.src_vocab_size, config.src_padding_id, **kwargs)
            self.tgt_embedding = TransformerEmbedding(config.hidden_size, config.tgt_vocab_size, config.tgt_padding_id, **kwargs)
        

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
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
        if self.bert_embedding:
            src_emb = self.src_embedding(src)
            tgt_emb = self.tgt_embedding(tgt)
        else:
            src_emb = self.positional_embedding(self.src_embedding(src))
            tgt_emb = self.positional_embedding(self.tgt_embedding(tgt))
        outputs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        return self.generator(outputs)

    def encode(self, 
               src: Tensor, 
               src_mask: Optional[Tensor] = None,
               src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        if self.bert_embedding:
            src_emb = self.src_embedding(src)
        else:
            src_emb = self.positional_embedding(self.src_embedding(src))
        return self.transformer.encoder(src_emb, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self,
              tgt: Tensor,
              memory: Tensor,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        if self.bert_embedding:
            tgt_emb = self.tgt_embedding(tgt)
        else:
            tgt_emb = self.positional_embedding(self.tgt_embedding(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

