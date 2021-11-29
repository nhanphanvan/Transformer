import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from packaging import version

class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        position_emb = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        omega = torch.exp(torch.arange(0, hidden_size, 2)*(-math.log(10000)/hidden_size))

        position_emb[:, 0::2] = torch.sin(position * omega)
        position_emb[:, 1::2] = torch.cos(position * omega)

        position_emb = position_emb.unsqueeze(0)
        self.register_buffer('position_emb', position_emb)

    def forward(self, embedding):
        embedding = embedding + Variable(self.position_emb[:, :embedding.size(1)], requires_grad=False)
        return self.dropout(embedding)


class TransformerEmbedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, padding_id, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_id, **kwargs)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.hidden_size)

class BertEmbedding(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 vocab_size, 
                 padding_id, 
                 max_len=1024, 
                 layer_norm_eps=1e-5, 
                 dropout=0.1, 
                 type_vocab_size=1, 
                 device=None, 
                 dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_id, **kwargs)
        self.position_embedding = nn.Embedding(max_len, hidden_size, **kwargs)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size, **kwargs)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(self, x, token_type_ids=None):
        input_shape = x.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, : seq_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        embedding = self.word_embedding(x) + self.position_embedding(position_ids) + self.token_type_embedding(token_type_ids)
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        return embedding