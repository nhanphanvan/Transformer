import torch
import torch.nn as nn
import math
from torch.autograd import Variable

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
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.hidden_size)