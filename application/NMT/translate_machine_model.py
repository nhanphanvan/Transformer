
from transformers import AutoTokenizer
import torch.nn as nn

from ...modules.config import TransformerConfig
from ...modules.seq2seq_transformer import Seq2SeqTransformer

class NMTModel(nn.Module):
    def __init__(self,
                 src_bos_id: int,
                 src_eos_id: int,
                 tgt_bos_id: int,
                 tgt_eos_id: int,
                 src_tokenizer: AutoTokenizer,
                 tgt_tokenizer: AutoTokenizer,
                 config: TransformerConfig,
                 transformer_model: Seq2SeqTransformer,
                 rdrsegmenter=None):

        super().__init__()
        self.src_bos_id = src_bos_id
        self.src_eos_id = src_eos_id
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.transformer_model = transformer_model
        self.rdrsegmenter = rdrsegmenter
        self.config = config
