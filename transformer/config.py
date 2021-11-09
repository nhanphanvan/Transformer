
class TransformerConfig:
    def __init__(self,
                src_vocab_size: int = 30522, 
                tgt_vocab_size: int = 30522, 
                hidden_size: int = 768,
                num_encoder_layers: int = 12,
                num_decoder_layers: int = 12,
                num_attention_heads: int = 12,
                feedforward_size: int = 2048,
                dropout: float = 0.1,
                activation: str = 'relu',
                layer_norm_eps: float = 1e-5,
                src_padding_id: int = 1,
                tgt_padding_id: int = 1,
                norm_first: bool = False) -> None:

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.src_padding_id = src_padding_id
        self.tgt_padding_id = tgt_padding_id
        self.norm_first = norm_first
