
class TransformerConfig:
    r"""
    This is the configuration class to store the configuration of the Transformer Model and
    Seq2SeqTransformer . It is used to instantiate a Transformer model according to the
    specified arguments.

    Args:
        src_vocab_size (`int`, *optional*, default = 30522):
            Vocabulary size of source language. Use for Seq2SeqTransformer model.
        tgt_vocab_size (`int`, *optional*, default = 30522):
            Vocabulary size of target language. Use for Seq2SeqTransformer model.
        hidden_size (`int`, *optional*, default = 768):
            The number of expected features in the encoder or decoder. Use for both models.
        num_encoder_layers (`int`, *optional*, default = 12):
            The number of sub-encoder in encoder block. Use for both models.
        num_decoder_layers ('`int`, *optional*, default = 12):
            The number of sub-decoder in decoder block. Use for both models.
        num_attention_heads (`int`, *optional*, default = 12):
            The number of heads in the multihead attention. Use for both models.
        feedforward_size (`int`, *optional*, default = 2048):
            The dimension of the feedforward network model. Use for both models.
        dropout (`float`, *optional*, default = 0.1):
            The dropout value. Use for both models.
        activation (`str`, *optional*, default = 'relu'):
            The activation function. Can be ("relu" or "gelu"). Use for both models.
        layer_norm_eps (`float`, *optional*, default = 1e-5) :
            The eps value in layer normalization components. Use for both models.
        src_padding_id (`int`, *optional*, default = 1):
            The index of padding token in source vocabulary. Use for batch training.
            User for both models.
        tgt_padding_id (`int`, *optional*, default = 1):
            The index of padding token in target vocabulary. Use for batch training.
            Use for both models.
        norm_first (`bool`, *optional*, default = False):
            If `True`, encoder and decoder layer will perform LayerNorm before.
            Use for both models.
        max_sequence_length (`int`, *optional*, default = 1024):
            Maximum number of tokens in one sentence. Use for declare positional embedding.
            Use for Seq2SeqTransformer model.
        bert_embedding (`bool`, *optional*, default = False):
            if `True`, Embedding layer will use Bert Embedding, else Transformer Embedding.
            Use for Seq2SeqTransformer model.
        type_vocab_size (`int`, *optional*, default = 1):
            Use in Bert Embedding. The vocabulary size of the `token_type_ids`.
            Use for Seq2SeqTransformer model.
        output_hidden_states (`bool`, *optional*, default = False):
            If `True`, return the hidden states of all layers in encoder and decoder.
            Use for both models.
        apply_layer_norm (`bool`, *optional*, default = False):
            Use if `output_hidden_state` = `True`. If `True`, apply LayerNorm for all
            hidden states. Use for both models.
    """
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
                norm_first: bool = False,
                max_sequence_length: int = 1024,
                bert_embedding: bool = False,
                type_vocab_size: int = 1,
                output_hidden_states: bool = False,
                apply_layer_norm: bool = False,
                device=None,
                dtype=None) -> None:

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
        self.max_sequence_length = max_sequence_length
        self.bert_embedding = bert_embedding
        self.type_vocab_size = type_vocab_size
        self.output_hidden_states = output_hidden_states
        self.apply_layer_norm = apply_layer_norm
        self.device = device
        self.dtype = dtype
