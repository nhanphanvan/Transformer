import torch
import torch.nn.functional as F
import numpy as np

from .data_store import Datastore
from .translate_machine_model import NMTModel

class TranslateMachine:
  def __init__(self, model: NMTModel,
               data_store: Datastore,
               look_up_array: np.array,
               gamma: float = 0.2,
               temperature: int = 10,
               use_layernorm: bool = True,
               device: torch.device = None):
    
    if not 0 <= gamma <= 1:
      raise ValueError(f'gamma must be between 0 and 1')
    self.model = model
    self.data_store = data_store
    self.look_up_array = look_up_array
    self.gamma = gamma
    self.temperature = temperature
    self.use_layernorm = use_layernorm
    if device is None:
      self.device = torch.device('cpu')
    else:
      self.device = device

  def apply_data_store(self, encoder_embeddings, num_knns):
    queries = encoder_embeddings.cpu().detach().numpy()
    # print(queries.shape)
    distances, indexes = self.data_store.get_knns(queries, num_knns)
    knn_vectors = self.data_store.get_vals_vectors(indexes, self.look_up_array)
    knn_distribution = self.data_store.construct_final_distribution(knn_vectors, distances, self.temperature, self.use_layernorm)
    knn_distribution = torch.tensor(knn_distribution, dtype=torch.float32, device=self.device)
    # combined_vectors = self.gamma*final_vectors + (1-self.gamma)*encoder_embeddings
    return knn_distribution

  @staticmethod
  def convert_ids_to_string(tokenizer, ids):
    """
      convert list ids (not tensor) to string
    """
    tokens = tokenizer.convert_ids_to_tokens(ids)
    sentence = " ".join(tokens).replace("@@ ", "").replace("<unk> ", "").replace("<s>", "").replace("</s>", "").strip()
    return sentence 

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 0).transpose(0, 1)
    return mask

  def _init_beam_search(self, src, src_mask, max_len, num_beams, num_knns, use_datastore):
    """
    init first beam search with BOS symbol and calculate topk candidates
    :param model:
    :param src: (N, S), batch size N = 1
    :param src_mask: (S, S)
    :param max_len: 
    :param start_symbol: BOS
    :param num_beams:
    :return: outputs (num_beams, max_len), memory (N,S,E), log_scores (N, num_beams) 
    """
    # encode src to get memory, tgt contains only one token - BOS token
    src = src.to(self.device)
    src_mask = src_mask.to(self.device)
    memory = self.model.transformer_model.encode(src, src_mask)
    batch_size, src_length, hidden_size = memory.shape
    tgt = torch.LongTensor([[self.model.tgt_bos_id]]).to(self.device)
    tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).type(torch.bool)
    outputs = self.model.transformer_model.decode(tgt, memory, tgt_mask)

    # outputs.shape = (N,S,E) -> outputs[:, -1] take (N,E) at last element in axis=1 -> apply data_store
    encoder_embeddings = outputs[:, -1]
    outputs = self.model.transformer_model.generator(encoder_embeddings)
    if use_datastore:
      knn_distribution = self.apply_data_store(encoder_embeddings, num_knns)
      # apply distribution of datastore and get final distribution
      mt_distribution = F.softmax(outputs, dim=-1)
      final_distribution = self.gamma*knn_distribution + (1 - self.gamma)*mt_distribution
    else:
      final_distribution = F.softmax(outputs, dim=-1)

    log_scores, index = torch.log(final_distribution).topk(num_beams)
    outputs = torch.zeros((num_beams, max_len), dtype=torch.int32, device=self.device)
    outputs[:, 0] = self.model.tgt_bos_id
    outputs[:, 1] = index[0]
    memory = memory.expand(num_beams, src_length, hidden_size)

    return outputs, memory, log_scores

  @staticmethod
  def _choose_topk(outputs, final_distribution, log_scores, i, num_beams):
    """
    choose topk candidates from kxk candidates
    """
    log_probs, index = torch.log(final_distribution).topk(num_beams)
    # log_scores.transpose(0,1) to add correct element to log_probs
    log_probs = log_probs + log_scores.transpose(0, 1)
    log_probs, k_index = log_probs.view(-1).topk(num_beams)

    # calculate rows, cols becasue log_probs now has shape (num_beams x num_beams)
    rows = torch.div(k_index, num_beams, rounding_mode='floor')
    cols = k_index % num_beams
    outputs[:, :i] = outputs[rows, :i]
    outputs[:, i] = index[rows, cols]
    
    # log_probs has shape (num_beams) -> (1, num_beams)
    log_scores = log_probs.unsqueeze(0)

    return outputs, log_scores

  def beam_search(self, src, src_mask, max_len, num_beams, num_knns, use_datastore=True):
    max_len = 256 if max_len > 256 else max_len
    chosen_sentence_index = 0
    outputs, memory, log_scores = self._init_beam_search(src, src_mask, max_len, num_beams, num_knns, use_datastore)
    for i in range(2, max_len):
      tgt_mask = self._generate_square_subsequent_mask(outputs[:, :i].size(1)).type(torch.bool)
      encoder_embeddings = self.model.transformer_model.decode(outputs[:, :i], memory, tgt_mask[:i, :i])

      # encoder_embeddings.shape = (N,S,E) -> encoder_embeddings[:, -1] take (N,E) at last element in axis=1 -> apply data_store
      encoder_embeddings = encoder_embeddings[:, -1]
      prob = self.model.transformer_model.generator(encoder_embeddings)
      if use_datastore:
        knn_distribution = self.apply_data_store(encoder_embeddings, num_knns)
        mt_distribution = F.softmax(prob, dim=-1)
        final_distribution = self.gamma*knn_distribution + (1 - self.gamma)*mt_distribution
      else:
        final_distribution = F.softmax(prob, dim=-1)

      outputs, log_scores = self._choose_topk(outputs, final_distribution, log_scores, i, num_beams)
      finished_sentences = (outputs == self.model.tgt_eos_id).nonzero()
      mark_eos = torch.zeros(num_beams, dtype=torch.int64, device=self.device)
      num_finished_sentences = 0
      for eos_symbol in finished_sentences:
        sentence_ind, eos_location = eos_symbol
        if mark_eos[sentence_ind] == 0:
          mark_eos[sentence_ind] = eos_location
          num_finished_sentences += 1
      
      if num_finished_sentences == num_beams:
        alpha = 0.7
        division = mark_eos.type_as(log_scores)**alpha
        _, chosen_sentence_index = torch.max(log_scores / division, 1)
        chosen_sentence_index = chosen_sentence_index[0]
        break
    
    sentence_length = (outputs[chosen_sentence_index] == self.model.tgt_eos_id).nonzero()
    sentence_length = sentence_length[0] if len(sentence_length) > 0 else -1
    return outputs[chosen_sentence_index][:sentence_length+1]
  
  def beam_translate(self, src_sentence: str, num_beams: int = 10, 
                     use_datastore: bool = True, num_knns: int = 10, 
                     num_token_factor: float = 2.0):
    self.model.transformer_model.eval()
    with torch.no_grad():
      src_encodings = self.model.src_tokenizer.batch_encode_plus([src_sentence], padding=True)
      src_ids = torch.tensor(src_encodings.get('input_ids'))
      num_tokens = src_ids.shape[1]
      src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
      max_len = int(num_tokens*num_token_factor+5)
      tgt_tokens = self.beam_search(src_ids, src_mask, max_len=max_len, num_beams=num_beams, num_knns=num_knns, use_datastore=use_datastore).flatten()
      return self.convert_ids_to_string(self.model.tgt_tokenizer, tgt_tokens.tolist())
