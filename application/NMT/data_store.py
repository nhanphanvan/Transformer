from typing import Optional

import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Datastore:
  def __init__(self,
               dimension: int,
               size_value_array: int,
               num_centroid: int,
               num_centroid_per_subvector: int = 8,
               num_bits : int = 8,
               nprobe: int = 16,
               load_file: Optional[str] = None):
    
    self.dimension = dimension
    self.size_value_array = size_value_array
    self.num_centroid = num_centroid
    self.num_centroid_per_subvector = num_centroid_per_subvector
    self.num_bits = num_bits
    self.nprobe = nprobe
    if load_file is None:
      quantizer = faiss.IndexFlatL2(dimension)  # we keep the same L2 distance flat index
      self.index = faiss.IndexIVFPQ(quantizer, dimension, num_centroid, num_centroid_per_subvector, num_bits)
    else:
      self.index = faiss.read_index(load_file + ".trained")
      self.index.make_direct_map()

  def build_datastore(self, embeddings):
    if self.index.is_trained:
      raise RuntimeError(f"Datastore is built")
    else:
      self.index.train(embeddings)
      self.index.add(embeddings)
  
  def save_index(self, save_path):
    faiss.write_index(self.index, save_path + ".trained")

  def get_knns(self, query, k):
    distances, k_nearest_indexes = self.index.search(query, k)
    return distances, k_nearest_indexes

  def get_vals_vectors(self, indexes, look_up_array):
    num_queries, k = indexes.shape
    all_vals_vectors = np.zeros((num_queries, k), dtype=np.int64)
    for query_index, ind in enumerate(indexes):
      vals_vectors = np.zeros((k), dtype=np.int64)
      for k_index, id in enumerate(ind):
        val = look_up_array[id]
        vals_vectors[k_index] = val
      all_vals_vectors[query_index] = vals_vectors
    return torch.tensor(all_vals_vectors, dtype=torch.int64)

  def construct_final_distribution(self, all_vals_vectors, all_distances, temperature=10, use_layernorm=False):
    # print(all_vals_vectors)
    # print(all_distances)
    with torch.no_grad():
      num_queries, k = all_vals_vectors.shape
      norm = nn.LayerNorm(k)
      final_vectors = np.zeros((num_queries, self.size_value_array))
      all_distances = torch.tensor(all_distances)
      for index, (vals_vectors, distances) in enumerate(zip(all_vals_vectors, all_distances)):
        if use_layernorm:
          logits = F.softmax(norm(-distances), dim=0)
        else:
          logits = F.softmax(-distances/temperature, dim=0)
        
        result_vector = np.zeros(self.size_value_array)
        for val_id, prob in zip(vals_vectors, logits):
          result_vector[val_id] += prob
        final_vectors[index] = result_vector
      return final_vectors