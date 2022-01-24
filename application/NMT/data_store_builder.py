
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .translate_machine_model import NMTModel
from .utils import CustomDataset

class DatastoreBuilder:
    def __init__(self, model: NMTModel, device: str = 'cpu'):
        self.model = model
        self.device = device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 0).transpose(0, 1)
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.model.config.src_padding_id)
        tgt_padding_mask = (tgt == self.model.config.tgt_padding_id)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def batch_forward_transformer(self, src: torch.Tensor, tgt: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
            logits = self.model.transformer_model(src, 
                                                  tgt_input, 
                                                  src_mask=src_mask, 
                                                  tgt_mask=tgt_mask, 
                                                  src_key_padding_mask=src_padding_mask, 
                                                  tgt_key_padding_mask=tgt_padding_mask, 
                                                  memory_key_padding_mask=src_padding_mask)
        return logits

    @staticmethod
    def get_data_store_length(dataset, collate_fn, end_index=None):
        dataloader = DataLoader(dataset[:end_index], batch_size=1, collate_fn=collate_fn)
        tgt_lengths = 0
        for src, tgt in tqdm(dataloader):
            tgt_lengths += tgt.shape[1] - 2
        return tgt_lengths

    def batch_create_features_file(self, src_path, tgt_path, batch_size=1, end_index=None, is_save=False):
        src_tokenizer = self.model.src_tokenizer
        tgt_tokenizer = self.model.tgt_tokenizer

        # function to collate data samples into batch tesors
        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for src_sample, tgt_sample in batch:
                src_batch.append(src_sample.rstrip("\n"))
                tgt_batch.append(tgt_sample.rstrip("\n"))
            src_encodings = src_tokenizer.batch_encode_plus(src_batch, padding=True)
            src_ids = torch.tensor(src_encodings.get('input_ids'))
            # src_attention_masks = torch.tensor(src_encodings.get('attention_mask'))
            
            tgt_encodings = tgt_tokenizer.batch_encode_plus(tgt_batch, padding=True)
            tgt_ids = torch.tensor(tgt_encodings.get('input_ids'))
            # tgt_attention_masks = torch.tensor(tgt_encodings.get('attention_mask'))
            
            # return (src_ids, src_attention_masks), (tgt_ids, tgt_attention_masks)
            return src_ids, tgt_ids

        dataset_iter = CustomDataset(src_path, tgt_path)
        dataloader = DataLoader(dataset_iter[:end_index], batch_size=batch_size, collate_fn=collate_fn)
        
        total_length = self.get_data_store_length(dataset_iter, collate_fn, end_index=end_index)
        keys = np.zeros((total_length, self.model.config.hidden_size), dtype=np.float32)
        vals = np.zeros((total_length), dtype=np.int64)
        marked_index = 0
        for src, tgt in tqdm(dataloader):
            extracted_feature = self.batch_forward_transformer(src, tgt)
            tgt_lengths = (tgt == self.model.tgt_eos_id).nonzero()[:, 1]
            encoder_outputs = extracted_feature.decoder_hidden_states[-1]
            for index, (encoder_output, tgt_length) in enumerate(zip(encoder_outputs, tgt_lengths)):
                outputs = encoder_output[:tgt_length-1, :].cpu().detach().numpy()
                val_index = tgt[index][1:tgt_length].cpu().detach().numpy()
                keys[marked_index:tgt_length-1+marked_index, :] = outputs
                vals[marked_index:tgt_length-1+marked_index] = val_index
                marked_index += tgt_length-1
        if is_save:
            np.save('keys_embedding', keys)
            np.save('vals_mapping', vals)
        return keys, vals