
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.data.metrics import bleu_score

from .translate_machine_model import NMTModel

class CustomDataset(Dataset):
    r"""
    Args:
        src_path:
            path to source dataset.
        tgt_path:
            path to target dataset.
        segment (*optional*, default = None):
            if `True`, use segmenter to segment source sentences,
            while `False` is provided, use segmenter to segment target sentences.
        rdrsegmenter (*optional*):
            segmenter to segment sentences. Only use when `segment` != `None`.
    """
    def __init__(self, src_path, tgt_path, segment=None, rdrsegmenter=None):
        with open(src_path, 'r', encoding='utf-8') as file:
            src = file.read().splitlines()
        with open(tgt_path, 'r', encoding='utf-8') as file:
            tgt = file.read().splitlines()
        if segment is not None and rdrsegmenter is not None:
            if segment:
                src = [self._segment_sentence(rdrsegmenter, sentence) for sentence in src]
            else:
                tgt = [self._segment_sentence(rdrsegmenter, sentence) for sentence in tgt]
        self.samples = list(zip(src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @staticmethod
    def _segment_sentence(rdrsegmenter, sentence):
        sentences = rdrsegmenter.tokenize(sentence) 
        sentences = [" ".join(sentence) for sentence in sentences]
        sentence = " ".join(sentences).strip()
        return sentence


def calculate_bleu_score(translate_machine: NMTModel,
                         src_path: str,
                         tgt_path: str,
                         end_index: int = None,
                         num_beams: int = 10,
                         num_knns: int = 10,
                         gamma: float = 0.3,
                         temperature: int = 10, 
                         use_layernorm: bool = True,
                         return_sentence: bool = False):
  
  if not 0 <= gamma <= 1:
    raise ValueError(f'gamma must be between 0 and 1')
  data_iter = CustomDataset(src_path, tgt_path)
  pred_sents = []
  tgt_sents = []
  translate_machine.gamma = gamma
  translate_machine.temperature = temperature
  translate_machine.use_layernorm = use_layernorm
  for src, tgt in tqdm(data_iter[:end_index], desc='Blue score'):
    pred_tgt = translate_machine.beam_translate(src, num_beams=num_beams, num_knns=num_knns)
    pred_sents.append(pred_tgt)
    tgt_sents.append(tgt)
  
  translation_sents = [sent.strip().replace('_', ' ').split() for sent in pred_sents]
  target_sents = [[sent.strip().replace('_', ' ').split()] for sent in tgt_sents]

  bleu = bleu_score(translation_sents, target_sents)
  return (bleu, pred_sents, tgt_sents) if return_sentence else bleu