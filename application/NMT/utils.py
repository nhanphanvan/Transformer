
from torch.utils.data import Dataset

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
