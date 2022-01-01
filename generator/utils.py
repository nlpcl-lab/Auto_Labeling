from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from datasets import load_metric

bleu = load_metric('bleu')

class DictObj:

    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

class Corpus:
    __tokenizer = None

    @classmethod
    def create_tokenizer(cls, tokenizer):
        cls.__tokenizer = tokenizer
        return cls.__tokenizer

    @classmethod
    def get_tokenizer(cls):
        assert cls.__tokenizer, 'failed to load'
        return cls.__tokenizer

class Custom_Data(Dataset):
    def __init__(self, dataset):
        super(Custom_Data, self).__init__()
        self.src = ["Make question from Passage: " + c + "Generated Question: " for c in dataset['context']]
        self.tgt = dataset['question']

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.src)



class Data(Dataset):
    def __init__(self, dataset_path, data_type: str):
        super(Data, self).__init__()
        filepath = dataset_path + data_type
        self.src = []
        self.tgt = []
        with open(filepath, 'rb') as f:
            _file = f.readlines()
        for line in _file:
            line = line.strip().decode('utf-8')
            line_split = line.split("\t")
            self.src.append(line_split[0].replace("text:",""))
            self.tgt.append(line_split[1].replace("labels:",""))

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.tgt)

def collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_sentences, tgt_sentences = map(list, zip(*batch))
    tokenizer = Corpus.get_tokenizer()
    # batch = tokenizer(src_sentences, padding=False, truncation=False, max_length=512, return_tensors='pt')
    batch = tokenizer(src_sentences)
    with tokenizer.as_target_tokenizer():
        # labels = tokenizer(tgt_sentences, padding=False, truncation=False, max_length=512, return_tensors='pt')
        labels = tokenizer(tgt_sentences)
    batch['labels'] = labels['input_ids']
    # batch['decoder_attention_mask'] = labels['attention_mask']
    batch = {k:v.to(device) for k,v in batch.items()}
    return batch

def _compute_metrics(p):

    preds, label_ids = p
    if isinstance(preds, tuple):
        pred_ids = np.argmax(preds[0],axis=2)
    else:
        pred_ids = preds

    tokenizer = Corpus.get_tokenizer()

    pred_sents = tokenizer.batch_decode(pred_ids)
    label_ids[label_ids == -100] = tokenizer.token_to_id('[PAD]')

    ref_sents = tokenizer.batch_decode(label_ids)
    prediction = [line.split() if len(line) != 0 else "" for line in pred_sents]
    reference = [[line.split()] for line in ref_sents]

    results = [round(bleu.compute(predictions=prediction, references=reference, max_order=i)['bleu']*100,2) for i in range(1,5)]

    return results, pred_sents, ref_sents

def compute_metrics(p):

    results, pred_sents, ref_sents = _compute_metrics(p)

    return {
        'BLEU-4' : results[3]
        # 'results' : zip(pred_sents, ref_sents)
    }