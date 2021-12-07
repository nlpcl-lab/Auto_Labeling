from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch import optim
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_metric
import argparse
import os
import math

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


class Data(Dataset):
    def __init__(self, dataset_path, data_type: str = "train"):
        super(Data, self).__init__()
        filepath = dataset_path + data_type
        self.src = []
        self.tgt = []
        with open(filepath, 'rb') as f:
            _file = f.readlines()
        for line in _file:
            line_split = line.split('\t')
            self.src.append(line_split[0].replace("text:"))
            self.tgt.append(line_split[1].replace("labels:"))

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.tgt)

def collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_sentences, tgt_sentences = map(list, zip(*batch))
    tokenizer = Corpus.get_tokenizer()
    inputs = tokenizer(src_sentences, return_tensors='pt',padding=True)
    outputs = tokenizer.encode_batch(tgt_sentences)
    batch = {
        'input_ids' : inputs.input_ids.to(device),
        'attention_mask' : inputs.attention_mask.to(device),
        'decoder_input_ids' : torch.as_tensor([output.ids for output in outputs]).to(device),
        'decoder_attention_mask' : torch.as_tensor([output.attention_mask for output in outputs]).to(device),
        'labels' : torch.as_tensor([output.ids for output in outputs]).to(device)
    }

    return batch

def _compute_metrics(p):

    preds, label_ids = p
    if isinstance(preds, tuple):
        pred_ids = np.argmax(preds[0],axis=2)
    else:
        pred_ids = preds

    _, gloss_tokenizer = Corpus.get_tokenizer()

    pred_sents = gloss_tokenizer.decode_batch(pred_ids)
    label_ids[label_ids == -100] = gloss_tokenizer.token_to_id('[PAD]')

    ref_sents = gloss_tokenizer.decode_batch(label_ids)
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

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Training')
    parser.add_argument('--random_seed','-rs',type=int,default=42)
    parser.add_argument('--configs','-cf',type=str,default="./conf/baseline.yaml")
    parser.add_argument('--model','-m',type=str,default='transformer')

    args=parser.parse_args()
    random_seed = args.random_seed
    dataset_path = "/home/tjrals/beir/gen_datasets/"
    torch.manual_seed(random_seed)
    bleu = load_metric('bleu')

    batch_size = 32
    epochs = 10

    train_dataset =  Data(dataset_path, 'train.txt')
    eval_dataset = Data(dataset_path, 'valid.txt')
    test_dataset = Data(dataset_path, 'test.txt')

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    special_tokens_dict = {'additional_special_tokens': ['__SUB__','__name__','__type__','__SEP__','__des__','__PRE__','__token__','__OBJ__']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    Corpus.create_tokenizer(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir = "gen_model/{}".format('KB_BART'),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        seed=random_seed,
        dataloader_pin_memory=False,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    trainer.save_model()

    test_results = trainer.predict(test_dataset, metric_key_prefix='test')