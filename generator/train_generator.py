from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric,load_dataset
import argparse
import numpy as np
import yaml
from utils import DictObj, Corpus, Custom_Data, collate_fn, compute_metrics


def tokenize_function(examples):
    output = tokenizer(examples["passage"])

    return output


if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Training')
    parser.add_argument('--configs','-cf',type=str)

    args=parser.parse_args()
    config_filepath = "../conf/gen_KB.yaml"
    with open(config_filepath) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = DictObj(configs)

    dataset_path = configs.gen_query.gen_path
    model_name = configs.gen_query.model_name
    model_path = configs.gen_query.model_path.format(model_name)
    random_seed = configs.gen_query.random_seed
    batch_size = configs.gen_query.batch_size
    epochs = configs.gen_query.epochs

    torch.manual_seed(random_seed)
    bleu = load_metric('bleu')

    # train_dataset = Data(dataset_path, 'train.txt')
    # eval_dataset = Data(dataset_path, 'valid.txt')
    # test_dataset = Data(dataset_path, 'test.txt')
    raw_dataset = load_dataset("squad_v2")
    train_dataset = Custom_Data(raw_dataset['train'])
    eval_dataset = Custom_Data(raw_dataset['validation'])

    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # special_tokens_dict = {'additional_special_tokens': ['__SUB__','__name__','__type__','__SEP__','__des__','__PRE__','__token__','__OBJ__']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    Corpus.create_tokenizer(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_path,
        save_strategy='epoch',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=epochs,
        seed=random_seed,
        dataloader_pin_memory=False,
        save_total_limit=1
    )

    # training_args = Seq2SeqTrainingArguments(
    #     output_dir = model_path,
    #     save_strategy='epoch',
    #     per_device_train_batch_size=batch_size,
    #     num_train_epochs=epochs,
    #     seed=random_seed,
    #     dataloader_pin_memory=False,
    #     save_total_limit=1
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics
    # )

    train_result = trainer.train()
    trainer.save_model()

    # test_results = trainer.predict(test_dataset, metric_key_prefix='test')