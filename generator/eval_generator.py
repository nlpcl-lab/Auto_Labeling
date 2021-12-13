from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
import argparse
import numpy as np
import yaml
from utils import DictObj, Corpus, Data, collate_fn, compute_metrics

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--configs','-cf',type=str)

    args=parser.parse_args()
    config_filepath = "./conf/" + args.configs
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

    test_dataset = Data(dataset_path, 'test.txt')

    model = BartForConditionalGeneration.from_pretrained(model_path, forced_bos_token_id=0)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    # special_tokens_dict = {'additional_special_tokens': ['__SUB__','__name__','__type__','__SEP__','__des__','__PRE__','__token__','__OBJ__']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    Corpus.create_tokenizer(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir = "../gen_model/{}".format('KB_BART'),
        do_predict=True,
        seed=random_seed,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    test_results = trainer.predict(test_dataset, metric_key_prefix='test')

    del model
    del trainer
    torch.cuda.empty_cache()